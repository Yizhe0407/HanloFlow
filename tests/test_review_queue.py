from __future__ import annotations

import json
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from taigi_converter.review_queue import (
    ReviewCommitDurabilityError,
    _commit_transaction_unlocked,
    _prepare_state_dir,
    append_review_item,
    apply_review_decisions,
    export_pending_reviews,
    load_review_queue,
    load_review_snapshot,
)
from tests.helpers import valid_entry, write_jsonl


def _append_worker(args: tuple[str, int]) -> None:
    data_dir, index = args
    append_review_item(Path(data_dir), {"kind": "test", "sequence": index})


def _append_duplicate_worker(data_dir: str) -> None:
    append_review_item(Path(data_dir), {"kind": "same", "evidence": {"input": "重複"}})


class ReviewQueueTests(unittest.TestCase):
    def test_prepare_state_dir_tolerates_transient_file_removal_during_chmod(self) -> None:
        transient_paths = (
            Path(".review_state/mirror_dirty.json"),
            Path(".review_transaction.json"),
        )
        original_chmod = __import__("os").chmod

        def make_race_chmod(raced_path: Path):
            def race_chmod(path: str | bytes | Path, mode: int) -> None:
                if Path(path) == raced_path:
                    raise FileNotFoundError(path)
                original_chmod(path, mode)

            return race_chmod

        for relative_path in transient_paths:
            with self.subTest(path=relative_path), tempfile.TemporaryDirectory() as temp:
                data_dir = Path(temp)
                raced_path = data_dir / relative_path

                with patch(
                    "taigi_converter.review_queue.os.chmod",
                    side_effect=make_race_chmod(raced_path),
                ):
                    _prepare_state_dir(data_dir)

                self.assertTrue((data_dir / ".review_state" / "generations").is_dir())

    def test_prepare_state_dir_does_not_hide_permission_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            dirty_path = data_dir / ".review_state" / "mirror_dirty.json"
            original_chmod = __import__("os").chmod

            def denied_chmod(path: str | bytes | Path, mode: int) -> None:
                if Path(path) == dirty_path:
                    raise PermissionError(path)
                original_chmod(path, mode)

            with (
                patch(
                    "taigi_converter.review_queue.os.chmod",
                    side_effect=denied_chmod,
                ),
                self.assertRaises(PermissionError),
            ):
                _prepare_state_dir(data_dir)

    def test_cross_process_appends_are_not_lost_or_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            context = multiprocessing.get_context("spawn")
            with context.Pool(4) as pool:
                pool.map(_append_worker, [(temp, index) for index in range(24)])
            rows = load_review_queue(data_dir)
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["sequence"] for row in rows}, set(range(24)))
            self.assertEqual(len({row["review_id"] for row in rows}), 24)

    def test_cross_process_pending_duplicates_are_coalesced_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            context = multiprocessing.get_context("spawn")
            with context.Pool(4) as pool:
                pool.map(_append_duplicate_worker, [temp] * 24)
            rows = load_review_queue(data_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["occurrence_count"], 24)
            self.assertTrue(rows[0]["review_id"].startswith("rq_"))
            self.assertTrue(rows[0]["fingerprint"].startswith("rfp_"))

    def test_pending_duplicate_reuses_canonical_item_but_resolved_item_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            payload = {"kind": "same", "evidence": {"b": 2, "a": 1}}
            first = append_review_item(data_dir, payload)
            second = append_review_item(data_dir, {"evidence": {"a": 1, "b": 2}, "kind": "same"})
            self.assertEqual(second["review_id"], first["review_id"])
            self.assertEqual(second["fingerprint"], first["fingerprint"])
            self.assertEqual(second["occurrence_count"], 2)
            self.assertEqual(len(load_review_queue(data_dir)), 1)

            write_jsonl(data_dir / "lexicon_entries.jsonl", [valid_entry(tier="base")])
            decisions = data_dir / "decisions.jsonl"
            write_jsonl(
                decisions,
                [{"review_id": first["review_id"], "decision": "reject", "owner": "tester"}],
            )
            self.assertEqual(apply_review_decisions(data_dir, decisions)["applied"], 1)

            third = append_review_item(data_dir, payload)
            self.assertNotEqual(third["review_id"], first["review_id"])
            self.assertEqual(third["fingerprint"], first["fingerprint"])
            self.assertEqual(len(load_review_queue(data_dir)), 2)

    def test_pre_phase4_legacy_rows_use_neutral_export_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            write_jsonl(
                data_dir / "review_queue.jsonl",
                [
                    {
                        "review_id": "rq_legacy",
                        "kind": "legacy",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "status": "pending",
                    },
                    {
                        "review_id": "rq_high",
                        "kind": "high",
                        "priority": 75,
                        "created_at": "2026-01-02T00:00:00+00:00",
                        "status": "pending",
                    },
                    {
                        "review_id": "rq_low",
                        "kind": "low",
                        "priority": 25,
                        "created_at": "2026-01-03T00:00:00+00:00",
                        "status": "pending",
                    },
                ],
            )
            output_path = data_dir / "pending.jsonl"

            self.assertEqual(export_pending_reviews(data_dir, output_path), 3)
            exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual([row["kind"] for row in exported], ["high", "legacy", "low"])
            self.assertNotIn("priority", next(row for row in exported if row["kind"] == "legacy"))

    def test_pre_phase4_online_review_dedupes_after_diagnostics_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            legacy = {
                "review_id": "rq_legacy",
                "fingerprint": "rfp_pre_phase4_schema",
                "kind": "online_low_confidence",
                "action": "add_override",
                "owner": "runtime",
                "reason": "auto_enqueued_by_runtime",
                "status": "pending",
                "evidence": {
                    "input": "完全未知內容",
                    "output": "完全未知內容",
                    "warnings": [],
                    "match_count": 0,
                    "match_entry_ids": [],
                },
            }
            write_jsonl(data_dir / "review_queue.jsonl", [legacy])

            upgraded = append_review_item(
                data_dir,
                {
                    "kind": "online_low_confidence",
                    "action": "add_override",
                    "owner": "runtime",
                    "reason": "auto_enqueued_by_runtime",
                    "priority": 78,
                    "evidence": {
                        "input": "完全未知內容",
                        "normalized_input": "完全未知內容",
                        "output": "完全未知內容",
                        "warnings": [],
                        "low_confidence_reasons": [
                            "no_transform_evidence",
                            "sparse_conversion_coverage",
                        ],
                        "confidence_score": 0.15,
                        "review_priority": 78,
                        "matched_span_ratio": 0.0,
                        "identity_ratio": 1.0,
                        "matches": [],
                        "rules_applied": [],
                    },
                },
            )

            rows = load_review_queue(data_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(upgraded["review_id"], "rq_legacy")
            self.assertEqual(upgraded["occurrence_count"], 2)
            self.assertEqual(rows[0]["occurrence_count"], 2)
            self.assertEqual(rows[0]["fingerprint"], upgraded["fingerprint"])

    def test_online_review_identity_keeps_distinct_outputs_and_actions_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            base = {
                "kind": "online_low_confidence",
                "action": "add_override",
                "owner": "runtime",
                "reason": "auto_enqueued_by_runtime",
                "evidence": {"input": "同一輸入", "output": "第一輸出"},
            }
            first = append_review_item(data_dir, base)
            different_output = append_review_item(
                data_dir,
                {**base, "evidence": {"input": "同一輸入", "output": "第二輸出"}},
            )
            different_action = append_review_item(
                data_dir,
                {**base, "action": "reject"},
            )

            rows = load_review_queue(data_dir)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["fingerprint"] for row in rows}), 3)
            self.assertEqual(
                {row["review_id"] for row in rows},
                {
                    first["review_id"],
                    different_output["review_id"],
                    different_action["review_id"],
                },
            )

    def test_legacy_duplicate_ids_and_pending_fingerprints_are_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            write_jsonl(
                data_dir / "review_queue.jsonl",
                [
                    {"review_id": "rq_legacy", "kind": "same", "status": "pending"},
                    {"review_id": "rq_alias", "kind": "same", "status": "pending"},
                    {"review_id": "rq_legacy", "kind": "different", "status": "pending"},
                ],
            )
            rows = load_review_queue(data_dir)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["review_id"] for row in rows}), 2)
            same = next(row for row in rows if row["kind"] == "same")
            self.assertEqual(same["occurrence_count"], 2)
            self.assertIn("rq_alias", same["coalesced_review_ids"])

            write_jsonl(data_dir / "lexicon_entries.jsonl", [valid_entry(tier="base")])
            decisions = data_dir / "decisions.jsonl"
            write_jsonl(decisions, [{"review_id": "rq_alias", "decision": "reject"}])
            self.assertEqual(apply_review_decisions(data_dir, decisions)["applied"], 1)
            resolved = next(row for row in load_review_queue(data_dir) if row["kind"] == "same")
            self.assertEqual(resolved["status"], "resolved")

    def test_decision_batch_updates_queue_lexicon_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            write_jsonl(data_dir / "lexicon_entries.jsonl", [valid_entry(tier="base")])
            item = append_review_item(data_dir, {"kind": "test"})
            decisions = data_dir / "decisions.jsonl"
            write_jsonl(
                decisions,
                [
                    {
                        "review_id": item["review_id"],
                        "decision": "add_override",
                        "src": "新詞",
                        "tgt": "新台語詞",
                        "owner": "tester",
                    }
                ],
            )
            summary = apply_review_decisions(data_dir, decisions)
            self.assertEqual(summary["applied"], 1)
            self.assertEqual(load_review_queue(data_dir)[0]["status"], "resolved")
            lexicon_rows = [json.loads(line) for line in (data_dir / "lexicon_entries.jsonl").read_text().splitlines()]
            self.assertIn("新詞", {row["src"] for row in lexicon_rows})
            audit_rows = [json.loads(line) for line in (data_dir / "review_audit.jsonl").read_text().splitlines()]
            self.assertEqual(audit_rows[0]["review_id"], item["review_id"])
            self.assertFalse((data_dir / ".review_transaction.json").exists())

    def test_invalid_semantic_decision_makes_entire_batch_noop(self) -> None:
        invalid_fields = {
            "level": "bogus",
            "tier": "bogus",
            "status": "bogus",
            "context": {"unknown": "value"},
            "context_regex": {"left_regex": "["},
            "priority": "100",
            "priority_bool": True,
            "score": 1.1,
            "score_type": "1.0",
        }
        for label, invalid_value in invalid_fields.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                data_dir = Path(temp)
                lexicon_path = data_dir / "lexicon_entries.jsonl"
                write_jsonl(lexicon_path, [valid_entry(tier="base")])
                first = append_review_item(data_dir, {"kind": "first"})
                second = append_review_item(data_dir, {"kind": "second"})
                queue_before = (data_dir / "review_queue.jsonl").read_bytes()
                lexicon_before = lexicon_path.read_bytes()
                invalid_key = label.removesuffix("_regex").removesuffix("_bool").removesuffix("_type")
                decisions = data_dir / "decisions.jsonl"
                write_jsonl(
                    decisions,
                    [
                        {
                            "review_id": first["review_id"],
                            "decision": "add_override",
                            "src": "有效詞",
                            "tgt": "有效台語詞",
                        },
                        {
                            "review_id": second["review_id"],
                            "decision": "add_override",
                            "src": "無效詞",
                            "tgt": "無效台語詞",
                            invalid_key: invalid_value,
                        },
                    ],
                )
                summary = apply_review_decisions(data_dir, decisions)
                self.assertEqual(summary["applied"], 0)
                self.assertTrue(summary["errors"])
                self.assertEqual((data_dir / "review_queue.jsonl").read_bytes(), queue_before)
                self.assertEqual(lexicon_path.read_bytes(), lexicon_before)
                self.assertFalse((data_dir / "review_audit.jsonl").exists())

    def test_non_finite_decision_is_rejected_before_any_commit(self) -> None:
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(non_finite=non_finite), tempfile.TemporaryDirectory() as temp:
                data_dir = Path(temp)
                lexicon_path = data_dir / "lexicon_entries.jsonl"
                write_jsonl(lexicon_path, [valid_entry(tier="base")])
                item = append_review_item(data_dir, {"kind": "test"})
                queue_before = (data_dir / "review_queue.jsonl").read_bytes()
                lexicon_before = lexicon_path.read_bytes()
                decisions = data_dir / "decisions.jsonl"
                decisions.write_text(
                    json.dumps(
                        {
                            "review_id": item["review_id"],
                            "decision": "add_override",
                            "src": "詞",
                            "tgt": "台語詞",
                            "score": non_finite,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "非有限數值"):
                    apply_review_decisions(data_dir, decisions)
                self.assertEqual((data_dir / "review_queue.jsonl").read_bytes(), queue_before)
                self.assertEqual(lexicon_path.read_bytes(), lexicon_before)
                self.assertFalse((data_dir / "review_audit.jsonl").exists())

    def test_entry_id_conflict_rejects_entire_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            lexicon_path = data_dir / "lexicon_entries.jsonl"
            write_jsonl(lexicon_path, [valid_entry(entry_id="lx_existing", tier="base")])
            item = append_review_item(data_dir, {"kind": "test"})
            queue_before = (data_dir / "review_queue.jsonl").read_bytes()
            lexicon_before = lexicon_path.read_bytes()
            decisions = data_dir / "decisions.jsonl"
            write_jsonl(
                decisions,
                [
                    {
                        "review_id": item["review_id"],
                        "decision": "add_override",
                        "entry_id": "lx_existing",
                        "src": "另一個詞",
                        "tgt": "另一個台語詞",
                    }
                ],
            )
            summary = apply_review_decisions(data_dir, decisions)
            self.assertEqual(summary["applied"], 0)
            self.assertRegex(summary["errors"][0], "拒絕覆寫")
            self.assertEqual((data_dir / "review_queue.jsonl").read_bytes(), queue_before)
            self.assertEqual(lexicon_path.read_bytes(), lexicon_before)

    def test_snapshot_commit_switches_all_files_at_one_pointer_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            old_files = {
                "review_queue.jsonl": [{"generation_marker": "old"}],
                "lexicon_entries.jsonl": [{**valid_entry(), "generation_marker": "old"}],
                "review_audit.jsonl": [{"generation_marker": "old"}],
            }
            for name, rows in old_files.items():
                write_jsonl(data_dir / name, rows)
            old_snapshot = load_review_snapshot(data_dir)

            new_files = {
                "review_queue.jsonl": [{"generation_marker": "new"}],
                "lexicon_entries.jsonl": [{**valid_entry(), "generation_marker": "new"}],
                "review_audit.jsonl": [{"generation_marker": "new"}],
            }
            pointer_ready = threading.Event()
            release_pointer = threading.Event()
            from taigi_converter import review_queue

            real_publish = review_queue._publish_pointer

            def paused_publish(path: Path, generation: str, digest: str) -> None:
                pointer_ready.set()
                self.assertTrue(release_pointer.wait(timeout=5))
                real_publish(path, generation, digest)

            error: list[BaseException] = []

            def writer() -> None:
                try:
                    _commit_transaction_unlocked(data_dir, new_files)
                except BaseException as exc:  # pragma: no cover - asserted below
                    error.append(exc)

            with patch(
                "taigi_converter.review_queue._publish_pointer",
                side_effect=paused_publish,
            ):
                thread = threading.Thread(target=writer)
                thread.start()
                self.assertTrue(pointer_ready.wait(timeout=5))
                during = load_review_snapshot(data_dir)
                self.assertEqual(during["generation"], old_snapshot["generation"])
                self.assertEqual(
                    {rows[0]["generation_marker"] for rows in during["files"].values()},
                    {"old"},
                )
                release_pointer.set()
                thread.join(timeout=5)

            self.assertFalse(thread.is_alive())
            self.assertEqual(error, [])
            after = load_review_snapshot(data_dir)
            self.assertNotEqual(after["generation"], old_snapshot["generation"])
            self.assertEqual(
                {rows[0]["generation_marker"] for rows in after["files"].values()},
                {"new"},
            )

    def test_failed_pointer_publish_leaves_previous_snapshot_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            first = append_review_item(data_dir, {"kind": "old"})
            before = load_review_snapshot(data_dir)
            new_files = {
                "review_queue.jsonl": [{"kind": "new"}],
                "lexicon_entries.jsonl": [],
                "review_audit.jsonl": [],
            }
            with (
                patch(
                    "taigi_converter.review_queue._publish_pointer",
                    side_effect=OSError("simulated pointer failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated pointer failure"),
            ):
                _commit_transaction_unlocked(data_dir, new_files)

            after = load_review_snapshot(data_dir)
            self.assertEqual(after["generation"], before["generation"])
            self.assertEqual(after["files"]["review_queue.jsonl"][0]["review_id"], first["review_id"])

    def test_mirror_failure_does_not_undo_committed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            load_review_snapshot(data_dir)
            new_files = {
                "review_queue.jsonl": [{"generation_marker": "committed"}],
                "lexicon_entries.jsonl": [],
                "review_audit.jsonl": [],
            }
            with patch(
                "taigi_converter.review_queue._sync_flat_mirrors",
                side_effect=OSError("simulated mirror failure"),
            ):
                _commit_transaction_unlocked(data_dir, new_files)
            snapshot = load_review_snapshot(data_dir)
            self.assertEqual(
                snapshot["files"]["review_queue.jsonl"][0]["generation_marker"],
                "committed",
            )

    def test_partial_mirror_failure_is_marked_and_repaired_on_next_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            load_review_snapshot(data_dir)
            new_files = {
                "review_queue.jsonl": [{"generation_marker": "new"}],
                "lexicon_entries.jsonl": [{**valid_entry(), "generation_marker": "new"}],
                "review_audit.jsonl": [{"generation_marker": "new"}],
            }
            from taigi_converter import review_queue

            real_write = review_queue.atomic_write_text
            call_count = 0

            def fail_second_write(path: Path, text: str, *, mode: int | None = None) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated partial mirror failure")
                real_write(path, text, mode=mode)

            with patch(
                "taigi_converter.review_queue.atomic_write_text",
                side_effect=fail_second_write,
            ):
                _commit_transaction_unlocked(data_dir, new_files)

            dirty_path = data_dir / ".review_state" / "mirror_dirty.json"
            self.assertTrue(dirty_path.exists())
            snapshot = load_review_snapshot(data_dir)
            self.assertFalse(dirty_path.exists())
            generation_dir = data_dir / ".review_state" / "generations" / snapshot["generation"]
            for name in new_files:
                self.assertEqual(
                    (data_dir / name).read_bytes(),
                    (generation_dir / name).read_bytes(),
                )

    def test_pointer_post_replace_failure_reports_committed_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            load_review_snapshot(data_dir)
            new_files = {
                "review_queue.jsonl": [{"generation_marker": "committed"}],
                "lexicon_entries.jsonl": [],
                "review_audit.jsonl": [],
            }
            from taigi_converter import review_queue

            real_publish = review_queue._publish_pointer

            def publish_then_fail(path: Path, generation: str, digest: str) -> None:
                real_publish(path, generation, digest)
                raise OSError("simulated post-replace fsync failure")

            with (
                patch(
                    "taigi_converter.review_queue._publish_pointer",
                    side_effect=publish_then_fail,
                ),
                self.assertRaisesRegex(
                    ReviewCommitDurabilityError,
                    "已切換",
                ) as raised,
            ):
                _commit_transaction_unlocked(data_dir, new_files)
            self.assertTrue(raised.exception.committed)
            snapshot = load_review_snapshot(data_dir)
            self.assertEqual(
                snapshot["files"]["review_queue.jsonl"][0]["generation_marker"],
                "committed",
            )

    def test_matching_legacy_journal_is_cleaned_after_generation_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            snapshot = load_review_snapshot(data_dir)
            journal_path = data_dir / ".review_transaction.json"
            journal_path.write_text(
                json.dumps({"version": 1, "files": snapshot["files"]}),
                encoding="utf-8",
            )
            loaded = load_review_snapshot(data_dir)
            self.assertEqual(loaded["generation"], snapshot["generation"])
            self.assertFalse(journal_path.exists())

    def test_legacy_journal_recovery_is_idempotent_after_commit_before_unlink_crash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            journal_path = data_dir / ".review_transaction.json"
            journal_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "files": {
                            "review_queue.jsonl": [
                                {
                                    "review_id": "rq_recovery",
                                    "kind": "legacy",
                                    "status": "pending",
                                }
                            ],
                            "lexicon_entries.jsonl": [],
                            "review_audit.jsonl": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            from taigi_converter import review_queue

            real_unlink = review_queue.durable_unlink

            def fail_journal_unlink(path: Path, *, missing_ok: bool = False) -> None:
                if path.name == ".review_transaction.json":
                    raise OSError("simulated crash before journal unlink")
                real_unlink(path, missing_ok=missing_ok)

            with (
                patch(
                    "taigi_converter.review_queue.durable_unlink",
                    side_effect=fail_journal_unlink,
                ),
                self.assertRaisesRegex(OSError, "simulated crash"),
            ):
                load_review_snapshot(data_dir)

            recovered = load_review_snapshot(data_dir)
            self.assertFalse(journal_path.exists())
            self.assertEqual(
                recovered["files"]["review_queue.jsonl"][0]["review_id"],
                "rq_recovery",
            )

    def test_conflicting_legacy_journal_cannot_overwrite_current_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            before = load_review_snapshot(data_dir)
            journal_path = data_dir / ".review_transaction.json"
            journal_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "files": {
                            "review_queue.jsonl": [{"kind": "stale"}],
                            "lexicon_entries.jsonl": [],
                            "review_audit.jsonl": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "拒絕覆蓋"):
                load_review_snapshot(data_dir)
            journal_path.unlink()
            after = load_review_snapshot(data_dir)
            self.assertEqual(after["generation"], before["generation"])
            self.assertEqual(after["files"], before["files"])

    def test_legacy_flat_files_are_migrated_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            write_jsonl(data_dir / "review_queue.jsonl", [{"kind": "original"}])
            first = load_review_snapshot(data_dir)
            write_jsonl(data_dir / "review_queue.jsonl", [{"kind": "stale-flat-edit"}])
            second = load_review_snapshot(data_dir)
            self.assertEqual(second["generation"], first["generation"])
            self.assertEqual(second["files"]["review_queue.jsonl"][0]["kind"], "original")

    def test_snapshot_checksum_corruption_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            snapshot = load_review_snapshot(data_dir)
            queue_path = data_dir / ".review_state" / "generations" / snapshot["generation"] / "review_queue.jsonl"
            queue_path.chmod(0o600)
            queue_path.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                load_review_snapshot(data_dir)

    def test_snapshot_pointer_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            load_review_snapshot(data_dir)
            pointer_path = data_dir / ".review_state" / "current.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["generation"] = "gen_../../outside"
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pointer 格式錯誤"):
                load_review_snapshot(data_dir)

    def test_interrupted_transaction_is_recovered_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            queue_rows = [{"review_id": "rq_recovered", "kind": "recovered", "status": "pending"}]
            journal = {
                "version": 1,
                "files": {
                    "review_queue.jsonl": queue_rows,
                    "lexicon_entries.jsonl": [valid_entry()],
                    "review_audit.jsonl": [],
                },
            }
            (data_dir / ".review_transaction.json").write_text(json.dumps(journal), encoding="utf-8")
            recovered = load_review_queue(data_dir)
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["review_id"], "rq_recovered")
            self.assertEqual(recovered[0]["kind"], "recovered")
            self.assertTrue(recovered[0]["fingerprint"].startswith("rfp_"))
            self.assertFalse((data_dir / ".review_transaction.json").exists())


if __name__ == "__main__":
    unittest.main()
