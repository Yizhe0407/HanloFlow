from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from taigi_converter.review_queue import (
    append_review_item,
    apply_review_decisions,
    load_review_snapshot,
)
from tests.helpers import valid_entry, write_jsonl


@unittest.skipUnless(os.name == "posix", "POSIX mode bits only")
class ReviewPermissionTests(unittest.TestCase):
    def assertMode(self, path: Path, expected: int) -> None:  # noqa: N802
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected, str(path))

    def test_review_state_and_files_are_private_after_append_and_atomic_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "state"
            data_dir.mkdir(mode=0o755)
            queue_path = data_dir / "review_queue.jsonl"
            queue_path.write_text("", encoding="utf-8")
            queue_path.chmod(0o644)

            item = append_review_item(data_dir, {"kind": "private"})
            self.assertMode(data_dir, 0o700)
            self.assertMode(queue_path, 0o600)
            self.assertMode(data_dir / ".review_queue.lock", 0o600)

            write_jsonl(data_dir / "lexicon_entries.jsonl", [valid_entry(tier="base")])
            decisions = data_dir / "decisions.jsonl"
            write_jsonl(
                decisions,
                [{"review_id": item["review_id"], "decision": "reject", "owner": "tester"}],
            )
            self.assertEqual(apply_review_decisions(data_dir, decisions)["applied"], 1)
            self.assertMode(queue_path, 0o600)
            self.assertMode(data_dir / "review_audit.jsonl", 0o600)
            self.assertMode(data_dir / ".review_queue.lock", 0o600)

            snapshot = load_review_snapshot(data_dir)
            state_dir = data_dir / ".review_state"
            generation_dir = state_dir / "generations" / snapshot["generation"]
            self.assertMode(state_dir, 0o700)
            self.assertMode(state_dir / "generations", 0o700)
            self.assertMode(generation_dir, 0o500)
            self.assertMode(state_dir / "current.json", 0o600)
            for name in (
                "manifest.json",
                "review_queue.jsonl",
                "lexicon_entries.jsonl",
                "review_audit.jsonl",
            ):
                self.assertMode(generation_dir / name, 0o400)

    def test_transaction_journal_is_private_even_if_recovery_is_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            journal_path = data_dir / ".review_transaction.json"
            journal_path.write_text(
                '{"version":1,"files":{"review_queue.jsonl":[],"lexicon_entries.jsonl":[],"review_audit.jsonl":[]}}',
                encoding="utf-8",
            )
            journal_path.chmod(0o644)
            with (
                patch(
                    "taigi_converter.review_queue._commit_transaction_unlocked",
                    side_effect=RuntimeError("simulated interruption"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                load_review_snapshot(data_dir)
            self.assertMode(journal_path, 0o600)


if __name__ == "__main__":
    unittest.main()
