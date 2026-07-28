from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from taigi_converter.review_queue import (
    append_review_item,
    apply_review_decisions,
    load_review_queue,
)
from tests.helpers import valid_entry, write_jsonl


def _append_worker(args: tuple[str, int]) -> None:
    data_dir, index = args
    append_review_item(Path(data_dir), {"kind": "test", "sequence": index})


class ReviewQueueTests(unittest.TestCase):
    def test_cross_process_appends_are_not_lost_or_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            context = multiprocessing.get_context("spawn")
            with context.Pool(4) as pool:
                pool.map(_append_worker, [(temp, index) for index in range(24)])
            rows = load_review_queue(data_dir)
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["sequence"] for row in rows}, set(range(24)))

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

    def test_interrupted_transaction_is_recovered_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            queue_rows = [{"review_id": "rq_recovered", "status": "pending"}]
            journal = {
                "version": 1,
                "files": {
                    "review_queue.jsonl": queue_rows,
                    "lexicon_entries.jsonl": [valid_entry()],
                    "review_audit.jsonl": [],
                },
            }
            (data_dir / ".review_transaction.json").write_text(json.dumps(journal), encoding="utf-8")
            self.assertEqual(load_review_queue(data_dir), queue_rows)
            self.assertFalse((data_dir / ".review_transaction.json").exists())


if __name__ == "__main__":
    unittest.main()
