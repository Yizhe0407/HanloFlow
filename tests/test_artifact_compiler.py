from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from taigi_converter.artifact_compiler import (
    _source_digest,
    compile_runtime_artifacts,
    ensure_runtime_ready,
)
from tests.helpers import make_source_data, valid_entry, write_jsonl


class ArtifactCompilerTests(unittest.TestCase):
    def test_artifacts_are_deterministic_and_checksummed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            out_a = root / "a"
            out_b = root / "b"
            manifest_a = compile_runtime_artifacts(source, output_data_dir=out_a)
            manifest_b = compile_runtime_artifacts(source, output_data_dir=out_b)
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["source_digest"], _source_digest(source))
            self.assertEqual(
                sorted(path.name for path in (out_a / "artifacts").iterdir()),
                sorted(path.name for path in (out_b / "artifacts").iterdir()),
            )
            for path_a in (out_a / "artifacts").iterdir():
                self.assertEqual(path_a.read_bytes(), (out_b / "artifacts" / path_a.name).read_bytes())

    def test_invalid_schema_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source")
            invalid = valid_entry()
            invalid.pop("trust")
            write_jsonl(source / "lexicon_entries.jsonl", [invalid])
            with self.assertRaisesRegex(ValueError, "schema"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_duplicate_entry_id_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry()
            source = make_source_data(Path(temp) / "source", entries=[entry, {**entry, "src": "其他詞"}])
            with self.assertRaisesRegex(ValueError, "重複 entry_id"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_same_rank_conflicting_targets_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = valid_entry(entry_id="lx_conflict0001", src="台鐵", tgt="台鐵")
            second = valid_entry(entry_id="lx_conflict0002", src="台鐵", tgt="臺鐵")
            source = make_source_data(Path(temp) / "source", entries=[first, second])
            with self.assertRaisesRegex(ValueError, "同順位"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_invalid_context_regex_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry()
            entry["context"] = {"right_regex": "["}
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "context.right_regex regex"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_non_integer_priority_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry()
            entry["priority"] = 1.5
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "priority 必須是整數"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_cross_tier_same_runtime_rank_conflict_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = valid_entry(entry_id="lx_conflict0001", src="同詞", tgt="目標甲", tier="manual")
            second = valid_entry(entry_id="lx_conflict0002", src="同詞", tgt="目標乙", tier="manual_hotfix")
            source = make_source_data(Path(temp) / "source", entries=[first, second])
            with self.assertRaisesRegex(ValueError, "同順位"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_ensure_runtime_ready_repairs_corrupted_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            compile_runtime_artifacts(source, output_data_dir=runtime)
            artifact = runtime / "artifacts" / "entry_table.json"
            expected = artifact.read_bytes()
            artifact.write_bytes(expected + b" ")
            ensure_runtime_ready(source, output_data_dir=runtime)
            self.assertEqual(artifact.read_bytes(), expected)

    def test_ensure_runtime_ready_repairs_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            compile_runtime_artifacts(source, output_data_dir=runtime)
            manifest = runtime / "artifacts" / "manifest.json"
            manifest.write_text("{invalid", encoding="utf-8")
            rebuilt = ensure_runtime_ready(source, output_data_dir=runtime)
            self.assertEqual(rebuilt["version"], 3)


if __name__ == "__main__":
    unittest.main()
