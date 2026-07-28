from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from taigi_converter import TaigiConverter
from tests.helpers import build_minimal_runtime, make_source_data


def snapshot_tree(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


class RuntimeBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        TaigiConverter._runtime_cache.clear()

    def test_default_runtime_never_writes_read_only_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            before = snapshot_tree(runtime)
            for path in runtime.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            converter = TaigiConverter(runtime)
            self.assertEqual(converter.convert("測試詞"), "試驗詞")
            self.assertEqual(before, snapshot_tree(runtime))

    def test_missing_artifacts_do_not_auto_compile(self) -> None:
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(FileNotFoundError):
            TaigiConverter(Path(temp))

    def test_explicit_auto_prepare_separates_source_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            converter = TaigiConverter(runtime, auto_prepare=True, source_data_dir=source)
            self.assertEqual(converter.convert("測試詞"), "試驗詞")
            self.assertTrue((runtime / "artifacts" / "manifest.json").exists())

    def test_manifest_version_and_checksum_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            manifest_path = runtime / "artifacts" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["v"] = -1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "版本不相容"):
                TaigiConverter(runtime)

    def test_runtime_cache_reuses_immutable_loaded_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            started = time.perf_counter()
            first = TaigiConverter(runtime)
            cold = time.perf_counter() - started
            started = time.perf_counter()
            second = TaigiConverter(runtime)
            warm = time.perf_counter() - started
            self.assertIs(first.entries_by_index, second.entries_by_index)
            self.assertIs(first.phrase_trie, second.phrase_trie)
            self.assertEqual(len(TaigiConverter._runtime_cache), 1)
            self.assertLess(warm, cold)
            self.assertEqual(second.convert("測試詞"), "試驗詞")

    def test_persistent_artifact_checksum_mismatch_fails_with_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            artifact = runtime / "artifacts" / "entry_table.json"
            artifact.write_bytes(artifact.read_bytes() + b" ")
            with (
                patch.object(TaigiConverter, "ARTIFACT_LOAD_RETRIES", 2),
                patch.object(TaigiConverter, "ARTIFACT_RETRY_INTERVAL_SECONDS", 0.001),
                self.assertRaisesRegex(RuntimeError, "checksum"),
            ):
                TaigiConverter(runtime)

    def test_mixed_artifact_generation_is_retried_until_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            artifact = runtime / "artifacts" / "entry_table.json"
            original = artifact.read_bytes()
            artifact.write_bytes(original + b" ")

            def restore() -> None:
                time.sleep(0.02)
                replacement = artifact.with_suffix(".replacement")
                replacement.write_bytes(original)
                os.replace(replacement, artifact)

            worker = threading.Thread(target=restore)
            worker.start()
            try:
                with (
                    patch.object(TaigiConverter, "ARTIFACT_LOAD_RETRIES", 100),
                    patch.object(TaigiConverter, "ARTIFACT_RETRY_INTERVAL_SECONDS", 0.005),
                ):
                    converter = TaigiConverter(runtime)
                self.assertEqual(converter.convert("測試詞"), "試驗詞")
            finally:
                worker.join()

    def test_review_queue_uses_explicit_writable_state_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            review = root / "state"
            build_minimal_runtime(source, runtime)
            converter = TaigiConverter(runtime, review_data_dir=review)
            converter.convert("完全未知內容", profile={"enqueue_review": True, "owner": "test"})
            self.assertTrue((review / "review_queue.jsonl").exists())
            self.assertFalse((runtime / "review_queue.jsonl").exists())

    def test_review_queue_never_falls_back_to_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            converter = TaigiConverter(runtime)
            with self.assertRaisesRegex(RuntimeError, "review_data_dir"):
                converter.convert("完全未知內容", profile={"enqueue_review": True})
            self.assertFalse((runtime / "review_queue.jsonl").exists())

    def test_runtime_cache_does_not_leak_instance_review_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            first = TaigiConverter(runtime, review_data_dir=root / "first")
            second = TaigiConverter(runtime, review_data_dir=root / "second")
            self.assertEqual(first.review_data_dir, root / "first")
            self.assertEqual(second.review_data_dir, root / "second")


if __name__ == "__main__":
    unittest.main()
