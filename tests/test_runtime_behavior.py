from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from taigi_converter import TaigiConverter
from taigi_converter.models import RuntimeLexiconEntry
from tests.helpers import build_minimal_runtime, make_source_data, valid_entry


def snapshot_tree(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def rewrite_artifact(runtime: Path, filename: str, document: object) -> None:
    artifact = runtime / "artifacts" / filename
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    artifact.write_bytes(payload)

    manifest_path = runtime / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ah"][filename] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


class RuntimeBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        TaigiConverter.clear_runtime_cache()

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

    def test_number_bearing_lexicon_term_is_preserved_until_phrase_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(
                root / "source",
                entries=[
                    valid_entry(
                        entry_id="lx_numericname01",
                        src="台北101",
                        tgt="臺北101",
                        trust="ai_reviewed",
                    )
                ],
            )
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            converter = TaigiConverter(runtime)

            self.assertEqual(converter.convert("去臺北101"), "去臺北101")
            self.assertEqual(converter.convert("價格101元"), "價格一百零一元")

    def test_context_predicate_sees_original_text_across_protected_span(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protected_identity = valid_entry(
                entry_id="lx_protectedctx1",
                src="同事",
                tgt="同事",
            )
            contextual = valid_entry(
                entry_id="lx_protectedctx2",
                src="拜訪",
                tgt="探訪",
            )
            contextual["context"] = {"right_literal": "同事"}
            source = make_source_data(
                root / "source",
                entries=[protected_identity, contextual],
            )
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            converter = TaigiConverter(runtime)

            self.assertEqual(converter.convert("拜訪同事"), "探訪同事")
            self.assertEqual(converter.convert("拜訪朋友"), "拜訪朋友")

    def test_higher_priority_specific_context_beats_broader_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            broad = valid_entry(
                entry_id="lx_contextpriority1",
                src="找",
                tgt="揣",
                priority=100,
                trust="human",
            )
            broad["context"] = {"right_regex": "^我"}
            specific = valid_entry(
                entry_id="lx_contextpriority2",
                src="找",
                tgt="找",
                priority=200,
                trust="ai_reviewed",
            )
            specific["context"] = {"right_regex": "^我[〇零一二三四五六七八九十百千萬億兩0-9]+元"}
            source = make_source_data(root / "source", entries=[broad, specific])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            converter = TaigiConverter(runtime)

            self.assertEqual(converter.convert("找我120元"), "找我一百二十元")
            self.assertEqual(converter.convert("找我一下"), "揣我一下")

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

    def test_runtime_loader_enforces_canonical_artifact_document_schema(self) -> None:
        cases = (
            ("rule_plan.json", lambda document: document.pop("pt")),
            ("rule_plan.json", lambda document: document.__setitem__("protected", {})),
            ("phrase_trie.json", lambda document: document.__setitem__("t", [])),
            ("entry_table.json", lambda document: document.__setitem__("i", [])),
        )
        for filename, mutate in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = make_source_data(root / "source")
                runtime = root / "runtime"
                build_minimal_runtime(source, runtime)

                artifact = runtime / "artifacts" / filename
                document = json.loads(artifact.read_text(encoding="utf-8"))
                mutate(document)
                rewrite_artifact(runtime, filename, document)

                with self.assertRaisesRegex(ValueError, "artifact schema contract 驗證失敗"):
                    TaigiConverter(runtime)

    def test_runtime_loader_rejects_compact_phrase_edge_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "phrase_trie.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["t"] = {
                "測試詞": {"": [0]},
                "測驗詞": {"": [0]},
            }
            rewrite_artifact(runtime, "phrase_trie.json", document)

            with self.assertRaisesRegex(ValueError, "phrase trie edge collision"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_duplicate_compact_entry_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = [
                {
                    "entry_id": "lx_duplicate001",
                    "src": "測試甲",
                    "tgt": "試驗甲",
                    "level": "phrase",
                    "tier": "manual",
                    "priority": 100,
                    "context": None,
                    "score": 1.0,
                    "status": "active",
                    "source": "test",
                    "trust": "human",
                    "updated_by": "test",
                    "updated_at": "2026-07-28T00:00:00+08:00",
                },
                {
                    "entry_id": "lx_duplicate002",
                    "src": "測試乙",
                    "tgt": "試驗乙",
                    "level": "phrase",
                    "tier": "manual",
                    "priority": 100,
                    "context": None,
                    "score": 1.0,
                    "status": "active",
                    "source": "test",
                    "trust": "human",
                    "updated_by": "test",
                    "updated_at": "2026-07-28T00:00:00+08:00",
                },
            ]
            source = make_source_data(root / "source", entries=entries)
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "entry_table.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["ix"] = {"0": "lx_duplicate001", "1": "lx_duplicate001"}
            rewrite_artifact(runtime, "entry_table.json", document)

            with self.assertRaisesRegex(ValueError, "duplicate entry ids"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_malformed_compact_rule_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rule = {
                "rule_id": "rl_test00000001",
                "pass_name": "normalization",
                "type": "literal",
                "pattern": "測試",
                "replacement": "試驗",
                "priority": 10,
                "enabled": True,
                "note": "test",
            }
            source = make_source_data(root / "source", rules=[rule])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "rule_plan.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["r"][0][0].append("unexpected")
            rewrite_artifact(runtime, "rule_plan.json", document)

            with self.assertRaisesRegex(ValueError, "Invalid compact runtime rule row"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_duplicate_rule_plan_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "rule_plan.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["rt"].append(document["rt"][0])
            rewrite_artifact(runtime, "rule_plan.json", document)

            with self.assertRaisesRegex(ValueError, "residual terms 含重複項目"):
                TaigiConverter(runtime)

    def test_runtime_context_decoder_rejects_malformed_payloads(self) -> None:
        invalid_payloads = (
            {},
            [],
            ["unknown", "value"],
            ["r", ""],
            ["l", ""],
            {"unknown": "value"},
            {"right_regex": "^$", "unknown": "value"},
            {"right_regex": ""},
            {"right_regex": "["},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "Invalid runtime context"):
                TaigiConverter._decode_runtime_context(payload)

        for payload in ({}, {"unknown": "value"}, {"right_regex": "["}):
            with self.subTest(model_payload=payload), self.assertRaisesRegex(ValueError, "Invalid runtime context"):
                RuntimeLexiconEntry(
                    entry_id="lx_badcontext01",
                    src="有空",
                    tgt="有閒",
                    level="phrase",
                    tier="base",
                    context=payload,
                )

    def test_runtime_loader_rejects_context_that_would_fail_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_contextguard1",
                "src": "有空",
                "tgt": "有閒",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": {"right_regex": "^後"},
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "entry_table.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["e"][0][5] = {"unknown": "value"}
            payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
            artifact.write_bytes(payload)

            manifest_path = runtime / "artifacts" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["ah"]["entry_table.json"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid runtime context.*未知欄位"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_explicit_null_context_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_contextguard3",
                "src": "有空",
                "tgt": "有閒",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": {"right_regex": "^後"},
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "entry_table.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["e"][0][5] = None
            rewrite_artifact(runtime, "entry_table.json", document)

            with self.assertRaisesRegex(ValueError, "explicit null context"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_oversized_compact_entry_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "entry_table.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["e"][0].extend([None] * (7 - len(document["e"][0])))
            rewrite_artifact(runtime, "entry_table.json", document)

            with self.assertRaisesRegex(ValueError, "Invalid compact entry row"):
                TaigiConverter(runtime)

    def test_runtime_reference_contract_rejects_contextless_contextual_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_contextguard4",
                "src": "有空",
                "tgt": "有閒",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": None,
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "override_index.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["c"] = [0]
            rewrite_artifact(runtime, "override_index.json", document)

            with self.assertRaisesRegex(ValueError, "contextual override 引用不相容詞條"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_invalid_phrase_reference_without_lossy_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "phrase_trie.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["t"] = {"測試詞": {"": [0, 999_999]}}
            rewrite_artifact(runtime, "phrase_trie.json", document)

            with self.assertRaisesRegex(ValueError, "reference out of range"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_invalid_char_reference_without_lossy_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_charref00001",
                "src": "測",
                "tgt": "試",
                "level": "char",
                "tier": "base",
                "priority": 10,
                "context": None,
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "seed",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "char_map.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["m"]["測"] = [0, 999_999]
            rewrite_artifact(runtime, "char_map.json", document)

            with self.assertRaisesRegex(ValueError, "reference out of range"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_invalid_contextual_reference_without_lossy_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_contextref01",
                "src": "有空",
                "tgt": "有閒",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": {"right_regex": "^後"},
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "override_index.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["c"].append(999_999)
            rewrite_artifact(runtime, "override_index.json", document)

            with self.assertRaisesRegex(ValueError, "reference out of range"):
                TaigiConverter(runtime)

    def test_runtime_loader_rejects_invalid_sentence_reference_without_lossy_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_sentenceref01",
                "src": "完整句。",
                "tgt": "規个句。",
                "level": "sentence",
                "tier": "manual",
                "priority": 100,
                "context": None,
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "override_index.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["s"]["完整句。"] = [0, 999_999]
            rewrite_artifact(runtime, "override_index.json", document)

            with self.assertRaisesRegex(ValueError, "reference out of range"):
                TaigiConverter(runtime)

    def test_runtime_reference_contract_rejects_duplicate_sentence_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_sentencedup01",
                "src": "完整句。",
                "tgt": "規个句。",
                "level": "sentence",
                "tier": "manual",
                "priority": 100,
                "context": None,
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "override_index.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["s"]["完整句。"] = [0, 0]
            rewrite_artifact(runtime, "override_index.json", document)

            with self.assertRaisesRegex(ValueError, "sentence override index 含重複引用"):
                TaigiConverter(runtime)

    def test_runtime_reference_contract_rejects_context_removed_with_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_contextguard5",
                "src": "有空",
                "tgt": "有閒",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": {"right_regex": "^後"},
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            entry_artifact = runtime / "artifacts" / "entry_table.json"
            entry_document = json.loads(entry_artifact.read_text(encoding="utf-8"))
            entry_document["e"][0] = entry_document["e"][0][:5]
            rewrite_artifact(runtime, "entry_table.json", entry_document)

            override_artifact = runtime / "artifacts" / "override_index.json"
            override_document = json.loads(override_artifact.read_text(encoding="utf-8"))
            override_document["c"] = []
            rewrite_artifact(runtime, "override_index.json", override_document)

            with self.assertRaisesRegex(ValueError, "phrase trie 與 runtime entries 不一致"):
                TaigiConverter(runtime)

    def test_runtime_reference_contract_rejects_phrase_at_wrong_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "phrase_trie.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["t"] = {"錯誤路徑": {"": [0]}}
            rewrite_artifact(runtime, "phrase_trie.json", document)

            with self.assertRaisesRegex(ValueError, "phrase trie 路徑與詞條來源不一致"):
                TaigiConverter(runtime)

    def test_runtime_reference_contract_rejects_missing_char_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_charguard001",
                "src": "測",
                "tgt": "試",
                "level": "char",
                "tier": "base",
                "priority": 10,
                "context": None,
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "seed",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "char_map.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["m"] = {}
            rewrite_artifact(runtime, "char_map.json", document)

            with self.assertRaisesRegex(ValueError, "char map 與 runtime entries 不一致"):
                TaigiConverter(runtime)

    def test_runtime_reference_contract_rejects_context_entry_in_phrase_trie(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_contextguard2",
                "src": "有空",
                "tgt": "有閒",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": {"right_regex": "^後"},
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)

            artifact = runtime / "artifacts" / "phrase_trie.json"
            document = json.loads(artifact.read_text(encoding="utf-8"))
            document["t"] = {"有空": {"": [0]}}
            payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
            artifact.write_bytes(payload)

            manifest_path = runtime / "artifacts" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["ah"]["phrase_trie.json"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "phrase trie 不可引用 context"):
                TaigiConverter(runtime)

    def test_context_match_without_context_fails_closed(self) -> None:
        self.assertFalse(TaigiConverter._context_match("有空", 0, 2, None))

    def test_blocked_phrase_reserves_span_during_single_char_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blocked = valid_entry(
                entry_id="lx_blocked00001",
                src="東西",
                tgt="東西",
                level="sentence",
                tier="blocked",
                priority=1000,
            )
            single_char_phrase = valid_entry(
                entry_id="lx_singlechar01",
                src="東",
                tgt="方",
                tier="core",
            )
            source = make_source_data(
                root / "source", entries=[blocked, single_char_phrase]
            )
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            converter = TaigiConverter(runtime)

            self.assertEqual(converter.convert("東門"), "方門")
            self.assertEqual(converter.convert("東西"), "東西")
            traced = converter.convert("東西", trace=True)
            self.assertIn("blocked:lx_blocked00001:東西", traced.warnings)

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
            self.assertIs(first.single_char_phrase_map, second.single_char_phrase_map)
            self.assertIs(
                first.contextual_entry_indexes_by_first_char,
                second.contextual_entry_indexes_by_first_char,
            )
            self.assertEqual(len(TaigiConverter._runtime_cache), 1)
            self.assertLess(warm, cold)
            self.assertEqual(second.convert("測試詞"), "試驗詞")

    def test_cached_runtime_is_deeply_immutable_and_mutation_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = {
                "entry_id": "lx_test00000001",
                "src": "測試詞",
                "tgt": "試驗詞",
                "level": "phrase",
                "tier": "manual",
                "priority": 100,
                "context": {"right_regex": "(?:$|[。])"},
                "score": 1.0,
                "status": "active",
                "source": "test",
                "trust": "human",
                "updated_by": "test",
                "updated_at": "2026-07-28T00:00:00+08:00",
            }
            source = make_source_data(root / "source", entries=[entry])
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            first = TaigiConverter(runtime)
            second = TaigiConverter(runtime)
            runtime_entry = first.entries["lx_test00000001"]

            with self.assertRaises(FrozenInstanceError):
                runtime_entry.tgt = "[MUTATED]"  # type: ignore[misc]
            with self.assertRaises(TypeError):
                first.entries["new"] = runtime_entry  # type: ignore[index]
            entry_index = first.entry_index_by_id["lx_test00000001"]
            self.assertEqual(first.contextual_entry_indexes_by_first_char["測"], (entry_index,))
            with self.assertRaises(TypeError):
                first.contextual_entry_indexes_by_first_char["新"] = (entry_index,)  # type: ignore[index]
            assert runtime_entry.context is not None
            with self.assertRaises(TypeError):
                runtime_entry.context["right_regex"] = ".*"  # type: ignore[index]
            with self.assertRaises(AttributeError):
                first.phrase_trie["c"].clear()
            with self.assertRaises(TypeError):
                first.single_char_phrase_map["測"] = (entry_index,)  # type: ignore[index]

            first.entries = {}
            self.assertEqual(second.convert("測試詞。"), "試驗詞。")
            third = TaigiConverter(runtime)
            self.assertEqual(third.convert("測試詞。"), "試驗詞。")
            self.assertIs(second._runtime, third._runtime)

    def test_same_root_cold_start_loads_artifacts_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            TaigiConverter.clear_runtime_cache()
            load_count = 0
            count_lock = threading.Lock()
            start_barrier = threading.Barrier(8)
            original_load = TaigiConverter._load_artifacts

            def counted_load(instance: TaigiConverter, manifest_bytes: bytes) -> None:
                nonlocal load_count
                with count_lock:
                    load_count += 1
                time.sleep(0.02)
                original_load(instance, manifest_bytes)

            def construct() -> TaigiConverter:
                start_barrier.wait()
                return TaigiConverter(runtime)

            with (
                patch.object(TaigiConverter, "_load_artifacts", counted_load),
                ThreadPoolExecutor(max_workers=8) as executor,
            ):
                converters = list(executor.map(lambda _: construct(), range(8)))

            self.assertEqual(load_count, 1)
            self.assertTrue(all(item.convert("測試詞") == "試驗詞" for item in converters))
            self.assertTrue(all(item._runtime is converters[0]._runtime for item in converters))
            self.assertEqual(TaigiConverter.runtime_cache_info()["loads_in_progress"], 0)

    def test_single_flight_wait_has_bounded_timeout_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source_data(root / "source")
            runtime = root / "runtime"
            build_minimal_runtime(source, runtime)
            artifact_root = str((runtime / "artifacts").resolve())
            stalled = threading.Event()
            with TaigiConverter._runtime_cache_lock:
                TaigiConverter._runtime_load_events[artifact_root] = stalled
            try:
                with (
                    patch.object(TaigiConverter, "RUNTIME_LOAD_WAIT_TIMEOUT_SECONDS", 0.001),
                    self.assertRaisesRegex(RuntimeError, "single-flight.*artifact_root"),
                ):
                    TaigiConverter(runtime)
            finally:
                with TaigiConverter._runtime_cache_lock:
                    TaigiConverter._runtime_load_events.pop(artifact_root, None)

    def test_runtime_cache_is_bounded_and_clearable(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(TaigiConverter, "RUNTIME_CACHE_MAX_ENTRIES", 2):
            root = Path(temp)
            for index in range(3):
                source = make_source_data(root / f"source-{index}")
                runtime = root / f"runtime-{index}"
                build_minimal_runtime(source, runtime)
                TaigiConverter(runtime)
            self.assertEqual(TaigiConverter.runtime_cache_info()["size"], 2)
            TaigiConverter.clear_runtime_cache()
            self.assertEqual(TaigiConverter.runtime_cache_info()["size"], 0)

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

    def test_review_enqueue_collects_matches_without_trace_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            review = Path(temp)
            converter = TaigiConverter(review_data_dir=review)
            with patch("taigi_converter.converter.append_review_item") as append:
                output = converter.convert(
                    "民眾說：「就列個東西告示牌，清楚告知使用規範。」",
                    profile={"enqueue_review": True, "owner": "test"},
                )

            self.assertIsInstance(output, str)
            append.assert_called_once()
            evidence = append.call_args.args[1]["evidence"]
            self.assertEqual(evidence["match_count"], 1)
            self.assertEqual(evidence["match_entry_ids"], ["lx_531000000012"])
            self.assertEqual(evidence["warnings"], ["核心漏轉:東西"])

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
