from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from taigi_converter import TaigiConverter
from taigi_converter.artifact_compiler import (
    ARTIFACT_SCHEMA_VERSIONS,
    LEXICON_STAGE,
    MANIFEST_VERSION,
    PROTECTED_LEGACY_CATEGORY,
    PROTECTED_TERM_MAX_LENGTH,
    RULE_PLAN_LEXICON_STAGE_KEY,
    RULE_PLAN_SCHEMA_VERSION,
    _pack_runtime_context,
    _source_digest,
    compile_runtime_artifacts,
    ensure_runtime_ready,
    migrate_explicit_protected_metadata,
    validate_artifact_documents,
)
from taigi_converter.lexicon_policy import RUNTIME_FILTER_UNTRUSTED_CONTEXT_ENTRY, runtime_exclusion_reason
from taigi_converter.models import LexiconEntry
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

    def test_active_private_use_target_fails_closed_but_disabled_history_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            active = valid_entry(src="噴香水", tgt="\uf5ea芳水")
            source = make_source_data(Path(temp) / "source", entries=[active])
            with self.assertRaisesRegex(ValueError, r"active tgt 含 Unicode private-use 字元 \(U\+F5EA\)"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

            active["status"] = "disabled"
            write_jsonl(source / "lexicon_entries.jsonl", [active])
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

    def test_empty_context_fails_closed_at_schema_policy_and_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry(tier="base", trust="seed", src="有空", tgt="有閒")
            entry["context"] = {}
            source = make_source_data(Path(temp) / "source", entries=[entry])
            output = Path(temp) / "out"

            with self.assertRaisesRegex(ValueError, "context 若有提供，至少必須包含一個條件欄位"):
                compile_runtime_artifacts(source, output_data_dir=output)

            self.assertFalse((output / "artifacts" / "manifest.json").exists())
            model = LexiconEntry.from_dict(entry)
            self.assertEqual(runtime_exclusion_reason(model), RUNTIME_FILTER_UNTRUSTED_CONTEXT_ENTRY)
            with self.assertRaisesRegex(ValueError, "context 若有提供"):
                _pack_runtime_context(model.context)

    def test_active_core_private_use_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source", entries=[])
            (source / "core_lexicon.json").write_text(
                json.dumps([{"src": "噴香水", "tgt": "\uf5ea芳水"}], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, r"core_lexicon.json 第 1 筆 active tgt 含 Unicode private-use 字元 \(U\+F5EA\)"
            ):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_trusted_core_context_is_indexed_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source", entries=[])
            (source / "core_lexicon.json").write_text(
                json.dumps(
                    [
                        {
                            "entry_id": "lx_corecontext01",
                            "src": "大家",
                            "tgt": "逐家",
                            "context": {"right_regex": "^."},
                            "trust": "human",
                        }
                    ],
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(temp) / "out"
            compile_runtime_artifacts(source, output_data_dir=output)

            converter = TaigiConverter(data_dir=output)
            self.assertEqual(converter.convert("大家來"), "逐家來")
            self.assertEqual(converter.convert("大家"), "大家")

    def test_untrusted_context_entry_is_explicitly_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry(tier="base", trust="seed", src="有空", tgt="有閒")
            entry["context"] = {"right_regex": "^$"}
            source = make_source_data(Path(temp) / "source", entries=[entry])
            output = Path(temp) / "out"
            manifest = compile_runtime_artifacts(source, output_data_dir=output)

            self.assertEqual(
                manifest["runtime_excluded_reasons"]["untrusted_context_entry"],
                1,
            )
            self.assertEqual(TaigiConverter(data_dir=output).convert("有空"), "有空")

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

    def test_protected_terms_require_explicit_auditable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            implicit = valid_entry(
                entry_id="lx_identity00001",
                src="一般詞",
                tgt="一般詞",
            )
            explicit = valid_entry(
                entry_id="lx_identity00002",
                src="台灣大學",
                tgt="台灣大學",
            )
            explicit["protected"] = {
                "category": "organization",
                "reason": "Stable institution name",
            }
            source = make_source_data(Path(temp) / "source", entries=[implicit, explicit])
            output = Path(temp) / "out"
            manifest = compile_runtime_artifacts(source, output_data_dir=output)
            rule_plan = json.loads((output / "artifacts" / "rule_plan.json").read_text())
            protected_terms = set(rule_plan["pt"].splitlines())

            self.assertNotIn("一般詞", protected_terms)
            self.assertIn("台灣大學", protected_terms)
            self.assertEqual(manifest["identity_passthrough_protected_entry_count"], 1)
            self.assertEqual(
                manifest["runtime_excluded_reasons"]["identity_passthrough_unprotected"],
                1,
            )
            self.assertEqual(
                manifest["runtime_excluded_reasons"]["identity_passthrough_masked"],
                1,
            )

    def test_strict_protected_term_wins_over_cross_boundary_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            strict_identity = valid_entry(
                entry_id="lx_strictprotect01",
                src="台灣大學",
                tgt="台灣大學",
                tier="manual",
                priority=900,
            )
            strict_identity["protected"] = {
                "category": "proper_noun",
                "reason": "Official institution name must not be split by adjacent lexical phrases",
                "enforcement": "strict",
            }
            overlapping_phrase = valid_entry(
                entry_id="lx_strictprotect02",
                src="大學測驗",
                tgt="大學考試",
                tier="base",
            )
            source = make_source_data(Path(temp) / "source", entries=[strict_identity, overlapping_phrase])
            output = Path(temp) / "out"
            compile_runtime_artifacts(source, output_data_dir=output)

            rule_plan = json.loads((output / "artifacts" / "rule_plan.json").read_text())
            self.assertEqual(rule_plan["sp"], "台灣大學")
            self.assertEqual(TaigiConverter(data_dir=output).convert("台灣大學測驗"), "台灣大學測驗")

    def test_non_strict_protected_identity_yields_to_longer_runtime_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            identity = valid_entry(
                entry_id="lx_non_strict_identity",
                src="善終",
                tgt="善終",
                tier="manual",
                priority=900,
            )
            identity["protected"] = {
                "category": "lexical_identity",
                "reason": "Preserve the polysemous term unless a longer curated phrase applies",
            }
            longer_phrase = valid_entry(
                entry_id="lx_longer_phrase",
                src="不得善終",
                tgt="歹死",
                tier="manual",
                priority=800,
            )
            source = make_source_data(Path(temp) / "source", entries=[identity, longer_phrase])
            output = Path(temp) / "out"
            compile_runtime_artifacts(source, output_data_dir=output)

            rule_plan = json.loads((output / "artifacts" / "rule_plan.json").read_text())
            self.assertIn("善終", rule_plan["pt"].splitlines())
            self.assertNotIn("善終", rule_plan["sp"].splitlines())

            converter = TaigiConverter(data_dir=output)
            self.assertEqual(converter.convert("善終"), "善終")
            self.assertEqual(converter.convert("不得善終"), "歹死")
            self.assertEqual(converter.convert("他不得善終。"), "他歹死。")

    def test_protected_metadata_rejects_unknown_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry(src="一般詞", tgt="一般詞")
            entry["protected"] = {
                "category": "lexical_identity",
                "reason": "Invalid metadata must fail source validation",
                "enforcement": "soft",
            }
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "enforcement"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_short_punctuated_work_title_can_be_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            title = valid_entry(src="飛吧！熊鷹", tgt="飛吧！熊鷹")
            title["protected"] = {
                "category": "work_title",
                "reason": "Official documentary title",
            }
            source = make_source_data(Path(temp) / "source", entries=[title])
            output = Path(temp) / "out"
            manifest = compile_runtime_artifacts(source, output_data_dir=output)
            protected_terms = set(json.loads((output / "artifacts" / "rule_plan.json").read_text())["pt"].splitlines())

            self.assertIn("飛吧！熊鷹", protected_terms)
            self.assertEqual(manifest["protected_term_lint_count"], 0)
            self.assertEqual(TaigiConverter(data_dir=output).convert("飛吧！熊鷹"), "飛吧！熊鷹")

    def test_work_title_punctuation_exception_stays_narrow(self) -> None:
        invalid_titles = [
            "這是一段完整句子。",
            "詞" * 8 + "！" + "詞" * 8,
        ]
        for title in invalid_titles:
            with self.subTest(title=title), tempfile.TemporaryDirectory() as temp:
                entry = valid_entry(src=title, tgt=title)
                entry["protected"] = {
                    "category": "work_title",
                    "reason": "Must not bypass sentence protection policy",
                }
                source = make_source_data(Path(temp) / "source", entries=[entry])
                with self.assertRaisesRegex(ValueError, "work_title"):
                    compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_overlong_protected_term_fails_schema_lint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            term = "詞" * (PROTECTED_TERM_MAX_LENGTH + 1)
            entry = valid_entry(src=term, tgt=term)
            entry["protected"] = {
                "category": "lexical_identity",
                "reason": "Test boundary",
            }
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "超過上限"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_sentence_identity_cannot_be_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry(src="這是完整句子。", tgt="這是完整句子。", level="sentence")
            entry["protected"] = {
                "category": "lexical_identity",
                "reason": "Must be a regression fixture instead",
            }
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "sentence identity"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_legacy_compatibility_is_always_rejected(self) -> None:
        metadata_cases = [
            {
                "category": PROTECTED_LEGACY_CATEGORY,
                "reason": "Legacy category without migration marker",
            },
            {
                "category": PROTECTED_LEGACY_CATEGORY,
                "reason": "Former identity-v1 migration shape",
                "migration": "identity-v1",
            },
            {
                "category": PROTECTED_LEGACY_CATEGORY,
                "reason": "Attempted future migration shape",
                "migration": "identity-v2",
            },
        ]
        for index, metadata in enumerate(metadata_cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                entry = valid_entry(src="一般詞", tgt="一般詞")
                entry["protected"] = metadata
                source = make_source_data(Path(temp) / "source", entries=[entry])

                with self.assertRaisesRegex(ValueError, "migration 已完成，不可再使用"):
                    compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_normal_build_reports_zero_legacy_protected_debt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source")
            output = Path(temp) / "out"
            manifest = compile_runtime_artifacts(source, output_data_dir=output)
            compact_manifest = json.loads((output / "artifacts" / "manifest.json").read_text())

            self.assertEqual(manifest["legacy_protected_debt_count"], 0)
            self.assertEqual(compact_manifest["ld"], 0)

    def test_historical_sealed_legacy_row_cannot_be_reintroduced(self) -> None:
        entry = valid_entry(
            entry_id="lx_378c000000003",
            src="終身不得再進入教保機構任職。",
            tgt="終身不得再進入教保機構任職。",
            level="phrase",
            priority=1598,
        )
        entry.update(
            source="curation:round378_cts_terms",
            protected={
                "category": PROTECTED_LEGACY_CATEGORY,
                "reason": ("Legacy identity migration from curation:round378_cts_terms: contains sentence punctuation"),
                "migration": "identity-v1",
            },
        )

        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "migration 已完成，不可再使用"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_protected_boolean_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            entry = valid_entry(src="一般詞", tgt="一般詞")
            entry["protected"] = True
            source = make_source_data(Path(temp) / "source", entries=[entry])
            with self.assertRaisesRegex(ValueError, "protected 必須是 object"):
                compile_runtime_artifacts(source, output_data_dir=Path(temp) / "out")

    def test_unsafe_legacy_allowlist_term_is_linted_and_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source")
            unsafe = "這是一段不應該被保護的完整長句，因為它會遮蔽真正轉換。"
            (source / "char_verified_allowlist.txt").write_text(unsafe + "\n", encoding="utf-8")
            output = Path(temp) / "out"
            manifest = compile_runtime_artifacts(source, output_data_dir=output)
            rule_plan = json.loads((output / "artifacts" / "rule_plan.json").read_text())

            self.assertNotIn(unsafe, rule_plan["pt"])
            self.assertEqual(manifest["protected_term_lint_count"], 1)
            self.assertIn("pl", rule_plan)

    def test_rule_plan_serializes_stage_and_schema_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source")
            output = Path(temp) / "out"
            manifest = compile_runtime_artifacts(source, output_data_dir=output)
            documents = {
                name: json.loads((output / "artifacts" / name).read_text()) for name in ARTIFACT_SCHEMA_VERSIONS
            }
            compact_manifest = json.loads((output / "artifacts" / "manifest.json").read_text())

            self.assertEqual(manifest["version"], MANIFEST_VERSION)
            self.assertEqual(compact_manifest["v"], MANIFEST_VERSION)
            self.assertEqual(compact_manifest["ls"], LEXICON_STAGE)
            self.assertEqual(compact_manifest["sv"], ARTIFACT_SCHEMA_VERSIONS)
            self.assertEqual(manifest["lexicon_stage"], LEXICON_STAGE)
            self.assertEqual(manifest["artifact_schema_versions"], ARTIFACT_SCHEMA_VERSIONS)
            self.assertEqual(documents["rule_plan.json"]["v"], RULE_PLAN_SCHEMA_VERSION)
            self.assertEqual(
                documents["rule_plan.json"][RULE_PLAN_LEXICON_STAGE_KEY],
                LEXICON_STAGE,
            )
            for name, expected_version in ARTIFACT_SCHEMA_VERSIONS.items():
                self.assertEqual(documents[name]["v"], expected_version)
            validate_artifact_documents(documents)

    def test_artifact_contract_rejects_wrong_schema_version_or_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source")
            output = Path(temp) / "out"
            compile_runtime_artifacts(source, output_data_dir=output)
            documents = {
                name: json.loads((output / "artifacts" / name).read_text()) for name in ARTIFACT_SCHEMA_VERSIONS
            }
            documents["rule_plan.json"]["v"] = RULE_PLAN_SCHEMA_VERSION - 1
            with self.assertRaisesRegex(ValueError, "schema version"):
                validate_artifact_documents(documents)

            documents["rule_plan.json"]["v"] = RULE_PLAN_SCHEMA_VERSION
            documents["rule_plan.json"][RULE_PLAN_LEXICON_STAGE_KEY] = "before_rules"
            with self.assertRaisesRegex(ValueError, RULE_PLAN_LEXICON_STAGE_KEY):
                validate_artifact_documents(documents)

    def test_artifact_contract_rejects_unknown_fields_and_wrong_required_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(Path(temp) / "source")
            output = Path(temp) / "out"
            compile_runtime_artifacts(source, output_data_dir=output)
            documents = {
                name: json.loads((output / "artifacts" / name).read_text()) for name in ARTIFACT_SCHEMA_VERSIONS
            }

            documents["phrase_trie.json"]["legacy"] = {}
            with self.assertRaisesRegex(ValueError, "含未知欄位"):
                validate_artifact_documents(documents)

            documents["phrase_trie.json"].pop("legacy")
            documents["char_map.json"]["m"] = []
            with self.assertRaisesRegex(ValueError, "型別不合法"):
                validate_artifact_documents(documents)

    def test_explicit_protected_migration_is_deterministic_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = make_source_data(
                Path(temp) / "source",
                entries=[
                    valid_entry(entry_id="lx_safe00000001", src="台北車站", tgt="台北車站"),
                    valid_entry(
                        entry_id="lx_long00000001",
                        src="詞" * (PROTECTED_TERM_MAX_LENGTH + 1),
                        tgt="詞" * (PROTECTED_TERM_MAX_LENGTH + 1),
                    ),
                ],
            )
            first = migrate_explicit_protected_metadata(source)
            second = migrate_explicit_protected_metadata(source)
            rows = [json.loads(line) for line in (source / "lexicon_entries.jsonl").read_text().splitlines()]

            self.assertEqual(first, {"migrated": 1, "skipped_identity": 1})
            self.assertEqual(second, {"migrated": 0, "skipped_identity": 1})
            self.assertEqual(rows[0]["protected"]["category"], "lexical_identity")
            self.assertNotIn("protected", rows[1])

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
            self.assertEqual(rebuilt["version"], MANIFEST_VERSION)


if __name__ == "__main__":
    unittest.main()
