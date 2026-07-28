from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_core_runtime_has_no_third_party_dependency(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["dependencies"], [])
        self.assertIn("msgpack>=1.1.2", project["optional-dependencies"]["taibun"])

    def test_console_script_and_package_artifacts_exist(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["scripts"]["taigi-converter"], "taigi_converter.cli:main")
        artifacts = ROOT / "taigi_converter" / "data" / "artifacts"
        required = {
            "char_map.json",
            "entry_table.json",
            "manifest.json",
            "override_index.json",
            "phrase_trie.json",
            "rule_plan.json",
        }
        self.assertTrue(required <= {path.name for path in artifacts.iterdir()})
        manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("ah", manifest)

    def test_source_data_is_not_duplicated_in_package(self) -> None:
        package_data = ROOT / "taigi_converter" / "data"
        for name in (
            "char_verified_allowlist.txt",
            "core_lexicon.json",
            "lexicon_entries.jsonl",
            "rule_entries.jsonl",
        ):
            self.assertFalse((package_data / name).exists())


if __name__ == "__main__":
    unittest.main()
