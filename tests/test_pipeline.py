from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from taigi_converter.models import ConversionResult
from taigi_converter.pipeline import (
    PipelineResult,
    TaibunRomanizer,
    convert_zh_to_taigi_taibun,
    run_llm_postprocess,
)


class _StubConverter:
    def __init__(self, result: str | ConversionResult) -> None:
        self.result = result
        self.calls: list[tuple[str, bool, dict | None]] = []

    def convert(self, text: str, *, trace: bool, profile: dict | None):
        self.calls.append((text, trace, profile))
        return self.result


class PipelineResultTests(unittest.TestCase):
    def test_canonical_result_field_and_legacy_dict_key(self) -> None:
        result = PipelineResult("input", "漢字", "tai5-gi2")
        self.assertEqual(result.romanized_text, "tai5-gi2")
        self.assertEqual(result.to_dict()["romanized_text"], "tai5-gi2")
        self.assertEqual(result.to_dict()["taibun_number_tone"], "tai5-gi2")

    def test_legacy_constructor_and_property_are_deprecated_but_compatible(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = PipelineResult("input", "漢字", taibun_number_tone="tai5-gi2")
            self.assertEqual(result.taibun_number_tone, "tai5-gi2")
            result.taibun_number_tone = "tai5-gi2-new"

        self.assertEqual(result.romanized_text, "tai5-gi2-new")
        self.assertEqual(len(caught), 3)
        self.assertTrue(all(item.category is DeprecationWarning for item in caught))

    def test_conflicting_canonical_and_legacy_values_are_rejected(self) -> None:
        with (
            self.assertWarns(DeprecationWarning),
            self.assertRaisesRegex(ValueError, "不可指定不同值"),
        ):
            PipelineResult("input", "漢字", "new", taibun_number_tone="old")


class ConversionPipelineTests(unittest.TestCase):
    def test_injected_romanizer_receives_converted_hanji_and_preserves_trace(self) -> None:
        conversion = ConversionResult(output="台語漢字", warnings=["warning"])
        converter = _StubConverter(conversion)

        result = convert_zh_to_taigi_taibun(
            "華語",
            converter=converter,  # type: ignore[arg-type]
            trace=True,
            profile={"name": "test"},
            romanize_fn=lambda text: f"ROMAN:{text}",
        )

        self.assertEqual(result.taigi_hanji, "台語漢字")
        self.assertEqual(result.romanized_text, "ROMAN:台語漢字")
        self.assertEqual(result.warnings, ["warning"])
        self.assertIs(result.conversion_trace, conversion)
        self.assertEqual(converter.calls, [("華語", True, {"name": "test"})])

    def test_string_conversion_result_has_no_trace(self) -> None:
        converter = _StubConverter("台語漢字")
        result = convert_zh_to_taigi_taibun(
            "華語",
            converter=converter,  # type: ignore[arg-type]
            trace=False,
            romanize_fn=str.upper,
        )
        self.assertEqual(result.romanized_text, "台語漢字".upper())
        self.assertIsNone(result.conversion_trace)

    def test_run_llm_postprocess_delegates_with_deprecation_warning(self) -> None:
        with (
            mock.patch("taigi_converter.pipeline.run_taibun_postprocess", return_value="result") as canonical,
            self.assertWarns(DeprecationWarning),
        ):
            result = run_llm_postprocess("text", taibun_repo_path="/tmp/taibun")

        self.assertEqual(result, "result")
        canonical.assert_called_once_with("text", taibun_repo_path="/tmp/taibun")


class TaibunImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_modules = {
            name: module for name, module in sys.modules.items() if name == "taibun" or name.startswith("taibun.")
        }
        self._clear_taibun_modules()

    def tearDown(self) -> None:
        self._clear_taibun_modules()
        sys.modules.update(self._saved_modules)
        importlib.invalidate_caches()

    @staticmethod
    def _clear_taibun_modules() -> None:
        for name in list(sys.modules):
            if name == "taibun" or name.startswith("taibun."):
                sys.modules.pop(name, None)

    @staticmethod
    def _make_repo(root: Path, marker: str, *, keyword_only: bool = False) -> Path:
        package = root / "taibun"
        package.mkdir(parents=True)
        if keyword_only:
            constructor = """
    def __init__(self, *, system, dialect, format, delimiter, sandhi, punctuation, convert_non_cjk):
        self.format = format
"""
        else:
            constructor = """
    def __init__(self, system, dialect, format, delimiter, sandhi, punctuation, convert_non_cjk):
        self.format = format
"""
        (package / "__init__.py").write_text(
            f"""MARKER = {marker!r}\nclass Converter:\n{constructor}\n    def get(self, text):\n        return "妳2:" + MARKER + ":" + text\n"""
        )
        return root

    def test_repo_import_restores_sys_path_and_uses_requested_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = self._make_repo(Path(temp_dir), "repo-one")
            original_path = sys.path[:]

            romanizer = TaibunRomanizer(taibun_repo_path=repo)

            self.assertEqual(sys.path, original_path)
            self.assertEqual(romanizer.romanize("漢字"), "li2:repo-one:漢字")
            self.assertTrue(Path(sys.modules["taibun"].__file__).resolve().is_relative_to(repo.resolve()))

    def test_package_directory_can_be_passed_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = self._make_repo(Path(temp_dir), "package-path")
            romanizer = TaibunRomanizer(taibun_repo_path=repo / "taibun")
            self.assertEqual(romanizer.romanize("漢字"), "li2:package-path:漢字")

    def test_keyword_only_converter_constructor_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = self._make_repo(Path(temp_dir), "keyword", keyword_only=True)
            romanizer = TaibunRomanizer(taibun_repo_path=repo, tone_format="mark")
            self.assertEqual(romanizer.romanize("漢字"), "lí2:keyword:漢字")

    def test_loaded_module_from_different_repo_fails_predictably(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_repo = self._make_repo(Path(first_dir), "first")
            second_repo = self._make_repo(Path(second_dir), "second")
            TaibunRomanizer(taibun_repo_path=first_repo)
            original_path = sys.path[:]

            with self.assertRaisesRegex(RuntimeError, "不同位置載入"):
                TaibunRomanizer(taibun_repo_path=second_repo)

            self.assertEqual(sys.path, original_path)

    def test_loaded_module_from_same_repo_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = self._make_repo(Path(temp_dir), "same")
            first_module = TaibunRomanizer._import_taibun_module(repo)
            second_module = TaibunRomanizer._import_taibun_module(repo)
            self.assertIs(first_module, second_module)

    def test_missing_repo_path_has_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "不存在"):
            TaibunRomanizer(taibun_repo_path="/definitely/not/a/taibun/repo")


if __name__ == "__main__":
    unittest.main()
