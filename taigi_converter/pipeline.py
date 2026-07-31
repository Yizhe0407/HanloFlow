from __future__ import annotations

import importlib
import re
import sys
import threading
import warnings as warnings_module
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .converter import TaigiConverter
from .models import ConversionResult

__all__ = [
    "PipelineResult",
    "TaibunRomanizer",
    "convert_zh_to_taigi_taibun",
    "run_taibun_postprocess",
    "run_llm_postprocess",
]


_TAIBUN_IMPORT_LOCK = threading.RLock()


DEFAULT_TAIBUN_PATCH_MAP_BY_FORMAT: dict[str, dict[str, str]] = {
    "number": {
        "妳": "li2",
        "您": "lin2",
        "她": "i1",
        "嗎": "ma0",
        "嘸": "bo5",
        "呣": "m7",
    },
    "mark": {
        "妳": "lí",
        "您": "lín",
        "她": "i",
        "嗎": "ma",
        "嘸": "bô",
        "呣": "m̄",
    },
    "strip": {
        "妳": "li",
        "您": "lin",
        "她": "i",
        "嗎": "ma",
        "嘸": "bo",
        "呣": "m",
    },
}


@dataclass(init=False)
class PipelineResult:
    input_text: str
    taigi_hanji: str
    romanized_text: str
    warnings: list[str] = field(default_factory=list)
    conversion_trace: ConversionResult | None = None

    def __init__(
        self,
        input_text: str,
        taigi_hanji: str,
        romanized_text: str | None = None,
        warnings: list[str] | None = None,
        conversion_trace: ConversionResult | None = None,
        *,
        taibun_number_tone: str | None = None,
    ) -> None:
        if taibun_number_tone is not None:
            warnings_module.warn(
                "PipelineResult.taibun_number_tone 已棄用；請改用 romanized_text。",
                DeprecationWarning,
                stacklevel=2,
            )
            if romanized_text is not None and romanized_text != taibun_number_tone:
                raise ValueError("romanized_text 與 taibun_number_tone 不可指定不同值")
            romanized_text = taibun_number_tone
        if romanized_text is None:
            raise TypeError("缺少必要參數: romanized_text")

        self.input_text = input_text
        self.taigi_hanji = taigi_hanji
        self.romanized_text = romanized_text
        self.warnings = list(warnings or [])
        self.conversion_trace = conversion_trace

    @property
    def taibun_number_tone(self) -> str:
        """Deprecated alias for :attr:`romanized_text`."""
        warnings_module.warn(
            "PipelineResult.taibun_number_tone 已棄用；請改用 romanized_text。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.romanized_text

    @taibun_number_tone.setter
    def taibun_number_tone(self, value: str) -> None:
        warnings_module.warn(
            "PipelineResult.taibun_number_tone 已棄用；請改用 romanized_text。",
            DeprecationWarning,
            stacklevel=2,
        )
        self.romanized_text = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "taigi_hanji": self.taigi_hanji,
            "romanized_text": self.romanized_text,
            # 保留舊 key，避免既有 JSON consumer 立即中斷。
            "taibun_number_tone": self.romanized_text,
            "warnings": self.warnings,
            "conversion_trace": self.conversion_trace.to_dict() if self.conversion_trace else None,
        }


@contextmanager
def _temporary_import_path(path: Path) -> Iterator[None]:
    original_path = sys.path[:]
    path_text = str(path)
    # 即使原本已存在，也暫時移到最高優先序；finally 完整還原順序與重複項。
    sys.path[:] = [path_text, *(item for item in original_path if item != path_text)]
    try:
        yield
    finally:
        sys.path[:] = original_path


def _module_origins(module: ModuleType) -> list[Path]:
    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.append(Path(module_file).resolve())

    module_path = getattr(module, "__path__", None)
    if module_path:
        origins.extend(Path(item).resolve() for item in module_path)
    return origins


def _module_is_within(module: ModuleType, root: Path) -> bool:
    resolved_root = root.resolve()
    return any(origin == resolved_root or origin.is_relative_to(resolved_root) for origin in _module_origins(module))


def _taibun_import_locations(repo_path: Path) -> tuple[Path, Path]:
    if not repo_path.is_dir():
        raise NotADirectoryError(f"taibun_repo_path 不是目錄: {repo_path}")

    # 同時兼容「repo root/taibun」和直接傳入 package directory。
    if repo_path.name == "taibun" and (repo_path / "__init__.py").is_file():
        return repo_path.parent, repo_path
    return repo_path, repo_path


def _raise_taibun_import_error(exc: ModuleNotFoundError) -> None:
    missing_pkg = getattr(exc, "name", None)
    if missing_pkg and missing_pkg != "taibun":
        raise ModuleNotFoundError(
            f"taibun 匯入失敗，缺少模組: {missing_pkg}。請先安裝 Taibun 所需依賴（例如 `pip install msgpack`）後再試。"
        ) from exc
    raise ModuleNotFoundError("找不到 taibun。請先 clone repo，並傳入 taibun_repo_path。") from exc


class TaibunRomanizer:
    def __init__(
        self,
        *,
        taibun_repo_path: str | Path | None = None,
        system: str = "Tailo",
        dialect: str = "south",
        tone_format: str = "number",
        delimiter: str = "-",
        sandhi: str = "none",
        punctuation: str = "none",
        convert_non_cjk: bool = True,
        patch_map: dict[str, str] | None = None,
    ) -> None:
        self.tone_format = tone_format
        self.patch_map = dict(
            DEFAULT_TAIBUN_PATCH_MAP_BY_FORMAT.get(
                tone_format,
                DEFAULT_TAIBUN_PATCH_MAP_BY_FORMAT["number"],
            )
        )
        if patch_map:
            self.patch_map.update(patch_map)

        taibun_module = self._import_taibun_module(taibun_repo_path)
        converter_cls = getattr(taibun_module, "Converter", None)
        if converter_cls is None:
            raise AttributeError("找不到 taibun.Converter，請確認 clone 的 repo 是否正確。")

        try:
            self.converter = converter_cls(
                system,
                dialect,
                tone_format,
                delimiter,
                sandhi,
                punctuation,
                convert_non_cjk,
            )
        except TypeError:
            # 兼容可能的 keyword constructor 版本
            self.converter = converter_cls(
                system=system,
                dialect=dialect,
                format=tone_format,
                delimiter=delimiter,
                sandhi=sandhi,
                punctuation=punctuation,
                convert_non_cjk=convert_non_cjk,
            )

    @staticmethod
    def _import_taibun_module(taibun_repo_path: str | Path | None) -> ModuleType:
        # sys.path 與 sys.modules 都是 process-global；鎖住完整檢查／匯入／還原流程，
        # 避免不同執行緒的 scoped import 互相覆寫。
        with _TAIBUN_IMPORT_LOCK:
            return TaibunRomanizer._import_taibun_module_locked(taibun_repo_path)

    @staticmethod
    def _import_taibun_module_locked(taibun_repo_path: str | Path | None) -> ModuleType:
        if taibun_repo_path is None:
            try:
                return importlib.import_module("taibun")
            except ModuleNotFoundError as exc:
                _raise_taibun_import_error(exc)

        repo_path = Path(taibun_repo_path).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"taibun_repo_path 不存在: {repo_path}")
        import_root, expected_root = _taibun_import_locations(repo_path)

        loaded_module = sys.modules.get("taibun")
        if loaded_module is not None:
            if not isinstance(loaded_module, ModuleType) or not _module_is_within(loaded_module, expected_root):
                origins = ", ".join(str(path) for path in _module_origins(loaded_module)) or "未知來源"
                raise RuntimeError(
                    "taibun 已從不同位置載入，無法安全切換 repo；"
                    f"目前來源: {origins}；要求來源: {expected_root}。請使用新的 Python process。"
                )
            return loaded_module

        modules_before = {
            name: module for name, module in sys.modules.items() if name == "taibun" or name.startswith("taibun.")
        }
        try:
            with _temporary_import_path(import_root):
                module = importlib.import_module("taibun")
            if not _module_is_within(module, expected_root):
                origins = ", ".join(str(path) for path in _module_origins(module)) or "未知來源"
                raise ImportError(f"taibun 載入來源不符：實際 {origins}；預期位於 {expected_root}")
            return module
        except ModuleNotFoundError as exc:
            TaibunRomanizer._restore_taibun_modules(modules_before)
            _raise_taibun_import_error(exc)
        except Exception:
            TaibunRomanizer._restore_taibun_modules(modules_before)
            raise

    @staticmethod
    def _restore_taibun_modules(modules_before: dict[str, ModuleType]) -> None:
        for name in list(sys.modules):
            if name == "taibun" or name.startswith("taibun."):
                sys.modules.pop(name, None)
        sys.modules.update(modules_before)

    def romanize(self, hanji_text: str) -> str:
        if hasattr(self.converter, "get"):
            out = self.converter.get(hanji_text)
        elif callable(self.converter):
            out = self.converter(hanji_text)
        else:
            raise TypeError("taibun Converter 無法呼叫（缺少 get()）")

        out_text = str(out)
        for src, roman in self.patch_map.items():
            if self.tone_format == "number":
                out_text = re.sub(rf"{re.escape(src)}[0-9]?", roman, out_text)
            else:
                out_text = out_text.replace(src, roman)
        return out_text


def convert_zh_to_taigi_taibun(
    text: str,
    *,
    converter: TaigiConverter | None = None,
    trace: bool = True,
    profile: dict[str, Any] | None = None,
    taibun_repo_path: str | Path | None = None,
    taibun_options: dict[str, Any] | None = None,
    taibun_patch_map: dict[str, str] | None = None,
    romanize_fn: Callable[[str], str] | None = None,
) -> PipelineResult:
    converter_instance = converter or TaigiConverter()
    conversion = converter_instance.convert(text, trace=trace, profile=profile)

    if isinstance(conversion, ConversionResult):
        taigi_hanji = conversion.output
        result_warnings = list(conversion.warnings)
        conversion_trace: ConversionResult | None = conversion
    else:
        taigi_hanji = conversion
        result_warnings = []
        conversion_trace = None

    if romanize_fn is not None:
        taibun_output = romanize_fn(taigi_hanji)
    else:
        options = dict(taibun_options or {})
        romanizer = TaibunRomanizer(
            taibun_repo_path=taibun_repo_path,
            patch_map=taibun_patch_map,
            **options,
        )
        taibun_output = romanizer.romanize(taigi_hanji)

    return PipelineResult(
        input_text=text,
        taigi_hanji=taigi_hanji,
        romanized_text=taibun_output,
        warnings=result_warnings,
        conversion_trace=conversion_trace,
    )


def run_taibun_postprocess(
    text: str,
    *,
    taibun_repo_path: str | Path,
) -> str:
    """華語 -> 台語漢字 -> Taibun（Tailo number）。"""
    result = convert_zh_to_taigi_taibun(
        text,
        trace=False,
        taibun_repo_path=taibun_repo_path,
        taibun_options={"system": "Tailo", "dialect": "south", "tone_format": "number"},
    )
    return result.romanized_text


def run_llm_postprocess(
    text: str,
    *,
    taibun_repo_path: str | Path,
) -> str:
    """Deprecated alias for :func:`run_taibun_postprocess`."""
    warnings_module.warn(
        "run_llm_postprocess() 未使用 LLM 且已棄用；請改用 run_taibun_postprocess()。",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_taibun_postprocess(text, taibun_repo_path=taibun_repo_path)
