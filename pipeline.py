from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Callable

from converter import TaigiConverter
from models import ConversionResult

__all__ = ["run_llm_postprocess"]


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


@dataclass
class PipelineResult:
    input_text: str
    taigi_hanji: str
    taibun_number_tone: str
    warnings: list[str] = field(default_factory=list)
    conversion_trace: ConversionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "taigi_hanji": self.taigi_hanji,
            "taibun_number_tone": self.taibun_number_tone,
            "warnings": self.warnings,
            "conversion_trace": self.conversion_trace.to_dict() if self.conversion_trace else None,
        }


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
    def _import_taibun_module(taibun_repo_path: str | Path | None):
        if taibun_repo_path:
            repo_path = Path(taibun_repo_path).expanduser().resolve()
            if not repo_path.exists():
                raise FileNotFoundError(f"taibun_repo_path 不存在: {repo_path}")
            repo_str = str(repo_path)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        try:
            import taibun  # type: ignore
        except ModuleNotFoundError as exc:
            missing_pkg = getattr(exc, "name", None)
            if missing_pkg and missing_pkg != "taibun":
                raise ModuleNotFoundError(
                    f"taibun 已載入，但缺少依賴套件: {missing_pkg}。"
                    "請先安裝依賴（例如 `pip install msgpack`）後再試。"
                ) from exc
            raise ModuleNotFoundError(
                "找不到 taibun。請先 clone repo，並傳入 taibun_repo_path。"
            ) from exc

        return taibun

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
        warnings = list(conversion.warnings)
        conversion_trace: ConversionResult | None = conversion
    else:
        taigi_hanji = conversion
        warnings = []
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
        taibun_number_tone=taibun_output,
        warnings=warnings,
        conversion_trace=conversion_trace,
    )


def run_llm_postprocess(
    text: str,
    *,
    taibun_repo_path: str | Path,
) -> str:
    """
    超短入口：
    華語 -> 台語漢字 -> Taibun（Tailo number）
    """
    result = convert_zh_to_taigi_taibun(
        text,
        trace=False,
        taibun_repo_path=taibun_repo_path,
        taibun_options={"system": "Tailo", "dialect": "south", "tone_format": "number"},
    )
    return result.taibun_number_tone
