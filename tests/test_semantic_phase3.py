from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from taigi_converter import ConversionResult, TaigiConverter
from taigi_converter.artifact_compiler import compile_runtime_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = REPO_ROOT / "data" / "lexicon_entries.jsonl"
ENTRY_ID_PREFIX = "lx_545000000"


@dataclass(frozen=True, slots=True)
class Phase3Case:
    entry_id: str
    source: str
    expected: str
    negative: str
    multiple: str


@dataclass(frozen=True, slots=True)
class Phase3OutputCase:
    source: str
    expected: str
    matched_entry_ids: tuple[str, ...] = ()
    excluded_entry_ids: tuple[str, ...] = ()


CASES = (
    Phase3Case("lx_545000000001", "順路買鹽。", "順紲買鹽。", "順路買賣契約。", "順路買鹽，順路買票。"),
    Phase3Case("lx_545000000002", "咳嗽三天。", "嗽三工。", "請勿咳嗽。", "咳嗽三天，咳嗽兩工。"),
    Phase3Case(
        "lx_545000000003",
        "傷口長出新皮。",
        "傷口發出新皮。",
        "他長出一口氣。",
        "傷口長出新皮，樹長出新芽。",
    ),
    Phase3Case(
        "lx_545000000004",
        "雨水滲進來。",
        "雨水漏入來。",
        "消息滲進來了。",
        "雨水從縫滲進來，海水也滲進來。",
    ),
    Phase3Case(
        "lx_545000000005",
        "我做不來。",
        "我做袂來。",
        "這是做不來電訪問。",
        "我做不來，你也做不來。",
    ),
    Phase3Case("lx_545000000006", "自己烘茶。", "家己焙茶。", "烘茶色布料。", "烘茶，烘茶技術。"),
    Phase3Case(
        "lx_545000000007",
        "他嘴硬。",
        "伊喙䫌。",
        "嘴硬糖很好吃。",
        "他嘴硬，但他還是嘴硬。",
    ),
    Phase3Case(
        "lx_545000000008",
        "他的手藝很好。",
        "伊的手路真好。",
        "手藝品展覽。",
        "他的手藝很好，師傅的手藝真好。",
    ),
    Phase3Case(
        "lx_545000000009",
        "新鞋開口了。",
        "新鞋裂喙矣。",
        "他開口說話。",
        "新鞋開口了，舊鞋仔開口矣。",
    ),
    Phase3Case(
        "lx_545000000010",
        "生意很好。",
        "生理真好。",
        "這是生意外事件。",
        "生意很好，生意越來越好。",
    ),
    Phase3Case("lx_545000000011", "乘車優惠。", "坐車優惠。", "乘車規定。", "乘車優惠，乘車方式。"),
    Phase3Case(
        "lx_545000000012",
        "公共自行車租借服務。",
        "公共跤踏車租借服務。",
        "自行車床控制器。",
        "自行車租借服務，公共自行車道。",
    ),
    Phase3Case(
        "lx_545000000013",
        "海纜搶修。",
        "海底電纜搶修。",
        "海纜魚標本。",
        "海纜搶修，海纜系統。",
    ),
    Phase3Case("lx_545000000014", "道路坍方。", "道路崩山。", "坍方風險。", "道路坍方，公路坍方。"),
    Phase3Case(
        "lx_545000000015",
        "蓄水率回升。",
        "貯水率回升。",
        "蓄水率先驗模型。",
        "蓄水率回升，蓄水率下降。",
    ),
    Phase3Case(
        "lx_545000000016",
        "最新資料。",
        "上新的資料。",
        "最新穎的設計。",
        "最新資料，最新版本。",
    ),
    Phase3Case(
        "lx_545000000025",
        "最新的消息。",
        "上新的消息。",
        "最新的設計。",
        "最新的消息，最新的資料。",
    ),
    Phase3Case(
        "lx_545000000017",
        "政策帶動產業發展。",
        "政策𤆬動產業發展。",
        "帶動整體觀光。",
        "政策帶動產業，計畫帶動店家。",
    ),
    Phase3Case(
        "lx_545000000018",
        "提供弱勢家庭。",
        "提供予弱勢家庭。",
        "提供服務。",
        "提供弱勢家庭，提供受災戶。",
    ),
    Phase3Case("lx_545000000019", "一整排樹。", "規排樹。", "一整排版面。", "一整排樹，一整排苦楝樹。"),
    Phase3Case(
        "lx_545000000020",
        "這塊地以前種菜。",
        "這坵地以前種菜。",
        "板塊地震。",
        "這塊地以前，那塊地目前。",
    ),
    Phase3Case(
        "lx_545000000021",
        "菜價有點硬。",
        "菜價有淡薄仔懸。",
        "石頭很硬。",
        "菜價有點硬，售價偏硬。",
    ),
    Phase3Case(
        "lx_545000000022",
        "方案還有討論的空間。",
        "方案猶有討論的空間。",
        "商店還有貨。",
        "方案還有討論的空間，這項工作還有討論的空間。",
    ),
    Phase3Case(
        "lx_545000000023",
        "連續三個月。",
        "連紲三个月。",
        "連續兩天。",
        "連續三個月，連續四季。",
    ),
    Phase3Case(
        "lx_545000000024",
        "最近開始下雨。",
        "這陣開始落雨。",
        "最近距離。",
        "她最近開始學習，他最近常跑醫院。",
    ),
)


CONTEXT_OUTPUT_CASES = (
    Phase3OutputCase(
        "雨水從縫滲進來，海水也滲進來。",
        "雨水對縫漏入來，海水也漏入來。",
        matched_entry_ids=("lx_545000000004",),
    ),
    Phase3OutputCase(
        "水果香氣從門外滲進來。",
        "水果香氣對門外滲入來。",
        excluded_entry_ids=("lx_545000000004",),
    ),
    Phase3OutputCase(
        "水軍把假消息滲進來。",
        "水軍把假消息滲入來。",
        excluded_entry_ids=("lx_545000000004",),
    ),
    Phase3OutputCase(
        "他嘴硬，但他還是嘴硬。",
        "伊喙䫌，但伊猶是喙䫌。",
        matched_entry_ids=("lx_545000000007",),
    ),
    Phase3OutputCase(
        "這隻烏龜的嘴硬。",
        "這隻烏龜的嘴硬。",
        excluded_entry_ids=("lx_545000000007",),
    ),
    Phase3OutputCase(
        "這隻螃蟹的嘴硬，但是腳很軟。",
        "這隻蟳仔的嘴硬，毋過腳真軟。",
        excluded_entry_ids=("lx_545000000007",),
    ),
    Phase3OutputCase(
        "新鞋開口了，舊鞋仔開口矣。",
        "新鞋裂喙矣，舊鞋仔裂喙矣。",
        matched_entry_ids=("lx_545000000009",),
    ),
    Phase3OutputCase(
        "鞋店老闆終於開口。",
        "鞋店頭家終於開口。",
        excluded_entry_ids=("lx_545000000009",),
    ),
    Phase3OutputCase(
        "布袋鎮長終於開口。",
        "布袋鎮長終於開口。",
        excluded_entry_ids=("lx_545000000009",),
    ),
    Phase3OutputCase(
        "最新資料，最新版本。",
        "上新的資料，上新的版本。",
        matched_entry_ids=("lx_545000000016",),
    ),
    Phase3OutputCase(
        "最新的消息已經公布。",
        "上新的消息已經公布。",
        matched_entry_ids=("lx_545000000025",),
    ),
    Phase3OutputCase(
        "這是最新的青年就業統計。",
        "這是上新的青年就業統計。",
        matched_entry_ids=("lx_545000000025",),
    ),
    Phase3OutputCase(
        "這是最新穎的設計。",
        "這是最新穎的設計。",
        excluded_entry_ids=("lx_545000000016", "lx_545000000025"),
    ),
)


@pytest.fixture(scope="module")
def converter(tmp_path_factory: pytest.TempPathFactory) -> TaigiConverter:
    runtime_dir = tmp_path_factory.mktemp("semantic-phase3-runtime")
    compile_runtime_artifacts(REPO_ROOT / "data", output_data_dir=runtime_dir)
    return TaigiConverter(runtime_dir)


def traced(converter: TaigiConverter, source: str) -> ConversionResult:
    result = converter.convert(source, trace=True)
    assert isinstance(result, ConversionResult)
    return result


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.entry_id)
def test_phase3_positive_context(
    converter: TaigiConverter,
    case: Phase3Case,
) -> None:
    result = traced(converter, case.source)

    assert result.output == case.expected
    assert case.entry_id in {match.entry_id for match in result.matches}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.entry_id)
def test_phase3_negative_or_word_boundary(
    converter: TaigiConverter,
    case: Phase3Case,
) -> None:
    result = traced(converter, case.negative)

    assert case.entry_id not in {match.entry_id for match in result.matches}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.entry_id)
def test_phase3_multiple_occurrences_are_occurrence_local(
    converter: TaigiConverter,
    case: Phase3Case,
) -> None:
    result = traced(converter, case.multiple)

    assert sum(match.entry_id == case.entry_id for match in result.matches) == 2


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.entry_id)
def test_phase3_positive_output_is_fixed_point(
    converter: TaigiConverter,
    case: Phase3Case,
) -> None:
    assert converter.convert(case.expected) == case.expected


@pytest.mark.parametrize(
    "case",
    CONTEXT_OUTPUT_CASES,
    ids=lambda case: case.source,
)
def test_phase3_context_guard_complete_outputs(
    converter: TaigiConverter,
    case: Phase3OutputCase,
) -> None:
    result = traced(converter, case.source)
    matched_entry_ids = {match.entry_id for match in result.matches}

    assert result.output == case.expected
    assert set(case.matched_entry_ids) <= matched_entry_ids
    assert set(case.excluded_entry_ids).isdisjoint(matched_entry_ids)


def test_phase3_entries_are_ai_reviewed_contextual_phrases() -> None:
    rows = [json.loads(line) for line in LEXICON_PATH.read_text(encoding="utf-8").splitlines() if line]
    phase3_rows = [row for row in rows if row["entry_id"].startswith(ENTRY_ID_PREFIX)]

    assert len(phase3_rows) == len(CASES) == 25
    assert {row["entry_id"] for row in phase3_rows} == {case.entry_id for case in CASES}
    for row in phase3_rows:
        assert row["level"] == "phrase"
        assert row["trust"] == "ai_reviewed"
        assert row["status"] == "active"
        assert row["source"] == "curation:round545_semantic_phase3_train_development"
        assert row["context"]
        assert row["governance_note"]
