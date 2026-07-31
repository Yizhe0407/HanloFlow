from __future__ import annotations

from typing import Any

from .unicode_policy import contains_han_ideograph

TRUST_HUMAN = "human"
TRUST_AI_REVIEWED = "ai_reviewed"
TRUST_MACHINE = "machine"
TRUST_SEED = "seed"
VALID_TRUSTS = {TRUST_HUMAN, TRUST_AI_REVIEWED, TRUST_MACHINE, TRUST_SEED}
CURATED_TRUSTS = frozenset({TRUST_HUMAN, TRUST_AI_REVIEWED})

MACHINE_UPDATERS = {"itaigi_full", "itaigi_bot", "codex"}
MACHINE_SOURCES = {"review_queue"}
LOW_TRUST_LONG_PHRASE_MIN_LEN = 8

RUNTIME_FILTER_NOOP_MANUAL_HOTFIX = "noop_manual_hotfix"
RUNTIME_FILTER_SINGLE_CHAR_MACHINE = "single_char_machine_override"
RUNTIME_FILTER_SINGLE_CHAR_LOW_TRUST_PHRASE = "single_char_low_trust_phrase"
RUNTIME_FILTER_LOW_TRUST_LONG_PHRASE = "low_trust_long_phrase"
RUNTIME_FILTER_LOW_TRUST_SEVERE_CONTRACTION = "low_trust_severe_contraction"
RUNTIME_FILTER_DEFINITION_LIKE_LOW_TRUST = "definition_like_low_trust_phrase"
RUNTIME_FILTER_UNSAFE_CONTEXT_FREE_EDGE = "unsafe_context_free_lexical_edge"
RUNTIME_FILTER_NON_HANJI_TARGET = "non_hanji_target"
RUNTIME_FILTER_UNTRUSTED_CONTEXT_ENTRY = "untrusted_context_entry"


_DEFINITION_STRONG_PREFIXES = (
    "泛指",
    "比喻",
    "形容",
    "表示",
    "指",
    "位於",
    "用來",
    "古書上",
    "傳說中",
)
_DEFINITION_SOFT_PREFIXES = (
    "一種",
    "一個",
    "一位",
    "一條",
    "一片",
    "一粒",
    "一隻",
    "一頭",
    "從事",
    "使人",
    "使在",
    "使勁",
    "使水",
)
_DEFINITION_CATEGORY_TERMS = frozenset({"病名", "疾病名", "魚名", "植物名", "動物名"})
_DEFINITION_CATEGORY_SUFFIXES = ("名詞", "用字", "之一")
_SHORT_REVERSE_DEFINITION_EDGES = frozenset(
    {
        ("一例", "一切"),
        ("六寸", "海參"),
        ("初三", "月眉"),
        ("創制", "政權"),
        ("加蒜", "冬菜"),
        ("千足", "純金"),
        ("卵生", "鳥類"),
        ("失音", "白喉"),
        ("多汁", "楊桃"),
        ("多肉", "菜蟳"),
        ("大眼", "蛤仔"),
        ("強酸", "鹽酸"),
        ("幼蟲", "尾蝶"),
        ("忌宜", "通書"),
        ("怕光", "杜猴"),
        ("攻防", "柔道"),
        ("殼薄", "珠螺"),
        ("水生", "菱角"),
        ("外乾", "噴漆"),
        ("大麥", "五穀"),
        ("八尺", "大麻"),
        ("口小", "喇叭"),
        ("嘴大", "杜鵑"),
        ("頭小", "烏毛"),
        ("指人", "內部"),
        ("產煤", "大同"),
        ("稍大", "翡翠"),
        ("即狗", "犬"),
        ("即矢", "箭"),
        ("即藕", "蓮"),
        ("即貍", "貉"),
        ("即鰈", "魬"),
        ("互生", "榛"),
        ("似槐", "檬"),
        ("寒滑", "蘋"),
        ("對生", "梔"),
        ("用肉", "羹"),
        ("長柄", "矛"),
    }
)
_SENSITIVE_LOW_TRUST_TARGETS = frozenset({"番仔", "阿啄仔", "啄鼻仔"})

_UNSAFE_CONTEXT_FREE_LEGACY_EDGES = frozenset(
    {
        ("得利", "受害"),
        ("大食", "小食"),
        ("奉贈", "似"),
        ("念誦", "表白"),
        ("刀術", "武術"),
        ("十位", "數位"),
        ("大宗", "大筆"),
        ("性質", "性"),
        ("少少", "濟濟"),
        ("大筏", "查"),
        ("憑借", "階"),
        ("長衣", "袍"),
        ("短衣", "褙"),
        ("美石", "玫"),
        ("玉光", "瑛"),
        ("煉鐵", "鋼"),
        ("竹簟", "笙"),
        ("置身", "廁"),
        ("舉起", "拯"),
        ("貪求", "殉"),
        ("雞蟲", "雞"),
        ("危殆", "圾"),
        ("多言", "諜"),
        ("很近", "咫"),
        ("楯脊", "瓦"),
        ("無義", "喲"),
        ("圓鍬", "鉛筆仔"),
        ("嚴寒", "生冷"),
        ("劈刀", "柴鍥"),
        ("四份", "四捻"),
        ("回轉", "斡輾轉"),
        ("乃是", "蓋"),
        ("佐助", "讚"),
        ("佞臣", "倖"),
        ("依託", "俚"),
        ("俸祿", "秩"),
        ("承受", "忍"),
        ("正值", "正"),
        ("拿去吃", "孝孤"),
        ("店家", "店頭家"),
        ("常見", "捷看"),
        ("常規", "紀綱"),
        ("承繼", "過房"),
        ("招集", "募集"),
        ("界限", "地界"),
        ("喜好", "興"),
        ("不只", "毋但"),
        ("不料", "無疑悟"),
        ("務必", "一定"),
        ("即使", "著算"),
        ("努力", "拚勢"),
        ("升高", "衝懸"),
        ("去世", "老去"),
        ("吃藥", "食藥仔"),
        ("包圍", "圍"),
        ("包容", "包涵"),
        ("同樣", "平平"),
        ("合適", "好勢"),
        ("不久", "無偌久"),
        ("不怕", "毋驚"),
        ("不肯", "毋肯"),
        ("丟失", "拍毋見"),
        ("刮痧", "掠痧"),
        ("力氣", "氣力"),
        ("勝過", "贏過"),
        ("勞神", "損神"),
        ("勞累", "疲勞"),
        ("勤勉", "骨力"),
        ("勾芡", "牽羹"),
        ("匆忙", "趕狂"),
        ("化痰", "去痰"),
        ("午飯", "日晝頓"),
        ("午飯", "中晝飯"),
        ("即將", "得欲"),
        ("即將", "咧欲"),
        ("原先", "原早"),
        ("厭惡", "討厭"),
        ("及早", "量早"),
        ("及早", "冗早"),
        ("取笑", "恥笑"),
        ("受到", "受著"),
        ("叢生", "密密生"),
        ("可惜", "無彩"),
        ("吃驚", "昂愕"),
        ("吃驚", "著驚"),
        ("吹風", "搧風"),
        ("啟發", "啟示"),
        ("埋怨", "怨嘆"),
        ("堵塞", "塞死"),
        ("塗抹", "糊"),
        ("填平", "坉塗"),
        ("母親", "阿母"),
        ("洋人", "番仔"),
        ("植物", "草木"),
        ("活著", "活咧"),
        ("呼喚", "呼"),
        ("呼喚", "叫"),
        ("回去", "轉去"),
        ("父母親", "爸母"),
        ("吹風", "放風聲"),
        ("埋怨", "怨慼"),
        ("堵塞", "滯滯"),
        ("塗抹", "抹"),
        ("塗抹", "抉"),
        ("塗抹", "挲"),
        ("填平", "坉平"),
        ("洋人", "西洋人"),
        ("洋人", "外國人"),
        ("呼喚", "喊"),
        ("各個", "逐个"),
        ("各自", "隨人"),
        ("名望", "聲望"),
        ("吵架", "冤家"),
        ("吵架", "相罵"),
        ("吻合", "搭峇"),
        ("吻合", "峇"),
        ("吻合", "合"),
        ("吻合", "符合"),
        ("吻合", "相符"),
        ("吻合", "一致"),
        ("周到", "點陳"),
        ("周遭", "周圍"),
        ("周遭", "四箍輾轉"),
        ("周遭", "四箍圍仔"),
        ("咀嚼", "哺食"),
        ("咀嚼", "哺"),
        ("咀嚼", "卯"),
        ("品性", "性地"),
        ("品性", "人格"),
        ("品行", "心行"),
        ("哽咽", "喉實"),
        ("哽咽", "喉滇"),
        ("唆使", "煽動"),
        ("唆使", "拐弄"),
        ("唆使", "使弄"),
        ("唆使", "呲"),
        ("唾棄", "呸瀾"),
        ("唾棄", "漚屎"),
        ("商榷", "參詳"),
        ("善後", "帕尾"),
        ("善後", "收尾"),
        ("善終", "大葩尾"),
        ("善終", "好尾景"),
        ("善終", "好尾梢"),
        ("喜劇", "喜齣"),
        ("喜劇", "笑詼齣"),
        ("喜歡", "合意"),
        ("喜歡", "佮意"),
        ("喜歡", "愛"),
        ("喜歡", "意愛"),
        ("喜餅", "盒仔餅"),
        ("喜餅", "大餅"),
        ("喝酒", "食酒"),
        ("喝酒", "啉酒"),
        ("喪志", "失志"),
        ("喪志", "餒志"),
        ("喪服", "麻衫"),
        ("喪服", "孝衫"),
        ("嗩吶", "鼓吹"),
        ("嗩吶", "噯仔"),
        ("嘆氣", "吐大氣"),
        ("嘆氣", "吐氣"),
    }
)

_REVERSE_EXAMPLE_CATEGORY_TARGETS = frozenset(
    {
        "代名詞",
        "代步",
        "大寫",
        "宗親",
        "家禽",
        "感官",
        "插枝",
        "方位",
        "文具",
        "海產",
        "液體",
        "潰瘍",
        "燃料",
        "物質",
        "礦物",
        "行業",
        "金屬",
        "鹼",
    }
)

_SHORT_REVERSE_DEFINITION_GLOSSES = frozenset(
    {
        "七瓣",
        "卵形",
        "國名",
        "嘴闊",
        "有節",
        "有青",
        "桑科",
        "牙細",
        "瘧蚊",
        "眼大",
        "羽白",
        "翼長",
        "肉美",
        "胸平",
        "色白",
        "色黑",
        "體長",
    }
)


def infer_trust_from_metadata(
    *,
    source: str | None,
    updated_by: str | None,
    tier: str | None,
) -> str:
    source_v = (source or "").strip()
    updated_by_v = (updated_by or "").strip()
    tier_v = (tier or "").strip()

    if source_v in MACHINE_SOURCES and updated_by_v in MACHINE_UPDATERS:
        return TRUST_MACHINE
    if source_v.startswith("user:"):
        return TRUST_HUMAN
    if tier_v in {"manual", "manual_hotfix"} and source_v == "review_queue":
        return TRUST_HUMAN
    return TRUST_SEED


def normalize_trust(
    *,
    trust: str | None,
    source: str | None,
    updated_by: str | None,
    tier: str | None,
) -> str:
    trust_v = (trust or "").strip()
    if trust_v in VALID_TRUSTS:
        return trust_v
    return infer_trust_from_metadata(source=source, updated_by=updated_by, tier=tier)


def is_trusted_manual_entry(entry: Any) -> bool:
    """Return whether a manual entry has explicit semantic review provenance.

    ``human`` and ``ai_reviewed`` are intentionally distinct provenance labels,
    but both are curated inputs rather than unreviewed machine suggestions.
    Precedence remains provenance-aware in :func:`runtime_layer_rank`.
    """

    return (
        getattr(entry, "tier", None) in {"manual", "manual_hotfix"} and getattr(entry, "trust", None) in CURATED_TRUSTS
    )


def is_trusted_context_entry(entry: Any) -> bool:
    context = getattr(entry, "context", None)
    return (
        context is not None
        and bool(context)
        and getattr(entry, "trust", None) in CURATED_TRUSTS
        and getattr(entry, "tier", None) in {"core", "manual", "manual_hotfix"}
    )


def is_machine_generated_override(entry: Any) -> bool:
    return getattr(entry, "tier", None) == "manual_hotfix" and getattr(entry, "trust", None) == TRUST_MACHINE


def is_sentence_manual_override(entry: Any) -> bool:
    return (
        getattr(entry, "level", None) == "sentence"
        and is_trusted_manual_entry(entry)
        and getattr(entry, "context", None) is None
        and getattr(entry, "status", None) == "active"
    )


def runtime_layer_rank(entry: Any) -> int:
    """Return the actual precedence layer used by runtime candidate selection."""

    tier = getattr(entry, "tier", None)
    level = getattr(entry, "level", None)
    trust = getattr(entry, "trust", None)
    if tier == "blocked":
        return 0
    if is_sentence_manual_override(entry):
        return 1 if trust == TRUST_HUMAN else 2
    if is_trusted_manual_entry(entry):
        return 3 if trust == TRUST_HUMAN else 4
    if tier == "core" and level in {"phrase", "sentence"}:
        return 5
    if tier == "domain" and level in {"phrase", "sentence"}:
        return 6
    if tier == "base" and level in {"phrase", "sentence"}:
        return 7
    if is_machine_generated_override(entry) and level in {"phrase", "sentence"}:
        return 8
    if level == "char":
        return 9
    return 99


def is_noop_manual_hotfix(entry: Any) -> bool:
    return (
        getattr(entry, "tier", None) == "manual_hotfix"
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and getattr(entry, "trust", None) == TRUST_MACHINE
        and getattr(entry, "src", "") == getattr(entry, "tgt", "")
    )


def is_single_char_machine_override(entry: Any) -> bool:
    return (
        getattr(entry, "trust", None) == TRUST_MACHINE
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and len(getattr(entry, "src", "")) == 1
    )


def is_single_char_low_trust_phrase(entry: Any) -> bool:
    return (
        getattr(entry, "tier", None) in {"base", "domain"}
        and getattr(entry, "trust", None) in {TRUST_SEED, TRUST_MACHINE}
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and len(getattr(entry, "src", "")) == 1
    )


def is_low_trust_severe_contraction(entry: Any) -> bool:
    """Reject gloss-shaped low-trust rewrites that collapse a clause into a headword.

    Legacy dictionary imports contain reversed definition fragments such as a
    four-to-seven character description mapped to a one- or two-character
    dictionary headword.  These rows are unsafe as context-free runtime
    translations.  Curated entries are intentionally exempt and can restore a
    reviewed contraction with explicit provenance.
    """

    src = getattr(entry, "src", "")
    tgt = getattr(entry, "tgt", "")
    return (
        getattr(entry, "tier", None) in {"base", "domain"}
        and getattr(entry, "trust", None) in {TRUST_SEED, TRUST_MACHINE}
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and ((len(src) == 3 and len(tgt) == 1) or (len(src) >= 4 and len(tgt) <= 2 and len(tgt) * 2 <= len(src)))
    )


def is_low_trust_long_phrase(entry: Any, min_len: int = LOW_TRUST_LONG_PHRASE_MIN_LEN) -> bool:
    return (
        getattr(entry, "tier", None) in {"base", "domain"}
        and getattr(entry, "trust", None) in {TRUST_SEED, TRUST_MACHINE}
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and len(getattr(entry, "src", "")) >= min_len
    )


def is_unsafe_context_free_low_trust_edge(entry: Any) -> bool:
    return (
        getattr(entry, "tier", None) in {"base", "domain", "manual_hotfix"}
        and getattr(entry, "trust", None) in {TRUST_SEED, TRUST_MACHINE}
        and getattr(entry, "level", None) in {"phrase", "sentence"}
        and (
            (getattr(entry, "src", ""), getattr(entry, "tgt", "")) in _UNSAFE_CONTEXT_FREE_LEGACY_EDGES
            or (
                getattr(entry, "src", "") != getattr(entry, "tgt", "")
                and getattr(entry, "tgt", "") in _SENSITIVE_LOW_TRUST_TARGETS
            )
        )
    )


def is_definition_like_low_trust_phrase(entry: Any) -> bool:
    if (
        getattr(entry, "tier", None) not in {"base", "domain"}
        or getattr(entry, "trust", None) not in {TRUST_SEED, TRUST_MACHINE}
        or getattr(entry, "level", None) not in {"phrase", "sentence"}
    ):
        return False

    src = getattr(entry, "src", "")
    tgt = getattr(entry, "tgt", "")
    if (
        (src, tgt) in _SHORT_REVERSE_DEFINITION_EDGES
        or (len(src) == 2 and src.startswith("如") and tgt in _REVERSE_EXAMPLE_CATEGORY_TARGETS)
        or src in _DEFINITION_CATEGORY_TERMS
        or src in _SHORT_REVERSE_DEFINITION_GLOSSES
        or src.endswith(_DEFINITION_CATEGORY_SUFFIXES)
    ):
        return True

    if len(src) < 4:
        return False

    if src.startswith(_DEFINITION_STRONG_PREFIXES):
        return True

    if src.startswith(_DEFINITION_SOFT_PREFIXES):
        return ("的" in src) or (len(src) >= 8)

    return False


def is_non_hanji_target(entry: Any) -> bool:
    if (
        getattr(entry, "tier", None) not in {"base", "domain", "manual_hotfix"}
        or getattr(entry, "trust", None) not in {TRUST_SEED, TRUST_MACHINE}
        or getattr(entry, "level", None) not in {"phrase", "sentence", "char"}
    ):
        return False

    tgt = getattr(entry, "tgt", "")
    if not tgt:
        return True
    return not contains_han_ideograph(tgt)


def runtime_exclusion_reason(entry: Any) -> str | None:
    if getattr(entry, "context", None) is not None and not is_trusted_context_entry(entry):
        return RUNTIME_FILTER_UNTRUSTED_CONTEXT_ENTRY
    if is_noop_manual_hotfix(entry):
        return RUNTIME_FILTER_NOOP_MANUAL_HOTFIX
    if is_single_char_machine_override(entry):
        return RUNTIME_FILTER_SINGLE_CHAR_MACHINE
    if is_single_char_low_trust_phrase(entry):
        return RUNTIME_FILTER_SINGLE_CHAR_LOW_TRUST_PHRASE
    if is_unsafe_context_free_low_trust_edge(entry):
        return RUNTIME_FILTER_UNSAFE_CONTEXT_FREE_EDGE
    if is_low_trust_long_phrase(entry):
        return RUNTIME_FILTER_LOW_TRUST_LONG_PHRASE
    if is_low_trust_severe_contraction(entry):
        return RUNTIME_FILTER_LOW_TRUST_SEVERE_CONTRACTION
    if is_definition_like_low_trust_phrase(entry):
        return RUNTIME_FILTER_DEFINITION_LIKE_LOW_TRUST
    if is_non_hanji_target(entry):
        return RUNTIME_FILTER_NON_HANJI_TARGET
    return None
