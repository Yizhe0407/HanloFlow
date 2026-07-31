from types import SimpleNamespace

from taigi_converter.lexicon_policy import (
    RUNTIME_FILTER_LOW_TRUST_SEVERE_CONTRACTION,
    RUNTIME_FILTER_UNSAFE_CONTEXT_FREE_EDGE,
    TRUST_AI_REVIEWED,
    TRUST_HUMAN,
    is_definition_like_low_trust_phrase,
    is_low_trust_severe_contraction,
    is_trusted_context_entry,
    is_unsafe_context_free_low_trust_edge,
    normalize_trust,
    runtime_exclusion_reason,
    runtime_layer_rank,
)


def _entry(*, trust: str, tier: str = "manual", level: str = "phrase", context=None):
    return SimpleNamespace(
        trust=trust,
        tier=tier,
        level=level,
        context=context,
        status="active",
    )


def test_ai_reviewed_is_a_first_class_explicit_provenance() -> None:
    assert (
        normalize_trust(
            trust=TRUST_AI_REVIEWED,
            source="curation:semantic_audit",
            updated_by="codex",
            tier="manual",
        )
        == TRUST_AI_REVIEWED
    )


def test_runtime_precedence_keeps_human_review_ahead_of_ai_review() -> None:
    human = _entry(trust=TRUST_HUMAN)
    ai_reviewed = _entry(trust=TRUST_AI_REVIEWED)
    core = _entry(trust=TRUST_HUMAN, tier="core")

    assert runtime_layer_rank(human) < runtime_layer_rank(ai_reviewed) < runtime_layer_rank(core)


def test_ai_reviewed_context_is_curated_but_not_mislabeled_human() -> None:
    entry = _entry(
        trust=TRUST_AI_REVIEWED,
        context={"right_regex": "^(?!其)"},
    )

    assert entry.trust != TRUST_HUMAN
    assert is_trusted_context_entry(entry)


def test_low_trust_gloss_fragments_are_excluded_by_general_shape() -> None:
    for source in (
        "位於苗栗縣內",
        "用來支撐身體",
        "病名",
        "植物名",
        "法律名詞",
        "地名用字",
        "內臟之一",
    ):
        entry = _entry(trust="seed", tier="base")
        entry.src = source
        entry.tgt = "錯誤特定詞"
        assert is_definition_like_low_trust_phrase(entry), source


def test_gloss_filter_does_not_treat_curated_or_ordinary_terms_as_definition_fragments() -> None:
    curated = _entry(trust=TRUST_AI_REVIEWED, tier="manual")
    curated.src = "位於苗栗縣內"
    curated.tgt = "位佇苗栗縣內"
    assert not is_definition_like_low_trust_phrase(curated)

    ordinary = _entry(trust="seed", tier="base")
    ordinary.src = "姓名"
    ordinary.tgt = "姓名"
    assert not is_definition_like_low_trust_phrase(ordinary)


def test_low_trust_severe_contraction_filters_reverse_dictionary_glosses() -> None:
    for source, target in (
        ("上有細溝", "唱片"),
        ("對人表示抱歉", "歹勢"),
        ("寒涼陰冷的風", "冷風"),
    ):
        entry = _entry(trust="seed", tier="base")
        entry.src = source
        entry.tgt = target
        assert is_low_trust_severe_contraction(entry), (source, target)
        assert runtime_exclusion_reason(entry) == RUNTIME_FILTER_LOW_TRUST_SEVERE_CONTRACTION


def test_three_to_one_low_trust_glosses_are_fail_closed() -> None:
    for source, target in (
        ("一種菜", "菲"),
        ("文體名", "箴"),
        ("狗叫聲", "汪"),
    ):
        entry = _entry(trust="seed", tier="base")
        entry.src = source
        entry.tgt = target
        assert is_low_trust_severe_contraction(entry), (source, target)
        assert runtime_exclusion_reason(entry) == RUNTIME_FILTER_LOW_TRUST_SEVERE_CONTRACTION


def test_short_biological_reverse_definition_glosses_are_fail_closed() -> None:
    for source, target in (
        ("七瓣", "葫"),
        ("國名", "葛"),
        ("桑科", "榕"),
        ("眼大", "貓"),
        ("體長", "鯉"),
    ):
        entry = _entry(trust="seed", tier="base")
        entry.src = source
        entry.tgt = target
        assert is_definition_like_low_trust_phrase(entry), (source, target)
        assert runtime_exclusion_reason(entry) == "definition_like_low_trust_phrase"


def test_low_trust_reverse_gloss_policy_covers_the_full_governed_source_data() -> None:
    import json
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "data" / "lexicon_entries.jsonl"
    entries = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    three_to_one = [
        entry
        for entry in entries
        if entry.get("status") == "active"
        and entry.get("tier") in {"base", "domain"}
        and entry.get("trust") in {"seed", "machine"}
        and entry.get("level") in {"phrase", "sentence"}
        and len(entry.get("src", "")) == 3
        and len(entry.get("tgt", "")) == 1
    ]
    assert len(three_to_one) == 63
    for row in three_to_one:
        entry = SimpleNamespace(**row)
        assert runtime_exclusion_reason(entry) == RUNTIME_FILTER_LOW_TRUST_SEVERE_CONTRACTION, row["entry_id"]

    short_sources = {
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
    short_rows = [entry for entry in entries if entry.get("status") == "active" and entry.get("src") in short_sources]
    assert {entry["src"] for entry in short_rows} == short_sources
    for row in short_rows:
        entry = SimpleNamespace(**row)
        assert runtime_exclusion_reason(entry) == "definition_like_low_trust_phrase", row["entry_id"]


def test_curated_three_to_one_entry_can_use_the_reviewed_escape_hatch() -> None:
    curated = _entry(trust=TRUST_AI_REVIEWED, tier="manual")
    curated.src = "白水煮"
    curated.tgt = "煠"
    assert not is_low_trust_severe_contraction(curated)
    assert runtime_exclusion_reason(curated) is None


def test_severe_contraction_filter_is_fail_closed_but_curated_entries_can_restore_reviewed_terms() -> None:
    ordinary = _entry(trust="seed", tier="base")
    ordinary.src = "名落孫山"
    ordinary.tgt = "落第"
    assert is_low_trust_severe_contraction(ordinary)

    curated = _entry(trust=TRUST_AI_REVIEWED, tier="manual")
    curated.src = "名落孫山"
    curated.tgt = "落第"
    assert not is_low_trust_severe_contraction(curated)

    non_contracted = _entry(trust="seed", tier="base")
    non_contracted.src = "垃圾"
    non_contracted.tgt = "糞埽"
    assert not is_low_trust_severe_contraction(non_contracted)


def test_unicode_17_extension_g_target_is_runtime_eligible_han_text() -> None:
    entry = _entry(trust="seed", tier="base")
    entry.src = "骯髒"
    entry.tgt = "癩𰣻"

    assert runtime_exclusion_reason(entry) is None


def test_private_use_detection_distinguishes_legacy_pua_from_extension_g() -> None:
    from taigi_converter.unicode_policy import private_use_code_points

    assert private_use_code_points("癩𰣻") == ()
    assert private_use_code_points("\uf5ea芳水") == (0xF5EA,)
    assert private_use_code_points("癩\U000ff5e7") == (0xFF5E7,)


def test_short_reverse_definition_edges_are_fail_closed_by_exact_direction() -> None:
    reverse_edges = (
        ("一例", "一切"),
        ("六寸", "海參"),
        ("初三", "月眉"),
        ("卵生", "鳥類"),
        ("失音", "白喉"),
        ("多汁", "楊桃"),
        ("多肉", "菜蟳"),
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
    )
    for source, target in reverse_edges:
        entry = _entry(trust="seed", tier="base")
        entry.src = source
        entry.tgt = target
        assert is_definition_like_low_trust_phrase(entry), (source, target)
        assert runtime_exclusion_reason(entry) == "definition_like_low_trust_phrase"

    safe_direction = _entry(trust="seed", tier="base")
    safe_direction.src = "冰糖"
    safe_direction.tgt = "糖霜"
    assert not is_definition_like_low_trust_phrase(safe_direction)

    # 防線只比對已驗證的 exact edge，不得粗暴封鎖所有含「即」詞組。
    for source, target in (("即使", "準做"), ("即刻", "隨即")):
        valid_rewrite = _entry(trust="machine", tier="manual_hotfix")
        valid_rewrite.src = source
        valid_rewrite.tgt = target
        assert not is_definition_like_low_trust_phrase(valid_rewrite)
        assert runtime_exclusion_reason(valid_rewrite) is None


def test_known_unsafe_short_edges_are_fail_closed_without_global_length_heuristics() -> None:
    unsafe_edges = (
        ("得利", "受害", "base", "seed"),
        ("大食", "小食", "base", "seed"),
        ("奉贈", "似", "base", "seed"),
        ("念誦", "表白", "base", "seed"),
        ("刀術", "武術", "base", "seed"),
        ("十位", "數位", "base", "seed"),
        ("大宗", "大筆", "base", "seed"),
        ("性質", "性", "manual_hotfix", "machine"),
        ("少少", "濟濟", "base", "seed"),
        ("大筏", "查", "base", "seed"),
        ("憑借", "階", "base", "seed"),
        ("長衣", "袍", "base", "seed"),
        ("短衣", "褙", "base", "seed"),
        ("美石", "玫", "base", "seed"),
        ("玉光", "瑛", "base", "seed"),
        ("煉鐵", "鋼", "base", "seed"),
        ("竹簟", "笙", "base", "seed"),
        ("置身", "廁", "base", "seed"),
        ("舉起", "拯", "base", "seed"),
        ("貪求", "殉", "base", "seed"),
        ("雞蟲", "雞", "base", "seed"),
        ("危殆", "圾", "base", "seed"),
        ("多言", "諜", "base", "seed"),
        ("很近", "咫", "base", "seed"),
        ("楯脊", "瓦", "base", "seed"),
        ("無義", "喲", "base", "seed"),
        ("圓鍬", "鉛筆仔", "manual_hotfix", "machine"),
        ("嚴寒", "生冷", "manual_hotfix", "machine"),
        ("劈刀", "柴鍥", "manual_hotfix", "machine"),
        ("四份", "四捻", "manual_hotfix", "machine"),
        ("回轉", "斡輾轉", "manual_hotfix", "machine"),
        ("乃是", "蓋", "base", "seed"),
        ("佐助", "讚", "base", "seed"),
        ("佞臣", "倖", "base", "seed"),
        ("依託", "俚", "base", "seed"),
        ("俸祿", "秩", "base", "seed"),
        ("承受", "忍", "manual_hotfix", "machine"),
        ("正值", "正", "manual_hotfix", "machine"),
        ("拿去吃", "孝孤", "manual_hotfix", "machine"),
        ("店家", "店頭家", "manual_hotfix", "machine"),
        ("常見", "捷看", "manual_hotfix", "machine"),
        ("常規", "紀綱", "manual_hotfix", "machine"),
        ("承繼", "過房", "manual_hotfix", "machine"),
        ("招集", "募集", "manual_hotfix", "machine"),
        ("界限", "地界", "manual_hotfix", "machine"),
        ("喜好", "興", "manual_hotfix", "machine"),
        ("不只", "毋但", "manual_hotfix", "machine"),
        ("不料", "無疑悟", "manual_hotfix", "machine"),
        ("務必", "一定", "manual_hotfix", "machine"),
        ("即使", "著算", "manual_hotfix", "machine"),
        ("努力", "拚勢", "manual_hotfix", "machine"),
        ("升高", "衝懸", "manual_hotfix", "machine"),
        ("去世", "老去", "manual_hotfix", "machine"),
        ("吃藥", "食藥仔", "manual_hotfix", "machine"),
        ("包圍", "圍", "manual_hotfix", "machine"),
        ("包容", "包涵", "manual_hotfix", "machine"),
        ("同樣", "平平", "manual_hotfix", "machine"),
        ("合適", "好勢", "manual_hotfix", "machine"),
        ("不久", "無偌久", "manual_hotfix", "machine"),
        ("不怕", "毋驚", "manual_hotfix", "machine"),
        ("不肯", "毋肯", "manual_hotfix", "machine"),
        ("丟失", "拍毋見", "manual_hotfix", "machine"),
        ("刮痧", "掠痧", "manual_hotfix", "machine"),
        ("力氣", "氣力", "manual_hotfix", "machine"),
        ("勝過", "贏過", "manual_hotfix", "machine"),
        ("勞神", "損神", "manual_hotfix", "machine"),
        ("勞累", "疲勞", "manual_hotfix", "machine"),
        ("勤勉", "骨力", "manual_hotfix", "machine"),
        ("勾芡", "牽羹", "manual_hotfix", "machine"),
        ("匆忙", "趕狂", "manual_hotfix", "machine"),
        ("化痰", "去痰", "manual_hotfix", "machine"),
        ("午飯", "日晝頓", "manual_hotfix", "machine"),
        ("午飯", "中晝飯", "manual_hotfix", "machine"),
        ("即將", "得欲", "manual_hotfix", "machine"),
        ("即將", "咧欲", "manual_hotfix", "machine"),
        ("原先", "原早", "manual_hotfix", "machine"),
        ("厭惡", "討厭", "manual_hotfix", "machine"),
        ("及早", "量早", "manual_hotfix", "machine"),
        ("及早", "冗早", "manual_hotfix", "machine"),
        ("取笑", "恥笑", "manual_hotfix", "machine"),
        ("受到", "受著", "manual_hotfix", "machine"),
        ("叢生", "密密生", "manual_hotfix", "machine"),
        ("可惜", "無彩", "manual_hotfix", "machine"),
        ("吃驚", "昂愕", "manual_hotfix", "machine"),
        ("吃驚", "著驚", "manual_hotfix", "machine"),
        ("吹風", "搧風", "manual_hotfix", "machine"),
        ("啟發", "啟示", "manual_hotfix", "machine"),
        ("埋怨", "怨嘆", "manual_hotfix", "machine"),
        ("堵塞", "塞死", "manual_hotfix", "machine"),
        ("塗抹", "糊", "manual_hotfix", "machine"),
        ("填平", "坉塗", "manual_hotfix", "machine"),
        ("母親", "阿母", "manual_hotfix", "machine"),
        ("洋人", "番仔", "manual_hotfix", "machine"),
        ("植物", "草木", "manual_hotfix", "machine"),
        ("活著", "活咧", "manual_hotfix", "machine"),
        ("呼喚", "呼", "manual_hotfix", "machine"),
        ("呼喚", "叫", "manual_hotfix", "machine"),
        ("回去", "轉去", "manual_hotfix", "machine"),
        ("父母親", "爸母", "manual_hotfix", "machine"),
        ("吹風", "放風聲", "manual_hotfix", "machine"),
        ("埋怨", "怨慼", "manual_hotfix", "machine"),
        ("堵塞", "滯滯", "manual_hotfix", "machine"),
        ("塗抹", "抹", "manual_hotfix", "machine"),
        ("塗抹", "抉", "manual_hotfix", "machine"),
        ("塗抹", "挲", "manual_hotfix", "machine"),
        ("填平", "坉平", "manual_hotfix", "machine"),
        ("洋人", "西洋人", "manual_hotfix", "machine"),
        ("洋人", "外國人", "manual_hotfix", "machine"),
        ("呼喚", "喊", "manual_hotfix", "machine"),
        ("各個", "逐个", "manual_hotfix", "machine"),
        ("各自", "隨人", "manual_hotfix", "machine"),
        ("名望", "聲望", "manual_hotfix", "machine"),
        ("吵架", "冤家", "manual_hotfix", "machine"),
        ("吵架", "相罵", "manual_hotfix", "machine"),
        ("吻合", "搭峇", "manual_hotfix", "machine"),
        ("吻合", "峇", "manual_hotfix", "machine"),
        ("吻合", "合", "manual_hotfix", "machine"),
        ("吻合", "符合", "manual_hotfix", "machine"),
        ("吻合", "相符", "manual_hotfix", "machine"),
        ("吻合", "一致", "manual_hotfix", "machine"),
        ("周到", "點陳", "manual_hotfix", "machine"),
        ("周遭", "周圍", "manual_hotfix", "machine"),
        ("周遭", "四箍輾轉", "manual_hotfix", "machine"),
        ("周遭", "四箍圍仔", "manual_hotfix", "machine"),
        ("咀嚼", "哺食", "manual_hotfix", "machine"),
        ("咀嚼", "哺", "manual_hotfix", "machine"),
        ("咀嚼", "卯", "manual_hotfix", "machine"),
        ("品性", "性地", "manual_hotfix", "machine"),
        ("品性", "人格", "manual_hotfix", "machine"),
        ("品行", "心行", "manual_hotfix", "machine"),
        ("哽咽", "喉實", "manual_hotfix", "machine"),
        ("哽咽", "喉滇", "manual_hotfix", "machine"),
        ("唆使", "煽動", "manual_hotfix", "machine"),
        ("唆使", "拐弄", "manual_hotfix", "machine"),
        ("唆使", "使弄", "manual_hotfix", "machine"),
        ("唆使", "呲", "manual_hotfix", "machine"),
        ("唾棄", "呸瀾", "manual_hotfix", "machine"),
        ("唾棄", "漚屎", "manual_hotfix", "machine"),
        ("商榷", "參詳", "manual_hotfix", "machine"),
        ("善後", "帕尾", "manual_hotfix", "machine"),
        ("善後", "收尾", "manual_hotfix", "machine"),
        ("善終", "大葩尾", "manual_hotfix", "machine"),
        ("善終", "好尾景", "manual_hotfix", "machine"),
        ("善終", "好尾梢", "manual_hotfix", "machine"),
        ("喜劇", "喜齣", "manual_hotfix", "machine"),
        ("喜劇", "笑詼齣", "manual_hotfix", "machine"),
        ("喜歡", "合意", "manual_hotfix", "machine"),
        ("喜歡", "佮意", "manual_hotfix", "machine"),
        ("喜歡", "愛", "manual_hotfix", "machine"),
        ("喜歡", "意愛", "manual_hotfix", "machine"),
        ("喜餅", "盒仔餅", "manual_hotfix", "machine"),
        ("喜餅", "大餅", "manual_hotfix", "machine"),
        ("喝酒", "食酒", "manual_hotfix", "machine"),
        ("喝酒", "啉酒", "manual_hotfix", "machine"),
        ("喪志", "失志", "manual_hotfix", "machine"),
        ("喪志", "餒志", "manual_hotfix", "machine"),
        ("喪服", "麻衫", "manual_hotfix", "machine"),
        ("喪服", "孝衫", "manual_hotfix", "machine"),
        ("嗩吶", "鼓吹", "manual_hotfix", "machine"),
        ("嗩吶", "噯仔", "manual_hotfix", "machine"),
        ("嘆氣", "吐大氣", "manual_hotfix", "machine"),
        ("嘆氣", "吐氣", "manual_hotfix", "machine"),
    )
    for source, target, tier, trust in unsafe_edges:
        entry = _entry(trust=trust, tier=tier)
        entry.src = source
        entry.tgt = target
        assert is_unsafe_context_free_low_trust_edge(entry), (source, target)
        assert runtime_exclusion_reason(entry) == RUNTIME_FILTER_UNSAFE_CONTEXT_FREE_EDGE

    sensitive_target = _entry(trust="seed", tier="base")
    sensitive_target.src = "外國人"
    sensitive_target.tgt = "番仔"
    assert is_unsafe_context_free_low_trust_edge(sensitive_target)
    assert runtime_exclusion_reason(sensitive_target) == RUNTIME_FILTER_UNSAFE_CONTEXT_FREE_EDGE

    curated_sensitive_identity = _entry(trust=TRUST_HUMAN, tier="manual")
    curated_sensitive_identity.src = "番仔"
    curated_sensitive_identity.tgt = "番仔"
    assert not is_unsafe_context_free_low_trust_edge(curated_sensitive_identity)

    valid_short_rewrite = _entry(trust="seed", tier="base")
    valid_short_rewrite.src = "垃圾"
    valid_short_rewrite.tgt = "糞埽"
    assert not is_unsafe_context_free_low_trust_edge(valid_short_rewrite)
    assert runtime_exclusion_reason(valid_short_rewrite) is None

    for source, target in (
        ("青椒", "大同仔"),
        ("點滴", "大筒"),
        ("小肚", "膀胱"),
        ("披著", "幔"),
        ("圓鍬", "沙挑"),
        ("嚴寒", "大寒"),
    ):
        valid_attested_rewrite = _entry(trust="seed", tier="base")
        valid_attested_rewrite.src = source
        valid_attested_rewrite.tgt = target
        assert not is_unsafe_context_free_low_trust_edge(valid_attested_rewrite)
        assert runtime_exclusion_reason(valid_attested_rewrite) is None


def test_reverse_example_category_family_is_fail_closed_without_blocking_ordinary_ru_terms() -> None:
    for source, target in (
        ("如壹", "大寫"),
        ("如我", "代名詞"),
        ("如水", "液體"),
        ("如煤", "燃料"),
        ("如金", "金屬"),
        ("如魚", "海產"),
    ):
        entry = _entry(trust="seed", tier="base")
        entry.src = source
        entry.tgt = target
        assert is_definition_like_low_trust_phrase(entry), (source, target)

    ordinary = _entry(trust="seed", tier="base")
    ordinary.src = "如何"
    ordinary.tgt = "怎樣"
    assert not is_definition_like_low_trust_phrase(ordinary)
