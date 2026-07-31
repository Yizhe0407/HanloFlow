from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "lexicon_entries.jsonl"
CORE_PATH = DATA_DIR / "core_lexicon.json"


def _active_rows() -> list[dict[str, Any]]:
    return [
        row
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("status") == "active"
    ]


class LexiconDataGovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _active_rows()

    def test_nursing_roles_are_not_narrowed_to_nurse_practitioner_title(self) -> None:
        failures: list[str] = []
        for row in self.rows:
            source = row["src"]
            target = row["tgt"]
            if "護理人員" in source and "護理人員" not in target:
                failures.append(f"{row['entry_id']}: {source} -> {target}")
            if "護士" in source and source != "辯護士" and "護士" not in target:
                failures.append(f"{row['entry_id']}: {source} -> {target}")
        self.assertEqual(failures, [])

    def test_legally_distinct_nursing_terms_are_explicitly_protected(self) -> None:
        by_source = {row["src"]: row for row in self.rows}
        for term in ("護士", "護理人員"):
            with self.subTest(term=term):
                row = by_source[term]
                self.assertEqual(row["tgt"], term)
                self.assertEqual(row["protected"]["category"], "technical_term")
                self.assertTrue(row["protected"]["reason"].strip())

    def test_invalid_reverse_translations_are_disabled(self) -> None:
        invalid_pairs = {
            ("人客", "主人"),
            ("會議時", "表決"),
            ("飲食", "食物件"),
            ("火車頭", "車母"),
            ("干焦", "只是"),
            ("進前", "以前"),
            ("唏嚇叫", "嚇嚇叫"),
            ("燒的", "涼的"),
            ("倒手", "正手"),
            ("開會時", "列席"),
            ("用餐時", "西餐"),
            ("袂記得", "會記得"),
            ("無夠", "有夠"),
            ("別款", "仝款"),
            ("彼站", "這站"),
            ("易燃", "硫"),
            ("走動", "行振動"),
            ("野球", "棒球"),
            ("雨天", "落雨天"),
            ("窗仔", "窗仔門"),
            ("拖仔", "淺拖仔"),
            ("一下就好", "一下仔就好"),
            ("天上", "天頂"),
            ("門跤口", "門口"),
            ("轉來", "倒轉來"),
            ("拍開", "展開"),
            ("等一下", "等咗"),
            ("外口", "外頭"),
            ("物仔", "物件"),
            ("偏心", "大細心"),
            ("倚近", "近倚"),
            ("使用", "用"),
            ("使用的", "用的"),
            ("遺失物", "毋見的物件"),
            ("目的地", "欲去的所在"),
            ("物品", "物仔"),
            ("車牌號碼", "車牌號"),
            ("本人", "家己"),
            ("大聲", "昂聲"),
            ("別人", "人"),
            ("一定", "定著"),
            ("等一下", "等咧"),
            ("喧嘩", "噓嚇叫"),
            ("播放", "播送"),
            ("按怎", "啥款"),
            ("記得", "記著"),
            ("拄著", "遭遇"),
            ("酒醉", "馬西馬西"),
            ("跋倒", "摔倒"),
            ("走相逐", "走相掠"),
            ("嘔吐", "吐"),
            ("包仔", "肉包"),
            ("封喙", "窒喙空"),
            ("情緒", "起毛"),
            ("擄獲", "俘"),
            ("毋但", "不止"),
            ("家私", "家伙"),
            ("越頭", "翻頭"),
            ("合股", "鬥股"),
            ("冤家", "對頭"),
            ("多謝", "感謝"),
            ("阻擋", "擋咧"),
            ("拍毋見", "拍無去"),
            ("古錐", "可愛"),
            ("價數", "價錢"),
            ("旁邊", "隔壁"),
            ("厲害", "利害"),
            ("票價是", "價錢是"),
            ("等等", "等陣"),
            ("等等", "等陣仔"),
            ("南下", "南落"),
            ("最後", "上後"),
            ("扯後腿", "上後跤"),
            ("如飢", "需要"),
            ("仝位", "別搭"),
            ("汁多", "梨"),
            ("跑路", "走路"),
            ("單性花", "杉"),
        }
        active_pairs = {(row["src"], row["tgt"]) for row in self.rows}
        self.assertTrue(invalid_pairs.isdisjoint(active_pairs))

    def test_cross_language_homograph_uses_context_instead_of_global_guest_rewrite(self) -> None:
        core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))
        guest_mapping = next(row for row in core_rows if row.get("src") == "客人" and row.get("tgt") == "人客")
        self.assertEqual(guest_mapping.get("status"), "disabled")

        hakka_mapping = next(row for row in self.rows if row["src"] == "客家人")
        self.assertEqual(hakka_mapping["tgt"], "客人")
        self.assertEqual(hakka_mapping["trust"], "seed")

        by_source = {row["src"]: row for row in self.rows}
        self.assertEqual(by_source["幫客人"]["tgt"], "替人客")
        self.assertEqual(by_source["協助客人"]["tgt"], "鬥相共人客")

    def test_active_lexicon_has_no_exact_rewrite_cycles(self) -> None:
        rewrites = {(row["src"], row["tgt"]) for row in self.rows if row["src"] != row["tgt"]}
        cycles = sorted((source, target) for source, target in rewrites if (target, source) in rewrites)
        self.assertEqual(cycles, [])

    def test_context_sensitive_entries_have_boundary_guards(self) -> None:
        by_id = {row["entry_id"]: row for row in self.rows}
        expected_contexts = {
            "lx_2056fc68536b": {"right_regex": "^(?!臨)"},
            "lx_12e03343f381": {"right_regex": "^(?!車|遠|後)"},
            "lx_514r000000025": {"right_regex": "^(?!仔)"},
            "lx_92a000000001": {"right_regex": "^(?!錢)"},
            "lx_661cbe37d371": {
                "left_regex": (
                    r"(?:^|[，。！？；：、\s]|患有|感染|罹患|得到|得了|預防|防治|治療|篩檢|檢查|避免|關於)$"
                ),
                "right_regex": "^(?!症|疾|變|灶|理|毒|菌)",
            },
            "lx_27092f5b0950": {"right_regex": "^(?!仔)"},
            "lx_79d48e388cff": {"right_regex": "^(?!仔)"},
            "lx_2e7c4ab15d90": {"left_regex": "(?<!上)$"},
            "lx_4c2e7ab15d90": {"left_regex": "(?<!上)$"},
            "lx_149c000000009": {"left_regex": "(?<![〇零一二三四五六七八九十廿卅0-9])$"},
            "lx_162c00000000b": {"left_regex": "(?<![〇零一二三四五六七八九十廿卅0-9])$"},
        }
        for entry_id, expected_context in expected_contexts.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["context"], expected_context)

    def test_verified_canonical_terms_are_protected(self) -> None:
        by_source = {row["src"]: row for row in self.rows}
        for term in (
            "方便",
            "差不多",
            "外口",
            "物仔",
            "偏心",
            "倚近",
            "使用",
            "目的地",
            "物品",
            "車牌號碼",
            "本人",
            "大聲",
            "別人",
            "一定",
            "定著",
            "無一定",
            "等一下",
            "等咧",
            "播放",
            "播送",
            "唏嚇叫",
            "按怎",
            "記得",
            "拄著",
            "酒醉",
            "跋倒",
            "走相逐",
            "封喙",
            "擄獲",
            "大小朋友",
            "毋但",
            "家私",
            "越頭",
            "合股",
            "冤家",
            "多謝",
            "拍毋見",
            "古錐",
            "毋但是",
            "價數",
        ):
            with self.subTest(term=term):
                row = by_source[term]
                self.assertEqual(row["tgt"], term)
                self.assertEqual(row["protected"]["category"], "lexical_identity")

    def test_round532_low_trust_gloss_cleanup_is_governed(self) -> None:
        legacy_pairs = {
            ("為人", "傷慢矣"),
            ("複姓", "公子"),
            ("國君", "中央"),
            ("故名", "倒吊"),
            ("教學活動中", "教科書"),
            ("病名", "中風"),
            ("魚名", "串仔"),
            ("植物名", "拍某菜"),
            ("動物名", "吳郭魚"),
            ("地名用字", "壩"),
            ("或入境證", "入境"),
            ("位於苗栗縣內", "公館鄉"),
            ("位於腰部", "腎"),
            ("昏昧不明", "聾"),
            ("姪兒", "孫仔"),
            ("姪女", "查某孫"),
            ("姪子", "孫仔"),
            ("法律名詞", "當事人"),
            ("疾病名", "疳積"),
            ("譯音用字", "迦"),
            ("比熟鐵質硬", "鋼"),
            ("用來支撐身體", "枴"),
            ("宋廢", "從事"),
            ("利用擴音器", "播出"),
            ("直接言明", "愕"),
            ("內臟之一", "脾"),
            ("前腳很短", "狽"),
            ("奔跑", "浪"),
        }
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        statuses = {(row["src"], row["tgt"]): row["status"] for row in all_rows}
        self.assertEqual(
            {pair: statuses.get(pair) for pair in legacy_pairs},
            {pair: "disabled" for pair in legacy_pairs},
        )

        expected_active = {
            "為人": "做人",
            "複姓": "複姓",
            "國君": "君王",
            "故名": "故名",
            "教學活動中": "教學活動中",
            "病名": "病名",
            "魚名": "魚名",
            "植物名": "植物名",
            "動物名": "動物名",
            "地名用字": "地名用字",
            "位於苗栗縣內": "位佇苗栗縣內",
            "位於腰部": "位佇腰部",
            "昏昧不明": "昏昧不明",
            "姪兒": "姪仔",
            "姪女": "姪女",
            "姪子": "姪仔",
            "法律名詞": "法律名詞",
            "疾病名": "疾病名",
            "譯音用字": "譯音用字",
            "比熟鐵質硬": "比熟鐵質較硬",
            "用來支撐身體": "用來支撐身軀",
            "利用擴音器": "利用擴音器",
            "直接言明": "直接講明",
            "內臟之一": "內臟之一",
            "前腳很短": "頭前跤足短",
            "奔跑": "踉",
        }
        round_rows = {
            row["src"]: row
            for row in self.rows
            if row.get("source") == "curation:round532_low_trust_gloss_root_cleanup"
        }
        self.assertEqual(set(round_rows), set(expected_active))
        for source, target in expected_active.items():
            with self.subTest(source=source):
                row = round_rows[source]
                self.assertEqual(row["tgt"], target)
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["trust"], "ai_reviewed")

        identity_sources = {source for source, target in expected_active.items() if source == target}
        self.assertEqual(
            {source: round_rows[source]["protected"]["category"] for source in identity_sources},
            {source: "lexical_identity" for source in identity_sources},
        )
        self.assertNotIn("或入境證", round_rows)
        self.assertNotIn("宋廢", round_rows)

    def test_round532_fixed_point_semantic_safety_entries_are_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        legacy = next(row for row in all_rows if row["entry_id"] == "lx_c90d37a13b40")
        self.assertEqual((legacy["src"], legacy["tgt"], legacy["status"]), ("婆婆", "大家", "disabled"))

        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_fixed_point_semantic_safety"
        }
        self.assertEqual(
            {source: row["tgt"] for source, row in round_rows.items()},
            {"還是要": "猶是欲", "婆婆": "婆婆"},
        )
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual(round_rows["婆婆"]["protected"]["category"], "lexical_identity")

    def test_round532_runtime_risk_root_cleanup_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_30b80e1cec2d",
            "lx_6724e21776d2",
            "lx_7d0696ab651c",
            "lx_9737b693441e",
            "lx_9472c7c1ac8c",
            "lx_cc5eb286e0b7",
            "lx_8da9d6810652",
            "lx_fb7052012851",
            "lx_266938a37298",
            "lx_1a6f082b3cbf",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {
            "萬二": "萬二",
            "互相洩底": "相卸代",
            "下蛋": "生卵",
            "下霜": "落霜",
            "下等": "下等",
            "氣候非常的冷": "氣候非常的冷",
            "藏身於錐形": "藏身於錐形",
            "廣泛傳播": "廣泛傳播",
            "自找死路": "自揣死路",
            "不壞": "不壞",
        }
        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_runtime_risk_root_cleanup"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2145)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                if row["src"] == row["tgt"]:
                    self.assertEqual(row["protected"]["category"], "lexical_identity")

    def test_round532_high_signal_risk_cleanup_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_26b108542cf6",
            "lx_24d4f133fbd7",
            "lx_3b780687a0a9",
            "lx_dc57a339473f",
            "lx_d797dc9541bf",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {
            "仝位": "仝位",
            "汁多": "汁濟",
            "跑路": "落跑",
            "單性花": "單性花",
        }
        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_high_signal_risk_cleanup"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2160)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")

        self.assertEqual(round_rows["仝位"]["protected"]["category"], "lexical_identity")
        self.assertEqual(round_rows["單性花"]["protected"]["category"], "technical_term")
        self.assertEqual(round_rows["單性花"]["protected"]["enforcement"], "strict")

    def test_round532_low_trust_contraction_policy_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_254be5691da6",
            "lx_a92c2c368a85",
            "lx_96bc476ed572",
            "lx_3c619f8f43f1",
            "lx_e4cd6053e9ba",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {
            "受到欺騙": "受騙",
            "資金的運轉": "資金的轉踅",
            "道德虧損": "失德",
            "涉及法律案件": "涉案",
        }
        round_rows = {
            row["src"]: row
            for row in self.rows
            if row.get("source") == "curation:round532_low_trust_contraction_policy"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2160)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")

    def test_round532_target_cascade_cleanup_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {"lx_d14f7b1c7257", "lx_3f7874a9deed"}
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {"凊嗽": "冷嗽", "冷嗽": "冷嗽", "涼水": "涼水"}
        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_target_cascade_cleanup"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2165)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
        for source in ("冷嗽", "涼水"):
            self.assertEqual(round_rows[source]["protected"]["category"], "lexical_identity")
            self.assertEqual(round_rows[source]["protected"]["enforcement"], "strict")

    def test_round532_machine_severe_contractions_are_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_30f7029beee1",
            "lx_8c6fe76db1f0",
            "lx_b756bc56f074",
            "lx_1e2854156aed",
            "lx_3787aa5c1251",
            "lx_ce3b8ff3407a",
            "lx_92c51021ea46",
            "lx_4582ef497dcb",
            "lx_a5cc0986ebb3",
            "lx_7fbd55779c7a",
            "lx_a03f3435b7ed",
            "lx_e7f1931bb2ce",
            "lx_492b16736073",
            "lx_3fdc9def0b0d",
            "lx_dd4a4c5d2ef7",
            "lx_16741981ffbe",
            "lx_678067643992",
            "lx_1345bf479e9f",
            "lx_df1809b84294",
            "lx_aa47ca2ad757",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {
            "不得善終": "歹死",
            "不良少年": "歹囝",
            "丟人現眼": "現世",
            "公共汽車": "巴士",
            "心狠手辣": "酷刑",
            "悶悶不樂": "鬱卒",
            "挑撥離間": "使弄",
            "月經來潮": "來洗",
            "有始有終": "透流",
            "那個時候": "彼站",
            "隨機應變": "變竅",
            "高抬貴手": "讓手",
            "心情煩躁": "心情煩躁",
            "心情鬱悶": "心情鬱悶",
            "恢復健康": "恢復健康",
            "料想不到": "料想不到",
            "江湖術士": "江湖術士",
            "穿越馬路": "穿越馬路",
            "自討苦吃": "自討苦吃",
            "連續不斷": "連續不斷",
        }
        round_rows = {
            row["src"]: row
            for row in self.rows
            if row.get("source") == "curation:round532_machine_severe_contraction_review"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2170)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                if row["src"] == "穿越馬路":
                    self.assertEqual(row["level"], "sentence")
                    self.assertNotIn("protected", row)
                elif row["src"] == row["tgt"]:
                    self.assertEqual(row["protected"]["category"], "lexical_identity")
                    self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round532_short_polysemy_context_fixes_are_governed(self) -> None:
        expected = {"大丈夫": "大丈夫", "下面": "下面", "不想": "無疑", "下風": "下風"}
        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_short_polysemy_context"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2175)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
        self.assertEqual(round_rows["大丈夫"]["protected"]["enforcement"], "strict")
        self.assertIn("right_regex", round_rows["下面"]["context"])
        self.assertIn("right_regex", round_rows["不想"]["context"])
        self.assertIn("right_regex", round_rows["下風"]["context"])

    def test_round532_short_context_free_machine_review_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_aea76a22bb45",
            "lx_b06a6ec7cca5",
            "lx_c3f2528f1351",
            "lx_c5c7e557e7b0",
            "lx_3464b7a2bb33",
            "lx_d7d20148c4bc",
            "lx_0395b1dd1eaa",
            "lx_fa75f1b18db7",
            "lx_518b61d0be32",
            "lx_97ea3260c458",
            "lx_1d64a310a734",
            "lx_28d5343bedfa",
            "lx_aacc2657f597",
            "lx_3a33b02333c8",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {
            "一生": "一世人",
            "丈夫": "查埔人",
            "上臂": "手股",
            "下回": "下擺",
            "下工": "放工",
            "下座": "下座",
            "下風": "輸勢",
            "不合": "不合",
            "不和": "不和",
            "不妨": "不妨",
            "不想": "無想欲",
            "世故": "世故",
            "下屬": "下屬",
            "下痢": "下痢",
        }
        round_rows = {
            row["src"]: row
            for row in all_rows
            if row.get("source") == "curation:round532_short_context_free_machine_review"
        }
        self.assertEqual({source: row["tgt"] for source, row in round_rows.items()}, expected)
        self.assertEqual({row["priority"] for row in round_rows.values()}, {2180})
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in round_rows.values()}, {"active"})

        for source in {"一生", "丈夫", "下風", "不想"}:
            with self.subTest(context_source=source):
                self.assertEqual(round_rows[source]["level"], "phrase")
                self.assertTrue(round_rows[source]["context"])
                self.assertNotIn("protected", round_rows[source])
        for source in {"不合", "不和", "世故"}:
            with self.subTest(exact_only_source=source):
                self.assertEqual(round_rows[source]["level"], "sentence")
                self.assertIsNone(round_rows[source]["context"])
                self.assertNotIn("protected", round_rows[source])
        for source in {"下座", "不妨", "下屬"}:
            with self.subTest(identity_source=source):
                self.assertEqual(round_rows[source]["protected"]["category"], "lexical_identity")
                self.assertEqual(round_rows[source]["protected"]["enforcement"], "strict")
        self.assertEqual(round_rows["下痢"]["protected"]["category"], "technical_term")
        self.assertEqual(round_rows["下痢"]["protected"]["enforcement"], "strict")

    def test_round532_second_short_context_free_machine_review_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_fc274b6a490a",
            "lx_faa784a44bbe",
            "lx_1e0cab78ddd3",
            "lx_9ef6e6d74eae",
            "lx_8ed1211b6b04",
            "lx_633195aa877e",
            "lx_2a8b4f20d8e9",
            "lx_eba1ee477e27",
            "lx_dc9eb9366a08",
            "lx_d11725fa1eda",
            "lx_d1dbb4c7636b",
            "lx_9a91caf4dabd",
            "lx_681a4ddebd45",
            "lx_d55d76e4070a",
            "lx_59ab24bffa08",
            "lx_a92ed5ce5935",
            "lx_c5b211e8f3de",
            "lx_d013521e81e8",
            "lx_6b79a2fa0a13",
            "lx_6f928ade8d2c",
            "lx_d040162707e8",
            "lx_2fc150ab0b44",
            "lx_2ba21587fe5e",
            "lx_f0a565bfd735",
            "lx_d0f0a36775cc",
            "lx_6a4c51f3beb4",
            "lx_fad76cdda490",
            "lx_c72ae972571c",
            "lx_75f946b0343f",
            "lx_bf256d6325f2",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        safe_mappings = {
            "中暑": "著痧",
            "中獎": "著獎",
            "乖戾": "聬儱",
            "乘涼": "歇涼",
            "乳缽": "研缽",
            "乾爽": "焦鬆",
            "乾癟": "脯脯",
            "些微": "峇微",
            "京劇": "京戲",
            "今晚": "下暗",
            "仍舊": "猶原",
        }
        fail_closed = {
            "丟掉",
            "丟棄",
            "丟臉",
            "中夜",
            "中筋",
            "中餐",
            "久遠",
            "乏味",
            "九孔",
            "乞求",
            "乳房",
            "乾枯",
            "乾涸",
            "乾燥",
            "亂說",
            "二胡",
            "些許",
            "交際",
            "亮麗",
        }
        round_rows = {
            row["src"]: row
            for row in all_rows
            if row.get("source") == "curation:round532_second_short_context_free_machine_review"
        }
        self.assertEqual(set(round_rows), set(safe_mappings) | fail_closed)
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in safe_mappings},
            safe_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in fail_closed},
            {source: source for source in fail_closed},
        )
        self.assertEqual({row["priority"] for row in round_rows.values()}, {2185})
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in round_rows.values()}, {"active"})
        self.assertEqual({round_rows[source]["level"] for source in safe_mappings}, {"phrase"})
        self.assertEqual({round_rows[source]["level"] for source in fail_closed}, {"sentence"})
        self.assertEqual({round_rows[source]["context"] for source in round_rows}, {None})
        self.assertFalse(any("protected" in round_rows[source] for source in fail_closed))

    def test_round532_third_short_context_free_machine_review_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_f23386d365f9",
            "lx_174c1a7cb33f",
            "lx_1818c03cf240",
            "lx_31fddf9a27b9",
            "lx_ef2a0185849c",
            "lx_b6acbf93ca11",
            "lx_c22b7e42540b",
            "lx_429b0dad341f",
            "lx_260e99ef771a",
            "lx_9163b60e30fc",
            "lx_ade137aeb22e",
            "lx_2154ea470873",
            "lx_6fce50614df5",
            "lx_597a38205493",
            "lx_62916165d331",
            "lx_a2088d1388dd",
            "lx_0f20d025e034",
            "lx_5fbd75401f94",
            "lx_a0bd9b6af4f9",
            "lx_391f60995f8a",
            "lx_0d6d6ee6f704",
            "lx_5276e6fc1e3d",
            "lx_239d6a384204",
            "lx_a6c14058e599",
            "lx_c10b04aa35a3",
            "lx_1336a54ec30d",
            "lx_0c5f8bbdaa63",
            "lx_e72798318a31",
            "lx_ce09d2f28069",
            "lx_647faeec70f7",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        global_mappings = {
            "仗勢": "靠勢",
            "仰泳": "死囡仔䖙",
            "伯母": "阿姆",
            "似乎": "敢若",
            "依舊": "猶原",
        }
        contextual_mappings = {
            "他日": "另日",
            "以便": "通好",
            "以往": "往時",
            "以後": "了後",
            "何必": "曷著",
            "何須": "曷著",
            "來生": "下世人",
            "供奉": "供",
            "供桌": "尪架桌",
            "依照": "照",
        }
        fail_closed = {
            "仔細",
            "仙草",
            "任憑",
            "仿效",
            "伉儷",
            "位移",
            "低垂",
            "作事",
            "作假",
            "作弄",
            "使勁",
            "使喚",
            "侍奉",
            "依次",
            "依附",
        }
        round_rows = {
            row["src"]: row
            for row in all_rows
            if row.get("source") == "curation:round532_third_short_context_free_machine_review"
        }
        self.assertEqual(set(round_rows), set(global_mappings) | set(contextual_mappings) | fail_closed)
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in global_mappings},
            global_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in contextual_mappings},
            contextual_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in fail_closed},
            {source: source for source in fail_closed},
        )
        self.assertEqual({row["priority"] for row in round_rows.values()}, {2190})
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in round_rows.values()}, {"active"})
        self.assertEqual({round_rows[source]["level"] for source in global_mappings}, {"phrase"})
        self.assertEqual({round_rows[source]["context"] for source in global_mappings}, {None})
        self.assertEqual({round_rows[source]["level"] for source in contextual_mappings}, {"phrase"})
        self.assertTrue(all(round_rows[source]["context"] for source in contextual_mappings))
        self.assertEqual({round_rows[source]["level"] for source in fail_closed}, {"sentence"})
        self.assertEqual({round_rows[source]["context"] for source in fail_closed}, {None})
        self.assertFalse(any("protected" in round_rows[source] for source in fail_closed))

    def test_round532_sensor_measurement_boundary_cleanup_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(by_id["lx_817ff75f50af"]["status"], "disabled")

        rows = [row for row in all_rows if row.get("source") == "curation:round532_sensor_measurement_boundary_cleanup"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["src"], row["tgt"]), ("器量", "度量"))
        self.assertEqual(row["level"], "phrase")
        self.assertEqual(row["priority"], 2191)
        self.assertEqual(row["trust"], "ai_reviewed")
        self.assertEqual(row["status"], "active")
        self.assertIn("right_regex", row["context"])

    def test_round532_fourth_short_context_free_machine_review_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_fb10feeb30f9",
            "lx_c22577f178d0",
            "lx_9bfbf268129b",
            "lx_990d7f4bb548",
            "lx_f68000c279da",
            "lx_ec5db1eb3e03",
            "lx_57ad35ac17af",
            "lx_88882c98fca8",
            "lx_9ac41190ee31",
            "lx_b159470cd693",
            "lx_768223e3c1d3",
            "lx_6106b56534ec",
            "lx_7b002c733806",
            "lx_e2d6a4ab52e7",
            "lx_bf94866f9110",
            "lx_fa479221c8f3",
            "lx_21a24ae76941",
            "lx_e587bfe3be14",
            "lx_d0371c28bb19",
            "lx_cfa696bfd984",
            "lx_132b703a8724",
            "lx_6f6b25ce131d",
            "lx_99e2358934c0",
            "lx_e1f9f15cb0d3",
            "lx_a980a054e083",
            "lx_e4d8164936c4",
            "lx_c9989a5984b0",
            "lx_0db803aad188",
            "lx_225a614c9960",
            "lx_585fc6b928dc",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        global_mappings = {"倒楣": "落衰", "倘若": "假使"}
        contextual_mappings = {
            "便秘": "祕結",
            "俏皮": "激骨",
            "信教": "入教",
            "信札": "批",
            "俯臥": "坦覆",
            "借住": "倚蹛",
            "倦怠": "厭𤺪",
            "假裝": "假影",
            "假錢": "假銀票",
            "偏旁": "字爿",
            "做愛": "相姦",
            "健忘": "無頭神",
            "健朗": "勇健",
            "偶爾": "有當時仔",
            "傍晚": "欲暗仔",
            "傳話": "寄聲",
            "傷神": "損神",
            "傷胃": "損胃",
            "傻瓜": "癮頭",
            "傾斜": "歪斜",
            "像樣": "成款",
            "儉省": "虯儉",
        }
        fail_closed = {"便衣", "倒流", "借錢", "偏激", "傳遞", "優劣"}
        round_rows = {
            row["src"]: row
            for row in all_rows
            if row.get("source") == "curation:round532_fourth_short_context_free_machine_review"
        }
        self.assertEqual(set(round_rows), set(global_mappings) | set(contextual_mappings) | fail_closed)
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in global_mappings},
            global_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in contextual_mappings},
            contextual_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in fail_closed},
            {source: source for source in fail_closed},
        )
        self.assertEqual({row["priority"] for row in round_rows.values()}, {2195})
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in round_rows.values()}, {"active"})
        self.assertEqual({round_rows[source]["level"] for source in global_mappings}, {"phrase"})
        self.assertEqual({round_rows[source]["context"] for source in global_mappings}, {None})
        self.assertEqual({round_rows[source]["level"] for source in contextual_mappings}, {"phrase"})
        self.assertTrue(all(round_rows[source]["context"] for source in contextual_mappings))
        self.assertEqual({round_rows[source]["level"] for source in fail_closed}, {"sentence"})
        self.assertEqual({round_rows[source]["context"] for source in fail_closed}, {None})
        self.assertFalse(any("protected" in round_rows[source] for source in fail_closed))

    def test_round532_fifth_short_context_free_machine_review_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        machine_ids = {
            "lx_576c1763bbfc",
            "lx_3c3cef59460b",
            "lx_24b098c246e3",
            "lx_76a239f412d5",
            "lx_3150e7575ad1",
            "lx_3803c90daa1d",
            "lx_7b59197501e7",
            "lx_795c543ef4bf",
            "lx_b9fc711532f9",
            "lx_0f1bc80585a9",
            "lx_232d8666821b",
            "lx_96a3be736c4a",
            "lx_808a8c7467aa",
            "lx_f46f6f5e3f66",
            "lx_f8d49b820883",
            "lx_63a67497a194",
            "lx_f0ae724b5ed9",
            "lx_a131c3b33bc6",
            "lx_f9eb013b6bf8",
            "lx_534d08ed9b1a",
            "lx_6cdaba929083",
            "lx_0c44035e9ee5",
            "lx_fcfb04633783",
            "lx_c374b4019600",
            "lx_62e11a4d3452",
            "lx_fbf632c7d900",
            "lx_8060449ffd4b",
            "lx_d59dae43aa95",
            "lx_6af5382d277f",
            "lx_17d33ed99bd3",
        }
        invalid_root_ids = {
            "lx_086681b558cb",
            "lx_7007628044c7",
            "lx_26e8e97511c8",
            "lx_704ddae65c19",
            "lx_8453eccac16f",
            "lx_6b36e3dd57a5",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = machine_ids | invalid_root_ids
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        global_mappings = {
            "冀望": "寄望",
            "冤屈": "枉屈",
            "冥紙": "銀紙",
        }
        contextual_mappings = {
            "元配": "原配",
            "兄長": "兄哥",
            "兆頭": "彩頭",
            "先前": "進前",
            "先鋒": "頭陣",
            "兒子": "後生",
            "內臟": "腹內",
            "全身": "規身軀",
            "兩邊": "兩爿",
            "公鴨": "鴨鵤",
            "兵卒": "兵仔",
            "具名": "徛名",
            "冒死": "胚命",
            "冬天": "寒天",
            "冬衣": "寒衫",
            "冰棒": "枝仔冰",
            "冰雹": "雹",
            "冷清": "稀微",
            "凡是": "見若",
            "凶日": "歹日",
            "凹陷": "塌窩",
        }
        fail_closed = {"充裕", "光亮", "免去", "入贅", "冷水", "凶惡"}
        round_rows = {
            row["src"]: row
            for row in all_rows
            if row.get("source") == "curation:round532_fifth_short_context_free_machine_review"
        }
        self.assertEqual(
            set(round_rows),
            set(global_mappings) | set(contextual_mappings) | fail_closed,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in global_mappings},
            global_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in contextual_mappings},
            contextual_mappings,
        )
        self.assertEqual(
            {source: round_rows[source]["tgt"] for source in fail_closed},
            {source: source for source in fail_closed},
        )
        self.assertEqual({row["priority"] for row in round_rows.values()}, {2200})
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in round_rows.values()}, {"active"})
        self.assertEqual({round_rows[source]["level"] for source in global_mappings}, {"phrase"})
        self.assertEqual({round_rows[source]["context"] for source in global_mappings}, {None})
        self.assertEqual(
            {round_rows[source]["level"] for source in contextual_mappings},
            {"phrase"},
        )
        self.assertTrue(all(round_rows[source]["context"] for source in contextual_mappings))
        self.assertEqual({round_rows[source]["level"] for source in fail_closed}, {"sentence"})
        self.assertEqual({round_rows[source]["context"] for source in fail_closed}, {None})
        self.assertFalse(any("protected" in round_rows[source] for source in fail_closed))

        core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))
        cold_row = next(row for row in core_rows if row.get("src") == "冷" and row.get("tgt") == "寒")
        self.assertTrue(cold_row["context"]["right_regex"])
        self.assertEqual(
            cold_row["source"],
            "curation:round532_fifth_root_context_governance",
        )

        but_row = by_id["lx_e1e2418c9776"]
        self.assertTrue(but_row["context"]["left_regex"])
        self.assertTrue(but_row["context"]["right_regex"])
        self.assertEqual(
            but_row["source"],
            "curation:round532_fifth_root_context_governance",
        )

        acquired_row = by_id["lx_532000000234"]
        self.assertEqual((acquired_row["src"], acquired_row["tgt"]), ("後天", "後日"))
        self.assertEqual(acquired_row["priority"], 2205)
        self.assertEqual(acquired_row["trust"], "ai_reviewed")
        self.assertTrue(acquired_row["context"]["left_regex"])
        self.assertTrue(acquired_row["context"]["right_regex"])

        disease_row = by_id["lx_532000000237"]
        self.assertEqual((disease_row["src"], disease_row["tgt"]), ("疾病", "病症"))
        self.assertEqual(disease_row["priority"], 2210)
        self.assertEqual(disease_row["trust"], "ai_reviewed")
        self.assertEqual(disease_row["context"], {"left_regex": "(?<!性)(?<!患)$"})

        protected_terms = {
            row["src"]: row for row in all_rows if row.get("entry_id") in {"lx_532000000235", "lx_532000000236"}
        }
        self.assertEqual(set(protected_terms), {"內臟脂肪", "全身性疾病"})
        for source, row in protected_terms.items():
            with self.subTest(protected_source=source):
                self.assertEqual(row["tgt"], source)
                self.assertEqual(row["priority"], 2210)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["protected"]["category"], "technical_term")
                self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round532_v5_generalized_root_fixes_are_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        disabled_ids = {
            "lx_9459bb8d9b1c",
            "lx_55108be99eb4",
            "lx_41cc10b42a07",
            "lx_458d6d9cf03a",
            "lx_6dee516d2b72",
            "lx_532000000053",
            "lx_532000000058",
            "lx_532000000059",
        }
        by_id = {row["entry_id"]: row for row in all_rows}
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        expected = {
            "我們": "阮",
            "袖子": "袖子",
            "一把剪刀": "一支鉸刀",
            "牛肉麵": "牛肉麵",
            "里長": "里長",
            "打電話給": "拍電話予",
            "風險評估": "風險評估",
            "一封信": "一張批",
            "線上": "線頂",
            "互相交換": "互相交換",
            "如果不要": "若毋愛",
        }
        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_v5_generalized_root_fixes"
        }
        self.assertEqual({src: row["tgt"] for src, row in round_rows.items()}, expected)
        for source, row in round_rows.items():
            with self.subTest(source=source):
                self.assertEqual(row["tier"], "manual")
                self.assertEqual(row["priority"], 2150)
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")

        self.assertIsInstance(round_rows["我們"].get("context"), dict)
        self.assertTrue(round_rows["我們"]["context"].get("right_regex"))

        for source in ("袖子", "牛肉麵", "里長", "風險評估", "互相交換"):
            with self.subTest(strict_source=source):
                self.assertEqual(round_rows[source]["protected"]["enforcement"], "strict")

        boundary_row = next(
            row for row in self.rows if row.get("source") == "curation:round532_strict_boundary_reconciliation"
        )
        self.assertEqual((boundary_row["src"], boundary_row["tgt"]), ("樓上", "樓頂"))
        self.assertEqual(boundary_row["context"], {"right_regex": "^(?!課)"})
        self.assertEqual(boundary_row["trust"], "ai_reviewed")
        self.assertEqual(boundary_row["status"], "active")

    def test_round524_ai_semantic_entries_are_not_mislabeled_human(self) -> None:
        reviewed_sources = {
            "放涼",
            "停止",
            "否則不要",
            "整理乾淨",
            "不要糖",
            "囥冷",
            "否則",
            "與",
        }
        matching = {row["src"]: row for row in self.rows if row["src"] in reviewed_sources}

        self.assertEqual(set(matching), reviewed_sources)
        self.assertEqual({row["trust"] for row in matching.values()}, {"ai_reviewed"})

    def test_independent_ai_review_entries_have_explicit_provenance(self) -> None:
        expected_provenance = {
            "每天打電話回家": "curation:round525_independent_ai_semantic_review",
            "內用還是外帶": "curation:round525_independent_ai_semantic_review",
            "資料還沒確認以前": "curation:round525_independent_ai_semantic_review",
            "他的聲音很低，我聽不清楚": ("curation:round525_independent_ai_semantic_review"),
            "破了這個案子": "curation:round531_semantic_precision",
            "電腦跑得很快": "curation:round525_independent_ai_semantic_review",
        }
        matching = {row["src"]: row for row in self.rows if row["src"] in expected_provenance}

        self.assertEqual(set(matching), set(expected_provenance))
        self.assertEqual({row["trust"] for row in matching.values()}, {"ai_reviewed"})
        self.assertEqual(
            {source: row["source"] for source, row in matching.items()},
            expected_provenance,
        )

    def test_legal_and_technical_terms_are_protected(self) -> None:
        by_source = {row["src"]: row for row in self.rows}
        for term in ("遺失物", "嘔吐袋", "情緒", "阻擋"):
            with self.subTest(term=term):
                row = by_source[term]
                self.assertEqual(row["tgt"], term)
                self.assertEqual(row["protected"]["category"], "technical_term")
                self.assertTrue(row["protected"]["reason"].strip())

    def test_contextual_synonym_choices_use_long_phrases(self) -> None:
        by_source = {row["src"]: row for row in self.rows}
        expected = {
            "等一下我先": "等咧我先",
            "等一下會": "等咧會",
            "等一下如果": "等咧若是",
            "等一下若是": "等咧若是",
            "等一下先": "等咧先",
            "不一定要": "無一定愛",
            "一定要": "一定愛",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(by_source[source]["tgt"], target)

    def test_round532_high_signal_root_governance_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_be656560d37b",
            "lx_ac84a5bcb999",
            "lx_4c698d8cf1ec",
            "lx_fddf2814f1be",
            "lx_2e6360007b03",
            "lx_8d2dfd02d71a",
            "lx_8cae6a86a038",
            "lx_05946c9b9ed8",
            "lx_6e5a0f7dab93",
            "lx_1dff7a530926",
            "lx_8e2078577da0",
            "lx_1a7b6ded05cc",
            "lx_f061b0c29567",
            "lx_532000000058",
            "lx_532000000253",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        round_rows = {
            row["src"]: row for row in self.rows if row.get("source") == "curation:round532_high_signal_root_governance"
        }
        expected = {
            "哪裡": "佗位",
            "下面": "下面",
            "詐騙": "詐騙",
            "周邊": "周邊",
            "教師": "教師",
            "民宅": "人家厝仔",
            "人家厝仔": "人家厝仔",
            "變聲": "轉聲",
            "書架": "冊架仔",
            "冊架仔": "冊架仔",
            "書櫃": "冊櫥",
            "冊櫥": "冊櫥",
            "一把鑰匙": "一支鎖匙",
            "二十一": "廿一",
            "哪裡哪裡": "毋敢當",
            "系統書櫃": "系統書櫃",
            "電子書櫃": "電子書櫃",
            "數位書櫃": "數位書櫃",
            "虛擬書櫃": "虛擬書櫃",
            "線上書櫃": "線上書櫃",
        }
        self.assertEqual({source: row["tgt"] for source, row in round_rows.items()}, expected)
        self.assertEqual({row["trust"] for row in round_rows.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in round_rows.values()}, {"active"})
        self.assertEqual({row["priority"] for row in round_rows.values()}, {2215, 2220})

        self.assertEqual(round_rows["周邊"]["level"], "sentence")
        self.assertNotIn("protected", round_rows["周邊"])
        for source in ("民宅", "變聲", "書架", "書櫃", "二十一", "哪裡哪裡"):
            with self.subTest(contextual_source=source):
                self.assertTrue(round_rows[source]["context"])

        strict_sources = {
            "下面",
            "詐騙",
            "教師",
            "人家厝仔",
            "冊架仔",
            "冊櫥",
            "系統書櫃",
            "電子書櫃",
            "數位書櫃",
            "虛擬書櫃",
            "線上書櫃",
        }
        for source in strict_sources:
            with self.subTest(strict_source=source):
                self.assertEqual(round_rows[source]["protected"]["enforcement"], "strict")

    def test_round532_machine_semantic_root_governance_batch2_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_ef253080ba47",
            "lx_4ebb051508dd",
            "lx_6752b7deda3d",
            "lx_77e8d6de09e6",
            "lx_c3ffdab62ed6",
            "lx_ccf2670cba21",
            "lx_39903e541422",
            "lx_2ece11b7999d",
            "lx_e0fda99c572c",
            "lx_d2d0691030b8",
            "lx_de7045376660",
            "lx_8496648bdc85",
            "lx_c6ea3c421f5e",
            "lx_064a12edda17",
            "lx_4aef78cc22a2",
            "lx_29563af5fcaf",
            "lx_171295834f6d",
            "lx_2e03cdb552e5",
            "lx_80bf97cbd296",
            "lx_c2b9479fa0bd",
            "lx_466f5d3bb64e",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        source = "curation:round532_machine_semantic_root_governance_batch2"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "出讓": "讓渡",
            "分蔥": "珠蔥",
            "劇目": "齣頭",
            "吆喝": "喝咻",
            "喝咻": "喝咻",
            "吝嗇": "凍霜",
            "吹牛": "膨風",
            "副詞": "副詞",
            "助選": "助選",
            "反常": "反常",
            "吐奶": "吐奶",
            "剛直": "剛直",
            "修飾": "修飾",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2225})

        strict_sources = {"喝咻", "副詞", "助選", "反常", "吐奶", "剛直", "修飾"}
        for src in strict_sources:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")

        fail_closed_sources = {
            "下輩",
            "出神",
            "分頭",
            "削除",
            "剩餘",
            "劃分",
            "勸導",
            "包攬",
            "名堂",
        }
        active_sources = {row["src"] for row in self.rows}
        self.assertTrue(fail_closed_sources.isdisjoint(active_sources))

    def test_round532_short_machine_boundary_governance_batch3_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_74e384d9a480",
            "lx_be0e14a0f5e3",
            "lx_8f6b3889f007",
            "lx_cac4e4743613",
            "lx_d8aa54a0cc6e",
            "lx_1c109037960c",
            "lx_cae9bfa66850",
            "lx_aa63c1ea7baa",
            "lx_a9475a850bb9",
            "lx_1ec5b65f26cb",
            "lx_358144611d91",
            "lx_4feeb0737bba",
            "lx_c3e57eacd4dc",
            "lx_06a99ba98d3c",
            "lx_fc63e6d3ead5",
            "lx_04e017ca3e40",
            "lx_532000000276",
            "lx_be5d0552ac44",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        source = "curation:round532_short_machine_boundary_governance_batch3"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "一同": "同齊",
            "一齊": "同齊",
            "上香": "燒香",
            "下注": "落注",
            "刨具": "剾仔",
            "刨刀": "剾刀",
            "刷子": "抿仔",
            "牙刷子": "齒抿仔",
            "創立": "創立",
            "加多": "加添",
            "分量": "分量",
            "一天": "一工",
            "不同": "無仝",
            "制止": "阻止",
            "刺癢": "刺疫",
            "刺眼": "鑿目",
            "分娩": "分娩",
            "不同意見": "無仝意見",
            "不同意義": "無仝意義",
            "不同意識": "無仝意識",
            "不同意象": "無仝意象",
            "不同意涵": "無仝意涵",
            "不同意境": "無仝意境",
            "香火鼎盛": "香火鼎盛",
            "供桌上": "尪架桌頂",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed.values()}, {"active"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2230})

        strict_sources = {"創立", "分量", "分娩", "香火鼎盛"}
        for src in strict_sources:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")

        shared_left_boundary = r"(?<![第任每某單唯逐統零〇一二三四五六七八九十百千萬億兆廿卅卌0-9])$"
        self.assertEqual(governed["一同"]["context"]["left_regex"], shared_left_boundary)
        self.assertEqual(governed["一齊"]["context"]["left_regex"], shared_left_boundary)
        for suffix in ("山", "港", "案", "檳", "包", "火", "芳"):
            with self.subTest(protected_aroma_suffix=suffix):
                self.assertIn(suffix, governed["上香"]["context"]["right_regex"])
        for suffix in ("意", "解", "釋", "音", "射", "入", "記", "明"):
            with self.subTest(protected_annotation_suffix=suffix):
                self.assertIn(suffix, governed["下注"]["context"]["right_regex"])
        self.assertEqual(governed["一天"]["context"]["right_regex"], r"^(?!然氣|線|文|候|體|氣)")
        self.assertEqual(governed["不同"]["context"]["right_regex"], r"^(?!意|居)")
        self.assertEqual(governed["制止"]["context"]["left_regex"], r"(?<![強管限控遏抑防])$")
        self.assertEqual(
            governed["刺眼"]["context"]["right_regex"],
            r"^(?!睛|球|周|角|膜|眶|窩|皮|底|內|外|部)",
        )
        for src in ("不同意見", "不同意義", "不同意識", "不同意象", "不同意涵", "不同意境"):
            with self.subTest(different_compound=src):
                self.assertEqual(governed[src]["context"]["left_regex"], r"(?<!有)$")

    def test_round532_machine_semantic_root_governance_batch4_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_4a074b70166f",
            "lx_77c83cdcb635",
            "lx_23e6128d85be",
            "lx_f846bc391898",
            "lx_a7522920c370",
            "lx_5d4a0d9e2aa5",
            "lx_b70047de954a",
            "lx_301131c7477c",
            "lx_36d86f56a7b0",
            "lx_732bbed26901",
            "lx_a0960de4f56b",
            "lx_fd214032b238",
            "lx_ae49dc4ede9d",
            "lx_aed19d8015d1",
            "lx_61320b058817",
            "lx_c164d935de9f",
            "lx_e92afb92a182",
            "lx_0ae910352493",
            "lx_0df6c66c6ce0",
            "lx_eaf5249d02a2",
            "lx_532000000295",
            "lx_532000000299",
            "lx_532000000302",
            "lx_532000000305",
            "lx_532000000306",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        source = "curation:round532_machine_semantic_root_governance_batch4"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "勾引": "勾引",
            "呼號": "呼號",
            "咆哮": "咆哮",
            "地名": "地名",
            "大號": "大號",
            "好事": "好事",
            "弄髒": "弄髒",
            "情婦": "情婦",
            "撞傷": "撞傷",
            "斷氣": "斷氣",
            "痊癒": "痊癒",
            "瘀血": "瘀血",
            "盤算": "盤算",
            "相撲": "相撲",
            "嗆到": "嗾著",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed.values()}, {"active"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2240})

        for src in set(expected) - {"嗆到"}:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")

        fail_closed_sources = {"天花", "打氣", "生鐵", "瘦弱", "發起"}
        active_sources = {row["src"] for row in self.rows}
        self.assertTrue(fail_closed_sources.isdisjoint(active_sources))
        for src in ("天花板", "打氣筒", "打氣嗝", "生鐵鍋", "瘦弱的童養媳", "發起互助會"):
            with self.subTest(longer_source=src):
                self.assertIn(src, active_sources)

    def test_round532_machine_semantic_root_governance_batch5_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_7621b2a36360",
            "lx_cd42d69b75e0",
            "lx_6a9c96a63031",
            "lx_2f1ef1886dd6",
            "lx_52c1370b55ba",
            "lx_f69866a2ce8d",
            "lx_308c61d3dde3",
            "lx_49a721704e52",
            "lx_4c8d4404d88e",
            "lx_4ffdebe940ef",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        source = "curation:round532_machine_semantic_root_governance_batch5"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "溺水": "駐水",
            "斤兩": "秤頭",
            "草灰": "草烌",
            "好轉": "起色",
            "居然": "居然",
            "量詞": "量詞",
            "小號": "小號",
            "消除": "消除",
            "土堆": "塗堆",
            "差錯": "差錯",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed.values()}, {"active"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2250})

        for src in {"居然", "量詞", "小號", "消除", "差錯"}:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")
        self.assertEqual(governed["量詞"]["protected"]["category"], "technical_term")

    def test_round532_machine_semantic_and_boundary_governance_batch6_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_147a31e602f6",
            "lx_5a5ddfcbe3eb",
            "lx_cde0a0ea499e",
            "lx_a9a352fe6914",
            "lx_ca00c6efc0c3",
            "lx_0f3d6dd873bc",
            "lx_a13c15edf8f8",
            "lx_626fd7d76b8f",
            "lx_1ec95f09a383",
            "lx_784b19e82d83",
            "lx_3fce19a2cc62",
            "lx_a8370bdef8f4",
            "lx_bca9a970911a",
            "lx_92506139d6ff",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )
        self.assertTrue(all(by_id[entry_id].get("governance_note") for entry_id in disabled_ids))

        source = "curation:round532_machine_semantic_and_boundary_governance_batch6"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "招請": "招請",
            "惡性腫瘤": "惡性瘤",
            "哺乳動物": "哺乳動物",
            "狠心": "狠心",
            "密合": "密合",
            "被套": "被單",
            "台階": "砛",
            "開支": "支出",
            "小米": "秮仔米",
            "相交": "交陪",
            "相反": "顛倒反",
            "連接": "敆倚",
            "子彈": "銃子",
            "扳手": "十扳仔",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed.values()}, {"active"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2260})

        for src in {"招請", "哺乳動物", "狠心", "密合"}:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")
        for src in {"哺乳動物", "密合"}:
            with self.subTest(technical_source=src):
                self.assertEqual(governed[src]["protected"]["category"], "technical_term")

        self.assertEqual(governed["被套"]["context"]["right_regex"], "^(?!(?:牢|住|著|在|佇|上|入|進|用|取|困|綁|覆))")
        self.assertEqual(governed["台階"]["context"], {"left_regex": "(?<!平)$", "right_regex": "^(?!段)"})
        self.assertEqual(governed["開支"]["context"], {"right_regex": "^(?!援)"})
        self.assertEqual(governed["子彈"]["context"], {"right_regex": "^(?!列車)"})
        self.assertEqual(governed["扳手"]["context"], {"right_regex": "^(?!腕)"})

    def test_round532_technical_and_compound_boundary_governance_batch7_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_b755d65f52bf",
            "lx_0797eb6fe989",
            "lx_0e58d6977918",
            "lx_ecc3516f2689",
            "lx_82c5ef7e3c98",
            "lx_e789deb51f34",
            "lx_375fc3f9021d",
            "lx_ef55e8a1f5ca",
            "lx_ed85fcb5e855",
            "lx_48ea5d1becaa",
            "lx_471d7c3a7ecb",
            "lx_4024e09e939f",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )
        self.assertTrue(all(by_id[entry_id].get("governance_note") for entry_id in disabled_ids))

        source = "curation:round532_technical_and_compound_boundary_governance_batch7"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "地面": "塗跤",
            "孫子": "孫仔",
            "孫子兵法": "孫子兵法",
            "小火": "䆀火",
            "小子": "查埔囝仔",
            "對角": "斜角",
            "對角線": "斜角線",
            "對角矩陣": "對角矩陣",
            "對角元素": "對角元素",
            "對角化": "對角化",
            "回聲": "應聲",
            "回聲定位": "回聲定位",
            "認識": "熟似",
            "賣淫": "賣淫",
            "詰問": "詰問",
            "陰道": "陰道",
            "胸部": "胸部",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed.values()}, {"active"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2270})

        strict_sources = {
            "孫子兵法",
            "對角矩陣",
            "對角元素",
            "對角化",
            "回聲定位",
            "賣淫",
            "詰問",
            "陰道",
            "胸部",
        }
        for src in strict_sources:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")
        for src in strict_sources - {"孫子兵法"}:
            with self.subTest(technical_source=src):
                self.assertEqual(governed[src]["protected"]["category"], "technical_term")

        self.assertEqual(governed["地面"]["context"], {"left_regex": "(?<![基接])$", "right_regex": "^(?!積)"})
        self.assertEqual(governed["小火"]["context"], {"right_regex": "^(?!箭)"})
        self.assertEqual(governed["小子"]["context"], {"right_regex": "^(?!彈)"})
        self.assertEqual(governed["認識"]["context"], {"left_regex": "(?<!辨)$", "right_regex": "^(?!別)"})

    def test_round532_medical_legal_and_technical_governance_batch8_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_410c45c52a91",
            "lx_875fadf1f538",
            "lx_d4377a939f19",
            "lx_28be5259c96f",
            "lx_81e9dc77c784",
            "lx_040c9958314b",
            "lx_331567403b2a",
            "lx_a282c0e7367b",
            "lx_8528d401fe3f",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )
        self.assertTrue(all(by_id[entry_id].get("governance_note") for entry_id in disabled_ids))

        source = "curation:round532_medical_legal_and_technical_governance_batch8"
        governed = {row["src"]: row for row in self.rows if row.get("source") == source}
        expected = {
            "臟腑": "臟腑",
            "抽空": "閬工",
            "抽空程序": "抽空程序",
            "腔體抽空": "腔體抽空",
            "泡沫": "沫",
            "房市泡沫": "房市泡沫",
            "金融泡沫": "金融泡沫",
            "泡沫滅火": "泡沫滅火",
            "泡沫塑膠": "泡沫塑膠",
            "水門": "水閘",
            "水門事件": "水門事件",
            "水門案": "水門案",
            "古人": "古早人",
            "古人類": "古人類",
            "耕地": "耕地",
            "混血兒": "半仿仔",
            "死亡": "過身",
            "死亡率": "死亡率",
            "死亡證明": "死亡證明",
            "死亡原因": "死亡原因",
            "死亡判定": "死亡判定",
            "陰莖": "𡳞鳥",
            "陰莖癌": "陰莖癌",
            "陰莖骨折": "陰莖骨折",
            "陰莖勃起功能障礙": "陰莖勃起功能障礙",
        }
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed.values()}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed.values()}, {"active"})
        self.assertEqual({row["priority"] for row in governed.values()}, {2280})

        strict_sources = {
            "臟腑",
            "抽空程序",
            "腔體抽空",
            "房市泡沫",
            "金融泡沫",
            "泡沫滅火",
            "泡沫塑膠",
            "水門事件",
            "水門案",
            "古人類",
            "耕地",
            "死亡率",
            "死亡證明",
            "死亡原因",
            "死亡判定",
            "陰莖癌",
            "陰莖骨折",
            "陰莖勃起功能障礙",
        }
        for src in strict_sources:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")
        for src in strict_sources - {"水門事件", "水門案"}:
            with self.subTest(technical_source=src):
                self.assertEqual(governed[src]["protected"]["category"], "technical_term")
        for src in {"水門事件", "水門案"}:
            with self.subTest(proper_name_source=src):
                self.assertEqual(governed[src]["protected"]["category"], "proper_noun")

        self.assertEqual(governed["古人"]["context"], {"right_regex": "^(?!類)"})
        self.assertEqual(governed["水門"]["context"], {"right_regex": "^(?!(?:事件|案|醜聞))"})
        self.assertEqual(governed["死亡"]["context"]["left_regex"], "(?<!判定)(?<!確認)$")
        self.assertEqual(governed["陰莖"]["context"]["left_regex"], "(?<!檢查)(?<!診斷)$")

        winding = by_id["lx_d479c3635e2d"]
        self.assertEqual(winding["status"], "active")
        self.assertEqual((winding["src"], winding["tgt"]), ("捲線", "經線"))

    def test_round532_reverse_gloss_and_domain_boundary_governance_batch9_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_a4eb3e94f3d9",
            "lx_7534e456c6ad",
            "lx_b4a23b3a7e43",
            "lx_2db8e8d2f4cd",
            "lx_02c9efb7e4f0",
            "lx_fa803162c406",
            "lx_6ab4858079e6",
            "lx_9672cf9db362",
            "lx_f3c9a8a3f718",
            "lx_0f508da7e6b6",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )
        self.assertTrue(all(by_id[entry_id].get("governance_note") for entry_id in disabled_ids))

        source = "curation:round532_reverse_gloss_and_domain_boundary_governance_batch9"
        governed_rows = [row for row in self.rows if row.get("source") == source]
        governed = {row["src"]: row for row in governed_rows}
        expected = {
            "水蒸氣": "水蒸氣",
            "專一": "專一",
            "受質專一性": "受質專一性",
            "酵素專一性": "酵素專一性",
            "深藍色": "紺色",
            "私生子": "私生子",
            "遺書": "遺書",
            "潛水": "藏水沬",
            "潛水艇": "潛水艇",
            "潛水艦": "潛水艦",
            "潛水鐘": "潛水鐘",
            "潛水醫學": "潛水醫學",
            "潛水裝備": "潛水裝備",
            "潛水作業": "潛水作業",
            "清潔": "清氣",
            "清潔劑": "清潔劑",
            "清潔用品": "清潔用品",
            "清潔設備": "清潔設備",
            "清潔程序": "清潔程序",
            "清潔標準": "清潔標準",
            "清潔作業": "清潔作業",
            "清潔消毒": "清潔消毒",
            "買賣": "生理",
            "買賣契約": "買賣契約",
            "買賣價金": "買賣價金",
            "買賣標的": "買賣標的",
            "買賣雙方": "買賣雙方",
            "買賣關係": "買賣關係",
            "不動產買賣": "不動產買賣",
            "土地買賣": "土地買賣",
            "浮標": "浮沉",
            "接口": "接喙",
            "程式接口": "程式接口",
            "軟體接口": "軟體接口",
            "網路接口": "網路接口",
            "硬體接口": "硬體接口",
        }
        self.assertEqual(len(governed_rows), len(expected))
        self.assertEqual({src: row["tgt"] for src, row in governed.items()}, expected)
        self.assertEqual({row["trust"] for row in governed_rows}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed_rows}, {"active"})
        self.assertEqual({row["priority"] for row in governed_rows}, {2290, 2291})

        contextual_sources = {"潛水", "清潔", "買賣", "浮標", "接口"}
        self.assertTrue(all(governed[src]["context"] for src in contextual_sources))
        strict_sources = set(expected) - contextual_sources - {"深藍色"}
        for src in strict_sources:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")
        self.assertEqual(governed["浮標"]["context"]["right_regex"], "^(?!沉)")
        self.assertIn("不動產", governed["買賣"]["context"]["left_regex"])
        self.assertIn("管線", governed["接口"]["context"]["left_regex"])

    def test_round532_professional_compound_boundary_governance_batch11_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_7fd552cb0829",
            "lx_ce92c64a85fe",
            "lx_71b423b47b3b",
            "lx_67ea1aa228e9",
            "lx_cf16ab5cf206",
            "lx_b88d16a93eba",
            "lx_923880b6b5f9",
            "lx_607ad7afb4e0",
            "lx_7c15672c1bbe",
            "lx_2228325d66d2",
            "lx_93949054a791",
            "lx_a2ff26b7904b",
            "lx_d65a92b1d89c",
            "lx_59c872af9088",
            "lx_c195d369cf07",
        }
        self.assertEqual(
            {entry_id: by_id[entry_id]["status"] for entry_id in disabled_ids},
            {entry_id: "disabled" for entry_id in disabled_ids},
        )

        source = "curation:round532_professional_compound_boundary_governance_batch11"
        governed_rows = [row for row in self.rows if row.get("source") == source]
        governed = {row["src"]: row for row in governed_rows}
        self.assertEqual(len(governed_rows), 54)
        self.assertEqual({row["trust"] for row in governed_rows}, {"ai_reviewed"})
        self.assertEqual({row["status"] for row in governed_rows}, {"active"})
        self.assertEqual({row["priority"] for row in governed_rows}, {2310, 2311})

        contextual_sources = {
            "委任",
            "疼痛",
            "職業",
            "工具",
            "哺乳",
            "永久",
            "育兒",
            "罰金",
            "攜帶",
            "工廠",
            "男子",
            "劇烈",
            "咽喉",
            "查帳",
        }
        self.assertTrue(all(governed[src]["context"] for src in contextual_sources))
        strict_sources = set(governed) - contextual_sources
        for src in strict_sources:
            with self.subTest(strict_source=src):
                self.assertEqual(governed[src]["src"], governed[src]["tgt"])
                self.assertEqual(governed[src]["protected"]["enforcement"], "strict")

        self.assertIn("契約", governed["委任"]["context"]["right_regex"])
        self.assertIn("指數", governed["疼痛"]["context"]["right_regex"])
        self.assertIn("安全衛生", governed["職業"]["context"]["right_regex"])
        self.assertIn("機", governed["工具"]["context"]["right_regex"])
        self.assertIn("期", governed["哺乳"]["context"]["right_regex"])
        self.assertIn("居留", governed["永久"]["context"]["right_regex"])
        self.assertIn("津貼", governed["育兒"]["context"]["right_regex"])
        self.assertIn("戶", governed["查帳"]["context"]["right_regex"])

        reverse_gloss = by_id["lx_7c15672c1bbe"]
        self.assertEqual(reverse_gloss["src"], "可食")
        self.assertEqual(reverse_gloss["tgt"], "石榴")
        self.assertEqual(reverse_gloss["status"], "disabled")

        reflection_gate = by_id["lx_532000000436"]
        self.assertIn("疲勞", reflection_gate["context"]["right_regex"])
        self.assertIn("壽命", reflection_gate["context"]["right_regex"])
        self.assertIn("軟板", reflection_gate["context"]["left_regex"])

    def test_library_fixed_term_is_preserved_across_overrides(self) -> None:
        matching_rows = [row for row in self.rows if "圖書館" in row["src"]]
        self.assertTrue(matching_rows)
        self.assertEqual(
            [
                f"{row['entry_id']}: {row['src']} -> {row['tgt']}"
                for row in matching_rows
                if "圖書館" not in row["tgt"] or "圖冊館" in row["tgt"]
            ],
            [],
        )
        identity = next(row for row in matching_rows if row["src"] == "圖書館")
        self.assertEqual(identity["tgt"], "圖書館")
        self.assertEqual(identity["protected"]["category"], "lexical_identity")

    def test_round532_authoritative_semantic_and_unicode_governance_batch12_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        for entry_id in (
            "lx_dcd97756106e",
            "lx_ec451241b3f8",
            "lx_999cf5146b4b",
            "lx_2111336b6840",
            "lx_f42e9e45d1f6",
            "lx_88e391dd8cbd",
            "lx_03e36b9a78d0",
            "lx_10d1362edef9",
        ):
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round532_authoritative_semantic_and_unicode_governance_batch12"
        governed = [row for row in all_rows if row.get("source") == source]
        self.assertEqual(len(governed), 10)
        governed_by_id = {row["entry_id"]: row for row in governed}
        superseded = {
            "lx_532000000493",
            "lx_532000000494",
            "lx_532000000495",
            "lx_532000000496",
            "lx_532000000498",
        }
        self.assertEqual(
            {entry_id for entry_id, row in governed_by_id.items() if row["status"] == "disabled"},
            superseded,
        )
        self.assertEqual(
            {entry_id for entry_id, row in governed_by_id.items() if row["status"] == "active"},
            {
                "lx_532000000497",
                "lx_532000000499",
                "lx_532000000500",
                "lx_532000000501",
                "lx_532000000502",
            },
        )

    def test_round532_precision_boundary_and_pua_governance_batch13_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_23142aab18d7",
            "lx_40e3c3308c72",
            "lx_de16af34ed1a",
            "lx_b7e64207fcde",
            "lx_698eb05ff0d4",
            "lx_c8a6664913cd",
            "lx_532000000493",
            "lx_532000000494",
            "lx_532000000495",
            "lx_532000000496",
            "lx_532000000498",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round532_precision_boundary_and_pua_governance_batch13"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(503, 545)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 42)

        contextual_ids = {
            "lx_532000000503",
            "lx_532000000524",
            "lx_532000000525",
            "lx_532000000526",
            "lx_532000000528",
            "lx_532000000529",
            "lx_532000000531",
        }
        strict_ids = set(range(504, 520)) | {520, 521, 522, 523} | set(range(532, 537)) | set(range(539, 545))
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertTrue(row["governance_note"].strip())
                if entry_id in contextual_ids:
                    self.assertTrue(row["context"])
                numeric_id = int(entry_id[-3:])
                if numeric_id in strict_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                    self.assertIsNone(row["context"])

        core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))
        things = [row for row in core_rows if row.get("src") == "事情" and row.get("tgt") == "代誌"]
        self.assertEqual(len(things), 1)
        self.assertEqual(things[0]["context"], {"left_regex": "(?<!戰)(?<!軍)(?<!人)$"})
        self.assertEqual(things[0]["source"], source)

        private_use_targets = []
        for row in self.rows:
            for character in row["tgt"]:
                code_point = ord(character)
                if (
                    0xE000 <= code_point <= 0xF8FF
                    or 0xF0000 <= code_point <= 0xFFFFD
                    or 0x100000 <= code_point <= 0x10FFFD
                ):
                    private_use_targets.append((row["entry_id"], f"U+{code_point:04X}"))
        self.assertEqual(private_use_targets, [])

    def test_round532_reverse_gloss_and_polysemy_governance_batch14_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_5cbafcb5ad26",
            "lx_28ccb4e8eb9c",
            "lx_2e1a82eaada3",
            "lx_710d3900476c",
            "lx_5f9dc4bc56a1",
            "lx_323b8f24dacb",
            "lx_4c6b8262f02e",
            "lx_bf557a7395d0",
            "lx_f282c5cbc9ee",
            "lx_8889faa7c9d9",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round532_reverse_gloss_and_polysemy_governance_batch14"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(545, 591)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 46)

        contextual_ids = {558, 559, 561, 562, 568, 569, 570, 585, 589}
        strict_ids = (
            set(range(545, 558)) | {560} | set(range(563, 568)) | set(range(571, 585)) | set(range(586, 589)) | {590}
        )
        self.assertEqual(contextual_ids | strict_ids, set(range(545, 591)))
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertTrue(row["governance_note"].strip())
                numeric_id = int(entry_id[-3:])
                if numeric_id in contextual_ids:
                    self.assertTrue(row["context"])
                    self.assertNotEqual(row["src"], row["tgt"])
                else:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                    self.assertIsNone(row["context"])

    def test_round533_short_edge_semantic_governance_batch15_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_32c293227c7a",
            "lx_36535bf767c2",
            "lx_c3f7b4b16ec6",
            "lx_bc2b6f3e6a59",
            "lx_412993a8a14a",
            "lx_cca66679a266",
            "lx_31d72437d8df",
            "lx_660f5828356b",
            "lx_db008192b166",
            "lx_27cd2b2bad79",
            "lx_da57d474a62c",
            "lx_1596892153a0",
            "lx_0f5cd93430bc",
            "lx_f718390f1f82",
            "lx_8e8d0f845862",
            "lx_410c000000019",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round533_short_edge_semantic_governance_batch15"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(591, 606)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 15)
        self.assertEqual({row["src"] for row in governed}, {row["tgt"] for row in governed})
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertTrue(row["governance_note"].strip())
                self.assertIsNone(row["context"])
                self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round534_reverse_definition_and_archaism_governance_batch16_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_94835572cb88",
            "lx_af13651d86a6",
            "lx_4785b116f26b",
            "lx_87da947528a6",
            "lx_d53c7d2ebce6",
            "lx_de8262a8430c",
            "lx_4740c80a4db4",
            "lx_8e0d27a5dc98",
            "lx_d8b53269d450",
            "lx_3e6004912ae0",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round534_reverse_definition_and_archaism_governance_batch16"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(606, 616)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 10)
        self.assertEqual({row["src"] for row in governed}, {row["tgt"] for row in governed})
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2310)
                self.assertTrue(row["governance_note"].strip())
                self.assertIsNone(row["context"])
                self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round535_definition_fragment_and_archaism_governance_batch17_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_c13fdaf0d5fe",
            "lx_0851bcbd0268",
            "lx_4b51279d9386",
            "lx_7fd063ec00ac",
            "lx_f41a3bc8bb36",
            "lx_6e8ba99d761e",
            "lx_204ac506a9cd",
            "lx_9b8eb8981406",
            "lx_9a4e7f6ee03d",
            "lx_bc9de2afc9bd",
            "lx_121d7d153a50",
            "lx_2ca4a3af70fe",
            "lx_f2f727ad5547",
            "lx_53c23d2fcca0",
            "lx_90b19395d6a3",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round535_definition_fragment_and_archaism_governance_batch17"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(616, 631)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 15)
        self.assertEqual({row["src"] for row in governed}, {row["tgt"] for row in governed})
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2310)
                self.assertTrue(row["governance_note"].strip())
                self.assertIsNone(row["context"])
                self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round536_archaic_gloss_and_truncated_target_governance_batch18_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_a8efb0034b53",
            "lx_e2399e01a69c",
            "lx_6c0907cbfb7a",
            "lx_323c038d3189",
            "lx_626cdc97a069",
            "lx_f5e73ec2c19d",
            "lx_d1ba7a107f80",
            "lx_b289cc7fc5ff",
            "lx_171419dc56fe",
            "lx_e441b0cdfe14",
            "lx_9fff2fb5d42a",
            "lx_e947b9646bbb",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round536_archaic_gloss_and_truncated_target_governance_batch18"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(631, 643)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 12)

        promoted = governed_by_id["lx_532000000640"]
        self.assertEqual((promoted["src"], promoted["tgt"]), ("披著", "幔"))
        self.assertNotIn("protected", promoted)

        identities = [row for row in governed if row["entry_id"] != "lx_532000000640"]
        self.assertEqual(len(identities), 11)
        self.assertEqual({row["src"] for row in identities}, {row["tgt"] for row in identities})
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2310)
                self.assertTrue(row["governance_note"].strip())
                self.assertIsNone(row["context"])
                if entry_id != "lx_532000000640":
                    self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round537_machine_override_and_archaic_gloss_governance_batch19_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_b9b50aaedb01",
            "lx_da4024f0289f",
            "lx_2bca09042b84",
            "lx_7a59f770b750",
            "lx_03ac2e0da5af",
            "lx_135418aa3581",
            "lx_417cb05f6eb1",
            "lx_f9728cd560d4",
            "lx_1394309da24f",
            "lx_0e809bf23c82",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round537_machine_override_and_archaic_gloss_governance_batch19"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(643, 653)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 10)

        promoted = {
            "lx_532000000643": ("圓鍬", "沙挑"),
            "lx_532000000644": ("嚴寒", "大寒"),
            "lx_532000000646": ("劈刀", "柴鍥"),
        }
        for entry_id, edge in promoted.items():
            with self.subTest(promoted_entry_id=entry_id):
                row = governed_by_id[entry_id]
                self.assertEqual((row["src"], row["tgt"]), edge)
                self.assertNotIn("protected", row)

        identities = [row for entry_id, row in governed_by_id.items() if entry_id not in promoted]
        self.assertEqual(len(identities), 7)
        self.assertEqual({row["src"] for row in identities}, {row["tgt"] for row in identities})
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2320)
                self.assertTrue(row["governance_note"].strip())
                if entry_id == "lx_532000000646":
                    self.assertIn("full_regex", row["context"])
                else:
                    self.assertIsNone(row["context"])
                if entry_id not in promoted:
                    self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round538_polysemy_and_register_governance_batch20_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_398392f3c5b8",
            "lx_1931d4ffc4b2",
            "lx_059f35f59d46",
            "lx_3b8f1798ef99",
            "lx_c55ddf3dce64",
            "lx_f7a7d8cfc73c",
            "lx_77305022480f",
            "lx_69bac66b5407",
            "lx_23e43dfb9fff",
            "lx_6ea5eb82f3b3",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")

        source = "curation:round538_polysemy_and_register_governance_batch20"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(653, 664)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 11)

        contextual_ids = {"lx_532000000653", "lx_532000000654", "lx_532000000656", "lx_532000000663"}
        for entry_id in contextual_ids:
            with self.subTest(contextual_entry_id=entry_id):
                row = governed_by_id[entry_id]
                self.assertEqual(len(row["context"]), 1)
                self.assertTrue({"left_regex", "right_regex"}.intersection(row["context"]))
                self.assertNotIn("protected", row)

        identities = [row for entry_id, row in governed_by_id.items() if entry_id not in contextual_ids]
        self.assertEqual(len(identities), 7)
        self.assertEqual({row["src"] for row in identities}, {row["tgt"] for row in identities})
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2330)
                self.assertTrue(row["governance_note"].strip())
                if entry_id not in contextual_ids:
                    self.assertIsNone(row["context"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")

    def test_round539_semantic_and_modal_governance_batch21_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_f20c09f28711",
            "lx_6ed7252f6dfe",
            "lx_d5dc05cf48f3",
            "lx_16a351736560",
            "lx_c7a6d91ef27e",
            "lx_226a3b6e1e30",
            "lx_0a1651158126",
            "lx_6d7c481701fe",
            "lx_3087ebda6bf3",
            "lx_1f6debb2b487",
            "lx_f4e36350a46c",
            "lx_0b196c532214",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")
                self.assertIn("Batch21", by_id[entry_id]["governance_note"])

        source = "curation:round539_semantic_and_modal_governance_batch21"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(664, 680)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 16)

        contextual_ids = {
            "lx_532000000665",
            "lx_532000000674",
            "lx_532000000675",
            "lx_532000000676",
            "lx_532000000677",
            "lx_532000000678",
        }
        identity_ids = {"lx_532000000672", "lx_532000000673", "lx_532000000679"}
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertIn(row["priority"], {2340, 2342, 2343})
                self.assertTrue(row["governance_note"].strip())
                if entry_id in contextual_ids:
                    self.assertIsInstance(row["context"], dict)
                    self.assertTrue(row["context"])
                    self.assertNotIn("full_regex", row["context"])
                    self.assertNotIn("protected", row)
                elif entry_id in identity_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertIsNone(row["context"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                else:
                    self.assertIsNone(row["context"])

    def test_round540_boundary_and_register_governance_batch22_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_d8db3bbbe2b6",
            "lx_9488f3cb4818",
            "lx_f69362966dd1",
            "lx_8caaf1ff24bf",
            "lx_f41dafffbbff",
            "lx_89d29645292f",
            "lx_13ef573a2d65",
            "lx_429c46db97fb",
            "lx_5343aea34828",
            "lx_286e5c9dbbe0",
            "lx_eac251c3e249",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")
                self.assertIn("Batch22", by_id[entry_id]["governance_note"])

        source = "curation:round540_boundary_and_register_governance_batch22"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(680, 691)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 11)

        contextual_ids = {
            "lx_532000000680",
            "lx_532000000681",
            "lx_532000000682",
            "lx_532000000683",
            "lx_532000000685",
            "lx_532000000687",
            "lx_532000000688",
            "lx_532000000689",
        }
        identity_ids = {"lx_532000000686"}
        promoted_ids = {"lx_532000000684", "lx_532000000690"}
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2350)
                self.assertTrue(row["governance_note"].strip())
                if entry_id in contextual_ids:
                    self.assertIsInstance(row["context"], dict)
                    self.assertTrue(row["context"])
                    self.assertNotIn("full_regex", row["context"])
                    self.assertNotIn("protected", row)
                elif entry_id in identity_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertIsNone(row["context"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                else:
                    self.assertIn(entry_id, promoted_ids)
                    self.assertIsNone(row["context"])
                    self.assertNotIn("protected", row)

    def test_round541_medical_temporal_and_boundary_governance_batch23_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_39d86974a95f",
            "lx_eeb763558486",
            "lx_4315a02fbeb7",
            "lx_b893ff760515",
            "lx_981f059cdece",
            "lx_5047d1181469",
            "lx_c592960a3cbd",
            "lx_dfb4fdbafc30",
            "lx_6698fe023db2",
            "lx_202c7ca9436b",
            "lx_92d731c73e41",
            "lx_d42b299578df",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")
                self.assertIn("Batch23", by_id[entry_id]["governance_note"])

        source = "curation:round541_medical_temporal_and_boundary_governance_batch23"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(691, 703)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 12)

        contextual_ids = {
            "lx_532000000691",
            "lx_532000000693",
            "lx_532000000694",
            "lx_532000000695",
            "lx_532000000696",
            "lx_532000000697",
            "lx_532000000698",
            "lx_532000000699",
            "lx_532000000701",
            "lx_532000000702",
        }
        identity_ids = {"lx_532000000692", "lx_532000000700"}
        promoted_ids: set[str] = set()
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2360)
                self.assertTrue(row["governance_note"].strip())
                if entry_id in contextual_ids:
                    self.assertIsInstance(row["context"], dict)
                    self.assertTrue(row["context"])
                    self.assertNotIn("full_regex", row["context"])
                    self.assertNotIn("protected", row)
                elif entry_id in identity_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertIsNone(row["context"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                else:
                    self.assertIn(entry_id, promoted_ids)
                    self.assertIsNone(row["context"])
                    self.assertNotIn("protected", row)

    def test_round542_register_boundary_and_polysemy_governance_batch24_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_a470c7564e6f",
            "lx_95cac1da1c5d",
            "lx_eb4abb6078e6",
            "lx_27ac6e59f279",
            "lx_203dce24dc36",
            "lx_6bc67beb3f66",
            "lx_023ac94d5031",
            "lx_93b0a73cac41",
            "lx_e3e95b632bd5",
            "lx_62a1afe40576",
            "lx_5f22e21627de",
            "lx_cff607e661bf",
            "lx_3aee7bc2a55d",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")
                self.assertIn("Batch24", by_id[entry_id]["governance_note"])

        source = "curation:round542_register_boundary_and_polysemy_governance_batch24"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(703, 716)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 13)

        contextual_ids = {
            "lx_532000000709",
            "lx_532000000712",
            "lx_532000000713",
            "lx_532000000714",
            "lx_532000000715",
        }
        identity_ids = {
            "lx_532000000703",
            "lx_532000000704",
            "lx_532000000705",
            "lx_532000000706",
            "lx_532000000707",
            "lx_532000000708",
            "lx_532000000710",
            "lx_532000000711",
        }
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2370)
                self.assertTrue(row["governance_note"].strip())
                if entry_id in contextual_ids:
                    self.assertIsInstance(row["context"], dict)
                    self.assertTrue(row["context"])
                    self.assertNotIn("full_regex", row["context"])
                    self.assertNotIn("protected", row)
                elif entry_id in identity_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertIsNone(row["context"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                else:
                    self.fail(f"unexpected Batch24 entry: {entry_id}")

    def test_round543_distribution_boundary_and_semantic_governance_batch25_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_c3ef566bc780",
            "lx_70a98bd67eec",
            "lx_6b283ac43c2b",
            "lx_e47d7d0394de",
            "lx_11cc488a3d5a",
            "lx_9efd341bf470",
            "lx_942b10adade3",
            "lx_c94c55dd6f60",
            "lx_c23d79e80643",
            "lx_079cc03dfd07",
            "lx_3f56694b4d94",
            "lx_cb029114cee5",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")
                self.assertIn("Batch25", by_id[entry_id]["governance_note"])

        source = "curation:round543_distribution_boundary_and_semantic_governance_batch25"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(716, 728)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 12)

        contextual_ids = {
            "lx_532000000716",
            "lx_532000000717",
            "lx_532000000719",
            "lx_532000000722",
            "lx_532000000726",
            "lx_532000000727",
        }
        identity_ids = {
            "lx_532000000718",
            "lx_532000000720",
            "lx_532000000721",
            "lx_532000000723",
            "lx_532000000724",
            "lx_532000000725",
        }
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2380)
                self.assertTrue(row["governance_note"].strip())
                if entry_id in contextual_ids:
                    self.assertIsInstance(row["context"], dict)
                    self.assertTrue(row["context"])
                    self.assertNotIn("full_regex", row["context"])
                    self.assertNotIn("protected", row)
                elif entry_id in identity_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertIsNone(row["context"])
                    self.assertEqual(row["protected"]["enforcement"], "strict")
                else:
                    self.fail(f"unexpected Batch25 entry: {entry_id}")

    def test_round544_semantic_scope_orthography_and_boundary_governance_batch26_is_governed(self) -> None:
        all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["entry_id"]: row for row in all_rows}
        disabled_ids = {
            "lx_a51e7200b0c0",
            "lx_78e1f58bfac8",
            "lx_67a032d6ae6a",
            "lx_0f537b4dc9ce",
            "lx_e7cd056d1694",
            "lx_a5c25696bc7d",
            "lx_2bd927dea66a",
            "lx_e3aa31c8e001",
            "lx_7c342129fbfc",
            "lx_33d9948a5201",
            "lx_9c58fd007e2b",
            "lx_f1fe8a12dc14",
        }
        for entry_id in disabled_ids:
            with self.subTest(disabled_entry_id=entry_id):
                self.assertEqual(by_id[entry_id]["status"], "disabled")
                self.assertIn("Batch26", by_id[entry_id]["governance_note"])

        source = "curation:round544_semantic_scope_orthography_and_boundary_governance_batch26"
        governed = [row for row in self.rows if row.get("source") == source]
        governed_by_id = {row["entry_id"]: row for row in governed}
        expected_ids = {f"lx_532000000{number}" for number in range(728, 742)}
        self.assertEqual(set(governed_by_id), expected_ids)
        self.assertEqual(len(governed), 14)

        identity_ids = {"lx_532000000731", "lx_532000000732"}
        strict_identity_ids = {"lx_532000000732"}
        high_priority_ids = {"lx_532000000734", "lx_532000000740"}
        for entry_id, row in governed_by_id.items():
            with self.subTest(entry_id=entry_id):
                self.assertEqual(row["trust"], "ai_reviewed")
                self.assertEqual(row["status"], "active")
                self.assertEqual(row["priority"], 2391 if entry_id in high_priority_ids else 2390)
                self.assertTrue(row["governance_note"].strip())
                if entry_id in identity_ids:
                    self.assertEqual(row["src"], row["tgt"])
                    self.assertIsNone(row["context"])
                    if entry_id in strict_identity_ids:
                        self.assertEqual(row["protected"]["enforcement"], "strict")
                    else:
                        self.assertNotIn("enforcement", row["protected"])
                else:
                    self.assertIsInstance(row["context"], dict)
                    self.assertTrue(row["context"])
                    self.assertNotIn("full_regex", row["context"])
                    self.assertNotIn("protected", row)

        longer_phrase = by_id["lx_532000000075"]
        self.assertEqual((longer_phrase["src"], longer_phrase["tgt"]), ("不得善終", "歹死"))
        self.assertEqual(longer_phrase["status"], "active")
        self.assertIsNone(longer_phrase["context"])

        self.assertEqual(by_id["lx_76a77f2523f5"]["tgt"], "上佮意")
        self.assertIn("Batch26", by_id["lx_76a77f2523f5"]["governance_note"])

        core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))
        drink_rows = [row for row in core_rows if row.get("src") == "喝" and row.get("tgt") == "啉"]
        self.assertEqual(len(drink_rows), 2)
        drink_by_source = {row["source"]: row for row in drink_rows}
        explicit_drink = drink_by_source["curation:round544_core_drink_boundary_governance_batch26"]
        self.assertEqual(set(explicit_drink["context"]), {"right_regex"})
        self.assertIn("完(?:奶|水|茶|湯", explicit_drink["context"]["right_regex"])
        anaphoric_drink = drink_by_source["curation:round544_core_anaphoric_drink_governance_batch26"]
        self.assertEqual(set(anaphoric_drink["context"]), {"left_regex", "right_regex"})
        self.assertIn("{0,12}", anaphoric_drink["context"]["left_regex"])
        self.assertNotIn("full_regex", anaphoric_drink["context"])
        for row in drink_rows:
            self.assertIn("Batch26", row["governance_note"])

        rule_rows = [
            json.loads(line)
            for line in (DATA_DIR / "rule_entries.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rule = next(row for row in rule_rows if row.get("rule_id") == "rl_5fe3ea2a6ecc")
        self.assertFalse(rule["enabled"])
        self.assertIn("Batch26", rule["note"])


if __name__ == "__main__":
    unittest.main()
