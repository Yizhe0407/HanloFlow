from __future__ import annotations

import unittest

from taigi_converter import ConversionResult, TaigiConverter


class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.converter = TaigiConverter()

    def test_basic_conversion(self) -> None:
        self.assertEqual(self.converter.convert("你在做什麼？"), "你咧做啥物？")

    def test_trace_result(self) -> None:
        result = self.converter.convert("公車到站了", trace=True)
        self.assertIsInstance(result, ConversionResult)
        assert isinstance(result, ConversionResult)
        self.assertEqual(result.output, "公車到站矣")
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertTrue(result.matches or result.rules_applied)

    def test_preserve_spacing(self) -> None:
        normal = self.converter.convert("  你   好  ")
        preserved = self.converter.convert(
            "  你   好  ",
            profile={"preserve_spacing": True},
        )
        self.assertNotEqual(normal, preserved)
        self.assertTrue(str(preserved).startswith("  "))
        self.assertTrue(str(preserved).endswith("  "))
        self.assertIn("   ", str(preserved))

    def test_taiwan_railway_orthography_is_deterministic(self) -> None:
        self.assertEqual(self.converter.convert("台鐵基隆站"), "臺鐵基隆站")

    def test_number_bearing_proper_names_reach_the_phrase_pipeline(self) -> None:
        self.assertEqual(
            self.converter.convert("請用 Google Maps 帶路去臺北101。"),
            "請用 Google Maps 帶路去臺北101。",
        )
        self.assertEqual(self.converter.convert("台北101"), "臺北101")
        self.assertEqual(self.converter.convert("價格是 101 元。"), "價錢是一百空一元。")

    def test_number_conversion_cleans_new_cjk_boundaries(self) -> None:
        expected = {
            "規畫 15日": "規畫十五日",
            "港府跟隨北京訂定首個5年規畫 15日起諮詢民意": ("港府跟隨北京訂定首個五年規畫十五日起諮詢民意"),
            "去臺北101，規畫 15日": "去臺北101，規畫十五日",
            "HTTP 429；Python 3.11；TLS 1.3": "HTTP 429；Python 3.11；TLS 1.3",
            "案件編號是 12345678": "案件編號是 12345678",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.converter.convert(source), target)

        self.assertEqual(
            self.converter.convert(
                "規畫 15日",
                profile={"preserve_spacing": True},
            ),
            "規畫 十五日",
        )

    def test_formal_news_connectors_preserve_compositional_meaning(self) -> None:
        expected = {
            "根據了解，會議明天召開。": "據所知，會議明仔載召開。",
            "想深化與美方關係": "想深化佮美方的關係",
            "深化與美方關係": "深化佮美方的關係",
            "沒有出席21日的會議": "無出席廿一日的會議",
            "這次我們在23日，下午的四點到六點，會非常熱鬧": ("這擺咱在二十三日，下晝的四點到六點，會非常鬧熱"),
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_final_spacing_and_overlapping_phrase_targets_are_fixed_points(self) -> None:
        expected = {
            "利率從 1.75% 調到 1.95%，不是增加 1.95%。": ("利率對 1.75% 調到 1.95%，毋是增加 1.95%。"),
            "責任還是要分清楚。": "責任猶是欲分清楚。",
            "婆婆今天會來。": "婆婆今仔日會來。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

        self.assertEqual(self.converter.convert("我婆婆"), "阮大家")

    def test_manual_phrase_targets_are_runtime_fixed_points(self) -> None:
        expected = {
            "球後莎芭蓮卡8強賽離奇吞敗": "球后莎芭蓮卡八強賽離奇吞敗",
            "觀看NVIDIA執行長黃仁勳主題演講6/1 11點線上直播": ("觀看NVIDIA執行長黃仁勳主題演講六/一十一點線頂直播"),
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_authoritative_legacy_machine_cleanup(self) -> None:
        expected = {
            "下疳": "疳瘡",
            "債權人": "債主",
            "驚醒": "拍醒",
            "門板": "門扇",
            "鹹稀飯": "鹹糜",
            "名落孫山": "落第",
            "行賄": "烏西",
            "伙計": "辛勞",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_low_trust_gloss_root_cleanup(self) -> None:
        expected = {
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
            "或入境證": "或入境證",
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
            "宋廢": "宋廢",
            "利用擴音器": "利用擴音器",
            "直接言明": "直接講明",
            "內臟之一": "內臟之一",
            "前腳很短": "頭前跤足短",
            "奔跑": "踉",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_high_signal_risk_cleanup(self) -> None:
        expected = {
            "如飢": "如飢",
            "如飢似渴": "如飢似渴",
            "仝位": "仝位",
            "這顆梨汁多": "這顆梨汁濟",
            "他跑路了": "伊落跑矣",
            "單性花": "單性花",
            "受到欺騙": "受騙",
            "增加資金": "增加資金",
            "資金的運轉": "資金的轉踅",
            "道德虧損": "失德",
            "涉及法律案件": "涉案",
            "凊嗽": "冷嗽",
            "冷嗽": "冷嗽",
            "涼水": "涼水",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_machine_severe_contractions_are_reviewed(self) -> None:
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
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_short_polysemy_context_fixes(self) -> None:
        expected = {
            "他是個頂天立地的大丈夫。": "伊是個頂天立地的大丈夫。",
            "下面我們討論第二個問題。": "下面咱討論第二个問題。",
            "桌子下面有一隻貓。": "桌仔下面有一隻貓。",
            "不想他竟然提早到了。": "無疑伊竟然較早到矣。",
            "我不想吃飯。": "我無想欲食飯。",
            "風從下風處吹來。": "風對下風處吹來。",
            "這場比賽我們甘拜下風。": "這場比賽咱甘拜輸勢。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_short_context_free_machine_review(self) -> None:
        expected = {
            "這是他的一生。": "這是伊的一世人。",
            "他一生氣就走。": "伊一受氣就走。",
            "她的丈夫回家了。": "伊的查埔人轉去厝裡矣。",
            "他有丈夫氣概。": "伊有丈夫氣魄。",
            "你的上臂受傷了。": "你的手股受傷矣。",
            "下回再見。": "下擺再會。",
            "我下工了。": "我放工矣。",
            "請你下座。": "請你下座。",
            "這個做法不合法。": "這个做法不合法。",
            "條件不合規定。": "條件不合規定。",
            "不合常理": "顛倒反",
            "鞋子不合腳": "鞋仔無合跤",
            "他們感情不和。": "怹感情不和。",
            "今天天氣不和暖。": "今仔日天氣不和暖。",
            "你不妨試試看。": "你不妨試看覓。",
            "這件事不妨礙交通。": "這層代誌不妨礙交通。",
            "我不想他會來。": "我不想伊會來。",
            "他很世故。": "伊真世故。",
            "他懂人情世故。": "伊懂人情世事。",
            "這位下屬很能幹。": "這位下屬真能幹。",
            "醫師診斷為下痢。": "醫師診斷為下痢。",
            "下輩子再見。": "後世人再會。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_second_short_context_free_machine_review(self) -> None:
        expected = {
            "丟掉": "丟掉",
            "丟棄": "丟棄",
            "丟臉": "丟面",
            "中夜": "中夜",
            "中暑": "著痧",
            "中獎": "著獎",
            "中筋": "中筋",
            "中餐": "中餐",
            "久遠": "久遠",
            "乏味": "乏味",
            "乖戾": "聬儱",
            "乘涼": "歇涼",
            "九孔": "九孔",
            "乞求": "乞求",
            "乳房": "乳房",
            "乳缽": "研缽",
            "乾枯": "乾枯",
            "乾涸": "乾涸",
            "乾燥": "乾燥",
            "乾爽": "焦鬆",
            "乾癟": "脯脯",
            "亂說": "亂說",
            "二胡": "二胡",
            "些微": "峇微",
            "些許": "些許",
            "交際": "交際",
            "京劇": "京戲",
            "亮麗": "亮麗",
            "今晚": "下暗",
            "仍舊": "猶原",
            "請用中筋麵粉。": "請用中筋麵粉。",
            "請安排乳房攝影檢查。": "請安排乳房攝影檢查。",
            "請開啟乾燥機。": "請開啟乾燥機。",
            "我們去跳交際舞。": "咱去跳交際舞。",
            "她穿得很亮麗。": "伊穿得真亮麗。",
            "不要亂說話。": "莫烏白講話。",
            "大家在樹下乘涼。": "逐家在樹跤歇涼。",
            "他的皮膚乾癟。": "伊的皮膚脯脯。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_third_short_context_free_machine_review(self) -> None:
        expected = {
            "仔細": "仔細",
            "他日": "另日",
            "仗勢": "靠勢",
            "仙草": "仙草",
            "以便": "通好",
            "以往": "往時",
            "以後": "以後",
            "仰泳": "死囡仔䖙",
            "任憑": "任憑",
            "仿效": "仿效",
            "伉儷": "伉儷",
            "伯母": "阿姆",
            "似乎": "敢若",
            "位移": "位移",
            "低垂": "低垂",
            "何必": "曷著",
            "何須": "曷著",
            "作事": "作事",
            "作假": "作假",
            "作弄": "作弄",
            "使勁": "使勁",
            "使喚": "使喚",
            "來生": "下世人",
            "侍奉": "侍奉",
            "供奉": "供奉",
            "供桌": "尪架桌",
            "依次": "依次",
            "依照": "照",
            "依舊": "猶原",
            "依附": "依附",
            "他日再會。": "另日再會。",
            "請選擇其他日期。": "請選擇其他日期。",
            "我要一碗仙草凍。": "我欲一碗仙草凍。",
            "請留下電話以便聯絡。": "請留下電話通好聯絡。",
            "所以便利商店很受歡迎。": "所以便利商店真受歡迎。",
            "請先存檔，以便於日後查驗。": "請先存檔，以便於後日查驗。",
            "與以往不同。": "佮往時無仝。",
            "旅客可以往返兩地。": "旅客會當來回兩地。",
            "我以後不會再犯。": "我以後袂再犯。",
            "畢業以後。": "畢業了後。",
            "遇到危險時可以後退。": "拄著危險時會當倒勼。",
            "你何必這樣做？": "你曷著按呢做？",
            "政府應採取任何必要措施。": "政府應採取任何必要措施。",
            "自己人何須多禮？": "家己人哪著遮厚禮數？",
            "任何須經核准的項目。": "任何須經核准的項目。",
            "請提交工作事故調查報告。": "請提交工作事故調查報告。",
            "他把工作弄錯了。": "伊把工作弄錯矣。",
            "即使勁敵來襲。": "準做勁敵來襲。",
            "即使喚醒回憶也沒用。": "準做喚醒回憶也無路用。",
            "我們來生再見。": "咱下世人再會。",
            "人工智慧會改變未來生活。": "人工智慧會改變未來生活。",
            "外來生物入侵。": "外來生物入侵。",
            "廟裡供奉媽祖。": "廟裡供媽祖。",
            "本會提供奉獻機會。": "本會提供奉獻機會。",
            "請把供桌擦乾淨。": "請把尪架桌擦清氣。",
            "會場免費提供桌椅。": "會場免費提供椅桌。",
            "請依照規定辦理。": "請照規定辦理。",
            "這是皈依照片。": "這是皈依相片。",
            "依附理論是心理學概念。": "依附理論是心理學概念。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_sensor_measurement_boundary_cleanup(self) -> None:
        expected = {
            "器量": "度量",
            "他的器量很大。": "伊的度量真大。",
            "感測器量測物體的位移。": "感測器量測物體的位移。",
            "儀器量測結果正常。": "儀器量測結果正常。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_fourth_short_context_free_machine_review(self) -> None:
        expected = {
            "便秘": "祕結",
            "便衣": "便衣",
            "俏皮": "激骨",
            "信教": "入教",
            "信札": "批",
            "俯臥": "坦覆",
            "倒楣": "落衰",
            "倒流": "倒流",
            "倘若": "假使",
            "借住": "倚蹛",
            "借錢": "借錢",
            "倦怠": "厭𤺪",
            "假裝": "假影",
            "假錢": "假銀票",
            "偏旁": "字爿",
            "偏激": "偏激",
            "做愛": "相姦",
            "健忘": "無頭神",
            "健朗": "勇健",
            "偶爾": "有當時仔",
            "傍晚": "欲暗仔",
            "傳話": "寄聲",
            "傳遞": "傳遞",
            "傷神": "損神",
            "傷胃": "損胃",
            "傻瓜": "癮頭",
            "傾斜": "歪斜",
            "像樣": "成款",
            "儉省": "虯儉",
            "優劣": "優劣",
            "醫師說我有慢性便秘。": "醫師講我有慢性祕結。",
            "這套系統方便秘書整理公文。": "這套系統方便秘冊整理公文。",
            "便衣警察正在執勤。": "便衣警察佇咧執勤。",
            "她說話很俏皮。": "伊講話真激骨。",
            "俊俏皮膚白皙。": "俊俏皮膚白皙。",
            "相信教育能改變社會。": "相信教育能改變社會。",
            "他從小信教。": "伊自細漢入教。",
            "這封信札保存完整。": "這封批保存完整。",
            "徵信札記已出版。": "徵信札記已出版。",
            "病人採俯臥姿勢。": "病人採坦覆姿勢。",
            "他每天做俯臥撐。": "伊逐日做俯臥撐。",
            "今天真倒楣。": "今仔日真衰。",
            "污水發生倒流。": "污水發生倒流。",
            "倘若下雨就取消。": "假使落雨就取消。",
            "他暫時借住朋友家。": "伊暫時倚蹛朋友家。",
            "租借住宿設備。": "租借歇暝設備。",
            "我向銀行借錢。": "我向銀行借錢。",
            "公司租借錢包服務。": "公司租借錢袋仔服務。",
            "工作倦怠需要休息。": "工作倦怠需要歇睏。",
            "旅途後感到倦怠。": "旅途後感覺厭𤺪。",
            "他假裝不知道。": "伊假影毋知。",
            "這是真假裝置比較。": "這是真假裝置比較。",
            "警方查獲假錢。": "警方查獲假銀票。",
            "不可造假錢包紀錄。": "不可造假錢袋仔紀錄。",
            "請找出漢字偏旁。": "請找出漢字字爿。",
            "位置偏旁邊一點。": "位置偏邊仔一點。",
            "他的言論很偏激。": "伊的言論真偏激。",
            "他們正在做愛。": "怹佇咧相姦。",
            "大家一起做愛心便當。": "逐家鬥陣做愛心便當。",
            "他最近很健忘。": "伊最近真無頭神。",
            "請做好保健忘年會規劃。": "請做好保健忘年會規劃。",
            "老人家身體健朗。": "老大人身體勇健。",
            "偶爾會下雨。": "有當時仔會落雨。",
            "配偶爾後到場。": "配偶爾後到場。",
            "傍晚一起散步。": "欲暗仔鬥陣散步。",
            "依傍晚輩生活。": "依傍序細生活。",
            "請替我傳話。": "請替我寄聲。",
            "宣傳話術要透明。": "宣傳話術要透明。",
            "資料會自動傳遞。": "資料會自動傳遞。",
            "這件事很傷神。": "這層代誌真損神。",
            "這種毒物會傷神經。": "這款毒物會傷神經。",
            "辣椒很傷胃。": "番仔薑真損胃。",
            "外傷胃部檢查。": "外傷胃部檢查。",
            "你這個傻瓜。": "你這个癮頭。",
            "這是傻瓜相機。": "這是傻瓜相機。",
            "塔身略微傾斜。": "塔身略微歪斜。",
            "請校正傾斜儀。": "請校正傾斜儀。",
            "總算有個像樣的成果。": "總算有個成款的成果。",
            "肖像樣本已建檔。": "肖像樣本已建檔。",
            "他一向儉省。": "伊一向虯儉。",
            "生活勤儉省錢。": "生活勤儉省錢。",
            "比較兩者優劣。": "較兩者優劣。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_fifth_short_context_free_machine_review(self) -> None:
        expected = {
            "元配": "原配",
            "兄長": "兄哥",
            "充裕": "充裕",
            "兆頭": "彩頭",
            "先前": "進前",
            "先鋒": "頭陣",
            "光亮": "光亮",
            "免去": "免去",
            "兒子": "後生",
            "入贅": "入贅",
            "內臟": "內臟",
            "全身": "規身軀",
            "兩邊": "兩爿",
            "公鴨": "鴨鵤",
            "兵卒": "兵仔",
            "具名": "徛名",
            "冀望": "寄望",
            "冒死": "胚命",
            "冤屈": "枉屈",
            "冥紙": "銀紙",
            "冬天": "寒天",
            "冬衣": "寒衫",
            "冰棒": "枝仔冰",
            "冰雹": "雹",
            "冷水": "冷水",
            "冷清": "冷清",
            "凡是": "見若",
            "凶惡": "凶惡",
            "凶日": "歹日",
            "凹陷": "塌窩",
            "她是他的元配。": "伊是伊的原配。",
            "多元配置已完成。": "多元配置已完成。",
            "美元配額增加。": "美元配額增加。",
            "我的兄長來了。": "我的兄哥來矣。",
            "師兄長期在海外。": "師兄長期在海外。",
            "兄長期照顧家人。": "兄長期照顧家人。",
            "時間很充裕。": "時間真充裕。",
            "物資充裕。": "物資充裕。",
            "這不是好兆頭。": "這毋是好彩頭。",
            "這是壞兆頭。": "這是壞彩頭。",
            "徵兆頭緒都不明。": "徵兆頭緒都不明。",
            "先前已經說過。": "進前已經說過。",
            "請先前往車站。": "請先去車站。",
            "他先前進會場。": "伊先前進會場。",
            "他是改革先鋒。": "伊是改革頭陣。",
            "先鋒部隊出發。": "頭陣部隊出發。",
            "先鋒科技公司。": "先鋒科技公司。",
            "房間很光亮。": "房間真光亮。",
            "拋光亮面處理。": "拋光亮面處理。",
            "日光亮度不足。": "日光亮度不足。",
            "免去他的職務。": "免去伊的職務。",
            "可免去手續。": "可免去手續。",
            "我的兒子長大了。": "我的後生大漢矣。",
            "幼兒子宮檢查。": "幼兒子宮檢查。",
            "孤兒子女補助。": "孤兒子女補助。",
            "兒子公司。": "兒子公司。",
            "他決定入贅女方家。": "伊決定入贅女方家。",
            "動物內臟。": "動物腹內。",
            "人體內臟。": "人體內臟。",
            "內臟脂肪指數。": "內臟脂肪指數。",
            "國內臟器移植。": "國內臟器徙栽。",
            "他全身濕透。": "伊規身軀濕透。",
            "全身麻醉。": "全身麻醉。",
            "全身性疾病。": "全身性疾病。",
            "安全身分驗證。": "安全身分驗證。",
            "道路兩邊。": "道路兩爿。",
            "兩邊都同意。": "兩爿都同意。",
            "兩邊形相似。": "兩邊形相似。",
            "這是一隻公鴨。": "這是一隻鴨鵤。",
            "公鴨嗓。": "公鴨嗓。",
            "公鴨母雞都在。": "鴨鵤雞母攏咧。",
            "古代兵卒。": "古代兵仔。",
            "象棋兵卒。": "象棋兵卒。",
            "閱兵卒業典禮。": "閱兵卒業典禮。",
            "請具名提出。": "請徛名提出。",
            "具名檢舉。": "徛名檢舉。",
            "家具名稱清單。": "家具名稱清單。",
            "道具名稱。": "道具名稱。",
            "我冀望成功。": "我寄望成功。",
            "父母冀望孩子平安。": "父母寄望囡仔平安。",
            "他冒死救人。": "伊胚命救人。",
            "冒死刑風險。": "冒死刑風險。",
            "他蒙受冤屈。": "伊蒙受枉屈。",
            "冤屈案件。": "枉屈案件。",
            "燒冥紙給亡者。": "燒銀紙給亡者。",
            "神明用金紙，亡者用冥紙。": "神明用金紙，亡者用銀紙。",
            "清明紙錢祭祖。": "清明紙錢祭祖。",
            "冬天很冷。": "寒天真寒。",
            "冬天候鳥來了。": "寒天渡鳥來矣。",
            "寒冬天氣。": "寒冬天氣。",
            "拿出冬衣。": "提出寒衫。",
            "冬衣物募集。": "冬衣物募集。",
            "吃冰棒。": "食枝仔冰。",
            "冰棒球隊。": "冰棒球隊。",
            "溜冰棒球課。": "溜冰棒球課。",
            "昨天下冰雹。": "昨昏落雹。",
            "冰雹災害。": "冰雹災害。",
            "喝冷水。": "啉冷水。",
            "冷水機。": "冷水機。",
            "冷水坑遊客中心。": "冷水坑遊客中心。",
            "市場很冷清。": "市場真冷清。",
            "感到冷清寂寞。": "感覺稀微寂寞。",
            "冷清洗設備。": "冷清洗設備。",
            "冷清單已更新。": "冷清單已更新。",
            "凡是學生都可參加。": "見若學生都可參加。",
            "平凡是美。": "平凡是美。",
            "非凡是我們的目標。": "非凡是阮的目標。",
            "規定凡是會員都要登記。": "規定見若會員攏愛登記。",
            "凶惡的歹徒。": "凶惡的歹徒。",
            "面貌凶惡。": "面腔凶惡。",
            "凶惡無禮。": "凶惡無禮。",
            "黃曆上的凶日。": "黃曆上的歹日。",
            "凶日子不宜嫁娶。": "凶日子不宜嫁娶。",
            "歹日": "歹日",
            "地面凹陷。": "塗跤塌窩。",
            "眼睛凹陷。": "目睭塌落。",
            "凹陷疤痕。": "凹陷疤痕。",
            "凹陷處積水。": "塌窩處積水。",
            "凹陷阱設計。": "凹陷阱設計。",
            "冷藏": "冷藏",
            "冷凍": "冷凍",
            "冷門": "冷門",
            "冷淡": "冷淡",
            "冷血": "冷血",
            "冷卻": "冷卻",
            "退卻": "退卻",
            "忘卻": "忘卻",
            "了卻": "了卻",
            "卻步": "卻步",
            "他卻不知道。": "伊煞毋知。",
            "急性病。": "急性病。",
            "慢性病。": "慢性病。",
            "性病防治。": "暗毿病防治。",
            "凶日不宜嫁娶。": "歹日不宜嫁娶。",
            "男性病患正在候診。": "男性病患佇咧候診。",
            "女性病房在二樓。": "女性病房佇二樓。",
            "個性病態扭曲。": "個性病態扭曲。",
            "他感染性病。": "伊感染暗毿病。",
            "中風徵兆頭暈。": "中風徵兆頭眩。",
            "家具名是什麼？": "家具名是啥？",
            "道具名是什麼？": "道具名是啥？",
            "立冬天氣轉冷。": "立冬天氣變寒。",
            "暖冬天氣持續。": "暖冬天氣持續。",
            "過冬衣服準備好了。": "過冬衫準備好矣。",
            "寒冬衣服要穿厚一點。": "寒冬衫要穿厚一點。",
            "行凶日子已查明。": "行凶日子已查明。",
            "不凡是他的特點。": "不凡是伊的特點。",
            "元配子女都到場。": "原配子女都到場。",
            "元配生下兩名子女。": "原配生下兩名子女。",
            "兄長最近回國。": "兄哥最近回國。",
            "兄長正在照顧家人。": "兄哥佇咧照顧家人。",
            "後天性疾病。": "後天性疾病。",
            "後天免疫缺乏症。": "後天免疫缺乏症。",
            "傳染性疾病。": "傳染性疾病。",
            "慢性疾病。": "慢性疾病。",
            "他罹患疾病。": "伊罹患疾病。",
            "先天與後天。": "先天佮後天。",
            "我們後天見。": "咱後日見。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_v5_generalized_semantic_root_fixes(self) -> None:
        expected = {
            "我們社區": "阮社區",
            "袖子": "袖子",
            "一把剪刀": "一支鉸刀",
            "牛肉麵包裝": "牛肉麵包裝",
            "長時間": "長時間",
            "里長大會": "里長大會",
            "打電話給客服": "拍電話予客服",
            "風險評估": "風險評估",
            "第一封信": "第一張批",
            "三樓上課": "三樓上課",
            "線上進行": "線頂進行",
            "互相交換答案": "互相交換解答",
            "如果不要飲料": "若毋愛飲料",
            "利率從 1.75% 調到 1.95%": "利率對 1.75% 調到 1.95%",
            "上升5%。": "上升百分之五。",
            "100%達成。": "百分之百達成。",
            "會議 13:40 在 502 室": "會議 13:40 在 502 室",
            "研習訂在 2026 年 8 月 3 日上午 10:20。": ("研習訂在 2026 年 8 月 3 日上晝 10:20。"),
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

        contrasted_keys = "一把鑰匙，卻找不到房門鑰匙"
        contrasted_output = "一支鎖匙，煞揣無房門鎖匙"
        first = self.converter.convert(contrasted_keys)
        self.assertEqual(first, contrasted_output)
        self.assertEqual(self.converter.convert(first), first)

    def test_round532_high_signal_root_governance(self) -> None:
        expected = {
            "哪裡哪裡，您過獎了。": "毋敢當，你過獎矣。",
            "你住在哪裡？": "你蹛佇佗位？",
            "以下面積公式計算。": "以下面積公式計算。",
            "請閱讀下面條款。": "請閱讀下面條款。",
            "上周邊境發生衝突。": "上周邊境發生衝突。",
            "本周邊境發生衝突。": "這禮拜邊境發生衝突。",
            "電腦周邊設備發生故障。": "電腦周邊設備發生故障。",
            "周邊神經病變。": "周邊神經病變。",
            "教師節與教師證。": "教師節佮教師證。",
            "他是一名國小教師。": "伊是一名國小教師。",
            "提供居民宅配服務。": "提供居民宅配服務。",
            "這一帶多為民宅。": "這角勢多為人家厝仔。",
            "改變聲音設定。": "改變聲音設定。",
            "語音經過變聲處理。": "語音經過變聲處理。",
            "青春期開始變聲。": "青春期開始轉聲。",
            "電子書架構設計。": "電子冊架構設計。",
            "書架仔放在房內。": "冊架仔囥佇房內。",
            "書架放在房內。": "冊架仔囥佇房內。",
            "系統書櫃功能。": "系統書櫃功能。",
            "書櫃放在房內。": "冊櫥囥佇房內。",
            "一把鑰匙，卻找不到房門鑰匙。": "一支鎖匙，煞揣無房門鎖匙。",
            "三百二十一人。": "三百二十一人。",
            "二千零二十一年。": "二千零二十一年。",
            "第二十一條。": "第二十一條。",
            "二十一分之三。": "二十一分之三。",
            "二十一世紀。": "廿一世紀。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_machine_semantic_root_governance_batch2(self) -> None:
        expected = {
            "我們下輩子再見。": "咱後世人再會。",
            "他是下輩中的長子。": "伊是下輩中的大囝。",
            "他看得出神。": "伊看得出神。",
            "這件作品出神入化。": "這件作品出神入化。",
            "公司決定出讓股權。": "公司決定讓渡股權。",
            "這是分蔥。": "這是珠蔥。",
            "大家分頭進行。": "逐家分頭進行。",
            "他從分頭線開始剪。": "伊對分頭線開始剪。",
            "請削除冗詞。": "請削除冗詞。",
            "他個性剛直。": "伊個性剛直。",
            "材料剛直度不足。": "材料剛直度不足。",
            "剩餘款項明日支付。": "剩餘錢項明日付錢。",
            "這個副詞修飾動詞。": "這个副詞修飾動詞。",
            "行政區重新劃分。": "行政區重新劃分。",
            "本季劇目已公布。": "本季齣頭已公布。",
            "他替候選人助選。": "伊替候選人助選。",
            "警方勸導民眾離開。": "警方勸導民眾離開。",
            "公司包攬所有工程。": "公司包攬所有工程。",
            "他一人包攬獎項。": "伊一人包攬獎項。",
            "包攬工事。": "貿工事。",
            "今天氣候反常。": "今仔日氣候反常。",
            "攤販沿街吆喝。": "攤販沿街喝咻。",
            "喝咻": "喝咻",
            "你在搞什麼名堂？": "你在搞啥名堂？",
            "嬰兒喝完奶後吐奶。": "紅嬰仔啉完奶後吐奶。",
            "他花錢很吝嗇。": "伊花錢真凍霜。",
            "他最愛吹牛。": "伊最愛膨風。",
            "修飾文章。": "修飾文章。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_short_machine_boundary_governance_batch3(self) -> None:
        expected = {
            "我們一同出發。": "咱同齊出發。",
            "十一同學一起出發。": "十一同學鬥陣出發。",
            "二十一同學報名。": "廿一同學報名。",
            "請統一同義詞的寫法。": "請統一同義詞的寫法。",
            "大家一齊來。": "逐家同齊來。",
            "三十一齊發車。": "三十一齊開車。",
            "統一齊頭式格式。": "統一齊頭式格式。",
            "他去廟裡上香。": "伊去廟裡燒香。",
            "架上香水很多。": "架上芳水誠濟。",
            "網路上香氛蠟燭熱賣。": "網路上香氛蠟燭熱賣。",
            "請向上香山前進。": "請向上香山前進。",
            "往上香港方向移動。": "往上香港方向移動。",
            "他在球賽下注。": "伊在球賽落注。",
            "請寫下注意事項。": "請寫下注意事項。",
            "請閱讀以下注解。": "請閱讀以下注解。",
            "分量很足。": "分量真足。",
            "微分量子場。": "微分量子場。",
            "積分量測。": "積分量測。",
            "向量的垂直分量。": "向量的垂直分量。",
            "木工刨具。": "木工剾仔。",
            "刨具體表面。": "刨具體表面。",
            "木工刨刀。": "木工剾刀。",
            "刨刀具磨損。": "刨刀具磨損。",
            "油漆刷子。": "油漆抿仔。",
            "鋼絲刷子。": "鋼絲抿仔。",
            "牙刷子。": "齒抿仔。",
            "公司於去年創立。": "公司於舊年創立。",
            "創立基金會。": "創立基金會。",
            "再加多一點。": "再加添一點。",
            "附加多媒體檔案。": "附加多媒體檔案。",
            "參加多場活動。": "參加多場活動。",
            "一天很長。": "一工真長。",
            "第一天上課。": "第一工上課。",
            "第一天然氣接收站。": "第一天然氣接收站。",
            "這兩種方法不同。": "這兩種方法無仝。",
            "他不同意這項提案。": "伊不同意這項提案。",
            "大家有不同意見。": "逐家有不同意見。",
            "他出面制止衝突。": "伊出面阻止衝突。",
            "政府強制止付該帳戶。": "政府強制止付該口座。",
            "醫院管制止痛藥。": "病院管制止痛藥。",
            "系統限制止損單。": "系統限制止損單。",
            "這件毛衣穿起來很刺癢。": "這件毛衣穿起來真刺疫。",
            "燈光太強，非常刺眼。": "燈光太強，非常鑿目。",
            "分娩機轉需要醫療監測。": "分娩機轉需要醫療監測。",
            "母牛分娩後需要觀察。": "牛母分娩後需要觀察。",
            "第一天線系統啟用。": "第一天線系統啟用。",
            "單一天線輸入。": "單一天線輸入。",
            "第一天文台開始觀測。": "第一天文台開始觀測。",
            "一天後回來。": "一工後轉來。",
            "夫妻已經不同居。": "夫妻已經不同居。",
            "大家提出不同意見。": "逐家提出無仝意見。",
            "兩者具有不同意義。": "兩者具有不同意義。",
            "研究不同意識狀態。": "研究無仝意識狀態。",
            "控制止血泵啟動。": "控制止血泵啟動。",
            "遏制止跌措施。": "遏制止跌措施。",
            "抑制止吐反應。": "抑制止吐反應。",
            "防制止詐宣導。": "防制止詐宣導。",
            "警方依法制止暴力。": "警方依法阻止暴力。",
            "針刺眼周穴位。": "針刺眼周穴位。",
            "異物刺眼球。": "異物刺眼球。",
            "尖刺眼角。": "尖刺眼角。",
            "強光很刺眼。": "強光真鑿目。",
            "第一同學上台。": "第一同學上台。",
            "任一同學都可以。": "任一同學都會當。",
            "單一同位素分析。": "單一同位素分析。",
            "唯一同意的方案。": "唯一同意的方案。",
            "逐一同意條款。": "逐一同意條款。",
            "第一齊射已完成。": "第一齊射已完成。",
            "單一齊次方程。": "單一齊次方程。",
            "唯一齊全的版本。": "唯一齊全的版本。",
            "端上香檳。": "端上香檳。",
            "送上香包。": "送上香芳。",
            "供桌上香火鼎盛。": "尪架桌頂香火鼎盛。",
            "寫下注釋。": "寫下注釋。",
            "輸入以下注音。": "輸入以下注音。",
            "按下注射鍵。": "按下注射鍵。",
            "按下注入按鈕。": "按下注入按鈕。",
            "請在表格下注明原因。": "請在表格下註明原因。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_machine_semantic_root_governance_batch4(self) -> None:
        expected = {
            "他勾引已婚者。": "伊勾引已婚者。",
            "飛行員的呼號是雷霆。": "飛行員的呼號是雷霆。",
            "獅子在咆哮。": "獅子在咆哮。",
            "請填寫出生地名。": "請填寫出生地名。",
            "這件衣服是大號。": "這件衫是大號。",
            "他上完大號。": "伊上完大號。",
            "大號樂手開始演奏。": "大號樂手開始演奏。",
            "天花病毒已被消滅。": "天花病毒已被消滅。",
            "請安裝天花板。": "請安裝天篷。",
            "這是一件好事。": "這是一件好事。",
            "他很好事。": "伊真好事。",
            "他喝水嗆到。": "伊啉水嗾著。",
            "煙太濃讓他嗆到。": "煙太濃予伊嗾著。",
            "別弄髒衣服。": "莫弄髒衫。",
            "她不是情婦。": "伊毋是情婦。",
            "大家替選手打氣。": "逐家替選手打氣。",
            "打氣筒。": "風灌。",
            "打氣嗝。": "呼噎仔。",
            "車禍造成乘客撞傷。": "車禍造成乘客撞傷。",
            "傷者已經斷氣。": "傷者已經斷氣。",
            "生鐵是煉鋼原料。": "生鐵是煉鋼原料。",
            "生鐵鍋。": "生鍋。",
            "病人已經痊癒。": "病人已經痊癒。",
            "瘀血與凝血是不同概念。": "瘀血佮凝血是無仝概念。",
            "病人身體瘦弱。": "病人身體瘦弱。",
            "瘦弱的童養媳。": "新婦仔癉。",
            "公司發起募款。": "公司發起募款。",
            "發起互助會。": "標會仔。",
            "他正在盤算成本。": "伊佇咧盤算成本。",
            "相撲是日本運動。": "相撲是日本運動。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_machine_semantic_root_governance_batch5(self) -> None:
        expected = {
            "他在河裡溺水。": "伊港口裡駐水。",
            "工程正在注水。": "工程佇咧注水。",
            "請核對商品斤兩。": "請核對商品秤頭。",
            "他做事很有斤兩。": "伊做事真有秤頭。",
            "爐裡留下草灰。": "爐裡留下草烌。",
            "牆面漆成黃灰色。": "牆面漆成黃灰色。",
            "病情逐漸好轉。": "病情逐漸起色。",
            "他聽了開始起氣。": "伊聽了開始起氣。",
            "他居然沒有來。": "伊居然無來。",
            "他說得真真正。": "伊講得真真正。",
            "請標出句中的量詞。": "請標出句中的量詞。",
            "我要一件小號衣服。": "我欲一件小號衫。",
            "他用小號演奏。": "伊用小號演奏。",
            "請消除安全隱患。": "請消除安全隱患。",
            "董事會決議解散公司。": "董事會決議解散公司。",
            "前方有一座土堆。": "頭前有一座塗堆。",
            "帕是計算土塊的量詞。": "帕是計算塗墼的量詞。",
            "不要出任何差錯。": "莫出任何差錯。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_machine_semantic_and_boundary_governance_batch6(self) -> None:
        expected = {
            "公司招請三位工程師。": "公司招請三位工程師。",
            "他先出招試探對手。": "伊先出招試探對手。",
            "醫師診斷為惡性腫瘤。": "醫師診斷為惡性瘤。",
            "不要散播惡毒物。": "莫散播惡毒物。",
            "鯨魚是哺乳動物。": "海翁是哺乳動物。",
            "農場飼奶動物。": "農場飼奶動物。",
            "他對孩子太狠心。": "伊對囡仔太狠心。",
            "他很有雄心。": "伊真有雄心。",
            "兩片零件已經密合。": "兩片零件已經密合。",
            "接頭的密合度。": "接頭的密合度。",
            "我今天清洗被套。": "我今仔日清洗被單。",
            "投資人高點買進後被套牢。": "投資人懸點買進後被套牢。",
            "玩偶被套在袋子裡。": "尪仔被套佇袋仔內底。",
            "他慢慢走上台階。": "伊沓沓仔行上砛。",
            "平台階段測試已完成。": "平台階段測試已完成。",
            "家庭每月開支很高。": "家庭每月支出真懸。",
            "系統已開支援模式。": "系統已開支援模式。",
            "早餐煮小米粥。": "早頓煮秮仔米粥。",
            "小米手機今天更新。": "小米手機今仔日更新。",
            "我買了小米。": "我買了小米。",
            "兩家相交多年。": "兩家交陪久年。",
            "兩條直線在原點相交。": "兩條直線在原點相交。",
            "集合相交。": "集合相交。",
            "兩人的方向相反。": "兩人的方向顛倒反。",
            "負數的相反數是正數。": "負數的相反數是正數。",
            "他的看法相反。": "伊的看法相反。",
            "請連接這兩條電線。": "請敆倚這兩條電線。",
            "USB連接埠沒有反應。": "USB連接埠無反應。",
            "請連接伺服器。": "請連接伺服器。",
            "警方找到一顆子彈。": "警方揣著一顆銃子。",
            "子彈列車準時進站。": "子彈列車準時入站。",
            "請拿扳手鎖緊螺絲。": "請拿十扳仔鎖緊螺絲。",
            "他贏得扳手腕比賽。": "伊贏得扳手腕比賽。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_technical_and_compound_boundary_governance_batch7(self) -> None:
        expected = {
            "地面很潮濕。": "塗跤真潮濕。",
            "基地面積很大。": "基地面積真大。",
            "接地面積不足。": "接地面積不足。",
            "地面積水。": "地面積水。",
            "他是我的孫子。": "伊是我的孫仔。",
            "我正在讀孫子兵法。": "我佇咧讀孫子兵法。",
            "孫子曰兵者詭道也。": "孫子曰兵者詭道也。",
            "用小火慢煮。": "用勻勻仔火落去煮。",
            "小火箭升空。": "小火箭升空。",
            "這個小子很聰明。": "這个查埔囝仔真聰明。",
            "箱子裡有小子彈。": "箱仔底有小銃子。",
            "這兩個角互為對角。": "這兩个角互為斜角。",
            "請畫出對角線。": "請畫出斜角線。",
            "這是對角矩陣。": "這是對角矩陣。",
            "計算對角元素。": "計算對角元素。",
            "矩陣可以對角化。": "矩陣會當對角化。",
            "山谷傳來回聲。": "山谷傳來應聲。",
            "蝙蝠使用回聲定位。": "夜婆使用回聲定位。",
            "請分析回聲訊號。": "請分析回聲訊號。",
            "我認識他。": "我熟似伊。",
            "請辨認識別證。": "請辨認識別證。",
            "辨認識別不同。": "辨認識別無仝。",
            "警方查獲賣淫行為。": "警方查獲賣淫行為。",
            "被告涉嫌媒介未成年人賣淫。": "被告涉嫌媒介未成年人賣淫。",
            "法官詰問證人。": "法官詰問證人。",
            "檢察官開始交互詰問證人。": "檢察官開始交互詰問證人。",
            "醫師檢查病人的陰道。": "醫師檢查病人的陰道。",
            "陰道超音波檢查需要預約。": "陰道超音波檢查需要預約。",
            "他用手護住胸部。": "伊用手護住胸部。",
            "病人接受胸部X光檢查。": "病人接受胸部X光檢查。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_medical_legal_and_technical_governance_batch8(self) -> None:
        expected = {
            "人的臟腑都在胸腹腔內。": "人的臟腑攏咧胸腹腔內。",
            "影像顯示胸腹臟腑損傷。": "影像顯示胸腹臟腑損傷。",
            "中醫臟腑辨證系統。": "漢醫臟腑考證系統。",
            "請你下午抽空過來。": "請你下晝閬工過來。",
            "真空槽完成抽空程序。": "真空槽完成抽空程序。",
            "幫浦正在對腔體抽空。": "幫浦佇咧對腔體抽空。",
            "設備抽空作業完成。": "設備抽空作業完成。",
            "肥皂產生很多泡沫。": "雪文產生誠濟沫。",
            "央行警告房市泡沫風險。": "央行警告房市泡沫風險。",
            "設備採用泡沫滅火系統。": "設備採用泡沫滅火系統。",
            "工廠生產泡沫塑膠。": "工場生產泡沫塑膠。",
            "金融泡沫破裂。": "金融泡沫破裂。",
            "工人打開灌溉水門。": "工人拍開淹田水閘。",
            "記者調查水門事件。": "記者調查水門事件。",
            "法院引用水門案判例。": "法院引用水門案判例。",
            "水門醜聞震驚全國。": "水門醜聞震驚全國。",
            "古人留下許多智慧。": "古早人留下許多智慧。",
            "她研究古人類學。": "伊研究古人類學。",
            "考古隊發現古人類化石。": "考古隊發現古人類化石。",
            "農民珍惜每一塊耕地。": "農民珍惜每一塊耕地。",
            "政府統計農業耕地面積。": "政府統計農業耕地面積。",
            "本案涉及耕地租用條例。": "本案涉及耕地租用條例。",
            "這名兒童是混血兒。": "這名兒童是半仿仔。",
            "土生仔不是混血兒。": "土生仔毋是半仿仔。",
            "她是一位台日混血兒演員。": "伊是一位台日半仿仔演員。",
            "那位老人昨天死亡。": "彼位老人昨昏過身。",
            "報告統計嬰兒死亡率。": "報告統計紅嬰仔死亡率。",
            "醫師開立死亡證明。": "醫師開立死亡證明。",
            "死亡原因尚待調查。": "死亡原因尚待調查。",
            "醫師判定死亡。": "醫師判定死亡。",
            "他不小心碰傷陰莖。": "伊不小心碰傷𡳞鳥。",
            "病人接受陰莖癌治療。": "病人接受陰莖癌治療。",
            "醫師診斷陰莖骨折。": "醫師診斷陰莖骨折。",
            "陰莖勃起功能障礙需要治療。": "陰莖勃起功能障礙需要治療。",
            "醫師檢查陰莖。": "醫師檢查陰莖。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_dictionary_verified_winding_translation_batch8(self) -> None:
        expected = {
            "請把延長線捲線收好。": "請共延長線經線收好。",
            "自動捲線機發生故障。": "自動經線機發生故障。",
            "馬達線圈正在進行捲線。": "馬達線圈佇咧進行經線。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_reverse_gloss_and_domain_boundary_governance_batch9(self) -> None:
        expected = {
            "這朵花有七瓣。": "這朵花有七瓣。",
            "葛是古代國名。": "葛是古代國名。",
            "花瓣呈卵形。": "花瓣呈卵形。",
            "榕樹屬於桑科。": "榕仔屬於桑科。",
            "那隻貓眼大。": "那隻貓眼大。",
            "一種菜。": "一種菜。",
            "文體名。": "文體名。",
            "狗叫聲。": "狗叫聲。",
            "沸水產生水蒸氣。": "沸水產生水蒸氣。",
            "實驗測量飽和水蒸氣壓。": "實驗測量飽和水蒸氣壓。",
            "他對感情非常專一。": "伊對感情非常專一。",
            "酵素具有受質專一性。": "酵素具有受質專一性。",
            "她穿著深藍色外套。": "伊穿著紺色外衫。",
            "影像以深藍色標示低溫區域。": "影像以紺色標示低溫區域。",
            "小說主角被稱為私生子。": "小說主角被稱為私生子。",
            "法律已避免使用私生子等歧視稱呼。": "法律已避免使用私生子等歧視稱呼。",
            "他喜歡到海邊潛水。": "伊佮意到海墘藏水沬。",
            "海軍的潛水艇正在下潛。": "海軍的潛水艇佇咧下潛。",
            "房間保持清潔。": "房間保持清氣。",
            "實驗室使用中性清潔劑。": "實驗室使用中性清潔劑。",
            "他在市場做買賣。": "伊在市場做生理。",
            "雙方簽訂不動產買賣契約。": "雙方簽訂不動產買賣契約。",
            "死者留下一封遺書。": "死者留下一封遺書。",
            "法院鑑定遺書是否本人書寫。": "法院鑑定遺書是否本人冊寫。",
            "港口設置導航浮標。": "港口設置導航浮標。",
            "請換釣魚用的浮標。": "請換釣魚用的浮沉。",
            "釣魚用的浮標沉下去了。": "釣魚用的浮標沉落去矣。",
            "系統提供程式接口。": "系統提供程式接口。",
            "網路接口發生錯誤。": "網路接口發生錯誤。",
            "兩個管線的接口漏水。": "兩个管線的接喙漏水。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_round532_reverse_gloss_policy_and_runtime_artifacts_stay_in_parity(self) -> None:
        import json
        from pathlib import Path

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
        data_path = Path(__file__).resolve().parents[1] / "data" / "lexicon_entries.jsonl"
        dangerous_rows = []
        for line in data_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            is_three_to_one = (
                row.get("status") == "active"
                and row.get("tier") in {"base", "domain"}
                and row.get("trust") in {"seed", "machine"}
                and row.get("level") in {"phrase", "sentence"}
                and len(row.get("src", "")) == 3
                and len(row.get("tgt", "")) == 1
            )
            is_short_reverse_gloss = row.get("status") == "active" and row.get("src") in short_sources
            if is_three_to_one or is_short_reverse_gloss:
                dangerous_rows.append(row)

        self.assertEqual(len(dangerous_rows), 80)
        for row in dangerous_rows:
            with self.subTest(entry_id=row["entry_id"], source=row["src"]):
                result = self.converter.convert(row["src"], trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertNotIn(row["entry_id"], {match.entry_id for match in result.matches})
                self.assertNotEqual(result.output, row["tgt"])

    def test_v3_semantic_disambiguation_and_identifier_protection(self) -> None:
        expected = {
            "垃圾車快到了，先把廚房那袋垃圾拿出去。": ("糞埽車欲到矣，先共廚房那袋糞埽提出去。"),
            "我不要甜的，請幫我換成無糖的。": "我無愛甜的，請共我換成無糖的。",
            "我太太不能吃辣，煮她的那份不要加辣椒。": ("我太太袂當食辣，煮伊的那份莫加番仔薑。"),
            "我還一本書給同事，不是又買一本。": "我還一本冊給同事，毋是又買一本。",
            "這台電腦打不開，不是外殼被鎖住。": ("這台電腦開袂起來，毋是外殼被鎖住。"),
            "明天搭 THSR 0821，不是 THSR 0812。": ("明仔載搭 THSR 0821，毋是 THSR 0812。"),
            "請在 GitHub issue #431 留言，不要關閉 PR-77。": ("請在 GitHub issue #431 留言，莫關閉 PR-77。"),
            "總共三百八十元，我付五百元，應該找我一百二十元。": ("總共三百八十元，我付五百元，應該找我一百二十元。"),
            "他今天在家帶小孩，不是帶孩子出門。": ("伊今仔日佇厝𤆬囡仔，毋是帶囡仔出門。"),
            "我找了半天，原來手機就在桌上。": "我揣規半工，原來手機就在桌頂。",
            "這個人很會說話。": "這个人真𠢕講話。",
            "我不是肚子痛，是右邊腰部在痛。": "我毋是腹肚疼，是正手爿腰部在痛。",
            "兒子如果比我們早到，叫他先打電話。": "後生若是比咱早到，叫伊先拍電話。",
            "你如果先到，就在門口等我，不要一直打電話。": ("你若是先到，就佇門跤口等我，莫一直拍電話。"),
            "你先把窗戶關起來，外面風很大。": "你先共窗仔門關起來，外面風真大。",
            "我對花生過敏，請確認醬料裡沒有花生。": ("我對塗豆過敏，請確認醬料裡無塗豆。"),
            "警察破了這個案子。": "警察破這个案子矣。",
            "你越著急，就越容易出錯。": "你越著急，就越容易走縒。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_legacy_protected_debt_root_fixes_are_semantic_and_idempotent(self) -> None:
        expected = {
            "飛吧！熊鷹": "飛吧！熊鷹",
            "杏和醫院：因故障已拆除": "杏和醫院：因故障已拆除",
            "但您聽過卡通角色跟您說掰掰嗎？": "但咱聽過卡通角色佮咱講再會無？",
            "好那你有哪裡受傷嗎？沒有沒有": "好那你有佗位受傷無？無啦無啦",
            "乘客說：「滿特別的啦，滿新奇的，心靈有一些撫慰的感覺，上下班可能會開心一點，希望多設一些，所有的人都可以搭得到。」": (
                "乘客講：「滿特別的啦，滿新奇的，心靈有一寡撫慰的感覺，上下班可能會開心一點，希望多設一寡，所有的人都可以搭得到。」"
            ),
            "畢竟那還是店家的東西，就是溫和一點去請店家拆下來，給我們就是讓我確認一下": (
                "畢竟那猶是店家的物件，就是溫和一點去請店頭家拆落來，予咱就是予我確認一下"
            ),
            "活動現場還特別打造沉浸式台式小吃場景，重現街頭熟悉景象": (
                "活動現場閣特別打造沉浸式台式小吃場景，重現街頭熟悉景象"
            ),
            "第一個主軸，是我們要跨時代，跨世代的方式，我們來演唱，或是詮釋，文夏老師的創作": (
                "第一個主軸，是阮欲跨時代，跨世代的方式，阮來演唱，抑是詮釋，文夏老師的創作"
            ),
            "搜救人員說：「我們已經找到，5名寮國籍民眾，他們全都平安，接下來我們將繼續，展開救援行動。」": (
                "搜救人員講：「阮已經揣著，五名寮國籍民眾，怹攏平安，紲落去阮欲繼續，展開救援行動。」"
            ),
            "我們持續地盤整，縣長跟鄉鎮市長，我們會連動地，來做全面的盤整的提名與徵召": (
                "阮持續地盤整，縣長佮鄉鎮市長，阮會連動地，來做全面的盤整的提名與徵召"
            ),
            "結果事後店員發現，包廂內的偵煙警報器與偵熱感應器，其中一顆，被誤以為內部藏有針孔遭硬拔下來": (
                "結果事後店員發現，包廂內的偵煙警報器與偵熱感應器，其中一顆，予人掠準內部藏有針孔遭硬拔下來"
            ),
            "當事店家說：「像我們正常是這樣子安裝是兩個設備。」": ("當事店頭家講：「像阮正常是按呢安裝是兩个設備。」"),
            "明明我們柯主席講得非常清楚，就是說各黨之間，因為我們就是所謂第三黨，要做議題合作": (
                "明明阮柯主席講得非常清楚，就是說各黨之間，因為阮就是所謂第三黨，要做議題合作"
            ),
            "美方盟友的大日子總統賴清德還特別準備3樣禮物": ("美方盟友大日子總統賴清德閣特別準備三樣禮物"),
            "為了讓民眾方便探索在地好味道，台中市工商發展投資策進會推出尋味台中，胃你導航數位美食平台": (
                "為著予民眾利便探索在地好滋味，台中市工商發展投資策進會推出尋味台中，胃你導航數位美食平台"
            ),
            "為了讓民眾更方便探索在地好味道": "為著予民眾閣較利便探索在地好滋味",
            "社會住宅，應該要馬上達標，讓青年人生育以後，有國家的免費住宅給他們居住": (
                "社會住宅，應該要連鞭達標，讓青年人生育以後，有國家的免費住宅給他們居住"
            ),
            "民眾說：「就列個東西告示牌，清楚告知使用規範。」": ("民眾講：「就列個東西告示牌，清楚告知使用規範。」"),
            "讓民眾直呼上下班會開心一點，不只車上處處有驚喜，因為聽聽這下車鈴": (
                "讓民眾直呼上下班會歡喜一點，不只車上處處有驚喜，因為聽聽這下車鈴"
            ),
            "惹哭全網：辛苦了": "惹哭全網：辛苦矣",
            "給我們的公司": "予阮的公司",
            "我們公司明天會再回覆您。": "阮公司明仔載會閣回覆咱。",
            "我婆婆和我媽媽同姓，但不是姐妹。": "阮大家佮阮阿母仝姓，毋過毋是姐妹。",
            "老闆說預算可以少一成，工期不能少一天。": ("頭家講預算會當少一成，工期袂當少一工。"),
            "你們兩個先走，我們三個留下來。": "恁兩个行先，阮三个留落來。",
            "他把書還我，卻還在生氣。": "伊共冊還我，毋過猶咧受氣。",
            "這道菜很下飯，但價格降不下來。": "這道菜真下飯，毋過價數落袂來。",
            "我們已經替您保留兩個座位，您不用再訂。": ("阮已經替咱留兩个座位，咱免閣訂。"),
            "你們家的狗認得我們，但不認得他們。": ("恁兜的狗認得阮，毋過無認得𪜶。"),
            "如果 build_26 失敗，就不要合併 PR#908。": ("若是 build_26 失敗，就莫合併 PR#908。"),
            "班長大後想當老師，不是一直當班長。": ("班長大漢了後想當老師，毋是一直當班長。"),
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_official_shared_term_is_not_globally_shortened(self) -> None:
        expected = {
            "發放": "發放",
            "發放津貼": "發放津貼",
            "0到18歲我們發放": "零到十八歲我們發放",
            "政府發放津貼": "政府發放津貼",
            "發放現金": "發予現錢",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.converter.convert(source), target)

    def test_v2_ai_semantic_audit_regressions(self) -> None:
        expected = {
            "請把房間打掃乾淨。": "請共房間摒掃予清氣。",
            "哥哥比我高，但是我跑得比他快。": "阿兄比我較懸，毋過我走比伊較緊。",
            "請把 PDF 寄給 OpenAI，案件編號是 12345678。": ("請把 PDF 寄給 OpenAI，案件編號是 12345678。"),
            "你要刷卡還是付現金？": "你欲刷卡抑是付現錢？",
            "請把統一編號打在發票上。": "請共統一編號拍佇發票頂面。",
            "我昨天花了三千元。": "我昨昏開三千箍。",
            "他每天開車上班。": "伊逐工駛車去上班。",
            "我在臺中高鐵站等你，到了用 LINE 告訴我。": ("我佇臺中高鐵站等你，到了用 LINE 共我講。"),
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.converter.convert(source), target)

    def test_verified_taiwanese_terms_are_idempotent(self) -> None:
        for text in ("人客", "會議時間", "飲食", "火車頭", "方便", "干焦", "進前", "差不多"):
            with self.subTest(text=text):
                self.assertEqual(self.converter.convert(text), text)

    def test_semantic_boundaries_preserve_distinct_lexical_meanings(self) -> None:
        expected = {
            "上下樓梯": "上下樓梯",
            "下樓梯": "落樓梯",
            "在旁邊": "佇邊仔",
            "旁邊": "邊仔",
            "厲害": "厲害",
            "好厲害": "真厲害",
            "更厲害": "閣較厲害",
            "利害關係": "利害關係",
            "太后": "太后",
            "鄉里": "鄉里",
            "千里眼": "千里眼",
            "票價是五十元": "票價是五十元",
            "雖然這班車停的站很多，但是票價比較便宜。": ("雖然這班車停的站真濟，毋過票價較俗。"),
            "如果你手機沒電，櫃檯旁邊有充電區。": ("若是你手機無電，櫃檯邊仔有充電區。"),
            "蘋果、香蕉等等": "蘋果、弓蕉等等",
            "等等我": "等我一下仔",
            "南下": "落南",
            "最後": "最後",
            "最後說了": "最後講矣",
            "最後一站": "最後一站",
            "扯後腿": "搝後跤",
            "客家人": "客人",
            "客人": "客人",
            "幫客人": "替人客",
            "可以幫客人換房嗎？": "會當替人客換房無？",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.converter.convert(source), target)

    def test_invalid_reverse_and_opposite_entries_do_not_mutate_canonical_text(self) -> None:
        for text in (
            "唏嚇叫",
            "燒的飲料",
            "倒手爿",
            "無夠",
            "彼站",
            "野球場",
            "門跤口",
            "轉來",
            "拍開",
            "窗仔",
            "窗仔門",
            "外口",
            "物仔",
            "偏心",
            "倚近",
            "使用",
            "遺失物",
            "目的地",
            "物品",
            "車牌號碼",
            "本人",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.converter.convert(text), text)
        self.assertNotEqual(self.converter.convert("袂記得"), "會記得")
        self.assertNotEqual(self.converter.convert("別款"), "仝款")

    def test_context_boundaries_prevent_substring_and_recursive_corruption(self) -> None:
        stable_texts = (
            "涉案人會面臨罰鍰",
            "我想欲改開會時間。",
            "我想欲改用餐時間。",
            "慢性病藥",
            "落雨天",
            "窗仔門",
            "外口",
            "物仔",
            "偏心",
            "倚近",
            "使用",
            "遺失物",
            "目的地",
            "物品",
            "車牌號碼",
            "本人",
            "問一下仔",
            "改用投現錢",
            "推嬰仔車上車進前",
            "這班車上遠只到海墘",
            "雨天上車",
            "舊曆四月初六",
            "一週五人",
            "會當替我買藥仔無？",
            "等我一下仔",
            "請毋通將窗仔門拍開。",
            "我轉來矣。",
            "我等你佇門跤口。",
            "醫生這馬猶在看上一位，你閣等一下仔。",
            "今仔日車站人誠濟，你先到邊仔等一下。",
            "大聲",
            "大聲話",
            "大聲呻",
            "別人",
            "別人的票卡",
            "一定",
            "定著",
            "無一定",
            "等一下",
            "等咧",
            "播放",
            "播送",
            "唏嚇叫",
        )
        for text in stable_texts:
            with self.subTest(text=text):
                self.assertEqual(self.converter.convert(text), text)

        self.assertEqual(self.converter.convert("問一下"), "問一下仔")
        self.assertEqual(self.converter.convert("改用投現"), "改用投現錢")
        self.assertEqual(self.converter.convert("車上真濟人"), "車頂真濟人")
        self.assertEqual(self.converter.convert("月初"), "月頭")
        self.assertEqual(self.converter.convert("週五"), "禮拜五")
        self.assertEqual(self.converter.convert("買藥"), "敆藥仔")
        self.assertEqual(self.converter.convert("等我一下"), "等我一下仔")
        self.assertEqual(self.converter.convert("等一下我先處理"), "等咧我先處理")
        self.assertEqual(self.converter.convert("等一下會開車"), "等咧會開車")
        self.assertEqual(self.converter.convert("等一下如果有空"), "等咧若是有閒")
        self.assertEqual(self.converter.convert("等一下先處理"), "等咧先處理")
        self.assertEqual(self.converter.convert("不一定"), "無一定")
        self.assertEqual(self.converter.convert("不一定要來"), "無一定愛來")
        self.assertEqual(self.converter.convert("一定要來"), "一定愛來")

    def test_formal_register_boundaries_preserve_legal_medical_and_engineering_terms(self) -> None:
        formal_expected = {
            "故意殺人罪的構成要件。": "故意殺人罪的構成要件。",
            "法院認定被告具有故意。": "法院認定被告具有故意。",
            "律師代寫告狀。": "律師代寫告狀。",
            "向法院遞交告狀。": "向法院遞交告狀。",
            "公司申請商標註冊。": "公司申請商標註冊。",
            "商標權受到侵害。": "商標權受著侵害。",
            "醫師進行氣管插管。": "醫師進行氣管插管。",
            "病人罹患氣管炎。": "病人罹患氣管炎。",
            "醫師診斷哮喘。": "醫師診斷哮喘。",
            "哮喘患者使用吸入器。": "哮喘患者使用吸入器。",
            "醫師檢查喉頭。": "醫師檢查喉頭。",
            "病人接受喉頭癌治療。": "病人接受喉頭癌治療。",
            "材料反折測試。": "材料反折測試。",
            "工程反折強度。": "工程反折強度。",
        }
        for source, expected in formal_expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, expected)
                self.assertEqual(self.converter.convert(first), first)

        colloquial_expected = {
            "伊是故意按呢做。": "伊是刁工按呢做。",
            "伊欲去告狀。": "伊欲去投。",
            "這是伊的商標。": "這是伊的標頭。",
            "氣管真重要。": "肺管真要緊。",
            "伊有哮喘。": "伊有痚呴。",
            "喉頭真疼。": "頷頸胿真疼。",
            "共伊反折。": "共伊撆。",
        }
        for source, expected in colloquial_expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.converter.convert(source), expected)

    def test_professional_compound_boundaries_preserve_terms_and_general_lexemes(self) -> None:
        professional_expected = {
            "這種植物可食。": "這款植物可食。",
            "律師接受委任。": "律師接受委任。",
            "雙方成立委任契約。": "雙方成立委任契約。",
            "受委任人應善盡義務。": "受委任人應善盡義務。",
            "醫師評估疼痛指數。": "醫師評估疼痛指數。",
            "慢性疼痛患者。": "慢性疼痛患者。",
            "職業安全衛生管理。": "職業安全衛生管理。",
            "職業災害統計。": "職業災害統計。",
            "工具機產業展。": "工具機產業展。",
            "軟體開發工具。": "軟體開發工具。",
            "哺乳期婦女用藥。": "哺乳期婦女用藥。",
            "永久居留證。": "永久居留證。",
            "永久性損傷。": "永久性損傷。",
            "育兒津貼發放。": "育兒津貼發放。",
            "刑法規定罰金刑。": "刑法規定罰金刑。",
            "攜帶式超音波設備。": "攜帶式超音波設備。",
            "智慧工廠自動化系統。": "智慧工廠自動化系統。",
            "男子單打冠軍。": "男子單打冠軍。",
            "患者出現劇烈疼痛。": "患者出現劇烈疼痛。",
            "咽喉癌篩檢。": "咽喉癌篩檢。",
            "請共我查帳戶餘額。": "請共我查口座餘額。",
            "法院判命返還本金。": "法院判命返還母錢。",
            "導線反折疲勞試驗。": "導線反折疲勞試驗。",
            "軟板反折壽命測試。": "軟板反折壽命測試。",
        }
        for source, expected in professional_expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, expected)
                self.assertEqual(self.converter.convert(first), first)

        general_expected = {
            "這件代誌委任伊處理。": "這件代誌交仗伊處理。",
            "我真疼痛。": "我真疼。",
            "伊的職業是醫師。": "伊的頭路是醫師。",
            "這是做工的工具。": "這是做工的家私。",
            "媽媽咧哺乳。": "阿母咧飼奶。",
            "愛永久記牢。": "愛永遠記牢。",
            "伊咧育兒。": "伊咧育囝。",
            "罰金真重。": "罰款真重。",
            "伊攜帶一支雨傘。": "伊帶一支雨傘。",
            "伊佇工廠做工。": "伊佇工場做工。",
            "彼个男子真懸。": "彼个查埔真懸。",
            "反應真劇烈。": "反應真激烈。",
            "伊的咽喉真疼。": "伊的嚨喉管真疼。",
            "會計師來查帳。": "會計師來查數。",
        }
        for source, expected in general_expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.converter.convert(source), expected)

    def test_authoritative_semantic_and_unicode_governance_batch12(self) -> None:
        expected = {
            "牙根疼痛。": "喙齒根疼。",
            "牙根尖手術。": "牙根尖手術。",
            "患者感到噁心。": "患者感覺噁心。",
            "噁心嘔吐。": "噁心嘔吐。",
            "化療引起噁心。": "化療引起噁心。",
            "這行為真噁心。": "這行為真噁心。",
            "噁心。": "噁心。",
            "環境真骯髒。": "環境真癩𰣻。",
            "媽祖進香活動。": "媽祖進香活動。",
            "動物園有企鵝。": "動物園有徛鵝。",
            "國際情勢分析。": "國際情形分析。",
            "法院判命返還本金。": "法院判命返還母錢。",
            "伊感覺反胃。": "伊感覺反腹。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_precision_boundary_and_pua_governance_batch13(self) -> None:
        expected = {
            # 支付：固定術語保留，一般付款動作仍轉換。
            "支付命令。": "支付命令。",
            "支付寶。": "支付寶。",
            "支付工具。": "支付工具。",
            "支付款項。": "支付款項。",
            "支付指示。": "支付指示。",
            "支付系統。": "支付系統。",
            "支付服務。": "支付服務。",
            "支付平台。": "支付平台。",
            "支付價款。": "支付價款。",
            "電子支付機構。": "電子支付機構。",
            "電子支付帳戶。": "電子支付帳戶。",
            "第三方支付。": "第三方支付。",
            "跨境支付。": "跨境支付。",
            "款項明日支付。": "錢項明日付錢。",
            "我來支付。": "我來付錢。",
            # 詞界與金融動詞。
            "戰事情勢。": "戰事情勢。",
            "軍事情勢。": "軍事情勢。",
            "人事情勢。": "人事情勢。",
            "事情真濟。": "代誌真濟。",
            "本金攤還。": "母錢攤還。",
            "分期攤還本金。": "分期攤還母錢。",
            "伊還我錢。": "伊還我錢。",
            "還你一本冊。": "還你一本冊。",
            "還有一本冊。": "閣有一本冊。",
            # 外幣、入帳與責任用語。
            "例外幣別。": "例外幣別。",
            "例外幣種。": "例外幣種。",
            "外幣存款。": "外票存款。",
            "輸入帳號。": "輸入帳號。",
            "登入帳戶。": "登入口座。",
            "匯入帳號。": "匯入帳號。",
            "記入帳簿。": "記入數簿。",
            "款項已入帳。": "錢項已入數。",
            "推卸責任。": "卸責任。",
            "不要推卸責任。": "莫卸責任。",
            # 進香／割香只在明確窄義轉換。
            "促進香火。": "促進香火。",
            "增進香客。": "增進香客。",
            "改進香爐。": "改進香爐。",
            "精進香料。": "精進芳料。",
            "媽祖進香活動。": "媽祖進香活動。",
            "信徒前往進香。": "信徒去進香。",
            "神像進香。": "神像割香。",
            "進香隊伍帶著神尊。": "割香隊伍帶著神尊。",
            # 一般牙根義、牙科 compound 與多義噁心。
            "我的牙根真疼。": "我的喙齒根真疼。",
            "患者有牙根尖周圍炎。": "患者有牙根尖周圍炎。",
            "影像顯示牙根吸收。": "影像顯示牙根吸收。",
            "牙根治療。": "牙根治療。",
            "我感覺噁心。": "我感覺噁心。",
            "患者出現噁心嘔吐。": "患者出現噁心嘔吐。",
            "術後噁心與嘔吐。": "術後噁心佮嘔吐。",
            "噁心量表用於評估。": "噁心量表用於評估。",
            "我噁心想吐。": "我喙凊。",
            "彼个噁心鬼。": "彼个癩𰣻鬼。",
            # PUA target 已由標準 Unicode 詞形取代。
            "她噴香水。": "伊澍芳水。",
            "噴香水。": "澍芳水。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)
                self.assertFalse(any(0xE000 <= ord(char) <= 0xF8FF for char in first))
                self.assertFalse(any(0xF0000 <= ord(char) <= 0xFFFFD for char in first))
                self.assertFalse(any(0x100000 <= ord(char) <= 0x10FFFD for char in first))

    def test_short_edge_semantic_governance_batch15(self) -> None:
        expected = {
            # 反義、資料腐敗與未證實近義不可全域改寫。
            "他因此得利。": "伊就按呢得利。",
            "受害者與得利者。": "受害者佮得利者。",
            "他的食量很大，是大食的人。": "伊的食量真大，是大食的人。",
            "本書奉贈讀者。": "本冊奉贈讀者。",
            "僧人正在念誦。": "僧人佇咧念誦。",
            # 辭典定義／例示片段不可反向覆蓋性質描述。
            "通書記載每日忌宜。": "通書記載每日忌宜。",
            "病人怕光。": "病人怕光。",
            "雙方攻防激烈。": "雙方攻防激烈。",
            "珠螺的殼薄。": "珠螺的殼薄。",
            "菱角是水生植物。": "菱角是水生植物。",
            "油漆外乾內未乾。": "油漆外乾內未乾。",
            "大麥是五穀之一。": "大麥是五穀之一。",
            # 下位類型、數學／量詞與專業複合詞保持精確語義。
            "刀術是武術的一種。": "刀術是武術的一種。",
            "這次有十位參賽者。": "這擺有十位參賽者。",
            "個位、十位與百位。": "個位、十位佮百位。",
            "大宗商品。": "大宗商品。",
            "這是一筆大宗交易。": "這是一筆大宗交易。",
            "材料性質。": "材料性質。",
            "化學性質。": "化學性質。",
            "工作性質。": "工作性質。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_reverse_definition_and_archaism_governance_batch16(self) -> None:
        expected = {
            # 辭典釋義的尺寸、動作與例句片段不可反向改成被定義詞。
            "這條繩子有八尺長。": "這條索仔有八尺長。",
            "他生得口小。": "伊生得口小。",
            "牠的嘴大。": "牠的嘴大。",
            "孩子頭小身體大。": "囡仔頭小身體大。",
            "他指人處理內部事務。": "伊指人處理內部事務。",
            "大同地區以產煤聞名。": "大同地區以產煤聞名。",
            "這件衣服稍大。": "這件衫稍大。",
            # 反義與古漢語訓詁不可洩漏到現代一般句。
            "少少就好。": "少少就好。",
            "大筏停在岸邊。": "大筏停在岸邊。",
            "他憑借經驗完成任務。": "伊憑借經驗完成任務。",
            # 已有權威詞義支持的臺語詞不得被粗暴短詞規則誤傷。
            "青椒與點滴。": "大同仔佮大筒。",
            "他的小肚不舒服。": "伊的膀胱無爽快。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_definition_fragment_and_archaism_governance_batch17(self) -> None:
        expected = {
            # 「即」是原句的判斷成分，不得連同辭典釋義殘片一起刪除。
            "這種動物即狗。": "這款動物即狗。",
            "古文記載此物即矢。": "古文記載此物即矢。",
            "地下莖即藕。": "地下莖即藕。",
            "古籍說此獸即貍。": "古籍說此獸即貍。",
            "此魚即鰈。": "此魚即鰈。",
            # 植物、藥性、料理與器物特徵不可反向縮成詞目。
            "葉片互生。": "葉片互生。",
            "外形似槐。": "外形似槐。",
            "表面寒滑。": "表面寒滑。",
            "葉序對生。": "葉序對生。",
            "這道菜用肉很多。": "這道菜用肉誠濟。",
            "長柄。": "長柄。",
            # 有古訓關係不等於適合現代 context-free 全域替換。
            "他穿著長衣。": "伊穿著長衣。",
            "這件短衣很輕。": "這件短衣真輕。",
            "雕刻使用美石。": "雕刻使用美石。",
            "寶石泛出玉光。": "寶石泛出玉光。",
            # 合法且有權威依據的短詞轉換不得被 exact-edge 防線誤傷。
            "青椒與點滴。": "大同仔佮大筒。",
            "他的小肚不舒服。": "伊的膀胱無爽快。",
            "垃圾很多。": "糞埽誠濟。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

    def test_archaic_gloss_and_truncated_target_governance_batch18(self) -> None:
        expected = {
            # 古典單字訓詁、通假字與釋義片段不得反向覆蓋現代一般用法。
            "情勢已到危殆地步。": "情形已到危殆地步。",
            "他因多言惹禍。": "伊因多言惹禍。",
            "目的地很近。": "目的地很近。",
            "屋頂的楯脊受損。": "厝頂的楯脊受損。",
            "這個語助詞本身無義。": "這个語助詞本身無義。",
            "工廠正在煉鐵。": "工場佇咧煉鐵。",
            "竹簟鋪在床上。": "竹簟鋪在床上。",
            "他置身事外。": "伊置身事外。",
            "他把箱子舉起。": "伊把箱仔舉起。",
            "不可貪求利益。": "不可貪求利益。",
            # 目標詞截斷不得造成宿主／寄生蟲混淆，也不得破壞成語。
            "雞蟲很多。": "雞蟲誠濟。",
            "雞蟲得失，不必計較。": "雞蟲得失，毋免窮分。",
            # 教育部臺語辭典直接支持的合法臺語動詞應提升 provenance，而非誤刪。
            "她披著一件外套。": "伊幔一件外衫。",
            "他披著披風。": "伊幔風幔。",
            "孩子披著毛毯。": "囡仔幔毛毯。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        promoted = self.converter.convert("她披著外套。", trace=True)
        assert isinstance(promoted, ConversionResult)
        self.assertTrue(any(match.entry_id == "lx_532000000640" for match in promoted.matches))

    def test_machine_override_and_archaic_gloss_governance_batch19(self) -> None:
        expected = {
            # 權威臺語詞彙應由 ai-reviewed entry 接手，不保留 machine provenance。
            "農夫拿圓鍬挖土。": "農夫拿沙挑挖塗。",
            "今天氣候嚴寒。": "今仔日氣候大寒。",
            "他拿劈刀砍柴。": "伊拿柴鍥剉柴。",
            "武術課正在練上弓步劈刀。": "武術課佇咧練上弓步劈刀。",
            "木工機的劈刀需要更換。": "木工機的劈刀需要換。",
            # 無法驗證或疑似混接的 machine target 應 fail closed。
            "蛋糕切成四份。": "雞卵糕切成四份。",
            "我們訂了四份餐點。": "咱訂了四份餐點。",
            "車子在路口回轉。": "車子佇路口回轉。",
            "請回轉身體。": "請回轉身軀。",
            # 古訓存在不代表適合現代 context-free 自動替換。
            "他乃是本案的證人。": "伊乃是本案的證人。",
            "宇智波佐助登場了。": "宇智波佐助登場矣。",
            "眾人佐助他完成工作。": "眾人佐助伊完成工作。",
            "朝廷被佞臣把持。": "朝廷被佞臣把持。",
            "這套系統依託雲端服務運作。": "這套系統依託雲端服務運作。",
            "官員每月領取俸祿。": "官員每月領取俸祿。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        promoted_ids = {
            "lx_532000000643",
            "lx_532000000644",
            "lx_532000000646",
        }
        for source in ("圓鍬", "嚴寒"):
            with self.subTest(promoted_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(promoted_ids.intersection(match.entry_id for match in result.matches))

        contextual = self.converter.convert("他拿劈刀砍柴。", trace=True)
        assert isinstance(contextual, ConversionResult)
        self.assertTrue(any(match.entry_id == "lx_532000000646" for match in contextual.matches))

    def test_polysemy_and_register_governance_batch20(self) -> None:
        expected = {
            # 身心承受義可保留權威臺語詞，載重能力必須 fail closed。
            "她承受很大的壓力。": "伊忍真大的壓力。",
            "這座橋能承受十噸重量。": "這座橋能承受十噸重量。",
            # 時間義與金額估值義必須分流。
            "正值颱風季節。": "正風颱季節。",
            "這台車正值十萬元。": "這台車正值十萬元。",
            # 粗魯命令語不可污染一般分享、勸食語境。
            "把水果拿去吃，不要浪費。": "把水果拿去吃，莫浪費。",
            # 店主語境可轉換，商店實體／組織語境應保留。
            "當事店家表示願意退款。": "當事店頭家表示肯退錢。",
            "平台整合店家資源。": "平台整合店家資源。",
            "店家設備需要維修。": "店家設備需要維修。",
            # 詞義窄化、詞性錯配與反向釋義抽取一律 fail closed。
            "這是常見疾病。": "這是常見病症。",
            "請按照常規流程處理。": "請照常規流程處理。",
            "他承繼家族事業。": "伊承繼家族事業。",
            "主管招集同仁開會。": "主管招集同仁開會。",
            "能力有其界限。": "能力有其界限。",
            "他的喜好是攝影。": "伊的喜好是攝影。",
            "他喜好甜食。": "伊喜好甜路。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        for source, entry_id in (
            ("她承受很大的壓力。", "lx_532000000653"),
            ("正值颱風季節。", "lx_532000000654"),
            ("當事店家表示願意退款。", "lx_532000000656"),
        ):
            with self.subTest(contextual_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))

    def test_contextual_phrase_can_override_shorter_protected_canonical_term(self) -> None:
        self.assertEqual(self.converter.convert("方便管理資料"), "利便管理資料")
        self.assertEqual(self.converter.convert("方便"), "方便")

    def test_rules_can_cross_a_protected_term_without_mutating_it(self) -> None:
        self.assertEqual(
            self.converter.convert("要去朝天宮的話，你在外圍下車就可以。"),
            "若是欲去朝天宮，你佇外圍落車就會當。",
        )

    def test_reverse_gloss_and_polysemy_governance_batch14(self) -> None:
        expected = {
            # 低信任辭典定義／例示片段不可反向覆蓋一般概念。
            "本研究提出一例。": "本研究提出一例。",
            "蛋糕直徑六寸。": "雞卵糕直徑六寸。",
            "正月初三。": "正月初三。",
            "他創制新制度。": "伊創制新制度。",
            "料理加蒜。": "料理加蒜。",
            "這是千足金。": "這是千足金。",
            "鳥類大多卵生。": "鳥類大多卵生。",
            "患者突然失音。": "患者忽然間失音。",
            "楊桃很多汁。": "楊桃真多汁。",
            "這株多肉植物。": "這株多肉植物。",
            "鹽酸是一種強酸。": "鹽酸是一種強酸。",
            "昆蟲幼蟲。": "蟲豸幼蟲。",
            "例如水、煤、紙、金屬和魚。": "例如水、煤、紙、金屬和魚。",
            "例如農業與船運。": "例如農業佮船運。",
            # 金額、一帶與上學的多義／跨詞界治理。
            "這枚硬幣是一元。": "這枚銀角仔是一箍。",
            "一元二次方程式。": "一元二次方程式。",
            "一元多項式。": "一元多項式。",
            "一元酸。": "一元酸。",
            "這一帶真鬧熱。": "這角勢鬧熱。",
            "一帶一路政策。": "一帶一路政策。",
            "囡仔去上學。": "囡仔去上課。",
            "上學期成績。": "上學期成績。",
            "向上學習。": "向上學習。",
            # 建築前庭、鐵路出軌與親屬稱謂只在已證實語境轉換。
            "厝的前庭真闊。": "厝的門跤口埕真闊。",
            "前庭花園。": "門跤口埕花園。",
            "前庭神經炎。": "前庭神經炎。",
            "前庭系統。": "前庭系統。",
            "列車出軌。": "列車敗馬。",
            "婚後出軌。": "婚後出軌。",
            "出軌事故。": "出軌事故。",
            "我的奶奶。": "我的阿媽。",
            "奶奶在洗碗。": "阿媽佇咧洗碗。",
            "牛奶奶油。": "牛奶奶油。",
            "鮮奶奶粉。": "鮮奶奶粉。",
            "牛奶奶茶。": "牛奶奶茶。",
            # 日常妄想與精神醫學術語、一般大小與技術複合詞分流。
            "你莫妄想。": "你莫戇想。",
            "妄想症。": "妄想症。",
            "被害妄想。": "被害妄想。",
            "妄想性障礙。": "妄想性障礙。",
            "物件大小。": "物件大細。",
            "大小便檢查。": "大小便檢查。",
            "大小寫字母。": "大小寫字母。",
            "大小端轉換。": "大小端轉換。",
            # 農曆大月與加拿大跨詞界、存貨與囤貨語義分離。
            "農曆大月有三十日。": "農曆月大有三十日。",
            "加拿大月份資料。": "加拿大月份資料。",
            "加拿大月報。": "加拿大月報。",
            "盤點存貨。": "盤點存貨。",
            "存貨成本。": "存貨成本。",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                first = self.converter.convert(source)
                self.assertEqual(first, target)
                self.assertEqual(self.converter.convert(first), first)

    def test_semantic_and_modal_governance_batch21(self) -> None:
        expected = {
            # 官方直接對應與詞界保護。
            "他不只會唱歌，也會跳舞。": "伊毋但會唱歌，也會跳舞。",
            "原以為會放晴，不料下午下雨。": "原掠準會放晴，無疑悟下晝落雨。",
            "這個人真不料量。": "這个人真不料量。",
            # 務必依肯定／否定情態分流，且第一輪即達 fixed point。
            "你務必要小心。": "你千萬愛小心。",
            "旅客務必攜帶護照。": "旅客千萬愛帶護照。",
            "務必不要洩漏密碼。": "千萬毋通洩漏密碼。",
            "務必不得進入。": "千萬毋通進入。",
            "務必勿靠近。": "千萬毋通靠近。",
            "即使下雨也要出門。": "準做落雨嘛欲出門。",
            # 詞性、強度與變化速率不明時 fail closed。
            "他努力工作。": "伊努力工作。",
            "他的努力值得肯定。": "伊的努力值得肯定。",
            "氣溫逐漸升高。": "氣溫逐漸升高。",
            "把桌面升高十公分。": "把桌面升高十公分。",
            # 死亡諱稱只在明確人物語境使用；寵物不套用人物 euphemism。
            "那位年輕作家意外去世。": "彼位年輕作家意外過身。",
            "三歲病童不幸去世。": "三歲病童不幸過身。",
            "他的寵物去世了。": "伊的寵物去世矣。",
            # occurrence-local 醫療、具體包圍、禮貌請求與「同樣是」語境。
            "醫生叫他吃藥，再吃藥水。": "醫生叫伊食藥仔，再食藥水。",
            "他因吸毒而吃藥成癮。": "伊因吸毒而食藥成癮。",
            "警察包圍歹徒，濃霧包圍山區。": "警察圍歹徒，濃霧包圍山區。",
            "突破包圍圈。": "突破包圍圈。",
            "請你包容，社會仍須包容多元文化。": "請你包涵，社會仍須包容多元文化。",
            "提升系統包容性。": "提升系統包容性。",
            "同樣是學生，買同樣的商品。": "平平是學生，買同樣的商品。",
            "同樣的道路。": "同樣的道路。",
            # 「好勢」多義且可能改變語氣；一般適宜義保留。
            "這個尺寸很合適。": "這个尺寸真合適。",
            "這藥合適兒童使用。": "這藥合適兒童使用。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        for source, entry_id in (
            ("原以為會放晴，不料下午下雨。", "lx_532000000665"),
            ("務必不要洩漏密碼。", "lx_532000000666"),
            ("你務必要小心。", "lx_532000000669"),
            ("即使下雨也要出門。", "lx_532000000671"),
            ("那位年輕作家意外去世。", "lx_532000000674"),
            ("醫生叫他吃藥。", "lx_532000000675"),
            ("警察包圍歹徒。", "lx_532000000676"),
            ("請你包容。", "lx_532000000677"),
            ("同樣是學生。", "lx_532000000678"),
        ):
            with self.subTest(contextual_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))

    def test_boundary_and_register_governance_batch22(self) -> None:
        expected = {
            # occurrence-local 邊界保護：同句中的安全 occurrence 轉換，風險 occurrence 保留。
            "不久他就回來了。久不久就會下雨。": "無偌久伊就轉來矣。久不久就會落雨。",
            "他不怕困難。不怕一萬，只怕萬一。": "伊毋驚困難。不怕一萬，只怕萬一。",
            "他不肯道歉，這個說法並不肯定。": "伊毋肯會失禮，這个說法並不肯定。",
            "他丟失錢包，系統丟失資料。": "伊拍毋見錢袋仔，系統丟失資料。",
            "他很有力氣，這是力氣活。": "伊真有氣力，這是力氣活。",
            # 未證成的「贏過」與正式／客套語域 fail closed。
            "行動勝過空談。取勝過程很辛苦。": "行動勝過空談。取勝過程真辛苦。",
            "他為這件事日夜勞神。勞神您代為轉交。": "伊為這層代誌日暝損神。勞神你代為轉交。",
            "長途奔波使大家很勞累。功勞累計三次。": "長途奔波使逐家真疲勞。功勞累計三擺。",
            "他工作一向很勤勉。董事勤勉義務。": "伊工作一向真骨力。董事勤勉義務。",
            # 教育部直接支持、詞性一致的烹調／療法詞採 promote。
            "師傅替他刮痧，也買了刮痧板。": "師傅替伊掠痧，也買了掠痧板。",
            "煮湯時要勾芡，先準備勾芡水。": "煮湯時要牽羹，先準備牽羹水。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        for source, entry_id in (
            ("不久他就回來了。", "lx_532000000680"),
            ("他不怕困難。", "lx_532000000681"),
            ("他不肯道歉。", "lx_532000000682"),
            ("他丟失錢包。", "lx_532000000683"),
            ("師傅替他刮痧。", "lx_532000000684"),
            ("他很有力氣。", "lx_532000000685"),
            ("他為這件事日夜勞神。", "lx_532000000687"),
            ("長途奔波使大家很勞累。", "lx_532000000688"),
            ("他工作一向很勤勉。", "lx_532000000689"),
            ("煮湯時要勾芡。", "lx_532000000690"),
        ):
            with self.subTest(contextual_or_promoted_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))

    def test_medical_temporal_and_boundary_governance_batch23(self) -> None:
        expected = {
            "他匆忙離開，行程十分匆忙。": "伊趕狂離開，行程十分匆忙。",
            "化痰藥有化痰作用。": "化痰藥有化痰作用。",
            "我們中午吃午飯，午飯後再出發。": "咱中晝食中晝飯，中晝飯後再出發。",
            "火車即將到站，立即將資料送出。": "火車咧欲到站，立即將資料送出。",
            "原先的計畫不變。": "原早的計畫不變。",
            "我厭惡暴力，風險厭惡程度上升。": "我討厭暴力，風險厭惡坎站上升。",
            "請及早準備，以及早期治療。": "請冗早準備，以及早期治療。",
            "不要取笑他，採取笑臉策略。": "莫恥笑伊，採取笑面策略。",
            "他受到表揚，接受到貨通知。": "伊受著表揚，接受到貨通知。",
            "雜草叢生，問題叢生，叢生植物很多。": "雜草叢生，問題叢生，叢生植物誠濟。",
            "真可惜，可惜成本很高。": "真無彩，可惜成本真懸。",
            "這結果令人吃驚。": "這結果令人著驚。",
            "他並不匆忙離開。": "伊並不匆忙離開。",
            "我並不厭惡暴力。": "我並不厭惡暴力。",
            "這其實很不可惜。": "這其實真不可惜。",
            "這實在太可惜。": "這實在太無彩。",
            "王醫師匆忙離開。": "王醫師趕狂離開。",
            "他匆忙地離開。": "伊趕狂地離開。",
            "他當即將文件交出。": "伊當即將文件交出。",
            "警方迅即將嫌犯逮捕。": "警方迅即將嫌犯逮捕。",
            "文件以及早準備的附件都已提交。": "文件以及早準備的附件都已提交。",
            "文件涉及早處理的條款。": "文件涉及早處理的條款。",
            "他蒙受到了損失。": "伊蒙受到了損失。",
            "他飽受到了批評。": "伊飽受到了批評。",
            "朋友只是開玩笑取笑他。": "朋友只是講耍笑取笑伊。",
            "同學互相取笑朋友。": "同學互相取笑朋友。",
            "甲午飯店今天開幕。": "甲午飯店今仔日開幕。",
            "抗原先前已完成檢驗。": "抗原進前已完成檢驗。",
            "請還原先後順序。": "請猶原先後順序。",
            "孩子第一次吃驚喜蛋。": "囡仔頭擺食驚喜蛋。",
            "這些小吃驚艷全場。": "遮的小食驚艷全場。",
            "他遂即將文件交出。": "伊遂即將文件交出。",
            "他便即將文件交出。": "伊便即將文件交出。",
            "該病原先是未知病毒。": "該病原先是未知病毒。",
            "青藏高原先有冰河覆蓋。": "青藏懸原先有冰河覆蓋。",
            "他領受到大家的關懷。": "伊領受到逐家的關懷。",
            "該批土地足可惜售。": "該批土地足可惜售。",
            "這道小吃驚了全場。": "這道小食驚了全場。",
            "這道小吃驚得評審說不出話。": "這道小食驚得評審說不出話。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        expected_resolution_ids = {
            "他匆忙離開。": "lx_532000000691",
            "午飯。": "lx_532000000693",
            "火車即將到站。": "lx_532000000694",
            "原先。": "lx_532000000695",
            "我厭惡暴力。": "lx_532000000696",
            "請及早準備。": "lx_532000000697",
            "不要取笑他。": "lx_532000000698",
            "他受到表揚。": "lx_532000000699",
            "真可惜。": "lx_532000000701",
            "吃驚。": "lx_532000000702",
        }
        for source, entry_id in expected_resolution_ids.items():
            with self.subTest(resolution_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))

    def test_register_boundary_and_polysemy_governance_batch24(self) -> None:
        expected = {
            "我在陽台吹風。": "我佇陽台吹風。",
            "他吹風笛。": "伊吹風笛。",
            "吹風機。": "吹風機。",
            "吹風胡哨。": "吹風胡哨。",
            "這次經驗很有啟發。": "這擺經驗真有啟發。",
            "啟發式搜尋。": "啟發式搜揣。",
            "啟發性程式。": "啟發性程式。",
            "啟發式教學法。": "啟發式教學法。",
            "他一直埋怨別人。": "伊一直埋怨別人。",
            "深埋怨恨。": "深埋怨恨。",
            "冠狀動脈堵塞。": "冠狀動脈堵塞。",
            "堵塞性肺病。": "堵塞性肺病。",
            "防堵塞車問題。": "防堵塞車問題。",
            "請塗抹藥膏。": "請塗抹藥膏。",
            "塗抹式介面。": "塗抹式介面。",
            "糊塗抹黑別人。": "糊塗抹黑別人。",
            "把坑洞填平。": "把窟仔填平。",
            "回填平整後夯實。": "回填平整後夯實。",
            "請填平均值。": "請填平均值。",
            "填平台資料。": "填平台資料。",
            "母親，請保重。": "阿母，請保重。",
            "「母親！我回來了。」": "「阿母！我轉來矣。」",
            "母親正在休息。": "母親佇咧歇睏。",
            "患者的母親已簽署同意書。": "患者的母親已簽署同意冊。",
            "母親節活動。": "母親節活動。",
            "他是一名洋人。": "伊是一名洋人。",
            "海洋人工智慧研究中心。": "海洋人工智慧研究中心。",
            "東洋人研究。": "東洋人研究。",
            "這種植物有毒。": "這款植物有毒。",
            "植物人仍在治療。": "植物人仍在治療。",
            "移植物排斥反應。": "移植物排斥反應。",
            "他還活著。": "伊猶活咧。",
            "傷者仍然活著！": "傷者猶原活咧！",
            "活著的人仍需要照顧。": "活著的人仍需要照顧。",
            "該系統以靈活著稱。": "該系統以靈活著稱。",
            "數位生活著作權指南。": "數位生活著作權指南。",
            "我大聲呼喚他。": "我大聲叫伊。",
            "老師正在呼喚你。": "老師佇咧叫你。",
            "時代的呼喚。": "時代的呼喚。",
            "這個稱呼喚起童年回憶。": "這个稱呼喚起童年回憶。",
            "我明天回去。": "我明仔載轉去。",
            "回去吧！": "轉去啦！",
            "他已經回去了。": "伊已經回去矣。",
            "請把資料傳回去。": "請把資料傳回去。",
            "努力挽回去年虧損。": "努力挽回舊年虧損。",
            "收回去年度資料。": "收回舊年度資料。",
            "撤回去職申請。": "撤回去職申請。",
            "我要把錢要回去。": "我欲把錢要回去。",
            "父母親都來參加。": "爸母都來參加。",
            "父母親友都來參加。": "父母親友都來參加。",
            "父母親子關係。": "父母親子關係。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        for source, entry_id in (
            ("母親，請保重。", "lx_532000000709"),
            ("他還活著。", "lx_532000000712"),
            ("我大聲呼喚他。", "lx_532000000713"),
            ("我明天回去。", "lx_532000000714"),
            ("父母親都來參加。", "lx_532000000715"),
        ):
            with self.subTest(contextual_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))

    def test_distribution_boundary_and_semantic_governance_batch25(self) -> None:
        expected = {
            "各個都看過了。": "逐个都看過矣。",
            "請檢查各個欄位。": "請檢查各個欄位。",
            "各個案。": "各個案。",
            "各個人資料。": "各個人資料。",
            "學生各自負責。": "學生隨人負責。",
            "兩部機器各自運轉。": "兩部機器各自運轉。",
            "雙方各自為政。": "雙方各自為政。",
            "各自治區。": "各自治區。",
            "各自動化產線。": "各自動化產線。",
            "他很有名望。": "伊真有名望。",
            "著名望遠鏡。": "著名望遠鏡。",
            "他們正在吵架。": "怹佇咧冤家。",
            "不要再吵架！": "莫再冤家！",
            "吵架事件。": "吵架事件。",
            "吵架原因。": "吵架原因。",
            "檢體DNA完全吻合。": "檢體DNA完全吻合。",
            "吻合度很高。": "吻合度真懸。",
            "接吻合照。": "接吻合照。",
            "服務十分周到。": "服務十分周到。",
            "考慮周到。": "考慮周到。",
            "上周到貨。": "上周到貨。",
            "本周到校。": "本周到校。",
            "周遭環境很安靜。": "周圍環境足靜的。",
            "觀察周遭變化。": "觀察周圍變化。",
            "一周遭遇。": "一周遭遇。",
            "每周遭逢事故。": "每周遭逢事故。",
            "病人咀嚼困難。": "病人咀嚼困難。",
            "咀嚼肌功能正常。": "咀嚼肌功能正常。",
            "咀嚼錠。": "咀嚼錠。",
            "咀嚼文字。": "咀嚼文字。",
            "他的品性良好。": "伊的品性良好。",
            "品性教育。": "品性教育。",
            "化學品性質。": "化學品性質。",
            "產品性能。": "產品性能。",
            "學生品行端正。": "學生品行端正。",
            "品行不良紀錄。": "品行不良紀錄。",
            "商品行銷。": "商品行銷。",
            "產品行銷。": "產品行銷。",
            "她哽咽著說不出話。": "伊喉滇著說不出話。",
            "聲音哽咽。": "聲音喉滇。",
            "患者出現哽咽與吞嚥困難。": "患者出現哽咽佮吞嚥困難。",
            "異物造成哽咽感。": "異物造成哽咽感。",
            "他唆使朋友去偷東西。": "伊煽動朋友去偷物件。",
            "他唆使未成年人犯罪。": "伊唆使未成年人犯罪。",
            "檢方指控他唆使證人作偽證。": "檢方指控伊唆使證人作偽證。",
            "教唆使人犯罪。": "教唆使人犯罪。",
            "囉唆使用者。": "囉唆使用者。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        for source, entry_id in (
            ("各個都看過了。", "lx_532000000716"),
            ("學生各自負責。", "lx_532000000717"),
            ("他們正在吵架。", "lx_532000000719"),
            ("周遭環境很安靜。", "lx_532000000722"),
            ("她哽咽著說不出話。", "lx_532000000726"),
            ("他唆使朋友去偷東西。", "lx_532000000727"),
        ):
            with self.subTest(contextual_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))

    def test_round544_semantic_scope_orthography_and_boundary_governance_batch26(self) -> None:
        expected = {
            "他的惡行令人唾棄。": "伊的惡行令人呸瀾。",
            "這種欺騙行為遭眾人唾棄。": "這款欺騙行為遭眾人呸瀾。",
            "本會嚴正唾棄暴力。": "本會嚴正唾棄暴力。",
            "請勿將咳唾棄置於地面。": "請毋通將咳唾棄置於塗跤。",
            "雙方將共同商榷解決方案。": "雙方將共同參詳解決方案。",
            "我想和你商榷這件事。": "我想和你參詳這層代誌。",
            "這項結論值得商榷。": "這項結論值得商榷。",
            "本案由廠商榷定報價。": "本案由廠商榷定報價。",
            "請留下來善後。": "請留落來帕尾。",
            "公司派人善後。": "公司派人帕尾。",
            "事故善後工作仍在進行。": "事故善後工作仍在進行。",
            "改善後續流程。": "改善後續流程。",
            "慈善後援會提供物資。": "慈善後援會提供物資。",
            "讓長者得以善終。": "讓長者得以善終。",
            "安寧善終照護。": "安寧善終照護。",
            "善始善終。": "善始善終。",
            "他不得善終。": "伊歹死。",
            "作惡的人不得善終。": "做歹的人歹死。",
            "改善終端照護制度。": "改善終端照護制度。",
            "這是一部喜劇。": "這是一部喜劇。",
            "喜劇片很好看。": "喜劇片硬掙。",
            "悲喜劇交織。": "悲喜劇交織。",
            "驚喜劇情接連出現。": "驚喜劇情接連出現。",
            "他喜歡到海邊潛水。": "伊佮意到海墘藏水沬。",
            "我喜歡這本書。": "我佮意這本冊。",
            "她喜歡上那位同學。": "伊喜歡上彼位同學。",
            "這個設計很討人喜歡。": "這个設計真討人喜歡。",
            "觀眾驚喜歡呼。": "觀眾驚喜歡呼。",
            "他最喜歡運動。": "伊上佮意運動。",
            "這盒西式喜餅。": "這盒西式盒仔餅。",
            "古早味喜餅。": "古早味大餅。",
            "他們分送喜餅。": "怹分送喜餅。",
            "歡喜餅乾。": "歡喜餅乾。",
            "他喝酒後開車。": "伊啉酒後開車。",
            "禁止喝酒。": "禁止啉酒。",
            "喝酒精很危險。": "喝酒精真危險。",
            "他一聲大喝酒杯便落地。": "伊一聲大喝酒杯便落地。",
            "這碗湯太燙了，先放涼再喝。": "這碗湯太燙矣，先囥冷再啉。",
            "大喝一聲，稍後再喝。": "大喝一聲，稍後再喝。",
            "不要因失敗而喪志。": "莫因失敗而失志。",
            "學生不可喪志。": "學生不可失志。",
            "玩物喪志。": "玩物喪志。",
            "治喪志工。": "治喪志工。",
            "喪志文學研究。": "喪志文學研究。",
            "家屬穿著喪服。": "家屬穿著孝衫。",
            "他換上喪服。": "伊換上孝衫。",
            "喪服制度。": "喪服制度。",
            "治喪服務。": "治喪服務。",
            "居喪服孝。": "居喪服孝。",
            "他吹奏嗩吶。": "伊吹奏鼓吹。",
            "嗩吶演奏家。": "鼓吹演奏家。",
            "嗩吶草是植物。": "嗩吶草是植物。",
            "《嗩吶》是片名。": "《嗩吶》是片名。",
            "他深深嘆氣。": "伊深深吐大氣。",
            "他無奈地嘆氣。": "伊無奈地吐氣。",
            "嘆氣聲很大。": "吐氣聲真大。",
            "感嘆氣候變遷。": "感嘆氣候變遷。",
            "讚嘆氣勢磅礴。": "讚嘆氣勢磅礴。",
        }
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
        for source, target in expected.items():
            with self.subTest(source=source):
                result = self.converter.convert(source, trace=True)
                self.assertIsInstance(result, ConversionResult)
                assert isinstance(result, ConversionResult)
                self.assertEqual(result.output, target)
                self.assertEqual(self.converter.convert(result.output), result.output)
                self.assertTrue(disabled_ids.isdisjoint(match.entry_id for match in result.matches))

        longer_result = self.converter.convert("他不得善終。", trace=True)
        assert isinstance(longer_result, ConversionResult)
        longer_match = next(match for match in longer_result.matches if match.entry_id == "lx_532000000075")
        self.assertEqual((longer_match.src, longer_match.tgt), ("不得善終", "歹死"))
        self.assertEqual((longer_match.start, longer_match.end), (1, 5))

        for source, entry_id in (
            ("他的惡行令人唾棄。", "lx_532000000728"),
            ("雙方將共同商榷解決方案。", "lx_532000000729"),
            ("請留下來善後。", "lx_532000000730"),
            ("他喜歡到海邊潛水。", "lx_532000000733"),
            ("這盒西式喜餅。", "lx_532000000734"),
            ("古早味喜餅。", "lx_532000000735"),
            ("他喝酒後開車。", "lx_532000000736"),
            ("不要因失敗而喪志。", "lx_532000000737"),
            ("家屬穿著喪服。", "lx_532000000738"),
            ("他吹奏嗩吶。", "lx_532000000739"),
            ("他深深嘆氣。", "lx_532000000740"),
            ("他無奈地嘆氣。", "lx_532000000741"),
        ):
            with self.subTest(contextual_source=source):
                result = self.converter.convert(source, trace=True)
                assert isinstance(result, ConversionResult)
                self.assertTrue(any(match.entry_id == entry_id for match in result.matches))


if __name__ == "__main__":
    unittest.main()
