from __future__ import annotations

import unittest

from taigi_converter.normalize import normalize_cjk_spacing, normalize_text


class NormalizeSpacingTests(unittest.TestCase):
    def test_canonicalizes_new_cjk_boundaries_without_changing_glyphs(self) -> None:
        self.assertEqual(normalize_cjk_spacing("臺鐵一點百分之七五 調整"), "臺鐵一點百分之七五調整")
        self.assertEqual(normalize_cjk_spacing("Google Maps 版本"), "Google Maps 版本")
        self.assertEqual(normalize_cjk_spacing("第一行\n 第二行"), "第一行\n 第二行")

    def test_extension_g_han_character_uses_cjk_spacing_policy(self) -> None:
        self.assertEqual(normalize_cjk_spacing("真 癩𰣻"), "真癩𰣻")
        self.assertEqual(normalize_cjk_spacing("癩𰣻 環境"), "癩𰣻環境")


class NormalizeNumberTests(unittest.TestCase):
    def test_small_numbers_keep_unit_based_conversion(self) -> None:
        self.assertEqual(normalize_text("價格是 1203.40 元"), "價格是一千二百零三點四零元")

    def test_valid_traditional_里_and_后_are_not_corrupted(self) -> None:
        source = "太后皇后天后宮球后歌后鄉里千里眼公里里長"
        self.assertEqual(normalize_text(source), source)

    def test_unambiguous_simplified_characters_are_still_normalized(self) -> None:
        self.assertEqual(normalize_text("这车让我们说"), "這車讓我們說")

    def test_number_conversion_does_not_create_second_pass_cjk_space_drift(self) -> None:
        source = "寮國尋金7人困洞一週 5人獲救"
        normalized = normalize_text(source)
        self.assertEqual(normalized, "寮國尋金七人困洞一週五人獲救")
        self.assertEqual(normalize_text(normalized), normalized)

    def test_integer_leading_zeroes_are_preserved_digit_by_digit(self) -> None:
        self.assertEqual(normalize_text("編號 00012"), "編號零零零一二")

    def test_decimal_integer_leading_zeroes_are_preserved(self) -> None:
        self.assertEqual(normalize_text("數值 000.50"), "數值零零零點五零")

    def test_explicit_numeric_identifier_labels_preserve_ascii_digits(self) -> None:
        cases = {
            "案件編號是 12345678": "案件編號是 12345678",
            "訂單編號：000123": "訂單編號：000123",
            "案號 987654": "案號 987654",
            "案件編號 = 24680": "案件編號 = 24680",
            "訂單編號為13579": "訂單編號為13579",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_explicit_identifier_labels_accept_fullwidth_digits(self) -> None:
        cases = {
            "案件編號是 １２３４５６７８": "案件編號是 12345678",
            "訂單編號：０００１２３": "訂單編號：000123",
            "案號　９８７６５４": "案號 987654",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_generic_or_non_identifier_labels_still_convert_numbers(self) -> None:
        cases = {
            "編號 00012": "編號零零零一二",
            "價格是 101 元": "價格是一百零一元",
            "案件數量是 12345678": "案件數量是一千二百三十四萬五千六百七十八",
            "訂單數量：000123": "訂單數量：零零零一二三",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_identifier_label_requires_an_explicit_value_separator(self) -> None:
        cases = {
            "案件編號共 12345678 筆": "案件編號共一千二百三十四萬五千六百七十八筆",
            "案號約 987654": "案號約九十八萬七千六百五十四",
            "訂單編號有 000123 筆": "訂單編號有零零零一二三筆",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_identifier_label_only_protects_a_pure_digit_value(self) -> None:
        cases = {
            "案件編號是 123-456": "案件編號是一百二十三-四百五十六",
            "案號 987654/2": "案號九十八萬七千六百五十四/二",
            "訂單編號：123.45": "訂單編號：一百二十三點四五",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_number_longer_than_python_int_limit_does_not_crash(self) -> None:
        digits = "1" + "0" * 5_000
        expected = "一" + "零" * 5_000
        self.assertEqual(normalize_text(digits), expected)

    def test_very_long_decimal_preserves_both_parts(self) -> None:
        integer_digits = "0" * 4_301
        fractional_digits = "1234567890" * 500
        result = normalize_text(f"-{integer_digits}.{fractional_digits}")
        self.assertEqual(
            result,
            "負" + "零" * len(integer_digits) + "點" + "一二三四五六七八九零" * 500,
        )

    def test_unicode_decimal_digits_are_supported(self) -> None:
        self.assertEqual(normalize_text("١٢.٣٠"), "十二點三零")

    def test_alphanumeric_code_is_not_converted(self) -> None:
        self.assertEqual(normalize_text("A12 B_34"), "A12 B_34")

    def test_technical_protocol_and_runtime_versions_preserve_digits(self) -> None:
        self.assertEqual(
            normalize_text("HTTP 429；Python 3.11；TLS 1.3"),
            "HTTP 429；Python 3.11；TLS 1.3",
        )

    def test_transit_and_hash_identifiers_preserve_digits(self) -> None:
        cases = (
            "明天搭 THSR 0821，不是 THSR 0812。",
            "請在 GitHub issue #431 留言，不要關閉 PR-77。",
            "如果 build_26 失敗，就不要合併 PR#908。",
            "請比對 GH#123 和 ticket#0007。",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_hyphenated_identifiers_and_iso_dates_preserve_digits(self) -> None:
        self.assertEqual(
            normalize_text("訂單 TW-2026-0007，日期 2026-07-28"),
            "訂單 TW-2026-0007，日期 2026-07-28",
        )

    def test_url_paths_and_queries_preserve_digits(self) -> None:
        cases = (
            "https://example.com/v2/items/429",
            "https://example.com/search?id=429&limit=20#page2",
            "ftp://files.example.net/releases/2026/archive-7.zip",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_ipv4_addresses_and_valid_ports_preserve_digits(self) -> None:
        cases = (
            "127.0.0.1:8080",
            "192.168.1.10:443",
            "255.255.255.255:65535",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_security_and_uuid_identifiers_preserve_digits(self) -> None:
        cases = (
            "CVE-2026-12345",
            "550e8400-e29b-41d4-a716-446655440000",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_unix_windows_and_unc_paths_preserve_digits(self) -> None:
        cases = (
            "/home/user/2026/input.json",
            r"C:\Users\user\2026\input.json",
            r"\\server\share\2026\input.json",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_standalone_query_strings_preserve_digits(self) -> None:
        cases = ("?id=429", "?page=2&limit=50", "&retry=3")
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_package_version_specs_preserve_digits(self) -> None:
        cases = (
            "package==3.11.2",
            "package>=3.11,<4",
            "requests[socks]~=2.32",
            "package@3.11.2",
            "@scope/package@2.0.1",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_docker_image_tags_preserve_digits(self) -> None:
        cases = (
            "nginx:1.27.0",
            "acme/api:2026.07",
            "ghcr.io/acme/api:2026.07.28",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_prose_number_syntax_is_not_mistaken_for_technical_spans(self) -> None:
        cases = {
            "價格是 1203.40 元": "價格是一千二百零三點四零元",
            "溫度是 -3.5 度": "溫度是負三點五度",
            "日期是 6/1": "日期是六/一",
            "範圍是 3-5 個": "範圍是三-五個",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_structured_prose_numbers_preserve_exact_surface(self) -> None:
        cases = (
            "利率 1.75%",
            "折扣 -3.5%",
            "時間 12:30",
            "會議 13:40 到 15:10",
            "地點 502 室",
            "研習訂在 2026 年 8 月 3 日上午 10:20。",
            "閏日是 2028 年 2 月 29 日。",
            "世紀閏日是 2000 年 2 月 29 日。",
            "月底是 2026 年 4 月 30 日。",
            "前導零是 2026 年 08 月 03 日。",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), source)

    def test_formal_date_protection_is_narrow_and_calendar_validated(self) -> None:
        cases = {
            "日期是 2026年8月3日": "日期是二千零二十六年八月三日",
            "日期是 2026 年 13 月 40 日": "日期是二千零二十六年十三月四十日",
            "日期是 2026 年 2 月 29 日": "日期是二千零二十六年二月二十九日",
            "日期是 2028 年 2 月 30 日": "日期是二千零二十八年二月三十日",
            "世紀閏日 1900 年 2 月 29 日": "世紀閏日一千九百年二月二十九日",
            "四月邊界 2026 年 4 月 31 日": "四月邊界二千零二十六年四月三十一日",
            "空白不完整 2026 年8月3日": "空白不完整二千零二十六年八月三日",
            "跨行 2026 年\n8 月 3 日": "跨行二千零二十六年\n八月三日",
            "2026台北市長選戰": "二千零二十六台北市長選戰",
            "日期是 6/1": "日期是六/一",
            "整數比例 5%": "整數比例五%",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_invalid_ipv4_port_or_clock_is_not_blanket_protected(self) -> None:
        cases = {
            "時間 29:99": "時間二十九:九十九",
            "端點 127.0.0.1:70000": "端點 127.0.0.1:七萬",
            "比較 3>=4": "比較三>=四",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)

    def test_signed_prose_number_still_converts(self) -> None:
        self.assertEqual(normalize_text("溫度是 -3.5 度"), "溫度是負三點五度")


if __name__ == "__main__":
    unittest.main()
