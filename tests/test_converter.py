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


if __name__ == "__main__":
    unittest.main()
