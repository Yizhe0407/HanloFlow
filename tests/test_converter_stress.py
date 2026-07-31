from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from taigi_converter import TaigiConverter
from taigi_converter.converter import _ProtectedText, _TextSegment
from tests.helpers import build_minimal_runtime, make_source_data, valid_entry


class ConverterStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        protected_entry = valid_entry(
            entry_id="lx_protected001",
            src="身體",
            tgt="身體",
            tier="core",
        )
        protected_entry["protected"] = {
            "category": "technical_term",
            "reason": "stress-test protected span",
        }
        source = make_source_data(
            root / "source",
            entries=[protected_entry, valid_entry()],
        )
        runtime = root / "runtime"
        build_minimal_runtime(source, runtime)
        cls.converter = TaigiConverter(runtime)

    @classmethod
    def tearDownClass(cls) -> None:
        TaigiConverter.clear_runtime_cache()
        cls.temp_dir.cleanup()

    def test_more_than_6400_protected_occurrences_have_no_token_limit(self) -> None:
        text = "身體" * 10_000
        self.assertEqual(self.converter.convert(text), text)

    def test_protected_identity_survives_capture_group_reordering(self) -> None:
        value = _ProtectedText(
            (
                _TextSegment("甲乙", protected=True),
                _TextSegment("中間"),
                _TextSegment("丙丁", protected=True),
            )
        )
        encoded = self.converter._encode_protected(value)
        self.assertIsNotNone(encoded)
        assert encoded is not None
        shadow, layout = encoded
        first, middle, second = shadow[:2], shadow[2:4], shadow[4:]

        decoded = self.converter._decode_protected(second + middle + first, layout)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.render(), "丙丁中間甲乙")

    def test_shadow_codec_preserves_backreference_duplication_and_deletion(self) -> None:
        protected = self.converter._protect_text("身體", respect_runtime_phrase_overlap=False)
        encoded = self.converter._encode_protected(protected)
        self.assertIsNotNone(encoded)
        assert encoded is not None
        shadow, layout = encoded

        duplicated = self.converter._decode_protected(shadow * 2, layout)
        deleted = self.converter._decode_protected("", layout)

        self.assertIsNotNone(duplicated)
        assert duplicated is not None
        self.assertEqual(duplicated.render(), "身體身體")
        self.assertIsNotNone(deleted)
        assert deleted is not None
        self.assertEqual(deleted.render(), "")

    def test_shadow_marker_exhaustion_falls_back_without_corruption(self) -> None:
        segments = []
        expected_parts = []
        for index in range(len(self.converter._shadow_marker_candidates()) + 1):
            value = chr(0x4E00 + index) * 2
            separator = f"-{index}-"
            segments.extend((_TextSegment(value, protected=True), _TextSegment(separator)))
            expected_parts.extend((value, separator))
        protected = _ProtectedText(tuple(segments))
        self.assertIsNone(self.converter._encode_protected(protected))

        output, _ = self.converter._apply_rules_to_protected(protected, collect_trace=False)

        self.assertEqual(output.render(), "".join(expected_parts))

    def test_private_use_characters_do_not_collide_with_protected_spans(self) -> None:
        bmp_private_use = "".join(chr(codepoint) for codepoint in range(0xE000, 0xF900))
        text = f"{bmp_private_use}身體測試詞{bmp_private_use}"
        expected = f"{bmp_private_use}身體試驗詞{bmp_private_use}"
        self.assertEqual(self.converter.convert(text), expected)


if __name__ == "__main__":
    unittest.main()
