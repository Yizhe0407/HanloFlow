from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from taigi_converter.cli import main

ROOT = Path(__file__).resolve().parents[1]

_SUBPROCESS_RUNNER = r"""
import sys
from taigi_converter import cli
from taigi_converter.models import ConversionResult

class StubConverter:
    def __init__(self, review_data_dir=None):
        self.review_data_dir = review_data_dir

    def convert(self, text, *, trace=False, profile=None):
        mapping = {"你在做什麼？": "你咧做啥物？", "公車到站了": "公車到站矣"}
        if profile and profile.get("preserve_spacing"):
            output = text
        else:
            output = mapping.get(text.strip(), text.strip())
        return ConversionResult(output=output) if trace else output

cli.TaigiConverter = StubConverter
raise SystemExit(cli.main(sys.argv[1:]))
"""


class _StubCliConverter:
    def __init__(self, review_data_dir=None) -> None:
        self.review_data_dir = review_data_dir

    def convert(self, text: str, *, trace: bool = False, profile: dict | None = None):
        mapping = {"你在做什麼？": "你咧做啥物？", "公車到站了": "公車到站矣"}
        output = text if profile and profile.get("preserve_spacing") else mapping.get(text.strip(), text.strip())
        if trace:
            from taigi_converter.models import ConversionResult

            return ConversionResult(output=output)
        return output


class _TTYInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class CliSubprocessTests(unittest.TestCase):
    def run_cli(self, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_RUNNER, *args],
            cwd=ROOT,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_preserve_spacing_keeps_outer_and_inner_spaces(self) -> None:
        completed = self.run_cli("--preserve-spacing", "  你   好  ")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "  你   好  \n")
        self.assertEqual(completed.stderr, "")

    def test_plain_positional_output_has_no_banner_or_label(self) -> None:
        completed = self.run_cli("你在做什麼？")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "你咧做啥物？\n")

    def test_non_tty_stdin_is_line_oriented_and_eof_is_success(self) -> None:
        completed = self.run_cli(input_text="你在做什麼？\n公車到站了\n")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "你咧做啥物？\n公車到站矣\n")
        self.assertNotIn("請輸入", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_empty_stdin_eof_is_success_without_output(self) -> None:
        completed = self.run_cli(input_text="")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_batch_trace_emits_one_json_document_per_input_line(self) -> None:
        completed = self.run_cli("--trace", input_text="你在做什麼？\n公車到站了\n")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        records = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([record["output"] for record in records], ["你咧做啥物？", "公車到站矣"])


class CliInteractiveTests(unittest.TestCase):
    def test_interactive_eof_returns_zero_without_exception(self) -> None:
        stdin = _TTYInput("你在做什麼？\n")
        stdout = io.StringIO()

        with mock.patch("taigi_converter.cli.TaigiConverter", _StubCliConverter):
            exit_code = main([], stdin=stdin, stdout=stdout)

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("華語 -> 台語漢字 轉換器", output)
        self.assertIn("你咧做啥物？", output)
        self.assertNotIn("Traceback", output)

    def test_interactive_exit_check_does_not_require_mutating_input(self) -> None:
        stdin = _TTYInput("  EXIT  \n")
        stdout = io.StringIO()
        with mock.patch("taigi_converter.cli.TaigiConverter", _StubCliConverter):
            self.assertEqual(main(["--preserve-spacing"], stdin=stdin, stdout=stdout), 0)
        self.assertNotIn("EXIT", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
