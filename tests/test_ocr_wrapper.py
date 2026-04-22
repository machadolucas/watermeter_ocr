"""Tests for the VisionOCR shell-wrapper.

We can't invoke the real Swift binary here (it depends on the system's Vision
framework and a compiled binary at ~/watermeter/bin/ocr). Instead we pass a
mock shell script that emits controlled output so we can exercise the contract:
- single-digit output flows through
- multi-char output is truncated to the first digit  (watermeter.py:163)
- non-zero exit produces "" (silent failure)
- non-digit characters are filtered out
"""
from __future__ import annotations

import pytest

from watermeter import VisionOCR


class TestVisionOCRBasic:
    def test_single_digit_passthrough(self, mock_ocr_bin):
        bin_path = mock_ocr_bin(output="5")
        ocr = VisionOCR(bin_path)
        full, top, bottom = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        assert full == "5"
        assert top == "5"
        assert bottom == "5"

    def test_multi_digit_is_truncated_to_first(self, mock_ocr_bin):
        # Vision sometimes returns two digits during rollover; callers expect
        # only the first (watermeter.py:163).
        bin_path = mock_ocr_bin(output="12")
        ocr = VisionOCR(bin_path)
        full, _, _ = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        assert full == "1"

    def test_nonzero_exit_returns_empty(self, mock_ocr_bin):
        bin_path = mock_ocr_bin(output="9", exit_code=1)
        ocr = VisionOCR(bin_path)
        full, top, bottom = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        # Swallowed via bare except → empty strings.
        assert full == ""
        assert top == ""
        assert bottom == ""

    def test_non_digit_output_filtered(self, mock_ocr_bin):
        bin_path = mock_ocr_bin(output="abc7def")
        ocr = VisionOCR(bin_path)
        full, _, _ = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        # Only the digit '7' survives the filter.
        assert full == "7"

    def test_empty_output_returns_empty(self, mock_ocr_bin):
        bin_path = mock_ocr_bin(output="")
        ocr = VisionOCR(bin_path)
        full, _, _ = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        assert full == ""

    def test_missing_binary_returns_empty(self):
        # No binary at all → subprocess error → silent empty.
        ocr = VisionOCR("/nonexistent/path/to/ocr")
        full, top, bottom = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        assert (full, top, bottom) == ("", "", "")


class TestVisionOCRFullTopBottom:
    def test_all_three_halves_invoked(self, mock_ocr_bin):
        # The mock emits the same digit for each call; we just confirm the wrapper
        # returns the 3-tuple shape regardless of which --half flag was passed.
        bin_path = mock_ocr_bin(output="3")
        ocr = VisionOCR(bin_path)
        result = ocr.full_top_bottom("/tmp/ignored.jpg", [0.1, 0.1, 0.1, 0.1])
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result == ("3", "3", "3")
