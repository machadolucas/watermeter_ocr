"""Tests for the digital (LCD) meter pipeline.

Coverage:
- parse_digital_total / parse_digital_flow
- is_valid_digital_view (cheap pre-validator for wrong-display-view frames)
- validate_digital_reading (big-jump / finite / negative guards)
- run_digital_cycle retry behaviour when the LCD is on a diagnostic view
- Digital HA discovery gating (flow_m3h sensor only in digital mode)
- draw_overlays_digital smoke test

Tests use injected monkey-patches so no ESP32 or Swift Vision binary is touched.
"""
from __future__ import annotations

import math
import re
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from watermeter import (
    MqttClient,
    apply_ocr_preprocess,
    build_digit_sub_rois,
    draw_overlays_digital,
    is_valid_digital_view,
    load_config,
    parse_digital_flow,
    parse_digital_total,
    read_region_per_digit,
    run_digital_cycle,
    validate_digital_reading,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

TOTAL_RE = re.compile(r"^\d{6}\.?\d{3}$")
FLOW_RE = re.compile(r"^\d{2,5}\.?\d{3}$")


def _cfg_stub(**over):
    """Minimal cfg-like object for pure parser/validator tests."""
    base = dict(
        digital_total_regex=TOTAL_RE,
        digital_flow_regex=FLOW_RE,
        digital_min_digits=6,
        digital_max_retries=2,
        digital_retry_delay_sec=0.01,
        big_jump_guard=2.0,
        retry_backoff_sec=0.01,
        monotonic_epsilon=0.0,
        image_path="/tmp/watermeter_test_raw.jpg",
        meter_type="digital",
        digital_total_roi=[0.1, 0.1, 0.8, 0.2],
        digital_total_int_roi=[],
        digital_total_frac_roi=[],
        digital_flow_roi=[0.1, 0.5, 0.8, 0.2],
        digital_ocr_preprocess="none",
        digital_total_int_digit_count=0,
        digital_total_frac_digit_count=0,
        digital_flow_digit_count=0,
        digital_per_digit_inset=0.10,
        digital_per_digit_auto_detect=False,  # default tests use equal subdivision
        digital_ocr_upscale_factor=1,
        digital_save_ocr_crops=False,
        debug_dir="/tmp",
        esp32_base_url="http://unused",
        image_timeout=1.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# parse_digital_total
# ---------------------------------------------------------------------------

class TestParseDigitalTotal:
    def test_dotted_form(self):
        assert parse_digital_total("000100.000", TOTAL_RE) == pytest.approx(100.0)

    def test_dotted_form_with_fractional(self):
        assert parse_digital_total("001234.567", TOTAL_RE) == pytest.approx(1234.567)

    def test_undotted_form_injects_decimal(self):
        # Vision sometimes misses the decimal; the parser places it at -3
        # because the physical display always has exactly 3 fractional digits.
        assert parse_digital_total("000100000", TOTAL_RE) == pytest.approx(100.0)
        assert parse_digital_total("001234567", TOTAL_RE) == pytest.approx(1234.567)

    def test_too_short_returns_none(self):
        assert parse_digital_total("12345", TOTAL_RE) is None

    def test_too_long_returns_none(self):
        assert parse_digital_total("00000000000", TOTAL_RE) is None

    def test_empty_returns_none(self):
        assert parse_digital_total("", TOTAL_RE) is None

    def test_none_returns_none(self):
        assert parse_digital_total(None, TOTAL_RE) is None

    def test_garbage_returns_none(self):
        assert parse_digital_total("abcdefghi", TOTAL_RE) is None


class TestParseDigitalFlow:
    def test_basic(self):
        assert parse_digital_flow("00.000", FLOW_RE) == pytest.approx(0.0)
        assert parse_digital_flow("12.345", FLOW_RE) == pytest.approx(12.345)

    def test_rejects_negative_sign(self):
        assert parse_digital_flow("-0.100", FLOW_RE) is None

    def test_dotless_form_injects_decimal(self):
        # Vision routinely drops the small baseline dot on 7-segment LCDs —
        # the parser must accept dotless forms and inject "." at position -3
        # (symmetric with parse_digital_total's behaviour).
        assert parse_digital_flow("00000", FLOW_RE) == pytest.approx(0.0)
        assert parse_digital_flow("12345", FLOW_RE) == pytest.approx(12.345)

    def test_rejects_too_short(self):
        # Below the minimum number of expected integer digits.
        assert parse_digital_flow("1234", FLOW_RE) is None
        assert parse_digital_flow("0", FLOW_RE) is None

    def test_none_returns_none(self):
        assert parse_digital_flow(None, FLOW_RE) is None


# ---------------------------------------------------------------------------
# is_valid_digital_view
# ---------------------------------------------------------------------------

class TestIsValidDigitalView:
    def test_numeric_view_accepted(self):
        assert is_valid_digital_view("000100.000", "00.000", _cfg_stub()) is True

    def test_none_inputs_rejected(self):
        assert is_valid_digital_view(None, "00.000", _cfg_stub()) is False
        assert is_valid_digital_view("000100.000", None, _cfg_stub()) is False

    def test_diag_view_too_few_digits_rejected(self):
        # A diagnostic screen might have labels like "Q3:1234 R400" or mostly blank.
        assert is_valid_digital_view("QR123", "", _cfg_stub()) is False

    def test_flow_has_no_digits_rejected(self):
        assert is_valid_digital_view("000100.000", "m3/h", _cfg_stub()) is False

    def test_flow_optional_when_roi_unconfigured(self):
        # If the user chose not to calibrate a flow ROI (e.g. flash glare), the
        # cheap validator must accept a None raw_flow — the caller also skipped
        # the read, so we never had a chance to see digits on that line.
        cfg = _cfg_stub(digital_flow_roi=[])
        assert is_valid_digital_view("000100.000", None, cfg) is True

    def test_total_still_required_when_flow_disabled(self):
        cfg = _cfg_stub(digital_flow_roi=[])
        assert is_valid_digital_view(None, None, cfg) is False
        assert is_valid_digital_view("QR123", None, cfg) is False

    def test_rejects_qalcosonic_alnum_diag_view(self):
        # Qalcosonic-style diagnostic screen: mixed letters/digits.
        # "F6A 1d426" has 5 digit chars; flow "CrC" has 0.
        assert is_valid_digital_view("F6A 1d426", "CrC", _cfg_stub()) is False

    def test_rejects_qalcosonic_short_number_diag_view(self):
        # Second diagnostic view: looks numeric but "401.06" has only 5 digit chars;
        # flow "wEr" has 0. Guards against a 5-digit reading being mis-parsed as
        # a short total.
        assert is_valid_digital_view("401.06", "wEr", _cfg_stub()) is False


# ---------------------------------------------------------------------------
# validate_digital_reading
# ---------------------------------------------------------------------------

class TestValidateDigitalReading:
    def test_first_read_accepted(self):
        assert validate_digital_reading(100.0, 0.125, None, _cfg_stub()) is True

    def test_within_big_jump_guard(self):
        assert validate_digital_reading(100.5, 0.1, 100.0, _cfg_stub()) is True

    def test_rejects_huge_forward_jump(self):
        assert validate_digital_reading(200.0, 0.1, 100.0, _cfg_stub()) is False

    def test_rejects_huge_backward_jump(self):
        assert validate_digital_reading(50.0, 0.1, 100.0, _cfg_stub()) is False

    def test_rejects_negative_flow(self):
        assert validate_digital_reading(100.0, -0.5, 100.0, _cfg_stub()) is False

    def test_rejects_non_finite(self):
        assert validate_digital_reading(float("inf"), 0.1, 100.0, _cfg_stub()) is False
        assert validate_digital_reading(100.0, float("nan"), 100.0, _cfg_stub()) is False

    def test_rejects_none(self):
        assert validate_digital_reading(None, 0.1, 100.0, _cfg_stub()) is False
        assert validate_digital_reading(100.0, None, 100.0, _cfg_stub()) is False

    def test_flow_none_accepted_when_roi_unconfigured(self):
        cfg = _cfg_stub(digital_flow_roi=[])
        assert validate_digital_reading(100.0, None, 99.5, cfg) is True
        # Big-jump guard still applies on the total even without flow.
        assert validate_digital_reading(200.0, None, 99.5, cfg) is False


# ---------------------------------------------------------------------------
# run_digital_cycle retry behaviour
# ---------------------------------------------------------------------------

class _FakeOCR:
    """VisionOCR double. Returns prescripted strings in call order.

    Handles both the line-mode and per-digit modes:
      - `read_line` is consumed by line OCR (one call per region per attempt).
      - `_run` is consumed by the single-digit OCR mode used for per-digit
        mode (N calls per region per attempt).
    Tests provide a flat list of expected strings; consumption order mirrors
    the pipeline. In per-digit mode each digit consumes one entry.
    """

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def read_line(self, img, roi):
        if self._i >= len(self._script):
            return None
        val = self._script[self._i]
        self._i += 1
        return val

    def _run(self, img, roi, half="full"):
        # Matches VisionOCR._run — returns a single digit string or "".
        if self._i >= len(self._script):
            return ""
        val = self._script[self._i]
        self._i += 1
        # The real Swift helper returns at most one char; mimic that.
        return val[:1] if val else ""


def _flatten(pairs):
    """Interleave a list of (total, flow) pairs into the call order the fake
    OCR expects. Convenience for tests that want to think in pairs."""
    out = []
    for t, f in pairs:
        out.append(t)
        out.append(f)
    return out


def _dummy_frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestRunDigitalCycle:
    def _patch_capture(self, monkeypatch, return_value=None, returns_seq=None):
        calls = {"n": 0}
        if returns_seq is not None:
            def fake(cfg, session, log):
                r = returns_seq[min(calls["n"], len(returns_seq) - 1)]
                calls["n"] += 1
                return r
        else:
            def fake(cfg, session, log):
                calls["n"] += 1
                return return_value if return_value is not None else _dummy_frame()
        monkeypatch.setattr("watermeter.capture_frame", fake)
        return calls

    def test_first_attempt_succeeds(self, monkeypatch):
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub()
        ocr = _FakeOCR(_flatten([("000100.000", "00.125")]))
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["reason"] == "ok"
        assert result["total"] == pytest.approx(100.0)
        assert result["flow_m3h"] == pytest.approx(0.125)
        assert result["raw_total"] == "000100.000"
        assert result["raw_flow"] == "00.125"

    def test_retries_past_diag_view(self, monkeypatch):
        calls = self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=2)
        ocr = _FakeOCR(_flatten([
            ("", ""),                    # diag 1 — empty
            ("meter info", "serial"),    # diag 2 — text without enough digits
            ("001234.567", "12.345"),    # numeric view finally
        ]))
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["total"] == pytest.approx(1234.567)
        assert calls["n"] == 3

    def test_returns_failure_dict_when_all_attempts_wrong_view(self, monkeypatch):
        calls = self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=2)
        # All three captures see diag screens — returns a failure dict with
        # the last-attempt state so callers can still draw a debug overlay.
        ocr = _FakeOCR(_flatten([("", ""), ("", ""), ("", "")]))
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "wrong_view"
        assert result["frame"] is not None   # last attempt's frame kept
        assert result["total"] is None
        assert result["flow_m3h"] is None
        assert calls["n"] == 3  # max_retries + 1 attempts

    def test_failure_dict_preserves_last_raw_strings(self, monkeypatch):
        # When attempts fail, the last attempt's raw OCR output is preserved
        # so draw_overlays_digital can render it in the debug JPEG.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=2)
        ocr = _FakeOCR(_flatten([
            ("first", "x"),
            ("second", "y"),
            ("last-attempt", "z"),
        ]))
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["raw_total"] == "last-attempt"
        assert result["raw_flow"] == "z"

    def test_no_frame_reason_when_every_capture_fails(self, monkeypatch):
        # Every capture returns None → no frame to overlay, reason="no_frame".
        self._patch_capture(monkeypatch, returns_seq=[None, None, None])
        cfg = _cfg_stub(digital_max_retries=2)
        ocr = _FakeOCR([])  # OCR never called
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "no_frame"
        assert result["frame"] is None

    def test_capture_failure_still_retries(self, monkeypatch):
        # Mix HTTP/decode failures with successful captures; the cycle must
        # not give up on a single capture_frame() == None.
        self._patch_capture(monkeypatch, returns_seq=[None, _dummy_frame(), _dummy_frame()])
        cfg = _cfg_stub(digital_max_retries=2)
        ocr = _FakeOCR(_flatten([("000100.000", "00.125")]))
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["total"] == pytest.approx(100.0)

    def test_rejects_when_parsed_value_violates_big_jump(self, monkeypatch):
        # Valid view with valid digits, but total jumped by more than big_jump_guard.
        # The cycle returns success=False with reason="validate_failed".
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=0, big_jump_guard=1.0)
        ocr = _FakeOCR(_flatten([("000200.000", "00.100")]))
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=100.0, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "validate_failed"
        # Parsed values are preserved so the overlay can annotate them.
        assert result["total"] == pytest.approx(200.0)
        assert result["flow_m3h"] == pytest.approx(0.1)

    def test_parse_failure_reason_on_garbage_digits(self, monkeypatch):
        # Cheap view-check accepts (≥6 digits) but the strict total regex fails.
        # Returns success=False reason="parse_failed".
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=0)
        ocr = _FakeOCR(_flatten([("1234567", "00000")]))  # total: 7 digits — regex rejects
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "parse_failed"
        assert result["raw_total"] == "1234567"

    def test_split_mode_concatenates_int_and_frac(self, monkeypatch):
        # When total_int + total_frac ROIs are both set, the cycle OCRs them
        # separately and joins with "." before parsing. Needed for LCDs whose
        # fractional digits are visibly smaller than the integer digits and
        # get dropped when captured in a single wide ROI.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(
            digital_total_int_roi=[0.10, 0.25, 0.45, 0.18],
            digital_total_frac_roi=[0.58, 0.30, 0.22, 0.14],
        )
        # Per attempt, OCR is called three times in order: int, frac, flow.
        ocr = _FakeOCR(["000000", "148", "00.000"])
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["total"] == pytest.approx(0.148)
        assert result["raw_total"] == "000000.148"
        assert result["raw_total_int"] == "000000"
        assert result["raw_total_frac"] == "148"

    def test_split_mode_parse_fails_when_frac_empty(self, monkeypatch):
        # Int read succeeds but frac comes back empty → concatenated
        # "000000." fails the total regex → parse_failed.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(
            digital_max_retries=0,
            digital_total_int_roi=[0.10, 0.25, 0.45, 0.18],
            digital_total_frac_roi=[0.58, 0.30, 0.22, 0.14],
        )
        ocr = _FakeOCR(["000000", "", "00.000"])
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "parse_failed"
        # The split-side reads are preserved on the result so the overlay
        # can annotate each sub-ROI independently.
        assert result["raw_total_int"] == "000000"
        assert result["raw_total_frac"] == ""

    def test_per_digit_int_concatenates(self, monkeypatch):
        # Per-digit OCR for the integer sub-ROI: each digit is recognized
        # independently via the Swift digit-mode helper (6 subprocess calls)
        # and concatenated. Frac stays in line mode. Useful when line OCR
        # drops narrow glyphs on the integer side.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(
            digital_total_int_roi=[0.10, 0.25, 0.45, 0.18],
            digital_total_frac_roi=[0.58, 0.30, 0.22, 0.14],
            digital_total_int_digit_count=6,  # per-digit
            digital_total_frac_digit_count=0,  # line mode
        )
        # 6 digit calls for int, 1 line call for frac, 1 line call for flow.
        ocr = _FakeOCR(["0", "0", "0", "0", "0", "0", "148", "00.000"])
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["total"] == pytest.approx(0.148)
        assert result["raw_total_int"] == "000000"
        assert result["raw_total_frac"] == "148"

    def test_per_digit_full_mode_all_regions(self, monkeypatch):
        # Per-digit everywhere: int (6) + frac (3) + flow (5).
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(
            digital_total_int_roi=[0.10, 0.25, 0.45, 0.18],
            digital_total_frac_roi=[0.58, 0.30, 0.22, 0.14],
            digital_total_int_digit_count=6,
            digital_total_frac_digit_count=3,
            digital_flow_digit_count=5,  # "00125" → injected dot → 0.125
        )
        ocr = _FakeOCR([
            "0", "0", "0", "0", "0", "0",
            "1", "1", "8",
            "0", "0", "1", "2", "5",
        ])
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["total"] == pytest.approx(0.118)
        assert result["flow_m3h"] == pytest.approx(0.125)

    def test_per_digit_missed_digit_fails_parse(self, monkeypatch):
        # One digit reads empty → concatenated int shortens from 6 to 5
        # chars → combined "00000.118" fails the 6-digit regex → parse_failed.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(
            digital_max_retries=0,
            digital_total_int_roi=[0.10, 0.25, 0.45, 0.18],
            digital_total_frac_roi=[0.58, 0.30, 0.22, 0.14],
            digital_total_int_digit_count=6,
            digital_total_frac_digit_count=3,
        )
        ocr = _FakeOCR([
            "0", "", "0", "0", "0", "0",  # 2nd digit missed
            "1", "1", "8",
            "00.000",
        ])
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "parse_failed"
        assert result["raw_total_int"] == "00000"

    def test_split_mode_wrong_view_when_both_empty(self, monkeypatch):
        # Both split reads come back empty — concat ".", digit count < 6.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(
            digital_max_retries=0,
            digital_total_int_roi=[0.10, 0.25, 0.45, 0.18],
            digital_total_frac_roi=[0.58, 0.30, 0.22, 0.14],
        )
        ocr = _FakeOCR(["", "", "00.000"])
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is False
        assert result["reason"] == "wrong_view"

    def test_on_attempt_called_once_per_attempt(self, monkeypatch):
        # Callback fires every attempt (wrong_view and success), giving the
        # main loop a chance to refresh the debug overlay between retries.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=2)
        ocr = _FakeOCR(_flatten([
            ("", ""),                    # attempt 1 — wrong_view
            ("meter info", "serial"),    # attempt 2 — wrong_view
            ("001234.567", "12.345"),    # attempt 3 — ok
        ]))
        seen = []
        run_digital_cycle(cfg, ocr, aligner=None, session=None,
                          prev_total=None, log=_StubLog(), sleep=_noop,
                          on_attempt=lambda s: seen.append(s["reason"]))
        assert seen == ["wrong_view", "wrong_view", "ok"]

    def test_on_attempt_gets_frame_for_overlay(self, monkeypatch):
        # The callback state must include the frame so the overlay can be
        # drawn — that's the whole point of per-attempt notification.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=0)
        ocr = _FakeOCR(_flatten([("", "")]))  # one attempt, wrong_view
        seen = []
        run_digital_cycle(cfg, ocr, aligner=None, session=None,
                          prev_total=None, log=_StubLog(), sleep=_noop,
                          on_attempt=lambda s: seen.append(s))
        assert len(seen) == 1
        assert seen[0]["frame"] is not None
        assert seen[0]["reason"] == "wrong_view"

    def test_on_attempt_callback_exception_does_not_abort_cycle(self, monkeypatch):
        # Callback raising shouldn't kill the cycle — main-loop logic must
        # still see a valid result even if overlay drawing explodes.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_max_retries=0)
        ocr = _FakeOCR(_flatten([("000100.000", "00.125")]))
        def bad(state):
            raise RuntimeError("overlay render failed")
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop,
                                   on_attempt=bad)
        assert result["success"] is True
        assert result["total"] == pytest.approx(100.0)

    def test_flow_disabled_only_reads_total(self, monkeypatch):
        # When digital_flow_roi is empty, run_digital_cycle must NOT call
        # ocr.read_line for flow — the fake only yields one string per attempt.
        self._patch_capture(monkeypatch)
        cfg = _cfg_stub(digital_flow_roi=[])
        ocr = _FakeOCR(["000100.000"])  # exactly one read expected, not two
        result = run_digital_cycle(cfg, ocr, aligner=None, session=None,
                                   prev_total=None, log=_StubLog(), sleep=_noop)
        assert result["success"] is True
        assert result["total"] == pytest.approx(100.0)
        assert result["flow_m3h"] is None
        assert result["raw_flow"] is None
        # Exhausted exactly one entry — no stray flow reads consumed.
        assert ocr._i == 1


def _noop(_):
    return None


class _StubLog:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def debug(self, *a, **kw): pass


# ---------------------------------------------------------------------------
# HA discovery gating
# ---------------------------------------------------------------------------

class _RecordingClient:
    """paho.mqtt.Client double that records publish() calls in-memory."""
    def __init__(self):
        self.published = []
        self.on_connect = None
        self.on_disconnect = None
    def username_pw_set(self, *a, **kw): pass
    def publish(self, topic, payload, retain=False):
        self.published.append((topic, payload, retain))


@pytest.fixture
def _recording_mqtt(monkeypatch):
    clients = []
    def make_client(*a, **kw):
        c = _RecordingClient()
        clients.append(c)
        return c
    monkeypatch.setattr("watermeter.mqtt.Client", make_client)
    return clients


class TestDiscoveryFlowM3h:
    def _cfg(self, tmp_path, meter_type, flow_roi=True):
        p = tmp_path / "config.yaml"
        parts = [
            "esp32:\n  base_url: \"http://localhost\"\n",
            "mqtt:\n  topic: \"home/watermeter\"\n",
        ]
        if meter_type == "digital":
            parts.append("meter:\n  type: digital\n")
            parts.append("rois:\n  digital:\n    total: [0.1, 0.1, 0.8, 0.2]\n")
            if flow_roi:
                parts.append("    flow:  [0.1, 0.5, 0.8, 0.2]\n")
        p.write_text("".join(parts))
        return load_config(str(p))

    def test_digital_with_flow_roi_publishes_flow_m3h(self, tmp_path, _recording_mqtt):
        cfg = self._cfg(tmp_path, "digital", flow_roi=True)
        client = MqttClient(cfg, _StubLog())
        client.connected = True  # bypass the gate so publish() actually records
        client.discovery()
        topics = [t for (t, _, _) in _recording_mqtt[0].published]
        assert any("water_flow_m3h/config" in t for t in topics)

    def test_digital_without_flow_roi_skips_flow_m3h(self, tmp_path, _recording_mqtt):
        # Flash glare etc. — the user captures only the total line. The HA
        # sensor should be suppressed so the entity list doesn't go stale.
        cfg = self._cfg(tmp_path, "digital", flow_roi=False)
        client = MqttClient(cfg, _StubLog())
        client.connected = True
        client.discovery()
        topics = [t for (t, _, _) in _recording_mqtt[0].published]
        assert not any("water_flow_m3h" in t for t in topics)

    def test_mechanical_does_not_publish_flow_m3h(self, tmp_path, _recording_mqtt):
        cfg = self._cfg(tmp_path, "mechanical")
        client = MqttClient(cfg, _StubLog())
        client.connected = True
        client.discovery()
        topics = [t for (t, _, _) in _recording_mqtt[0].published]
        assert not any("water_flow_m3h" in t for t in topics)


# ---------------------------------------------------------------------------
# Overlay smoke test
# ---------------------------------------------------------------------------

class TestDrawOverlaysDigital:
    def _cfg(self):
        # draw_overlays_digital reads a handful of overlay attributes off cfg.
        return SimpleNamespace(
            overlay_font_scale=1.0,
            overlay_font_thickness=2,
            overlay_outline_thickness=4,
            overlay_line_thickness=2,
        )

    def test_draws_both_rects_and_labels(self):
        img = np.full((240, 640, 3), 30, dtype=np.uint8)
        ov = draw_overlays_digital(
            img,
            total_roi_abs=(40, 50, 400, 40),
            flow_roi_abs=(40, 120, 400, 40),
            raw_total_str="000100.000",
            raw_flow_str="00.125",
            parsed_total=100.0, parsed_flow=0.125,
            aligned_ok=True, cfg=self._cfg(),
        )
        assert ov.shape == img.shape
        # Green rectangles when aligned_ok=True — at least one green pixel must
        # exist on the rect border. Sanity-only; exact contents are noisy.
        assert (ov[:, :, 1] > 150).any()

    def test_handles_missing_raw_strings(self):
        img = np.full((240, 640, 3), 30, dtype=np.uint8)
        # Diagnostic view captured: no raw strings, no parsed values.
        ov = draw_overlays_digital(
            img,
            total_roi_abs=(40, 50, 400, 40),
            flow_roi_abs=(40, 120, 400, 40),
            raw_total_str=None, raw_flow_str=None,
            parsed_total=None, parsed_flow=None,
            aligned_ok=False, cfg=self._cfg(),
        )
        assert ov.shape == img.shape

    def test_failure_banner_rendered_when_reason_set(self):
        # On failure, a coloured banner is drawn across the top of the image
        # so the debug JPEG visibly distinguishes rejected cycles. We don't
        # assert exact pixels — just that the top strip changed from the
        # baseline grey we painted.
        img = np.full((240, 640, 3), 30, dtype=np.uint8)
        ov = draw_overlays_digital(
            img,
            total_roi_abs=(40, 50, 400, 40),
            flow_roi_abs=(40, 120, 400, 40),
            raw_total_str="F6A 1d426", raw_flow_str="CrC",
            parsed_total=None, parsed_flow=None,
            aligned_ok=True, cfg=self._cfg(),
            reason="wrong_view",
        )
        top_strip = ov[0:8, :, :]
        # Banner is red-ish (BGR (30, 30, 200)); at minimum, the red channel
        # should dominate the top strip's mean, unlike the baseline grey.
        assert top_strip[:, :, 2].mean() > top_strip[:, :, 0].mean() + 50

    def test_ok_reason_does_not_render_banner(self):
        # When reason is "ok" (or None), no banner is drawn.
        img = np.full((240, 640, 3), 30, dtype=np.uint8)
        ov = draw_overlays_digital(
            img,
            total_roi_abs=(40, 50, 400, 40),
            flow_roi_abs=(40, 120, 400, 40),
            raw_total_str="000100.000", raw_flow_str="00.125",
            parsed_total=100.0, parsed_flow=0.125,
            aligned_ok=True, cfg=self._cfg(),
            reason="ok",
        )
        top_strip = ov[0:8, :, :]
        # Baseline grey stays roughly equal across channels.
        assert abs(top_strip[:, :, 2].mean() - top_strip[:, :, 0].mean()) < 20


# ---------------------------------------------------------------------------
# read_region_per_digit (per-digit OCR helper)
# ---------------------------------------------------------------------------

class TestReadRegionPerDigit:
    class _CollectingOCR:
        """Minimal VisionOCR stand-in — records each sub-ROI it's asked to
        recognize and returns a scripted digit per call so tests can assert
        both the concatenation and the geometric subdivision."""
        def __init__(self, script):
            self.script = list(script)
            self.calls = []

        def _run(self, img, roi, half="full"):
            self.calls.append(list(roi))
            return self.script.pop(0) if self.script else ""

    def test_empty_roi_returns_empty(self):
        ocr = self._CollectingOCR(["9"])
        assert read_region_per_digit(ocr, "p", [], 6) == ""
        assert ocr.calls == []  # never invoked

    def test_zero_count_returns_empty(self):
        ocr = self._CollectingOCR(["9"])
        assert read_region_per_digit(ocr, "p", [0.1, 0.2, 0.6, 0.1], 0) == ""
        assert ocr.calls == []

    def test_concatenates_all_digits(self):
        ocr = self._CollectingOCR(["1", "2", "3", "4", "5", "6"])
        result = read_region_per_digit(
            ocr, "p", [0.0, 0.0, 0.6, 0.1], 6, inset=0.0,
        )
        assert result == "123456"
        assert len(ocr.calls) == 6

    def test_missed_digit_yields_shorter_string(self):
        # Individual digit OCR failure just drops that character from the
        # join — downstream regex catches the length mismatch.
        ocr = self._CollectingOCR(["1", "", "3"])
        result = read_region_per_digit(ocr, "p", [0.0, 0.0, 0.3, 0.1], 3, inset=0.0)
        assert result == "13"

    def test_subdivisions_span_base_roi_evenly(self):
        # Sub-ROIs should collectively cover the base ROI with inset-respecting
        # gaps. Specifically: x-coords are monotonic; first starts after the
        # inset; last ends before the ROI's right edge minus the inset.
        # auto_detect=False to force equal-subdivision path in this test.
        ocr = self._CollectingOCR(["0"] * 6)
        read_region_per_digit(ocr, "p", [0.10, 0.20, 0.60, 0.10], 6, inset=0.10,
                              auto_detect=False)
        xs = [c[0] for c in ocr.calls]
        assert xs == sorted(xs)  # monotonic left-to-right
        # First sub-ROI's x is at least inset into the first cell.
        assert xs[0] >= 0.10
        # Last sub-ROI's right edge (x + w) is within the base ROI.
        last_x, _, last_w, _ = ocr.calls[-1]
        assert last_x + last_w <= 0.10 + 0.60 + 1e-9


# ---------------------------------------------------------------------------
# build_digit_sub_rois — auto-detection vs equal subdivision
# ---------------------------------------------------------------------------

class TestBuildDigitSubRois:
    def _synth_digits_image(self, tmp_path, count, gap_px=6, digit_w=14, digit_h=40):
        """Create a synthetic LCD-ish crop: `count` dark rectangular 'digits'
        on a light background, separated by clear gaps. Returns the saved
        path and the ROI in normalized coords."""
        import cv2
        pad_x = 10
        pad_y = 10
        total_w = pad_x * 2 + count * digit_w + (count - 1) * gap_px
        total_h = pad_y * 2 + digit_h
        img = np.full((total_h, total_w, 3), 220, dtype=np.uint8)  # light bg
        for i in range(count):
            x0 = pad_x + i * (digit_w + gap_px)
            img[pad_y:pad_y + digit_h, x0:x0 + digit_w] = 40  # dark digit
        path = str(tmp_path / "synth.jpg")
        cv2.imwrite(path, img)
        # Base ROI covers the whole synthetic image.
        return path, [0.0, 0.0, 1.0, 1.0]

    def test_auto_detect_finds_correct_count(self, tmp_path):
        # Synthetic 6-digit image with clear gaps — projection should find
        # exactly 6 runs and return 6 sub-ROIs in left-to-right order.
        path, base = self._synth_digits_image(tmp_path, count=6)
        subs = build_digit_sub_rois(path, base, expected_count=6, inset=0.10,
                                    auto_detect=True)
        assert len(subs) == 6
        xs = [s[0] for s in subs]
        assert xs == sorted(xs)
        # Sub-ROIs stay within the base ROI.
        for (x, y, w, h) in subs:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
            assert x + w <= 1.0 + 1e-9
            assert y + h <= 1.0 + 1e-9

    def test_auto_detect_falls_back_when_wrong_count(self, tmp_path):
        # Synthetic image has 4 digits, but we ask for 6 → projection returns
        # 4 runs → detection declines to return → fallback to equal
        # subdivision of the base ROI. Should still return 6 sub-ROIs.
        path, base = self._synth_digits_image(tmp_path, count=4)
        subs = build_digit_sub_rois(path, base, expected_count=6, inset=0.10,
                                    auto_detect=True)
        assert len(subs) == 6
        # Equal subdivision: each cell is (base_w/6) wide minus inset padding
        cell_w = (1.0 / 6) * (1 - 2 * 0.10)
        for sub in subs:
            assert abs(sub[2] - cell_w) < 1e-6

    def test_auto_detect_disabled_uses_equal_subdivision(self, tmp_path):
        path, base = self._synth_digits_image(tmp_path, count=6)
        auto = build_digit_sub_rois(path, base, 6, auto_detect=True)
        equal = build_digit_sub_rois(path, base, 6, auto_detect=False)
        # When disabled, sub-ROIs are uniform — all widths identical.
        widths = [s[2] for s in equal]
        assert max(widths) - min(widths) < 1e-6
        # Auto-detected widths depend on the synthesized digit bounds and
        # will differ from the uniform equal-subdivision case.
        auto_widths = [s[2] for s in auto]
        # At least one width should differ noticeably between the two modes.
        assert any(abs(a - e) > 1e-4 for a, e in zip(auto_widths, widths))

    def test_auto_detect_missing_file_falls_back(self, tmp_path):
        # Non-existent image → imread returns None → fallback to equal
        # subdivision.
        subs = build_digit_sub_rois(str(tmp_path / "missing.jpg"),
                                    [0.1, 0.2, 0.6, 0.1], 6, auto_detect=True)
        assert len(subs) == 6

    def test_auto_detect_handles_hollow_seven_segment_zero(self, tmp_path):
        # 7-segment "0" is drawn as a hollow rectangle — left + right vertical
        # strokes with no ink in the middle. Without morphological closing,
        # vertical projection would see each "0" as TWO runs (one per stroke),
        # tripping the count check. With closing, each "0" projects as ONE
        # continuous run, matching the expected digit count.
        import cv2
        total_w, total_h = 360, 60
        img = np.full((total_h, total_w, 3), 220, dtype=np.uint8)
        # Draw six hollow "0"s: 50 px wide, gap of 10 px, with 4 px strokes.
        pad_x, pad_y = 10, 10
        digit_w = 50
        gap = 10
        stroke = 4
        for i in range(6):
            x = pad_x + i * (digit_w + gap)
            # left stroke
            img[pad_y:pad_y + 40, x:x + stroke] = 40
            # right stroke
            img[pad_y:pad_y + 40, x + digit_w - stroke:x + digit_w] = 40
            # top + bottom strokes
            img[pad_y:pad_y + stroke, x:x + digit_w] = 40
            img[pad_y + 40 - stroke:pad_y + 40, x:x + digit_w] = 40
        path = str(tmp_path / "hollow.jpg")
        cv2.imwrite(path, img)
        subs = build_digit_sub_rois(path, [0.0, 0.0, 1.0, 1.0], 6,
                                    auto_detect=True)
        assert len(subs) == 6
        # Sub-ROIs should be roughly one digit-cell wide each (not half, which
        # would indicate we'd detected 12 strokes instead of 6 digits).
        widths = [s[2] * total_w for s in subs]
        # Each detected digit width should be close to digit_w (+ 2*inset
        # padding of the detected run itself).
        for w in widths:
            assert digit_w * 0.5 < w < digit_w * 1.4


# ---------------------------------------------------------------------------
# apply_ocr_preprocess (opt-in contrast enhancement before Vision)
# ---------------------------------------------------------------------------

class TestApplyOcrPreprocess:
    def test_none_is_noop(self, tmp_path):
        src = str(tmp_path / "in.jpg")
        dst = str(tmp_path / "out.jpg")
        import cv2
        cv2.imwrite(src, np.full((20, 20, 3), 128, dtype=np.uint8))
        assert apply_ocr_preprocess(src, dst, "none") is False

    def test_unknown_mode_is_noop(self, tmp_path):
        assert apply_ocr_preprocess("x", "y", "sharpen") is False

    def test_missing_source_returns_false(self, tmp_path):
        dst = str(tmp_path / "out.jpg")
        # File at this path doesn't exist; cv2.imread returns None.
        assert apply_ocr_preprocess(str(tmp_path / "missing.jpg"), dst, "clahe") is False

    def test_clahe_produces_output_file(self, tmp_path):
        import cv2
        src = str(tmp_path / "in.jpg")
        dst = str(tmp_path / "out.jpg")
        # Low-contrast synthetic image: gray background with slightly-lighter
        # vertical bars mimicking faint 7-segment digits.
        img = np.full((80, 160, 3), 110, dtype=np.uint8)
        for x in (20, 50, 80, 110):
            img[20:60, x:x+8] = 130
        cv2.imwrite(src, img)
        ok = apply_ocr_preprocess(src, dst, "clahe")
        assert ok is True
        out = cv2.imread(dst)
        assert out is not None
        assert out.shape == img.shape
        # CLAHE stretches local contrast, so the output's standard deviation
        # should be noticeably higher than the input's. Margin is small
        # because JPEG re-encoding smooths some of the gain.
        assert out.std() > img.std() + 2
