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
    draw_overlays_digital,
    is_valid_digital_view,
    load_config,
    parse_digital_flow,
    parse_digital_total,
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
        digital_flow_roi=[0.1, 0.5, 0.8, 0.2],
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
    """VisionOCR double. Returns prescripted line strings in call order.

    When the cycle runs with both total + flow ROIs configured, each "attempt"
    calls read_line twice (total then flow). With flow disabled it calls once.
    Test fixtures should pass a flat list of expected strings in call order.
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
