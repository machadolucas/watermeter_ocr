"""Acceptance gate, staleness/availability, LCD view phase, classifier fixtures.

Covers the 2026-11-04 changes:
- ReadingGate replaces the fixed big_jump_guard for the digital path;
- MQTT availability + LWT so HA shows `unavailable` for a stale reading;
- ViewPhase schedules captures inside the LCD's numeric view;
- the 7-segment probe geometry that read "3" as "9".
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import watermeter as W
from watermeter import (
    Config, MqttClient, ReadingGate, StaleTracker, ViewPhase,
    alignment_quality_ok, handle_digital_result, to_litres,
)

FIXTURES = Path(__file__).parent / "fixtures" / "lcd"


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


# ---------------------------------------------------------------------------
# ReadingGate
# ---------------------------------------------------------------------------

class TestReadingGate:
    def test_first_reading_is_baseline(self):
        g = ReadingGate()
        d = g.evaluate(57300, 1000.0)
        assert d.accepted and d.kind == "baseline" and g.last_l == 57300

    def test_normal_flow_accepted(self):
        g = ReadingGate(last_l=57300, last_ts=0.0)
        # 23 L/min (the highest real rate seen) for one 20 s cycle = 7.7 L
        d = g.evaluate(57308, 20.0)
        assert d.accepted and d.kind == "ok"

    def test_misread_plus_60_in_20s_rejected(self):
        # 2026-11-03 21:59:28: 57.234 -> 57.294 one cycle later (3 read as 9)
        g = ReadingGate(last_l=57234, last_ts=_ts("2026-11-03 21:59:08"))
        d = g.evaluate(57294, _ts("2026-11-03 21:59:28"))
        assert not d.accepted and d.kind == "too_fast"
        assert g.last_l == 57234

    def test_small_dip_is_hold(self):
        g = ReadingGate(last_l=57300, last_ts=0.0)
        d = g.evaluate(57299, 20.0)
        assert d.accepted and d.kind == "hold" and d.litres == 57300
        assert g.last_ts == 20.0

    def test_large_step_needs_confirmation(self):
        g = ReadingGate(last_l=57300, last_ts=0.0)
        # +60 L after 5 min: under the 40 L/min hard bound (201 L) but over
        # the immediate bound (25 L/min x 75 s cap + 1 = 32 L).
        d1 = g.evaluate(57360, 300.0)
        assert not d1.accepted and d1.kind == "candidate"
        d2 = g.evaluate(57365, 320.0)
        assert d2.accepted and d2.kind == "confirmed" and g.last_l == 57365

    def test_confirmation_must_be_independent_capture(self):
        g = ReadingGate(last_l=57300, last_ts=0.0, confirm_min_gap_sec=15)
        g.evaluate(57360, 300.0)
        d = g.evaluate(57360, 305.0)   # 5 s later: same rotation, not independent
        assert not d.accepted and d.kind == "candidate"

    def test_confirmation_must_not_go_backwards(self):
        g = ReadingGate(last_l=57300, last_ts=0.0)
        g.evaluate(57900, 1800.0)      # +600 L after 30 min: candidate
        d = g.evaluate(57360, 1820.0)  # next reading disagrees
        assert not d.accepted

    def test_long_outage_does_not_widen_immediate_accept(self):
        # After 11 h the old guard allowed +0.53 m3 at once; now +60 L must
        # be confirmed by a second capture.
        g = ReadingGate(last_l=57234, last_ts=0.0)
        d = g.evaluate(57294, 11 * 3600.0)
        assert not d.accepted and d.kind == "candidate"

    def test_backward_run_rebaselines(self):
        # Trapped above the meter by an accepted misread: correct readings
        # are all lower and agree with each other.
        g = ReadingGate(last_l=57294, last_ts=0.0, rebaseline_count=3,
                        rebaseline_min_span_sec=60)
        assert g.evaluate(57236, 20.0).kind == "backward"
        assert g.evaluate(57236, 50.0).kind == "backward"
        d = g.evaluate(57237, 80.0)   # span 60 s, 3 consistent readings
        assert d.accepted and d.kind == "rebaseline" and g.last_l == 57237

    def test_backward_run_needs_min_span(self):
        g = ReadingGate(last_l=57294, last_ts=0.0, rebaseline_min_span_sec=60)
        for t in (10.0, 20.0, 30.0):
            d = g.evaluate(57236, t)
        assert not d.accepted

    def test_inconsistent_backward_readings_do_not_rebaseline(self):
        g = ReadingGate(last_l=57294, last_ts=0.0)
        g.evaluate(57236, 20.0)
        g.evaluate(56000, 50.0)        # wild misread in between
        d = g.evaluate(57237, 80.0)
        assert not d.accepted

    def test_accepted_reading_resets_backward_run(self):
        g = ReadingGate(last_l=57294, last_ts=0.0)
        g.evaluate(57236, 20.0)
        g.evaluate(57236, 50.0)
        assert g.evaluate(57294, 60.0).accepted     # hold/ok resets the run
        assert not g.evaluate(57236, 90.0).accepted

    def test_preview_does_not_mutate(self):
        g = ReadingGate(last_l=57300, last_ts=0.0)
        g.preview(57360, 300.0)
        assert g.candidate is None and g.last_l == 57300

    def test_from_config_converts_m3(self):
        cfg = Config(esp32_base_url="x")
        g = ReadingGate.from_config(cfg, 57.295, 123.0)
        assert g.last_l == 57295 and g.last_ts == 123.0
        assert g.max_flow_lpm == 40.0 and g.confirm_above_lpm == 25.0

    def test_to_litres_avoids_float_boundary(self):
        # 57.295 - 57.294 in float is slightly below 0.001 (Codex F25)
        assert to_litres(57.295) - to_litres(57.294) == 1

    def test_replay_2026_11_03_night(self):
        """The night that lost the dishwasher + washer water, replayed.

        Truth: quiet at 57.234 m3 after 21:59, two misreads (21:59:28 -> 57.294,
        00:25:12 -> 57.295), then real use from 08:00 to 57.297 by 11:43. The
        old guard accepted the first misread and rejected every true reading
        until 11:43. The gate must reject both misreads and follow the truth.
        """
        g = ReadingGate(last_l=57231, last_ts=_ts("2026-11-03 21:58:48"))
        seq = [("2026-11-03 21:59:08", 57234, True),
               ("2026-11-03 21:59:28", 57294, False)]   # misread
        t = _ts("2026-11-03 22:00:00")
        while t < _ts("2026-11-04 00:25:00"):          # quiet night, 60 s cycles
            seq.append((t, 57234, True)); t += 60
        seq.append(("2026-11-04 00:25:12", 57295, False))  # misread
        t = _ts("2026-11-04 00:26:00")
        while t < _ts("2026-11-04 08:00:00"):
            seq.append((t, 57234, True)); t += 60
        litres = 57234
        while litres < 57297:                          # real use, ~6 L/min
            t += 20; litres += 2
            seq.append((t, min(litres, 57297), True))
        published = []
        for when, val, should_accept in seq:
            ts = _ts(when) if isinstance(when, str) else when
            d = g.evaluate(val, ts)
            assert d.accepted == should_accept, (when, val, d)
            if d.accepted:
                published.append(d.litres)
        assert max(published) == 57297
        assert published == sorted(published)          # never published a misread


# ---------------------------------------------------------------------------
# StaleTracker
# ---------------------------------------------------------------------------

class TestStaleTracker:
    def test_no_reading_yet_is_stale(self):
        assert StaleTracker(600, None).is_stale(0.0)

    def test_boundaries(self):
        s = StaleTracker(600, 1000.0)
        assert not s.is_stale(1599.0)
        assert s.is_stale(1600.0)
        s.mark_fresh(1600.0)
        assert not s.is_stale(1600.0)

    def test_zero_disables(self):
        assert not StaleTracker(0, None).is_stale(1e9)


# ---------------------------------------------------------------------------
# MQTT availability + discovery
# ---------------------------------------------------------------------------

@pytest.fixture
def mq():
    cfg = Config(esp32_base_url="http://localhost", mqtt_main_topic="home/watermeter",
                 meter_type="digital")
    with patch("watermeter.mqtt.Client") as C:
        C.return_value = MagicMock()
        c = MqttClient(cfg, logging.getLogger("t"))
        yield c


def _published(c):
    return [(a.args[0], a.args[1], a.kwargs) for a in c.client.publish.call_args_list]


class TestAvailability:
    def test_lwt_is_retained_offline(self, mq):
        mq.client.will_set.assert_called_once_with(
            "home/watermeter/status", "offline", qos=1, retain=True)

    def test_custom_availability_topic(self):
        cfg = Config(esp32_base_url="x", mqtt_availability_topic="a/b")
        with patch("watermeter.mqtt.Client"):
            assert MqttClient(cfg, logging.getLogger("t")).availability_topic == "a/b"

    def test_connect_is_async_with_paho_owning_reconnects(self, mq):
        mq.connect()
        mq.client.connect_async.assert_called_once()
        mq.client.loop_start.assert_called_once()
        mq.client.connect.assert_not_called()
        mq.ensure_connection()
        mq.client.reconnect.assert_not_called()

    def test_discovery_carries_availability_and_total_state_class(self, mq):
        mq.client.on_connect(mq.client, None, {}, 0)
        cfgs = {t: json.loads(p) for t, p, _ in _published(mq) if t.endswith("/config")}
        total = cfgs["homeassistant/sensor/water_total/config"]
        assert total["state_class"] == "total"
        for key in ("water_total", "water_rate", "water_rate_lpm"):
            c = cfgs[f"homeassistant/sensor/{key}/config"]
            assert c["availability_topic"] == "home/watermeter/status"
            assert c["payload_not_available"] == "offline"
        assert "availability_topic" not in cfgs["homeassistant/camera/water_overlay/config"]

    def test_on_connect_republishes_current_availability_after_discovery(self, mq):
        mq.available = True
        mq.client.on_connect(mq.client, None, {}, 0)
        pubs = _published(mq)
        assert pubs[-1][0] == "home/watermeter/status" and pubs[-1][1] == "online"
        assert pubs[-1][2] == {"retain": True, "qos": 1}

    def test_set_available_publishes_only_on_change(self, mq):
        mq.client.on_connect(mq.client, None, {}, 0)
        mq.client.publish.reset_mock()
        mq.set_available(False)            # already False
        assert mq.client.publish.call_count == 0
        mq.set_available(True)
        mq.set_available(True)
        assert [p[1] for p in _published(mq)] == ["online"]


# ---------------------------------------------------------------------------
# handle_digital_result: streak -> offline -> catch-up -> online
# ---------------------------------------------------------------------------

class _FakeMqtt:
    def __init__(self):
        self.msgs = []
        self.available = True

    def publish(self, t, p, retain=False, qos=0):
        self.msgs.append((t, p))

    def set_available(self, v):
        if bool(v) != self.available:
            self.available = bool(v)
            self.msgs.append(("home/watermeter/status", "online" if v else "offline"))


def _ok(litres, ts):
    return {"reason": "ok", "litres": litres, "ts": ts, "flow_m3h": None,
            "total_int_digits_list": [], "total_frac_digits_list": [], "raw_total": None}


def _fail(reason, ts, litres=None):
    return {"reason": reason, "litres": litres, "ts": ts, "flow_m3h": None,
            "total_int_digits_list": list("00005"), "total_frac_digits_list": list("7_8"),
            "raw_total": None}


class TestHandleDigitalResult:
    @pytest.fixture
    def env(self, tmp_path):
        cfg = SimpleNamespace(mqtt_main_topic="home/watermeter", meter_type="digital",
                              state_path=str(tmp_path / "state.json"))
        return cfg, _FakeMqtt(), ReadingGate(), StaleTracker(600, None)

    def test_streak_goes_offline_then_catch_up_publishes_true_total(self, env):
        cfg, mq, gate, stale = env
        log = logging.getLogger("t")
        handle_digital_result(cfg, mq, gate, stale, _ok(57300, 0.0), log, now=0.0)
        assert ("home/watermeter/main/value", "57.300") in mq.msgs and mq.available

        mq.msgs.clear()
        for t in range(20, 600, 20):             # < 10 min of give-ups
            handle_digital_result(cfg, mq, gate, stale, _fail("wrong_view", t), log, now=t)
        assert mq.available and not mq.msgs

        for t in range(600, 3 * 3600, 20):       # past 10 min: offline once
            handle_digital_result(cfg, mq, gate, stale, _fail("wrong_view", t), log, now=t)
        assert mq.msgs == [("home/watermeter/status", "offline")]

        # Recovery 3 h later, 300 L used meanwhile: first reading is a
        # candidate (stays offline), the next independent capture confirms
        # it and the true cumulative total goes out before "online".
        mq.msgs.clear()
        t = 3 * 3600.0
        handle_digital_result(cfg, mq, gate, stale, _ok(57600, t), log, now=t)
        assert not mq.available and not mq.msgs
        handle_digital_result(cfg, mq, gate, stale, _ok(57601, t + 20), log, now=t + 20)
        assert mq.msgs[0] == ("home/watermeter/main/value", "57.601")
        assert mq.msgs[-1] == ("home/watermeter/status", "online")
        assert json.load(open(cfg.state_path))["total_l"] == 57601

    def test_rebaseline_publishes_lower_value(self, env):
        cfg, mq, gate, stale = env
        log = logging.getLogger("t")
        gate.last_l, gate.last_ts = 57294, 0.0
        stale.mark_fresh(0.0)
        for t, v in ((20, 57236), (50, 57236)):
            handle_digital_result(cfg, mq, gate, stale, _fail("implausible", t, v), log, now=t)
        handle_digital_result(cfg, mq, gate, stale, _fail("implausible", 80, 57237), log, now=80)
        assert ("home/watermeter/main/value", "57.237") in mq.msgs
        assert ("home/watermeter/main/rate_lpm", "0.000") in mq.msgs

    def test_unparsed_cycle_does_not_touch_gate(self, env):
        cfg, mq, gate, stale = env
        handle_digital_result(cfg, mq, gate, stale, _fail("parse_failed", 5.0),
                              logging.getLogger("t"), now=5.0)
        assert gate.last_l is None


# ---------------------------------------------------------------------------
# ViewPhase
# ---------------------------------------------------------------------------

def _lcd_numeric(t, phi, period=20.0, dwell=11.0):
    return (t - phi) % period < dwell


class TestViewPhase:
    def test_learns_phase_and_schedules_inside_numeric_window(self):
        rng = random.Random(1)
        phi = 7.3
        p = ViewPhase(20.0, 11.0, margin_sec=1.0, capture_lead_sec=0.0)
        t = 1000.0 + rng.random() * 20
        for _ in range(4):                       # a few unsynchronised captures
            p.observe(t, _lcd_numeric(t, phi))
            t += 5.5 + rng.random()
        assert p.confident()
        start = p.next_start(t)
        assert start >= t
        assert (start - phi) % 20.0 >= 1.0 and (start - phi) % 20.0 <= 10.0

    @staticmethod
    def _simulate(true_period, cycles=400, phi=3.0, lead=1.4, interval=20.0):
        """Mirror main() + run_digital_cycle(): schedule, capture, retry on miss."""
        p = ViewPhase(20.0, 11.0, margin_sec=1.0, capture_lead_sec=lead)
        t = 0.0
        first_miss = no_reading = 0
        for cycle in range(cycles):
            start = p.next_start(t) if p.confident() else None
            if start is None:
                start = p.explore_start(t)
            tt = start
            got = False
            for attempt in range(3):
                e = tt + lead
                num = _lcd_numeric(e, phi, true_period)
                p.observe(e, num)
                if cycle >= 20 and attempt == 0 and not num:
                    first_miss += 1
                if num:
                    got = True
                    break
                nxt = p.next_start(e) if p.confident() else None
                tt = e + (min(max(0.0, nxt - e), 20.0) if nxt is not None else 5.5)
            if cycle >= 20 and not got:
                no_reading += 1
            t = start + interval
        return first_miss / (cycles - 20), no_reading

    @pytest.mark.parametrize("true_period,max_first_miss", [
        # First-capture misses are the periodic re-probes once the phase
        # uncertainty outgrows the window (~4 %); unsynchronised it is ~45 %.
        (20.0, 0.06),
        (20.01, 0.06),   # ~2.5x the measured period error
        (20.05, 0.06),   # stress: 12x; adaptive drift must keep up
    ])
    def test_scheduled_captures_stay_numeric_with_period_drift(self, true_period, max_first_miss):
        miss_rate, no_reading = self._simulate(true_period)
        assert miss_rate <= max_first_miss, miss_rate
        assert no_reading == 0

    def test_unsynchronised_baseline_is_worse(self):
        # Sanity: a plain 20 s cadence on a 20.0 s rotation phase-locks; if it
        # locks onto a diagnostic view every first capture misses.
        phi, misses = 6.4, 0
        for c in range(100):
            misses += not _lcd_numeric(c * 20.0 + 1.4, phi)
        assert misses == 100

    def test_contradiction_restarts_from_latest_observation(self):
        p = ViewPhase(20.0, 11.0)
        p.observe(0.0, True)
        p.observe(0.0 + 0.05, False)             # impossible: same instant
        assert p.feasible.any()

    def test_not_confident_without_observations(self):
        p = ViewPhase()
        assert not p.confident() and p.next_start(0.0) is None


# ---------------------------------------------------------------------------
# 7-segment classifier on real LCD frames
# ---------------------------------------------------------------------------

def test_lcd_fixtures_read_correctly():
    spec = json.loads((FIXTURES / "lcd_fixtures.json").read_text())
    for name, expected in spec["frames"].items():
        img = cv2.cvtColor(cv2.imread(str(FIXTURES / name), cv2.IMREAD_GRAYSCALE),
                           cv2.COLOR_GRAY2BGR)
        got = "".join(W._classify_7seg_from_path(None, r, img=img) or "_"
                      for r in spec["rois"])
        assert got == expected, name


def test_ambiguous_segment_returns_empty():
    # A "3" with a faint top-left ghost (between unlit and lit) must not be
    # guessed as "9".
    h, w = 60, 30
    img = np.full((h, w), 255, np.uint8)
    t = 5
    img[0:t, :] = 0                       # top
    img[(h - t) // 2:(h + t) // 2, :] = 0  # middle
    img[h - t:h, :] = 0                   # bottom
    img[:, w - t:w] = 0                   # right side (both)
    # Faint ghost spanning top bar to middle bar on the left (touching both,
    # so it stays part of the digit's ink cluster), ~17 % ink.
    ghost = img[t:(h - t) // 2, 0:int(w * 0.30)]
    ghost[::2, ::3] = 0
    assert W.classify_7seg(img) == ""
    clean = img.copy()
    clean[int(h * 0.20):int(h * 0.38), 0:int(w * 0.30)] = 255
    assert W.classify_7seg(clean) == "3"


# ---------------------------------------------------------------------------
# Alignment quality gate
# ---------------------------------------------------------------------------

def test_alignment_quality_gate():
    cfg = SimpleNamespace(min_inliers=25, min_inlier_ratio=0.5, max_translation_frac=0.10)
    M = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])
    good_mask = np.ones((100, 1), np.uint8)
    assert alignment_quality_ok(M, good_mask, 100, (480, 640), cfg)
    assert not alignment_quality_ok(M, np.ones((20, 1), np.uint8), 20, (480, 640), cfg)
    half = np.zeros((100, 1), np.uint8); half[:40] = 1
    assert not alignment_quality_ok(M, half, 100, (480, 640), cfg)
    far = np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 0.0]])
    assert not alignment_quality_ok(far, good_mask, 100, (480, 640), cfg)


def test_cycle_budget_stops_retries(monkeypatch):
    t = {"now": 0.0}
    def fake_capture(cfg, session, log, stamp=None):
        t["now"] += 30.0           # every capture takes 30 s
        return None
    monkeypatch.setattr("watermeter.capture_frame", fake_capture)
    cfg = SimpleNamespace(digital_max_retries=4, retry_backoff_sec=0.0,
                          digital_retry_delay_sec=0.0, digital_cycle_budget_sec=45.0)
    calls = []
    def log_warning(*a, **k): calls.append(a)
    log = SimpleNamespace(warning=log_warning, debug=lambda *a, **k: None)
    r = W.run_digital_cycle(cfg, None, None, None, None, log, sleep=lambda s: None,
                            clock=lambda: t["now"])
    assert r["reason"] == "no_frame"
    assert t["now"] == 60.0       # 2 attempts, then the budget stops the third
