"""Offline timing/integration checks; also runnable with stdlib unittest.

No broker, camera, runtime configuration or Swift subprocess is used.
"""
import json
import logging
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import watermeter as W
from tests.view_sync_simulation import simulate


class TestViewSyncRuntime(unittest.TestCase):
    def test_config_defaults_and_custom_view_rotation(self):
        raw = {"esp32": {"base_url": "http://unused"}}
        with patch.object(W, "load_yaml", return_value=raw):
            cfg = W.load_config("unused")
            self.assertIsNone(cfg.view_dwells_sec)
            self.assertEqual(cfg.view_drift_per_sec, 0.0003)
            self.assertEqual(cfg.view_drift_decay_observations, 180)
            self.assertEqual(cfg.view_observation_slack_sec, 0.25)
            raw["digital"] = {"view_sync": {"view_dwells_sec": [11, 9],
                              "drift_per_sec": 0.0005, "drift_decay_observations": 240,
                              "observation_slack_sec": 0.3}}
            cfg = W.load_config("unused")
            self.assertEqual(cfg.view_dwells_sec, [11, 9])
            self.assertEqual(cfg.view_drift_per_sec, 0.0005)
            self.assertEqual(cfg.view_drift_decay_observations, 240)
            self.assertEqual(cfg.view_observation_slack_sec, 0.3)

    def test_three_view_simulations_including_jitter_and_quiet_hours(self):
        for period in (20.0, 20.004, 20.01, 20.05):
            with self.subTest(period=period, jitter=0):
                misses, unread = simulate(period)
                self.assertLessEqual(misses, 0.06 if period == 20.05 else 0.02)
                self.assertEqual(unread, 0)
            for interval in (20.0, 60.0):
                with self.subTest(period=period, interval=interval, jitter=0.2):
                    misses, unread = simulate(period, cycles=1200, interval=interval,
                                              timing_jitter=0.2)
                    self.assertLessEqual(misses, 0.06 if period == 20.05 else 0.025)
                    self.assertEqual(unread, 0)

    def test_capture_proxy_excludes_body_flash_off_and_processing(self):
        cfg = W.Config(esp32_base_url="http://unused", esp32_capture_mode="esphome")
        now = [0.0]
        calls = []

        def advance(seconds):
            now[0] += seconds

        def response():
            r = MagicMock()
            def body(chunk_size):
                advance(0.6)
                yield b"jpeg"
            r.iter_content.side_effect = body
            return r

        def post(url, **kwargs):
            calls.append("on" if "turn_on" in url else "off")
            advance(0.1 if "turn_on" in url else 0.4)
            return response()

        def get(url, **kwargs):
            calls.append("kept" if kwargs.get("stream") else "discard")
            advance(0.2)
            return response()

        session = MagicMock(post=post, get=get)
        stamp = {}
        with patch.object(W.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(W.time, "sleep", side_effect=advance):
            content = W._fetch_jpeg(cfg, session, logging.getLogger("capture-test"), stamp)
        self.assertEqual(content, b"jpeg")
        self.assertEqual(calls, ["on", "discard", "kept", "off"])
        self.assertAlmostEqual(stamp["t_frame"], 1.4)
        self.assertAlmostEqual(now[0], 2.5)

    def _cycle(self, totals, phase, aligner=None, gate=None, **cfg_fields):
        cfg = W.Config(esp32_base_url="http://unused", meter_type="digital",
                       digital_total_roi=[0, 0, 1, 1], digital_flow_roi=[])
        for name, value in cfg_fields.items():
            setattr(cfg, name, value)
        now = [0.0]
        sources = iter(totals)
        ocr = MagicMock()
        ocr.read_line.side_effect = lambda *a: next(sources)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        def capture(cfg, session, log, stamp):
            stamp["t_frame"] = now[0] + 1.4
            now[0] += 1.8
            return frame

        def sleep(seconds):
            now[0] += seconds

        log = MagicMock()
        with patch.object(W, "capture_frame", side_effect=capture):
            result = W.run_digital_cycle(cfg, ocr, aligner, None, None, log,
                                         phase=phase, gate=gate, clock=lambda: now[0],
                                         wall=lambda: 1000 + now[0], sleep=sleep)
        return result, log, ocr

    def test_first_miss_logs_once_despite_multiple_retries(self):
        p = W.ViewPhase()
        p.explore_start(0.0)
        result, log, _ = self._cycle(["abc", "abc", "000057308"], p)
        self.assertTrue(result["success"])
        self.assertEqual(log.info.call_count, 1)
        args = log.info.call_args.args
        self.assertEqual(args[1:3], ("wrong_view", "exploring"))
        self.assertEqual(p._counts["cycles"], 1)
        self.assertEqual(p._counts["first_numeric"], 0)
        self.assertAlmostEqual(p.lead, 1.4512)

    def test_aligned_failure_logs_without_observing_phase_or_ocr(self):
        p = W.ViewPhase()
        aligner = MagicMock()
        aligner.align.return_value = (np.zeros((10, 10, 3)), None, False)
        result, log, ocr = self._cycle([], p, aligner)
        self.assertEqual(result["reason"], "align_failed")
        self.assertIsNone(p.last_obs)
        ocr.read_line.assert_not_called()
        self.assertEqual(log.info.call_count, 1)
        self.assertEqual(log.info.call_args.args[1], "align_failed")

    def test_parse_failure_logs_without_observing_phase(self):
        p = W.ViewPhase()
        with patch.object(W, "is_valid_digital_view", return_value=True), \
                patch.object(W, "parse_digital_total", return_value=None):
            result, log, _ = self._cycle(["000057308"] * 3, p)
        self.assertEqual(result["reason"], "parse_failed")
        self.assertIsNone(p.last_obs)
        self.assertEqual(log.info.call_count, 1)
        self.assertEqual(log.info.call_args.args[1], "parse_failed")

    def test_numeric_first_attempt_is_quiet_and_counted(self):
        p = W.ViewPhase()
        result, log, _ = self._cycle(["000057308"], p)
        self.assertTrue(result["success"])
        log.info.assert_not_called()
        self.assertEqual(p._counts["first_numeric"], 1)

    def test_gate_rejection_still_counts_numeric_and_does_not_accept_total(self):
        p = W.ViewPhase()
        gate = W.ReadingGate(last_l=57308, last_ts=1000.0)
        result, log, _ = self._cycle(["000057908"] * 3, p, gate=gate)
        self.assertEqual(result["reason"], "implausible")
        self.assertEqual(gate.last_l, 57308)
        self.assertEqual(p._counts["first_numeric"], 1)
        log.info.assert_not_called()

    def test_cycle_forwards_cell_boundaries_for_diagnostic_identity(self):
        p = W.ViewPhase()
        p.observe = MagicMock(wraps=p.observe)
        b = list("__40106__")
        n = list("000057308")
        results = [("".join(c for c in cells if c != "_"),
                    ["" if c == "_" else c for c in cells])
                   for cells in (b[:6], b[6:], n[:6], n[6:])]
        with patch.object(W, "read_region_individual_digits", side_effect=results):
            result, _, ocr = self._cycle([], p, digital_total_int_digits=[[0, 0, 1, 1]] * 6,
                                         digital_total_frac_digits=[[0, 0, 1, 1]] * 3)
        self.assertTrue(result["success"])
        first = p.observe.call_args_list[0]
        self.assertFalse(first.kwargs["numeric"])
        self.assertEqual(first.kwargs["cells"], ["" if c == "_" else c for c in b])
        ocr.read_line.assert_not_called()

    def test_hourly_counters_and_prediction_precede_updates(self):
        p = W.ViewPhase()
        log = MagicMock()
        p.feasible[:] = False
        p.feasible[0:20] = True
        p.last_obs = 0.0
        self.assertIsNotNone(p.next_start(0.0))
        source = p.begin_cycle(0.0)
        p.first_attempt(1.4, "wrong_view", source, log)
        self.assertEqual(log.info.call_args.args[1:3], ("wrong_view", "synced"))
        # First-miss logging has not changed the lead or feasible phase.
        self.assertEqual(p.lead, 1.5)
        self.assertEqual(p.feasible.sum(), 20)
        p.observe(5.0, False)  # contradiction, visible in the hourly summary
        p.summary(3599, log)
        self.assertEqual(log.info.call_count, 1)
        p.summary(3600, log)
        summary = log.info.call_args.args
        message = summary[0] % summary[1:]
        for text in ("cycles=1", "first_numeric=0", "wrong_view_synced=1",
                     "wrong_view_exploring=0", "explore_probes=0", "contradictions=1"):
            self.assertIn(text, message)
        p.summary(3601, log)
        self.assertEqual(log.info.call_count, 2)

    def test_recorded_identities_and_fractional_drift(self):
        spec = json.loads((Path(__file__).parent / "fixtures/lcd/view_sequence.json").read_text())
        p = W.ViewPhase()
        identified = {1: 0, 2: 0}
        for f in spec["samples"]:
            got = p.observe(f["t_sec"], f["view"] == "numeric", list(f["digits"]))
            expected = {"numeric": 0, "B": 1, "A": 2}.get(f["expected_view"])
            if got is not None and expected is not None:
                self.assertEqual(got, expected, f)
                if got:
                    identified[got] += 1
        self.assertGreaterEqual(identified[1], 30)
        self.assertGreaterEqual(identified[2], 60)
        self.assertEqual(p.contradictions, 0)
        p = W.ViewPhase()
        p.feasible[:] = p.coarse[:] = False
        p.feasible[0] = p.coarse[0] = True
        for _ in range(100):
            p._dilate(20 * p.drift)
        self.assertLessEqual(p.feasible.sum(), 13)

    def test_adaptive_drift_decays_with_hour_of_consistency(self):
        p = W.ViewPhase()
        p.feasible[:] = p.coarse[:] = False
        p.feasible[0] = p.coarse[0] = True
        p.last_obs = 0.0
        p.observe(5.0, False)
        self.assertTrue(math.isclose(p.drift, 0.0006))
        for i in range(1, 362):
            p.observe(5.0 + i * 20.0, False)
        self.assertEqual(p.contradictions, 1)
        self.assertTrue(math.isclose(p.drift, 0.0003))

    def test_mask_needs_independent_rotations_and_abstains_on_conflict(self):
        p = W.ViewPhase()
        p.feasible[:] = p.coarse[:] = False
        p.feasible[0] = p.coarse[0] = True
        cells = list("__98765__")  # no meter-specific digit values
        for t in (13.5, 13.6, 13.7, 33.5):
            self.assertIsNone(p.observe(t, False, cells))
        self.assertEqual(p.observe(53.5, False, cells), 1)
        self.assertIsNone(p.observe(78.0, False, cells))  # same mask in A
        self.assertIsNone(p.observe(93.5, False, cells))
        self.assertIsNone(p.observe(113.5, False, list("_________")))
        self.assertIsNone(p.observe(133.5, False, None))  # no cell boundaries

    def test_ordered_dwells_support_other_meter_layouts(self):
        p = W.ViewPhase(view_dwells_sec=[11.0, 2.0, 3.0, 4.0])
        masks = ["123456789", "__12345__", "_1_2_345_", "___123456"]
        identified = set()
        for t in np.arange(0.0, 160.0, 0.5):
            off = t % 20.0
            actual = 0 if off < 11 else 1 if off < 13 else 2 if off < 16 else 3
            got = p.observe(t, actual == 0, list(masks[actual]))
            if got is not None:
                self.assertEqual(got, actual)
                identified.add(got)
        self.assertEqual(identified, {0, 1, 2, 3})
        for dwells in ([11, 3.9, 5], [10, 10], [11, -1, 10]):
            with self.assertRaises(ValueError):
                W.ViewPhase(view_dwells_sec=dwells)

    def test_learned_identity_narrows_miss_and_conflict_uses_broad_constraint(self):
        p = W.ViewPhase()
        p.feasible[:] = p.coarse[:] = False
        p.feasible[0] = p.coarse[0] = True
        cells = list("__98765__")
        for t in (13.5, 33.5, 53.5):
            p.observe(t, False, cells)
        p.feasible[:] = p.coarse[:] = True
        self.assertEqual(p.observe(73.5, False, cells), 1)
        self.assertLess(p.feasible.sum(), p.coarse.sum())
        # Main phase says A while broad phase is uncertain: learned B must
        # abstain rather than force a phase reset or a numeric acceptance.
        p.feasible[:] = False
        p.feasible[155] = True  # origin 15.5 -> offset 18 at t=93.5
        p.coarse[:] = True
        self.assertIsNone(p.observe(93.5, False, cells))
        self.assertTrue(p.feasible[155])
        self.assertEqual(p.contradictions, 0)
        self.assertFalse(p._signatures)

    def test_forecast_expands_both_arc_edges(self):
        p = W.ViewPhase(drift_per_sec=0.001)
        p.feasible[:] = False
        p.feasible[50:70] = True
        p.last_obs = 0.0
        a, w = p._arc()
        predicted_a, predicted_w = p._predicted_arc(100.0)
        self.assertAlmostEqual(predicted_a, a - 0.1)
        self.assertAlmostEqual(predicted_w, w + 0.2)


if __name__ == "__main__":
    unittest.main()
