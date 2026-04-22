"""Tests for small pure utilities in watermeter.py."""
from __future__ import annotations

from datetime import datetime

import pytest

from watermeter import (
    Config,
    DialConfig,
    _hhmm_to_min,
    _in_window_local,
    adjust_dial_roi,
    build_digit_rois,
    circular_blend,
    circular_dist,
    clip01,
    norm_to_abs,
)


class TestClip01:
    def test_inside_range(self):
        assert clip01(0.5) == 0.5

    def test_below_zero(self):
        assert clip01(-0.1) == 0.0

    def test_above_one(self):
        assert clip01(1.1) == 1.0

    def test_exact_bounds(self):
        assert clip01(0.0) == 0.0
        assert clip01(1.0) == 1.0

    def test_negative_large(self):
        assert clip01(-100.0) == 0.0


class TestNormToAbs:
    def test_full_frame(self):
        assert norm_to_abs([0.0, 0.0, 1.0, 1.0], 640, 480) == (0, 0, 640, 480)

    def test_centered_quarter(self):
        x, y, w, h = norm_to_abs([0.25, 0.25, 0.5, 0.5], 640, 480)
        assert (x, y, w, h) == (160, 120, 320, 240)

    def test_truncation_with_odd_dims(self):
        x, y, w, h = norm_to_abs([0.5, 0.5, 0.5, 0.5], 641, 481)
        assert x == int(0.5 * 641)
        assert y == int(0.5 * 481)
        assert w == int(0.5 * 641)
        assert h == int(0.5 * 481)

    def test_zero_size_roi(self):
        assert norm_to_abs([0.1, 0.1, 0.0, 0.0], 640, 480) == (64, 48, 0, 0)


class TestHhmmToMin:
    @pytest.mark.parametrize("s,expected", [
        ("00:00", 0),
        ("07:00", 420),
        ("07:30", 450),
        ("12:00", 720),
        ("23:59", 1439),
    ])
    def test_valid(self, s, expected):
        assert _hhmm_to_min(s) == expected

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            _hhmm_to_min("nope")


class TestInWindowLocal:
    @pytest.mark.parametrize("hour,minute,expected", [
        (6, 59, False),
        (7, 0, True),
        (12, 0, True),
        (16, 59, True),
        (17, 0, False),
        (23, 0, False),
    ])
    def test_normal_window(self, hour, minute, expected):
        now = datetime(2025, 1, 1, hour, minute)
        assert _in_window_local(now, "07:00", "17:00") is expected

    @pytest.mark.parametrize("hour,minute,expected", [
        (23, 0, True),
        (23, 30, True),
        (0, 0, True),
        (0, 30, True),
        (0, 59, True),
        (1, 0, False),
        (6, 0, False),
        (12, 0, False),
    ])
    def test_wraparound_window(self, hour, minute, expected):
        now = datetime(2025, 1, 1, hour, minute)
        assert _in_window_local(now, "23:00", "01:00") is expected

    def test_quiet_hours_from_config(self):
        now = datetime(2025, 1, 1, 3, 30)
        assert _in_window_local(now, "00:00", "07:00") is True
        now2 = datetime(2025, 1, 1, 7, 0)
        assert _in_window_local(now2, "00:00", "07:00") is False


class TestBuildDigitRois:
    def _cfg(self, window=(0.2, 0.1, 0.5, 0.1), count=5, inset=0.1):
        return Config(
            esp32_base_url="http://localhost",
            digits_count=count,
            digits_window=list(window),
            per_digit_inset=inset,
        )

    def test_count_matches(self):
        rois = build_digit_rois(self._cfg(count=5), 640, 480)
        assert len(rois) == 5

    def test_rois_inside_window(self):
        win_x, win_y, win_w, win_h = 0.2, 0.1, 0.5, 0.1
        rois = build_digit_rois(self._cfg(window=(win_x, win_y, win_w, win_h), count=5), 640, 480)
        for x, y, w, h in rois:
            assert x >= win_x - 1e-9
            assert y >= win_y - 1e-9
            assert x + w <= win_x + win_w + 1e-9
            assert y + h <= win_y + win_h + 1e-9

    def test_non_overlapping_centers(self):
        rois = build_digit_rois(self._cfg(count=5), 640, 480)
        centers_x = [r[0] + r[2] / 2 for r in rois]
        # Sorted ascending and strictly increasing
        assert centers_x == sorted(centers_x)
        for i in range(len(centers_x) - 1):
            assert centers_x[i + 1] - centers_x[i] > 0.001

    def test_inset_shrinks(self):
        loose = build_digit_rois(self._cfg(inset=0.0), 640, 480)
        tight = build_digit_rois(self._cfg(inset=0.2), 640, 480)
        # Widths and heights are smaller when inset is larger
        assert tight[0][2] < loose[0][2]
        assert tight[0][3] < loose[0][3]


class TestAdjustDialRoi:
    def test_zero_offset_is_identity(self):
        out = adjust_dial_roi([0.5, 0.5, 0.1, 0.1], (0.0, 0.0), 640, 480, smoothing_alpha=0.3)
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(0.5)

    def test_positive_offset_shifts_right_and_down(self):
        out = adjust_dial_roi([0.3, 0.3, 0.1, 0.1], (50.0, 40.0), 640, 480, smoothing_alpha=1.0)
        # dx_norm = 50 * 0.1 / 640; alpha=1 applies fully
        assert out[0] > 0.3
        assert out[1] > 0.3

    def test_alpha_zero_is_identity(self):
        out = adjust_dial_roi([0.3, 0.3, 0.1, 0.1], (200.0, 200.0), 640, 480, smoothing_alpha=0.0)
        assert out[0] == pytest.approx(0.3)
        assert out[1] == pytest.approx(0.3)

    def test_clamped_lower_bound(self):
        out = adjust_dial_roi([0.01, 0.01, 0.1, 0.1], (-10000.0, -10000.0), 640, 480, smoothing_alpha=1.0)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.0)

    def test_clamped_upper_bound(self):
        out = adjust_dial_roi([0.85, 0.85, 0.1, 0.1], (10000.0, 10000.0), 640, 480, smoothing_alpha=1.0)
        # must be within [0, 1-w]
        assert out[0] == pytest.approx(1.0 - 0.1)
        assert out[1] == pytest.approx(1.0 - 0.1)

    def test_width_and_height_preserved(self):
        out = adjust_dial_roi([0.4, 0.4, 0.13, 0.17], (20.0, 10.0), 640, 480, smoothing_alpha=0.3)
        assert out[2] == pytest.approx(0.13)
        assert out[3] == pytest.approx(0.17)


class TestCircularDist:
    def test_zero_distance(self):
        assert circular_dist(3.0, 3.0) == pytest.approx(0.0)

    def test_linear_distance_in_range(self):
        assert circular_dist(2.0, 5.0) == pytest.approx(3.0)

    def test_wraparound_is_shorter(self):
        # 9.5 → 0.5 is 1.0 via wrap, not 9.0 linear.
        assert circular_dist(9.5, 0.5) == pytest.approx(1.0)
        assert circular_dist(0.5, 9.5) == pytest.approx(1.0)

    def test_halfway_point(self):
        assert circular_dist(0.0, 5.0) == pytest.approx(5.0)

    def test_custom_period(self):
        # 350° to 10° is 20° apart on a 360 ring.
        assert circular_dist(350.0, 10.0, period=360.0) == pytest.approx(20.0)


class TestCircularBlend:
    def test_equal_inputs_return_input(self):
        assert circular_blend(3.0, 3.0, alpha=0.5) == pytest.approx(3.0)

    def test_linear_blend_within_ring(self):
        # 2.0 and 4.0 are in the same half-ring; blend is near 3.0.
        result = circular_blend(2.0, 4.0, alpha=0.5)
        assert result == pytest.approx(3.0, abs=0.01)

    def test_wraparound_blend_stays_on_short_arc(self):
        # Linear blend of 9.9 and 0.1 would give ~5.0 (garbage).
        # Circular blend should stay near 0.0 (the true midpoint across the wrap).
        result = circular_blend(9.9, 0.1, alpha=0.5)
        # Must be close to 0.0 (or 10.0 ≡ 0.0), not around 5.0.
        assert circular_dist(result, 0.0) < 0.1

    def test_weighted_blend_biases_to_alpha(self):
        # alpha=0.9 puts most of the weight on a.
        result = circular_blend(9.9, 0.1, alpha=0.9)
        # Should sit much closer to 9.9 than to 0.1.
        assert circular_dist(result, 9.9) < circular_dist(result, 0.1)

    def test_alpha_one_returns_first_value(self):
        assert circular_blend(3.7, 8.2, alpha=1.0) == pytest.approx(3.7, abs=0.01)

    def test_alpha_zero_returns_second_value(self):
        assert circular_blend(3.7, 8.2, alpha=0.0) == pytest.approx(8.2, abs=0.01)
