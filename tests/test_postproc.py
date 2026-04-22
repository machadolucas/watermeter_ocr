"""Tests for the postprocessing helpers: rollover inference and temporal validation."""
from __future__ import annotations

import pytest

from watermeter import (
    Config,
    estimate_total_from_dials,
    predict_expected_reading,
    validate_reading_with_history,
)


def _cfg(rolling_threshold_down=0.08):
    return Config(
        esp32_base_url="http://localhost",
        rolling_threshold_down=rolling_threshold_down,
    )


# ---- estimate_total_from_dials --------------------------------------------

class TestEstimateTotalFromDials:
    def test_none_prev_returns_none(self):
        assert estimate_total_from_dials(None, 0.5, _cfg()) is None

    def test_no_rollover_keeps_integer(self):
        # prev=100.4, frac_pub=0.5 → integer stays at 100.
        assert estimate_total_from_dials(100.4, 0.5, _cfg()) == pytest.approx(100.5)

    def test_forward_rollover_increments_integer(self):
        # prev_frac=0.98 (above high threshold), frac_pub=0.02 (below low) → integer+1.
        cfg = _cfg()
        result = estimate_total_from_dials(100.98, 0.02, cfg)
        assert result == pytest.approx(101.02)

    def test_backward_rollback_decrements_integer(self):
        # prev_frac=0.02 (below low), frac_pub=0.98 (above high) → integer-1.
        cfg = _cfg()
        result = estimate_total_from_dials(100.02, 0.98, cfg)
        assert result == pytest.approx(99.98)

    def test_backward_rollback_clamped_at_zero(self):
        cfg = _cfg()
        # integer would go from 0 to -1, must clamp at 0.
        result = estimate_total_from_dials(0.02, 0.98, cfg)
        assert result == pytest.approx(0.98)

    def test_rolling_threshold_down_affects_window(self):
        # With a larger rolling_threshold_down, the low window is wider.
        cfg_narrow = _cfg(rolling_threshold_down=0.05)   # low clamps at min 0.1? No: min(0.25, max(0.05, 0.1)) = 0.1
        cfg_wide = _cfg(rolling_threshold_down=0.20)      # low = min(0.25, max(0.05, 0.4)) = 0.25
        # prev_frac=0.85 ("just above high=0.75" for wide, NOT for narrow where high=0.90)
        # frac_pub=0.05 (below both low thresholds)
        r_narrow = estimate_total_from_dials(100.85, 0.05, cfg_narrow)
        r_wide = estimate_total_from_dials(100.85, 0.05, cfg_wide)
        # Narrow: high=0.9, prev_frac=0.85 < 0.9 → no rollover → stays 100.
        # Wide:  high=0.75, prev_frac=0.85 > 0.75 → rolls → 101.
        assert r_narrow == pytest.approx(100.05)
        assert r_wide == pytest.approx(101.05)

    def test_small_change_no_rollover(self):
        result = estimate_total_from_dials(50.5, 0.51, _cfg())
        assert result == pytest.approx(50.51)


# ---- validate_reading_with_history ----------------------------------------

class TestValidateReadingWithHistory:
    def test_empty_history_full_confidence(self):
        reading, conf = validate_reading_with_history(5.0, [])
        assert reading == 5.0
        assert conf == 1.0

    def test_none_history_full_confidence(self):
        reading, conf = validate_reading_with_history(5.0, None)
        assert reading == 5.0
        assert conf == 1.0

    def test_small_change_full_confidence(self):
        # last reading 4.2, current 4.5 → diff < 1.0 → conf=1.0.
        reading, conf = validate_reading_with_history(4.5, [4.0, 4.1, 4.2])
        assert reading == 4.5
        assert conf == 1.0

    def test_medium_change_consistent_trend_high_confidence(self):
        # Diff must be >= 1.0 (and < 3.0) with a consistent trend to hit the
        # 0.9 branch. History rising by 1.0 per step; current=+1.2 from last.
        reading, conf = validate_reading_with_history(5.2, [1.0, 2.0, 3.0, 4.0])
        assert conf == pytest.approx(0.9)

    def test_large_sudden_change_reduces_confidence(self):
        # diff of 5 between last and current → flagged as suspicious.
        reading, conf = validate_reading_with_history(8.0, [2.5, 2.8, 3.0])
        assert conf == pytest.approx(0.5)

    def test_medium_change_inconsistent_trend_reduces_confidence(self):
        # diff is 2.5 (between 1 and 3), but trend was flat → not trend-consistent → 0.5
        reading, conf = validate_reading_with_history(5.0, [2.0, 2.1, 2.0, 2.5])
        # current_change=2.5, avg_change≈0.16 → |2.5-0.16|=2.34 > 2.0 → 0.5
        assert conf == pytest.approx(0.5)


# ---- predict_expected_reading ---------------------------------------------

class TestPredictExpectedReading:
    def test_insufficient_history_returns_none(self):
        assert predict_expected_reading([]) is None
        assert predict_expected_reading([1.0]) is None
        assert predict_expected_reading([1.0, 2.0]) is None

    def test_linear_trend_predicts_next(self):
        # Steady +1 per step; next should be 5.
        predicted = predict_expected_reading([1.0, 2.0, 3.0, 4.0])
        assert predicted == pytest.approx(5.0)

    def test_wrapping_at_ten(self):
        # Steady +1 per step climbing past 10 → wraps via %10.
        predicted = predict_expected_reading([7.0, 8.0, 9.0, 10.0])
        assert predicted == pytest.approx(1.0)  # 11 % 10

    def test_flat_trend_predicts_same(self):
        predicted = predict_expected_reading([5.0, 5.0, 5.0, 5.0])
        assert predicted == pytest.approx(5.0)
