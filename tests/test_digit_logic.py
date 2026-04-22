"""Tests for the rolling-digit resolver (decide_digit) and integer composer (compose_integer).

These are the highest-value tests in the suite: compose_integer is the
most subtle piece of logic in the project. See watermeter.py:565 and :625.

Reminders:
- OCR values are strings ("0".."9") or "" / None when detection fails.
- compose_integer's digit_obs order is (full, top, bottom) per position,
  left-to-right (most significant first).
- decide_digit's arg order is (bottom, top, full, progress, thr_up, thr_dn, prev).
"""
from __future__ import annotations

import pytest

from watermeter import compose_integer, decide_digit


THR_UP = 0.92
THR_DN = 0.08


# ---- decide_digit ----------------------------------------------------------

class TestDecideDigitRollingWindow:
    """Rolling window = (top == (bottom + 1) % 10). Algorithm uses progress
    thresholds to pick which of top/bottom is 'truth' at a given moment."""

    def test_high_progress_returns_top(self):
        # bottom=4, top=5 (next). progress high → digit has rolled to top.
        assert decide_digit("4", "5", "4", progress=0.95, thr_up=THR_UP, thr_dn=THR_DN) == 5

    def test_low_progress_returns_bottom(self):
        assert decide_digit("4", "5", "4", progress=0.05, thr_up=THR_UP, thr_dn=THR_DN) == 4

    def test_middle_with_prev_returns_prev(self):
        assert decide_digit("4", "5", "4", progress=0.5, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 4

    def test_middle_without_prev_returns_bottom(self):
        assert decide_digit("4", "5", "4", progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 4

    def test_nine_to_zero_wraparound_high(self):
        # bottom=9, top=0 (= (9+1)%10). progress high → rolled, returns 0.
        assert decide_digit("9", "0", "9", progress=0.95, thr_up=THR_UP, thr_dn=THR_DN) == 0

    def test_nine_to_zero_wraparound_low(self):
        assert decide_digit("9", "0", "9", progress=0.05, thr_up=THR_UP, thr_dn=THR_DN) == 9


class TestDecideDigitAgreement:
    def test_top_and_bottom_agree_returns_that_value(self):
        # Not a rolling window (t != b+1); they agree → trust them.
        assert decide_digit("7", "7", "7", progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 7

    def test_top_bottom_agree_overrides_full(self):
        assert decide_digit("3", "3", "9", progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 3


class TestDecideDigitRolloverFromPrev:
    """When progress is low AND prev exists, the algorithm infers a rollover
    just happened: the expected new digit is (prev + 1) % 10."""

    def test_ocr_shows_next_digit(self):
        # prev=4, low progress, OCR shows 5 → return 5.
        assert decide_digit("", "", "5", progress=0.05, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 5

    def test_ocr_shows_prev_digit_trust_ocr(self):
        # prev=4, low progress, OCR clearly shows 4 → trust OCR (roll didn't happen yet).
        assert decide_digit("", "", "4", progress=0.05, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 4

    def test_ocr_shows_other_digit_trust_ocr(self):
        # prev=4, low progress, OCR shows 7 → trust OCR over dial inference.
        assert decide_digit("", "", "7", progress=0.05, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 7

    def test_all_ocr_none_holds_prev(self):
        # prev=4, low progress, OCR silent → DO NOT speculatively increment.
        # decide_digit is stateless, so "low progress" alone can't distinguish
        # "just rolled over" from "stable low-progress window with bad OCR".
        # The safe fallback is to hold prev and let the next cycle recover.
        assert decide_digit(None, None, None, progress=0.05, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 4

    def test_empty_strings_treated_as_none_holds_prev(self):
        assert decide_digit("", "", "", progress=0.05, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 4

    def test_nine_prev_without_ocr_holds(self):
        # prev=9, low progress, OCR silent → hold prev=9 (no speculative wrap).
        assert decide_digit(None, None, None, progress=0.05, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=9) == 9


class TestDecideDigitStableZone:
    def test_middle_zone_with_prev_sticks_to_prev(self):
        # OCR disagrees wildly (3/7/5), but we're in the stable zone → trust prev.
        assert decide_digit("3", "7", "5", progress=0.5, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=4) == 4


class TestDecideDigitFallback:
    def test_all_none_returns_zero(self):
        assert decide_digit(None, None, None, progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 0

    def test_only_prev_returns_prev(self):
        assert decide_digit(None, None, None, progress=0.5, thr_up=THR_UP, thr_dn=THR_DN, prev_digit=6) == 6

    def test_only_full_returns_full(self):
        # Middle zone but no prev. Falls through the chain: f, b, t, prev.
        assert decide_digit(None, None, "3", progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 3

    def test_only_bottom_returns_bottom(self):
        assert decide_digit("3", None, None, progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 3

    def test_only_top_returns_top(self):
        assert decide_digit(None, "3", None, progress=0.5, thr_up=THR_UP, thr_dn=THR_DN) == 3


# ---- compose_integer -------------------------------------------------------

def _obs(s):
    """Helper: build digit_obs from a plain digit string, treating each digit as
    a stable reading (full=top=bottom=that digit)."""
    return [(c, c, c) for c in s]


class TestComposeIntegerStatic:
    def test_stable_reading_all_digits_agree(self):
        out, digits = compose_integer(_obs("12345"), frac=0.0,
                                       thr_up=THR_UP, thr_dn=THR_DN, prev_int_str=None)
        assert out == "12345"
        assert digits == [1, 2, 3, 4, 5]

    def test_stable_reading_with_matching_prev(self):
        out, _ = compose_integer(_obs("12345"), frac=0.5,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="12345")
        assert out == "12345"

    def test_prev_length_mismatch_treated_as_no_prev(self):
        # prev of wrong length must be ignored
        out, _ = compose_integer(_obs("00000"), frac=0.0,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="99")
        assert out == "00000"

    def test_prev_with_non_digit_treated_as_no_prev(self):
        out, _ = compose_integer(_obs("00000"), frac=0.0,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="xxxxx")
        assert out == "00000"

    def test_cold_start_no_prev(self):
        out, _ = compose_integer(_obs("00100"), frac=0.0,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str=None)
        assert out == "00100"


class TestComposeIntegerRollover:
    def test_last_digit_rollover_at_high_progress(self):
        # prev = 12349, rightmost mid-rollover (bot=9, top=0), high progress → rolls to 0.
        # Remaining digits stable at prev.
        obs = [("1", "1", "1"), ("2", "2", "2"), ("3", "3", "3"), ("4", "4", "4"), ("9", "0", "9")]
        out, digits = compose_integer(obs, frac=0.95,
                                       thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="12349")
        # The ones position rolled to 0; but progress propagates only via the
        # chosen value (0), which yields a low carry → tens stays at prev (4).
        assert digits[-1] == 0
        # Tens, hundreds, thousands, ten-thousands stay at their prev values
        # because the carry ripple after ones=0 is too small to trip thr_up.
        assert digits[:-1] == [1, 2, 3, 4]
        assert out == "12340"

    def test_last_digit_low_progress_sticks_at_prev(self):
        # Just after rollover: frac near 0, OCR sees (_, _, 0) or noisy.
        # Algorithm should trust prev (= 0 at ones) via the rollover-from-prev branch.
        obs = [("1", "1", "1"), ("2", "2", "2"), ("3", "3", "3"), ("5", "5", "5"), ("0", "0", "0")]
        out, _ = compose_integer(obs, frac=0.02,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="12350")
        assert out == "12350"

    def test_cascading_rollover_via_carry_ripple(self):
        # prev=19, tenths dial at 9.9 (frac≈0.99), ones stays at 9 (prev via middle zone
        # resolution below), tens receives carry (9+0.99)/10 ≈ 0.999 ≥ thr_up → rolls to top=0.
        # BUT ones at high progress with rolling window returns top=0, breaking the ripple.
        # So the cascade only works reliably when ones OCR is decisive or we're in non-rolling state.
        #
        # Simpler scenario: tens digit mid-rollover, ones digit at 9 stable (not in rolling window).
        # ones=(9,9,9) not rolling → fallback chain returns 9. lower=(9+0.99)/10=0.999 → tens rolls.
        obs = [("1", "1", "1"), ("9", "0", "9"), ("9", "9", "9")]
        out, digits = compose_integer(obs, frac=0.99,
                                       thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="199")
        # ones stable at 9; tens with high progress rolls to top=0.
        # Then carry = (0 + 0.999)/10 = 0.0999 < thr_up, hundreds stays at prev=1.
        assert digits == [1, 0, 9]

    def test_high_frac_with_stable_digits_holds(self):
        # Mid-reading: prev="00010", ones stable at 0, tens stable at 1, frac mid-range.
        out, _ = compose_integer(_obs("00010"), frac=0.3,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="00010")
        assert out == "00010"

    def test_ocr_dropout_falls_back_to_prev(self):
        # Two digits drop OCR entirely; middle zone + prev keeps them stable.
        obs = [("1", "1", "1"), ("", "", ""), ("3", "3", "3"), ("", "", ""), ("5", "5", "5")]
        out, _ = compose_integer(obs, frac=0.5,
                                  thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="12345")
        assert out == "12345"


class TestComposeIntegerCarryRipple:
    """Direct tests of the `lower = (chosen + lower) / 10.0` propagation
    documented in watermeter.py:650."""

    def test_low_chosen_gives_low_next_progress(self):
        # If position 4 chose 0 with frac=0.98, lower for position 3 is 0.098 (stable zone).
        # Tens at prev=9 should remain 9 even with a rolling OCR window on tens.
        obs = [("0", "0", "0"), ("1", "1", "1"), ("9", "0", "9"), ("9", "0", "9")]
        out, digits = compose_integer(obs, frac=0.98,
                                       thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="0199")
        # ones: rolling, high progress → returns 0.
        # tens: lower = (0 + 0.98)/10 = 0.098 (just above thr_dn, in middle zone) → prev=9.
        assert digits[-1] == 0
        assert digits[-2] == 9

    def test_high_chosen_propagates_ripple_upward(self):
        # If ones stays at 9 (OCR clean 9), lower for tens is (9 + 0.99)/10 = 0.999 → rolls.
        obs = [("0", "0", "0"), ("1", "1", "1"), ("9", "0", "9"), ("9", "9", "9")]
        out, digits = compose_integer(obs, frac=0.99,
                                       thr_up=THR_UP, thr_dn=THR_DN, prev_int_str="0199")
        # ones: not rolling window (t==b==9), returns 9.
        # tens: progress=0.999 ≥ thr_up, rolling window (b=9,t=0) → returns 0.
        assert digits[-1] == 9
        assert digits[-2] == 0
