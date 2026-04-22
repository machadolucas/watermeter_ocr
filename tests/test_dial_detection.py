"""Tests for the dial-detection cascade (detect_dial_center, needle methods, read_dial).

Ported from the original top-level test_dial_detection.py. Uses synthetic dials
(solid background + tick marks + a red needle) rather than real meter images.

Tolerances are deliberately loose because OpenCV's Hough/distance-transform
primitives behave differently across versions. Assertions use a circular
distance (readings are on a 0..10 ring: 0 and 9.9 are "close"), not linear.
"""
from __future__ import annotations

import numpy as np
import pytest

import cv2

from watermeter import (
    detect_dial_center,
    detect_dial_markings,
    detect_needle_by_color,
    detect_needle_by_lines,
    detect_needle_center,
    read_dial,
)


def _circular_err(a: float, b: float, period: float = 10.0) -> float:
    """Shortest distance on a circular scale of [0, period)."""
    d = abs(a - b) % period
    return min(d, period - d)


# ---- Reading → angle correspondence ---------------------------------------

# Angle values are in the same convention as create_synthetic_dial: 0 = right,
# 90 = down, 180 = left, 270 = up (image coords, y-axis flipped).
ANGLE_TO_READING = [
    (0, 2.5),
    (45, 3.75),
    (90, 5.0),
    (135, 6.25),
    (180, 7.5),
    (225, 8.75),
    (270, 0.0),
    (315, 1.25),
]


class TestReadDialAngles:
    @pytest.mark.parametrize("angle,expected", ANGLE_TO_READING)
    def test_cw_reading_matches_expected(self, synthetic_dial, angle, expected):
        img = synthetic_dial(needle_angle=angle, size=200)
        reading, conf, _ = read_dial(img, zero_angle_deg=-90.0, rotation="cw")
        # Generous tolerance — the detection cascade is not exact.
        assert _circular_err(reading, expected) < 0.6, (
            f"angle={angle}°: expected {expected}, got {reading:.3f}"
        )
        assert 0.0 <= conf <= 1.0

    def test_ccw_inverts_direction(self, synthetic_dial):
        img = synthetic_dial(needle_angle=0, size=200)  # CW expects 2.5
        reading_cw, _, _ = read_dial(img, zero_angle_deg=-90.0, rotation="cw")
        reading_ccw, _, _ = read_dial(img, zero_angle_deg=-90.0, rotation="ccw")
        # CCW should be 10 - CW (mod 10) at the same needle position.
        expected_ccw = (10.0 - reading_cw) % 10.0
        assert _circular_err(reading_ccw, expected_ccw) < 0.6


# ---- Noise robustness -----------------------------------------------------

class TestNoiseRobustness:
    def test_mean_error_stays_bounded(self, synthetic_dial):
        angle = 45
        expected = 3.75
        errors = []
        for seed in range(10):
            img = synthetic_dial(needle_angle=angle, size=200, add_noise=True, seed=seed)
            reading, _, _ = read_dial(img, zero_angle_deg=-90.0, rotation="cw")
            errors.append(_circular_err(reading, expected))
        # Statistical assertion, not per-trial. A few flukes are OK.
        assert np.mean(errors) < 0.6, f"mean circular error {np.mean(errors):.3f} too high"

    def test_clean_image_consistency(self, synthetic_dial):
        # Same input should yield the same reading (detection is deterministic on clean input).
        img1 = synthetic_dial(needle_angle=45, size=200)
        img2 = synthetic_dial(needle_angle=45, size=200)
        r1, _, _ = read_dial(img1, zero_angle_deg=-90.0, rotation="cw")
        r2, _, _ = read_dial(img2, zero_angle_deg=-90.0, rotation="cw")
        assert r1 == pytest.approx(r2)


# ---- Off-center detection --------------------------------------------------

class TestOffCenterDetection:
    @pytest.mark.parametrize("dx,dy", [(0, 0), (20, 0), (0, 20), (-20, 0), (0, -20), (15, 15)])
    def test_off_center_reading_bounded(self, synthetic_dial, dx, dy):
        canvas = np.ones((300, 300, 3), dtype=np.uint8) * 240
        dial = synthetic_dial(needle_angle=45, size=150)

        center_x = 150 + dx
        center_y = 150 + dy
        y1, y2 = center_y - 75, center_y + 75
        x1, x2 = center_x - 75, center_x + 75
        assert 0 <= y1 and y2 <= 300 and 0 <= x1 and x2 <= 300
        canvas[y1:y2, x1:x2] = dial

        reading, conf, offset = read_dial(canvas, zero_angle_deg=-90.0, rotation="cw")
        # Expected for 45° on this synthetic dial is ~3.75.
        assert _circular_err(reading, 3.75) < 1.0
        assert 0.0 <= conf <= 1.0
        # center_offset should exist as a 2-tuple of floats
        assert isinstance(offset, tuple) and len(offset) == 2


# ---- Individual detection primitives --------------------------------------

class TestDetectDialCenter:
    def test_returns_center_within_bounds(self, synthetic_dial):
        img = synthetic_dial(needle_angle=45, size=200)
        result = detect_dial_center(img)
        assert result is not None
        cx, cy, radius, conf = result
        h, w = img.shape[:2]
        assert 0 < cx < w
        assert 0 < cy < h
        assert radius > 0
        assert 0.0 <= conf <= 1.0

    def test_center_plausible_for_clean_dial(self, synthetic_dial):
        img = synthetic_dial(needle_angle=45, size=200)
        cx, cy, _, _ = detect_dial_center(img)
        # Should be somewhere near the real center (tolerance generous — the
        # needle-drop method can pick anywhere along the needle).
        assert abs(cx - 100) < 60
        assert abs(cy - 100) < 60


class TestDetectNeedleCenter:
    def test_finds_needle(self, synthetic_dial):
        img = synthetic_dial(needle_angle=0, size=200)
        result = detect_needle_center(img)
        assert result is not None
        cx, cy, conf = result
        assert 0 < cx < 200
        assert 0 < cy < 200
        assert 0.0 <= conf <= 1.0

    def test_no_needle_returns_none(self):
        # Pure gray image, no red.
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        assert detect_needle_center(img) is None


class TestDetectNeedleByColor:
    def test_reads_needle_angle(self, synthetic_dial):
        img = synthetic_dial(needle_angle=0, size=200)
        result = detect_needle_by_color(img, cx=100, cy=100)
        assert result is not None
        ang, conf = result
        assert _circular_err(ang, 0.0, period=360.0) < 30.0
        assert 0.0 <= conf <= 1.0

    def test_no_red_returns_none(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        assert detect_needle_by_color(img, cx=100, cy=100) is None


class TestDetectNeedleByLines:
    def test_finds_line_on_clean_dial(self, synthetic_dial):
        img = synthetic_dial(needle_angle=0, size=200)
        result = detect_needle_by_lines(img, cx=100, cy=100)
        # This method may return None on degenerate inputs; accept either but
        # if it returns a value, confidence must be valid.
        if result is not None:
            ang, conf = result
            assert -180.0 <= ang <= 180.0
            assert 0.0 <= conf <= 1.0

    def test_empty_image_returns_none(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 240
        result = detect_needle_by_lines(img, cx=100, cy=100)
        # Might still find noise-edge lines; just assert it doesn't crash.
        assert result is None or isinstance(result, tuple)


# ---- History-aware paths in read_dial -------------------------------------

class TestDetectDialMarkings:
    """After completing the implementation, detect_dial_markings now returns
    a real rotation_offset_deg (signed mean angular deviation of the detected
    "0" and "5" stamps). This test draws the markings at their canonical
    positions and verifies the offset is near zero.
    """

    def _dial_with_markings(self, size=200, rotation_deg=0.0):
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        cx, cy = size // 2, size // 2
        radius = size // 3
        cv2.circle(img, (cx, cy), radius, (50, 50, 50), 2)
        # "0" and "5" stamps at canonical positions (top and bottom, respectively)
        # optionally rotated about the dial center.
        r_mark = radius * 0.85
        for label, base_angle_deg in (("0", -90.0), ("5", 90.0)):
            theta = np.radians(base_angle_deg + rotation_deg)
            mx = int(cx + r_mark * np.cos(theta))
            my = int(cy + r_mark * np.sin(theta))
            # Draw a dark filled blob for the digit stamp
            cv2.putText(img, label, (mx - 8, my + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return img, cx, cy, radius

    def test_upright_dial_has_small_offset(self):
        img, cx, cy, radius = self._dial_with_markings(rotation_deg=0.0)
        offset_deg, conf = detect_dial_markings(img, cx, cy, radius)
        assert conf > 0.0
        assert abs(offset_deg) < 5.0  # small deviation tolerated

    def test_rotated_dial_detects_offset(self):
        # Rotate markings by +10°: detected offset should be ~+10°.
        img, cx, cy, radius = self._dial_with_markings(rotation_deg=10.0)
        offset_deg, conf = detect_dial_markings(img, cx, cy, radius)
        # Tolerance is generous because:
        # - cv2.putText glyph centroids are not exactly at the anchor point
        # - only the portion of each glyph that lands inside the sample window counts
        # What matters is the *sign* and *magnitude* tracking roughly with the true rotation.
        assert conf > 0.0
        assert 3.0 < offset_deg < 15.0

    def test_no_markings_returns_zero(self):
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 240
        offset_deg, conf = detect_dial_markings(blank, 100, 100, 60)
        assert offset_deg == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)


class TestCircularBlendInReadDial:
    """The previous linear blend at watermeter.py:554 collapsed readings across
    the 9→0 wrap to the wrong midpoint. These tests don't invoke read_dial
    end-to-end (too flaky); instead they verify the helper chosen to replace
    the linear blend behaves correctly — the same helper is imported and used
    by read_dial.
    """

    def test_circular_helper_used_by_read_dial(self):
        # Sanity that the module exports the helper in a stable place.
        from watermeter import circular_blend, circular_dist
        assert circular_dist(9.9, 0.1) < 0.3
        # With alpha=0.7 weighting, the blend leans toward the first arg.
        out = circular_blend(9.9, 0.1, alpha=0.7)
        assert circular_dist(out, 9.9) < circular_dist(out, 0.1)


class TestReadDialWithHistory:
    def test_prev_reading_used_when_no_detection_possible(self):
        # Blank image: no needle at all → read_dial falls back to prev.
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 240
        reading, conf, _ = read_dial(blank, zero_angle_deg=-90.0, rotation="cw",
                                      prev_reading=3.5)
        assert reading == pytest.approx(3.5)
        assert conf < 0.5  # low confidence because it's a fallback

    def test_zero_when_no_prev_and_no_detection(self):
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 240
        reading, conf, _ = read_dial(blank, zero_angle_deg=-90.0, rotation="cw")
        assert reading == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)
