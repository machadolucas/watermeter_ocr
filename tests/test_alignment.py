"""Tests for the Aligner (ORB+RANSAC similarity-transform alignment).

Uses synthetic images rich in features (text, rectangles, circles) so ORB
has something to latch onto. Alignment tolerances are generous: ORB is not
deterministic across OpenCV versions for degenerate match sets.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from watermeter import AlignConfig, Aligner


def _featureful_image(width: int = 400, height: int = 300) -> np.ndarray:
    """A high-contrast textured image with plenty of corner/edge features."""
    rng = np.random.default_rng(42)
    # Random noise background so ORB has texture everywhere.
    img = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    # Sharp geometric features.
    cv2.rectangle(img, (50, 40), (150, 100), (255, 255, 255), 3)
    cv2.rectangle(img, (220, 180), (360, 260), (0, 0, 255), 3)
    cv2.circle(img, (100, 200), 30, (0, 255, 0), 3)
    cv2.line(img, (10, 10), (width - 10, height - 10), (255, 0, 0), 2)
    cv2.line(img, (width - 10, 10), (10, height - 10), (0, 255, 255), 2)
    # Text gives ORB rich keypoints.
    cv2.putText(img, "WATER METER", (60, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    cv2.putText(img, "01234", (180, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2)
    return img


def _make_align_cfg(tmp_path: Path, **overrides) -> AlignConfig:
    ac = AlignConfig(
        enabled=True,
        reference_path=str(tmp_path / "reference.jpg"),
        use_mask=False,      # tests don't supply anchor ROIs
        nfeatures=2000,
        ratio_test=0.75,
        min_matches=10,      # loosened for small synthetic images
        ransac_thresh_px=3.0,
        max_scale_change=0.2,
        max_rotation_deg=20.0,
        warp_mode="similarity",
        write_debug_aligned=False,
    )
    for k, v in overrides.items():
        setattr(ac, k, v)
    return ac


@pytest.fixture
def logger():
    return logging.getLogger("test-aligner")


# ---- Reference creation ---------------------------------------------------

class TestReferenceCreation:
    def test_first_frame_creates_reference_when_missing(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path)
        ref_path = Path(cfg.reference_path)
        assert not ref_path.exists()

        aligner = Aligner(cfg, logger)
        img = _featureful_image()
        ok = aligner.ensure_reference(img)

        assert ok is True
        assert ref_path.exists()

    def test_existing_reference_is_loaded(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path)
        img = _featureful_image()
        cv2.imwrite(cfg.reference_path, img)

        aligner = Aligner(cfg, logger)
        assert aligner.ref_img is None
        ok = aligner.ensure_reference(np.zeros((10, 10, 3), dtype=np.uint8))  # arg ignored

        assert ok is True
        assert aligner.ref_img is not None
        assert aligner.ref_img.shape == img.shape


# ---- Alignment transforms -------------------------------------------------

class TestAlignmentTransforms:
    def test_identity_alignment(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path)
        aligner = Aligner(cfg, logger)
        img = _featureful_image()
        aligner.ensure_reference(img)

        aligned, M, ok = aligner.align(img.copy())
        assert ok is True
        assert M is not None
        # Scale ~ 1.0, rotation ~ 0.
        a, b = M[0, 0], M[0, 1]
        scale = math.sqrt(a * a + b * b)
        rot = math.degrees(math.atan2(b, a))
        assert scale == pytest.approx(1.0, abs=0.02)
        assert abs(rot) < 1.0

    def test_translated_copy_is_recovered(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path)
        aligner = Aligner(cfg, logger)
        ref = _featureful_image()
        aligner.ensure_reference(ref)

        # Translate the reference by (dx=10, dy=-5).
        dx, dy = 10, -5
        H, W = ref.shape[:2]
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(ref, T, (W, H))

        aligned, M, ok = aligner.align(shifted)
        assert ok is True
        # The recovered transform warps shifted → ref, so it must apply approximately
        # the INVERSE translation (-dx, -dy).
        assert M is not None
        assert M[0, 2] == pytest.approx(-dx, abs=2.0)
        assert M[1, 2] == pytest.approx(-dy, abs=2.0)

    def test_rotated_copy_is_recovered(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path)
        aligner = Aligner(cfg, logger)
        ref = _featureful_image()
        aligner.ensure_reference(ref)

        # Rotate reference by +5° around the center.
        H, W = ref.shape[:2]
        rot_deg = 5.0
        R = cv2.getRotationMatrix2D((W / 2, H / 2), rot_deg, 1.0)
        rotated = cv2.warpAffine(ref, R, (W, H))

        aligned, M, ok = aligner.align(rotated)
        assert ok is True
        assert M is not None
        a, b = M[0, 0], M[0, 1]
        recovered_rot = math.degrees(math.atan2(b, a))
        # The recovered transform should rotate BACK by ~5°.
        assert abs(recovered_rot + rot_deg) < 2.0


# ---- Failure modes --------------------------------------------------------

class TestAlignmentFailures:
    def test_solid_color_image_fails_alignment(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path)
        aligner = Aligner(cfg, logger)
        ref = _featureful_image()
        aligner.ensure_reference(ref)

        blank = np.ones_like(ref) * 128  # no features at all
        aligned, M, ok = aligner.align(blank)
        assert ok is False
        assert M is None
        # On failure the input image is returned unchanged.
        assert aligned is blank or np.array_equal(aligned, blank)

    def test_too_few_matches_fails(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path, min_matches=10000)
        aligner = Aligner(cfg, logger)
        ref = _featureful_image()
        aligner.ensure_reference(ref)

        aligned, M, ok = aligner.align(ref.copy())
        assert ok is False
        assert M is None

    def test_excessive_scale_rejected(self, tmp_path, logger):
        cfg = _make_align_cfg(tmp_path, max_scale_change=0.01)
        aligner = Aligner(cfg, logger)
        ref = _featureful_image()
        aligner.ensure_reference(ref)

        # Scale the image by 1.3x — well outside max_scale_change=0.01.
        H, W = ref.shape[:2]
        S = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1.3)
        scaled = cv2.warpAffine(ref, S, (W, H))

        aligned, M, ok = aligner.align(scaled)
        # The recovered transform could be found but then rejected.
        assert ok is False
