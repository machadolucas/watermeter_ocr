"""Shared fixtures for the watermeter_ocr test suite."""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_synthetic_dial(size: int = 200, needle_angle: float = 45.0,
                         add_noise: bool = False, seed: int | None = 0) -> np.ndarray:
    """Render a synthetic dial image. Ported from the original test_dial_detection.py.

    needle_angle is in degrees (0 = right, 90 = down, since OpenCV's y-axis is flipped).
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    center = (size // 2, size // 2)
    radius = size // 3

    cv2.circle(img, center, radius, (50, 50, 50), 2)

    for i in range(10):
        a = np.radians(i * 36 - 90)
        x1 = int(center[0] + radius * 0.85 * np.cos(a))
        y1 = int(center[1] + radius * 0.85 * np.sin(a))
        x2 = int(center[0] + radius * 0.95 * np.cos(a))
        y2 = int(center[1] + radius * 0.95 * np.sin(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    needle_rad = np.radians(needle_angle)
    needle_length = radius * 0.8
    end_x = int(center[0] + needle_length * np.cos(needle_rad))
    end_y = int(center[1] + needle_length * np.sin(needle_rad))
    # Needle line
    cv2.line(img, center, (end_x, end_y), (0, 0, 200), 3)
    # Pivot "drop" at the dial center — thicker than the needle so the
    # distance-transform-based center detector has a clear global maximum
    # right at the pivot. Without this, max_loc can wander along the needle
    # and produce a 180°-ambiguous center (see detect_needle_center).
    cv2.circle(img, center, 7, (0, 0, 200), -1)

    if add_noise:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def synthetic_dial():
    """Factory fixture: call with (angle, size, add_noise, seed) to build a dial image."""
    return _make_synthetic_dial


@pytest.fixture
def minimal_config_yaml(tmp_path):
    """Smallest valid config — only esp32.base_url is set; everything else defaults."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n"
        "  base_url: \"http://192.0.2.1\"\n"
    )
    return p


@pytest.fixture
def full_config_yaml(tmp_path):
    """A copy of the repo's config.yaml placed in tmp_path so we can mutate it safely."""
    src = REPO_ROOT / "config.yaml"
    dst = tmp_path / "config.yaml"
    shutil.copyfile(src, dst)
    return dst


@pytest.fixture
def cfg(full_config_yaml):
    """Loaded Config dataclass from the repo's config.yaml."""
    from watermeter import load_config
    return load_config(str(full_config_yaml))


@pytest.fixture
def mock_ocr_bin(tmp_path):
    """Factory fixture: write an executable shell script at <tmp>/ocr_mock that
    produces a given stdout when invoked. Returns the absolute path.

    The script ignores all CLI arguments.
    """
    counter = {"n": 0}

    def _make(output: str = "5", exit_code: int = 0) -> str:
        counter["n"] += 1
        path = tmp_path / f"ocr_mock_{counter['n']}.sh"
        # Using a heredoc-free body so the shell doesn't try to expand user input.
        body = (
            "#!/usr/bin/env bash\n"
            f"echo {output!r}\n"
            f"exit {int(exit_code)}\n"
        )
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return str(path)

    return _make
