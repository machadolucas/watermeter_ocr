"""Tests for the ESP32 capture modes (AI-on-the-edge vs ESPHome)."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

import watermeter
from watermeter import Config, capture_frame


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _Resp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, get_status=200):
        self.calls = []
        self.get_status = get_status

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        return _Resp(_jpeg(), self.get_status)

    def post(self, url, timeout=None):
        self.calls.append(("POST", url))
        return _Resp()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(watermeter.time, "sleep", lambda s: None)


def _cfg(tmp_path, **kw):
    return Config(esp32_base_url="http://192.0.2.1", image_path=str(tmp_path / "raw.jpg"), **kw)


def test_default_mode_uses_capture_with_flashlight(tmp_path):
    s = _Session()
    frame = capture_frame(_cfg(tmp_path), s, logging.getLogger("t"))
    assert frame is not None
    assert s.calls == [("GET", "http://192.0.2.1/capture_with_flashlight")]


def test_esphome_mode_flashes_around_snapshot(tmp_path):
    s = _Session()
    frame = capture_frame(_cfg(tmp_path, esp32_capture_mode="esphome"), s, logging.getLogger("t"))
    assert frame is not None and frame.shape[:2] == (48, 64)
    assert s.calls == [
        ("POST", "http://192.0.2.1/light/Flash/turn_on?brightness=102&r=255&g=255&b=255"),
        ("GET", "http://192.0.2.1:8080/"),
        ("POST", "http://192.0.2.1/light/Flash/turn_off"),
    ]


def test_esphome_mode_turns_flash_off_when_snapshot_fails(tmp_path):
    s = _Session(get_status=500)
    frame = capture_frame(_cfg(tmp_path, esp32_capture_mode="esphome"), s, logging.getLogger("t"))
    assert frame is None
    assert s.calls[-1] == ("POST", "http://192.0.2.1/light/Flash/turn_off")
