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

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, get_status=200, body=None, post_statuses=None):
        self.calls = []
        self.get_status = get_status
        self.body = _jpeg() if body is None else body
        self.post_statuses = list(post_statuses or [])

    def get(self, url, timeout=None, stream=False):
        self.calls.append(("GET", url))
        return _Resp(self.body, self.get_status)

    def post(self, url, timeout=None):
        self.calls.append(("POST", url))
        return _Resp(status=self.post_statuses.pop(0) if self.post_statuses else 200)


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
        ("POST", "http://192.0.2.1/light/Flash/turn_on?r=255&g=255&b=255"),
        ("GET", "http://192.0.2.1:8080/"),  # discarded: may predate the flash
        ("GET", "http://192.0.2.1:8080/"),
        ("POST", "http://192.0.2.1/light/Flash/turn_off"),
    ]


def test_esphome_mode_turns_flash_off_when_snapshot_fails(tmp_path):
    s = _Session(get_status=500)
    frame = capture_frame(_cfg(tmp_path, esp32_capture_mode="esphome"), s, logging.getLogger("t"))
    assert frame is None
    assert s.calls[-1] == ("POST", "http://192.0.2.1/light/Flash/turn_off")


def test_empty_body_is_capture_failure(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        frame = capture_frame(_cfg(tmp_path), _Session(body=b""), logging.getLogger("t"))
    assert frame is None
    assert "Capture failed" in caplog.text


def test_undecodable_body_is_capture_failure(tmp_path):
    assert capture_frame(_cfg(tmp_path), _Session(body=b"not a jpeg"), logging.getLogger("t")) is None


def test_oversized_body_is_capture_failure(tmp_path, caplog):
    cfg = _cfg(tmp_path, max_image_bytes=100)
    with caplog.at_level(logging.WARNING):
        assert capture_frame(cfg, _Session(), logging.getLogger("t")) is None
    assert "Capture failed" in caplog.text


def test_oversized_body_esphome_is_capture_failure(tmp_path):
    cfg = _cfg(tmp_path, esp32_capture_mode="esphome", max_image_bytes=100)
    assert capture_frame(cfg, _Session(), logging.getLogger("t")) is None


def test_valid_jpeg_returns_ndarray_and_writes_file(tmp_path):
    cfg = _cfg(tmp_path)
    frame = capture_frame(cfg, _Session(), logging.getLogger("t"))
    assert isinstance(frame, np.ndarray)
    assert (tmp_path / "raw.jpg").read_bytes() == _jpeg()


def test_flash_off_http_error_retries_logs_and_still_returns_image(tmp_path, caplog):
    s = _Session(post_statuses=[200, 500, 500])
    with caplog.at_level(logging.WARNING):
        frame = capture_frame(_cfg(tmp_path, esp32_capture_mode="esphome"), s, logging.getLogger("t"))
    assert frame is not None
    assert "Flash off failed" in caplog.text
    assert [c for c in s.calls if c[0] == "POST"].count(("POST", "http://192.0.2.1/light/Flash/turn_off")) == 2


def test_flash_off_success_needs_no_retry(tmp_path, caplog):
    s = _Session()
    with caplog.at_level(logging.WARNING):
        capture_frame(_cfg(tmp_path, esp32_capture_mode="esphome"), s, logging.getLogger("t"))
    assert "Flash off failed" not in caplog.text
    assert s.calls.count(("POST", "http://192.0.2.1/light/Flash/turn_off")) == 1


def test_flash_off_retry_can_succeed(tmp_path, caplog):
    s = _Session(post_statuses=[200, 500, 200])
    with caplog.at_level(logging.WARNING):
        assert capture_frame(_cfg(tmp_path, esp32_capture_mode="esphome"), s, logging.getLogger("t")) is not None
    assert "Flash off failed" not in caplog.text
