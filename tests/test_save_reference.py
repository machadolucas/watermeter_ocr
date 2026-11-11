"""save_reference.py: captures through the service's helper and checks every step."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import requests
import yaml

import save_reference
import watermeter


def _jpeg(value=128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, dtype=np.uint8))
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
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    """Records calls. get_statuses/bodies are consumed in order (last one repeats)."""

    def __init__(self, bodies=None, get_statuses=None):
        self.calls = []
        self.bodies = list(bodies or [_jpeg()])
        self.get_statuses = list(get_statuses or [200])

    def _next(self, seq):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def get(self, url, timeout=None, stream=False):
        self.calls.append(("GET", url))
        return _Resp(self._next(self.bodies), self._next(self.get_statuses))

    def post(self, url, timeout=None):
        self.calls.append(("POST", url))
        return _Resp()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(watermeter.time, "sleep", lambda s: None)


def _config(tmp_path, mode="esphome"):
    ref = tmp_path / "ref" / "reference.jpg"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "esp32": {"base_url": "http://192.0.2.1", "capture_mode": mode,
                  "flash_settle_sec": 0},
        "alignment": {"reference_path": str(ref)},
    }))
    return p, ref


def _seed_old_reference(ref):
    ref.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref), np.full((10, 10, 3), 7, np.uint8))
    return ref.read_bytes()


def test_esphome_flash_on_discard_snapshot_flash_off(tmp_path):
    cfg, ref = _config(tmp_path)
    s = _Session()
    assert save_reference.main([str(cfg)], session=s) == 0
    assert s.calls == [
        ("POST", "http://192.0.2.1/light/Flash/turn_on?r=255&g=255&b=255"),
        ("GET", "http://192.0.2.1:8080/"),      # discarded pre-flash frame
        ("GET", "http://192.0.2.1:8080/"),
        ("POST", "http://192.0.2.1/light/Flash/turn_off"),
    ]
    img = cv2.imread(str(ref))
    assert img is not None and img.shape == (48, 64, 3)


def test_ai_on_the_edge_endpoint(tmp_path):
    cfg, ref = _config(tmp_path, mode="ai_on_the_edge")
    s = _Session()
    assert save_reference.main([str(cfg)], session=s) == 0
    assert s.calls == [("GET", "http://192.0.2.1/capture_with_flashlight")]
    assert ref.exists()


def test_snapshot_failure_still_turns_flash_off_and_keeps_old_reference(tmp_path, capsys):
    cfg, ref = _config(tmp_path)
    old = _seed_old_reference(ref)
    s = _Session(get_statuses=[200, 503])
    assert save_reference.main([str(cfg)], session=s) == 1
    assert s.calls[-1] == ("POST", "http://192.0.2.1/light/Flash/turn_off")
    assert ref.read_bytes() == old
    assert "reference was not changed" in capsys.readouterr().err


def test_empty_body_fails(tmp_path, capsys):
    cfg, ref = _config(tmp_path, mode="ai_on_the_edge")
    assert save_reference.main([str(cfg)], session=_Session(bodies=[b""])) == 1
    assert not ref.exists()
    assert "empty image body" in capsys.readouterr().err


def test_undecodable_body_fails(tmp_path, capsys):
    cfg, ref = _config(tmp_path, mode="ai_on_the_edge")
    old = _seed_old_reference(ref)
    assert save_reference.main([str(cfg)], session=_Session(bodies=[b"garbage"])) == 1
    assert ref.read_bytes() == old
    assert "do not decode" in capsys.readouterr().err


def test_oversized_body_fails(tmp_path):
    cfg, ref = _config(tmp_path, mode="ai_on_the_edge")
    doc = yaml.safe_load(cfg.read_text())
    doc["processing"] = {"max_image_bytes": 10}
    cfg.write_text(yaml.safe_dump(doc))
    assert save_reference.main([str(cfg)], session=_Session()) == 1
    assert not ref.exists()


def test_write_failure_is_reported_and_old_reference_kept(tmp_path, monkeypatch, capsys):
    cfg, ref = _config(tmp_path, mode="ai_on_the_edge")
    old = _seed_old_reference(ref)
    monkeypatch.setattr(save_reference.cv2, "imwrite", lambda *a, **k: False)
    assert save_reference.main([str(cfg)], session=_Session()) == 1
    assert ref.read_bytes() == old
    assert list(ref.parent.glob("*.tmp.jpg")) == []
    assert "could not write" in capsys.readouterr().err


def test_success_keeps_a_backup_of_the_previous_reference(tmp_path, capsys):
    cfg, ref = _config(tmp_path, mode="ai_on_the_edge")
    old = _seed_old_reference(ref)
    assert save_reference.main([str(cfg)], session=_Session()) == 0
    backups = list(ref.parent.glob("reference.jpg.bak.*"))
    assert len(backups) == 1 and backups[0].read_bytes() == old
    assert cv2.imread(str(ref)).shape == (48, 64, 3)
    assert "Previous reference kept" in capsys.readouterr().out


def test_invalid_config_exits_78(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("esp32:\n  capture_mode: rtsp\n")
    assert save_reference.main([str(p)], session=_Session()) == 78
