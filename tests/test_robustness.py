"""Tests for robustness fixes: aligner short pairs, flow predicate, logging, QoS, state loading."""
from __future__ import annotations

import json
import logging
import logging.handlers
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import watermeter
from watermeter import AlignConfig, Aligner, Config, MqttClient, flow_tracked, load_config, load_state, setup_logging


def test_aligner_tolerates_short_knn_pairs(monkeypatch):
    al = Aligner(AlignConfig(min_matches=10, ratio_test=0.75), logging.getLogger("t"))
    al.ref_img = np.zeros((20, 20, 3), np.uint8)  # skip reference loading
    al.orb = MagicMock()
    al.orb.detectAndCompute.return_value = ([object()] * 20, np.zeros((20, 32), np.uint8))
    al.descR = np.zeros((20, 32), np.uint8)
    al.kpR = [object()] * 20
    al.mask = None
    m = SimpleNamespace(distance=1.0, queryIdx=0, trainIdx=0)
    n = SimpleNamespace(distance=100.0, queryIdx=0, trainIdx=0)
    bf = MagicMock()
    bf.knnMatch.return_value = [(m,), (), (m, n)]
    monkeypatch.setattr(cv2, "BFMatcher", lambda *a, **k: bf)
    img = np.zeros((20, 20, 3), np.uint8)
    out, M, ok = al.align(img)  # must not raise ValueError
    assert ok is False and M is None  # only 1 good match < min_matches


def test_flow_tracked_predicate():
    assert not flow_tracked(Config(esp32_base_url="x"))
    assert flow_tracked(Config(esp32_base_url="x", digital_flow_roi=[0, 0, 1, 1]))
    assert flow_tracked(Config(esp32_base_url="x", digital_flow_digits=[[0, 0, 1, 1]]))


def _client(**kw):
    cfg = Config(esp32_base_url="http://x", meter_type="digital", **kw)
    with patch("watermeter.mqtt.Client") as C:
        C.return_value = MagicMock()
        c = MqttClient(cfg, logging.getLogger("t"))
    c.connected = True
    return c


def test_discovery_publishes_flow_with_only_flow_digits_and_qos1():
    c = _client(digital_flow_digits=[[0.1, 0.1, 0.1, 0.1]])
    c.client.publish.reset_mock()
    c.discovery()
    topics = [call.args[0] for call in c.client.publish.call_args_list]
    assert any(t.endswith("water_flow_m3h/config") for t in topics)
    assert all(call.kwargs.get("qos") == 1 for call in c.client.publish.call_args_list)


def test_discovery_no_flow_when_nothing_configured():
    c = _client()
    c.client.publish.reset_mock()
    c.discovery()
    assert not any(call.args[0].endswith("water_flow_m3h/config") for call in c.client.publish.call_args_list)


def test_publish_default_qos0_and_passthrough():
    c = _client()
    c.client.publish.reset_mock()
    c.publish("t", "p")
    c.publish("t", "p", retain=True, qos=1)
    assert c.client.publish.call_args_list[0].kwargs == {"retain": False, "qos": 0}
    assert c.client.publish.call_args_list[1].kwargs == {"retain": True, "qos": 1}


def test_setup_logging_attaches_rotating_handler(tmp_path):
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    h = None
    try:
        h = setup_logging(str(tmp_path / "w.log"), 1234, 7)
        assert isinstance(h, logging.handlers.RotatingFileHandler)
        assert h in root.handlers
        assert h.maxBytes == 1234 and h.backupCount == 7
        assert root.level == logging.INFO
    finally:
        if h:
            root.removeHandler(h)
            h.close()
        root.handlers[:] = before
        root.setLevel(level)


def test_log_rotation_config_defaults_and_yaml(tmp_path):
    c = Config(esp32_base_url="x")
    assert (c.log_max_bytes, c.log_backup_count, c.max_image_bytes) == (10_000_000, 5, 2_000_000)
    p = tmp_path / "c.yaml"
    p.write_text('esp32:\n  base_url: "http://x"\npaths:\n  log_max_bytes: 5\n  log_backup_count: 2\n'
                 'processing:\n  max_image_bytes: 99\n')
    cfg = load_config(str(p))
    assert (cfg.log_max_bytes, cfg.log_backup_count, cfg.max_image_bytes) == (5, 2, 99)


def test_quiet_hours_dataclass_default_false():
    assert Config(esp32_base_url="x").quiet_hours_enabled is False


# --- load_state ---------------------------------------------------------

LOG = logging.getLogger("state-test")
DEFAULTS = {"total": None, "ts": None, "meter_type": None, "dial_histories": {}}


def _write(tmp_path, content):
    p = tmp_path / "state.json"
    p.write_text(content if isinstance(content, str) else json.dumps(content))
    return str(p)


def test_load_state_missing_file(tmp_path):
    assert load_state(str(tmp_path / "nope.json"), LOG) == DEFAULTS


@pytest.mark.parametrize("content", ["[1,2]", "42", "not json", '"str"'])
def test_load_state_non_dict_or_garbage(tmp_path, content):
    assert load_state(_write(tmp_path, content), LOG) == DEFAULTS


@pytest.mark.parametrize("bad", ['"12.5"', "NaN", "Infinity", "-1", "true", "[1]"])
def test_load_state_rejects_bad_total(tmp_path, bad, caplog):
    path = _write(tmp_path, '{"total": %s, "ts": 100.0, "meter_type": "digital"}' % bad)
    with caplog.at_level(logging.WARNING):
        st = load_state(path, LOG)
    assert st["total"] is None and st["ts"] is None
    assert st["meter_type"] == "digital"
    assert "total" in caplog.text


def test_load_state_valid(tmp_path):
    st = load_state(_write(tmp_path, {"total": 12.5, "ts": 1700000000.5, "meter_type": "digital",
                                      "dial_histories": {"dial_0": [1]}}), LOG)
    assert st == {"total": 12.5, "ts": 1700000000.5, "meter_type": "digital",
                  "dial_histories": {"dial_0": [1]}}


@pytest.mark.parametrize("bad", ['"x"', "NaN", "null"])
def test_load_state_rejects_bad_ts(tmp_path, bad):
    st = load_state(_write(tmp_path, '{"total": 3, "ts": %s}' % bad), LOG)
    assert st["total"] == 3 and st["ts"] is None


@pytest.mark.parametrize("bad", ["[1]", '"x"', "null", "5"])
def test_load_state_rejects_bad_dial_histories(tmp_path, bad):
    st = load_state(_write(tmp_path, '{"total": 3, "dial_histories": %s}' % bad), LOG)
    assert st["dial_histories"] == {}
