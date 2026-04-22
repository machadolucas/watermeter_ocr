"""Tests for load_config."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from watermeter import Config, DialConfig, load_config


def test_minimal_yaml_loads_with_defaults(minimal_config_yaml):
    cfg = load_config(str(minimal_config_yaml))
    assert cfg.esp32_base_url == "http://192.0.2.1"
    assert cfg.interval_sec == 10
    assert cfg.retry_backoff_sec == 5
    assert cfg.big_jump_guard == pytest.approx(2.0)
    assert cfg.dials == []
    assert cfg.align.enabled is True
    assert cfg.mqtt_main_topic == "home/watermeter"


def test_full_yaml_loads_dials(full_config_yaml):
    cfg = load_config(str(full_config_yaml))
    assert len(cfg.dials) == 4

    # Dial order is load-bearing: dials[0] must be the tenths wheel.
    assert cfg.dials[0].name == "dial_0_1"
    assert cfg.dials[0].factor == pytest.approx(0.1)
    assert cfg.dials[-1].factor == pytest.approx(0.0001)

    for d in cfg.dials:
        assert isinstance(d, DialConfig)
        assert d.rotation in ("cw", "ccw")
        assert len(d.roi) == 4
        assert all(0.0 <= v <= 1.0 for v in d.roi)


def test_full_yaml_loads_alignment(full_config_yaml):
    cfg = load_config(str(full_config_yaml))
    assert cfg.align.enabled is True
    assert cfg.align.nfeatures == 1200
    assert cfg.align.warp_mode == "similarity"
    assert len(cfg.align.anchor_rois) == 3


def test_full_yaml_loads_overlay_topic_from_mqtt_default(full_config_yaml):
    cfg = load_config(str(full_config_yaml))
    # The repo config.yaml sets an explicit camera_topic; verify it's honored.
    assert cfg.overlay_camera_topic == "home/watermeter/debug/overlay"


def test_overlay_topic_defaults_when_missing(tmp_path):
    """When overlay.camera_topic is omitted, it must default to {mqtt.topic}/debug/overlay."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n"
        "  base_url: \"http://localhost\"\n"
        "mqtt:\n"
        "  topic: \"some/custom/topic\"\n"
    )
    cfg = load_config(str(p))
    assert cfg.overlay_camera_topic == "some/custom/topic/debug/overlay"


def test_digits_count_respected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n  base_url: \"http://localhost\"\n"
        "digits:\n  count: 7\n"
    )
    cfg = load_config(str(p))
    assert cfg.digits_count == 7


def test_paths_get_expanduser(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n  base_url: \"http://localhost\"\n"
        "paths:\n"
        "  state_path: \"~/foo_state.json\"\n"
        "  log_path: \"~/foo.log\"\n"
        "  ocr_bin: \"~/foo_bin/ocr\"\n"
    )
    cfg = load_config(str(p))
    home = os.path.expanduser("~")
    assert cfg.state_path == os.path.join(home, "foo_state.json")
    assert cfg.log_path == os.path.join(home, "foo.log")
    assert cfg.ocr_bin == os.path.join(home, "foo_bin/ocr")


def test_alignment_reference_path_expanduser(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n  base_url: \"http://localhost\"\n"
        "alignment:\n"
        "  reference_path: \"~/ref.jpg\"\n"
    )
    cfg = load_config(str(p))
    assert cfg.align.reference_path == os.path.expanduser("~/ref.jpg")


def test_quiet_hours_defaults_to_disabled_when_missing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("esp32:\n  base_url: \"http://localhost\"\n")
    cfg = load_config(str(p))
    assert cfg.quiet_hours_enabled is False


def test_quiet_hours_loaded_from_full_config(full_config_yaml):
    cfg = load_config(str(full_config_yaml))
    assert cfg.quiet_hours_enabled is True
    assert cfg.quiet_start == "00:00"
    assert cfg.quiet_end == "07:00"
    assert cfg.quiet_interval_sec == 60


def test_auto_centering_loaded(full_config_yaml):
    cfg = load_config(str(full_config_yaml))
    assert cfg.auto_center_dials is True
    assert cfg.center_smoothing_alpha == pytest.approx(0.3)
    assert cfg.min_confidence_threshold == pytest.approx(0.4)


def test_meter_type_defaults_to_mechanical(minimal_config_yaml):
    cfg = load_config(str(minimal_config_yaml))
    assert cfg.meter_type == "mechanical"
    assert cfg.digital_total_roi == []
    assert cfg.digital_flow_roi == []
    # Defaults must still compile and accept canonical formats.
    assert cfg.digital_total_regex.match("000100.000")
    assert cfg.digital_total_regex.match("000100000")
    assert cfg.digital_flow_regex.match("00.000")
    assert cfg.digital_max_retries == 2
    assert cfg.digital_retry_delay_sec == pytest.approx(5.5)
    assert cfg.digital_min_digits == 6


def test_digital_meter_config_loads(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n  base_url: \"http://localhost\"\n"
        "meter:\n  type: digital\n"
        "processing:\n  interval_sec: 15\n"
        "rois:\n"
        "  digital:\n"
        "    total: [0.10, 0.25, 0.80, 0.18]\n"
        "    flow:  [0.25, 0.50, 0.55, 0.15]\n"
        "digital:\n"
        "  total_regex: \"^\\\\d{6}\\\\.?\\\\d{3}$\"\n"
        "  flow_regex:  \"^\\\\d{1,3}\\\\.\\\\d{3}$\"\n"
        "  max_retries: 3\n"
        "  retry_delay_sec: 6.0\n"
        "  min_digits: 5\n"
    )
    cfg = load_config(str(p))
    assert cfg.meter_type == "digital"
    assert cfg.interval_sec == 15
    assert cfg.digital_total_roi == [0.10, 0.25, 0.80, 0.18]
    assert cfg.digital_flow_roi == [0.25, 0.50, 0.55, 0.15]
    assert cfg.digital_max_retries == 3
    assert cfg.digital_retry_delay_sec == pytest.approx(6.0)
    assert cfg.digital_min_digits == 5
    # Regex accepts both dotted and undotted totals (Vision may or may not find the decimal).
    assert cfg.digital_total_regex.match("000100.000")
    assert cfg.digital_total_regex.match("000100000")
    assert cfg.digital_flow_regex.match("12.345")
