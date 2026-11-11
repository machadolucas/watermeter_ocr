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


# ---------------------------------------------------------------------------
# Validation (ConfigError) — the 2026-11-04 audit follow-up
# ---------------------------------------------------------------------------

import yaml as _yaml

import watermeter as _W
from watermeter import ConfigError, digit_count_problem, valid_total_digit_counts


def _write(tmp_path, doc):
    p = tmp_path / "config.yaml"
    p.write_text(_yaml.safe_dump(doc, sort_keys=False))
    return p


def _problems(tmp_path, doc):
    with pytest.raises(ConfigError) as ei:
        load_config(str(_write(tmp_path, doc)))
    return ei.value.problems


def _cells(n, x0=0.05, w=0.05):
    return [[round(x0 + i * (w + 0.01), 4), 0.3, w, 0.2] for i in range(n)]


def _digital(int_n=6, frac_n=3, **extra):
    doc = {
        "esp32": {"base_url": "http://192.0.2.1", "capture_mode": "esphome"},
        "meter": {"type": "digital"},
        "rois": {"digital": {"total_int_digits": _cells(int_n),
                             "total_frac_digits": _cells(frac_n, x0=0.5)}},
    }
    for k, v in extra.items():
        doc[k] = v
    return doc


class TestValidation:
    def test_live_shape_digital_config_loads(self, tmp_path):
        cfg = load_config(str(_write(tmp_path, _digital())))
        assert cfg.meter_type == "digital"
        assert len(cfg.digital_total_int_digits) == 6

    def test_repo_config_and_minimal_still_load(self, full_config_yaml, minimal_config_yaml):
        load_config(str(full_config_yaml))
        load_config(str(minimal_config_yaml))

    def test_empty_file_and_empty_sections_load(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("")
        load_config(str(p))
        p.write_text("esp32:\n  base_url: http://192.0.2.1\ndigital:\nprocessing:\n")
        load_config(str(p))

    @pytest.mark.parametrize("section,key,value", [
        ("meter", "type", "analog"),
        ("esp32", "capture_mode", "rtsp"),
        ("digital", "ocr_preprocess", "sharpen"),
        ("alignment", "warp_mode", "homography"),
    ])
    def test_bad_enum(self, tmp_path, section, key, value):
        doc = {"esp32": {"base_url": "http://192.0.2.1"}}
        doc.setdefault(section, {})[key] = value
        probs = _problems(tmp_path, doc)
        assert any(p.startswith(f"{section}.{key}: must be one of") and repr(value) in p
                   for p in probs), probs

    def test_bad_dial_rotation(self, tmp_path, full_config_yaml):
        doc = _yaml.safe_load(full_config_yaml.read_text())
        doc["rois"]["dials"][2]["rotation"] = "left"
        probs = _problems(tmp_path, doc)
        assert any(p.startswith("rois.dials[2].rotation") for p in probs)

    @pytest.mark.parametrize("section,key,value,fragment", [
        ("processing", "interval_sec", 0, "must be > 0"),
        ("processing", "image_timeout", -1, "must be > 0"),
        ("mqtt", "port", 70000, "must be <= 65535"),
        ("mqtt", "port", "1883", "must be an integer"),
        ("digital", "max_retries", -1, "must be >= 0"),
        ("digital", "ocr_upscale_factor", 0, "must be >= 1"),
        ("overlay", "jpeg_quality", 0, "must be >= 1"),
        ("alignment", "ratio_test", 1.5, "must be <= 1"),
        ("postproc", "rebaseline_count", 0, "must be >= 1"),
        ("esp32", "flash_settle_sec", -0.5, "must be >= 0"),
    ])
    def test_bad_range(self, tmp_path, section, key, value, fragment):
        doc = {"esp32": {"base_url": "http://192.0.2.1"}}
        doc.setdefault(section, {})[key] = value
        probs = _problems(tmp_path, doc)
        assert any(p.startswith(f"{section}.{key}: {fragment}") for p in probs), probs

    def test_boolean_string_rejected(self, tmp_path):
        probs = _problems(tmp_path, {"alignment": {"enabled": "false"}})
        assert any(p.startswith("alignment.enabled: must be true or false") for p in probs)

    def test_unquoted_quiet_time_rejected(self, tmp_path):
        p = tmp_path / "config.yaml"
        # YAML 1.1 reads an unquoted 17:30 as the base-60 integer 1050.
        p.write_text("processing:\n  quiet_hours:\n    start: 17:30\n")
        with pytest.raises(ConfigError) as ei:
            load_config(str(p))
        assert "HH:MM" in str(ei.value)

    def test_thresholds_ordered(self, tmp_path):
        probs = _problems(tmp_path, {"digits": {"rolling_threshold_up": 0.3,
                                                "rolling_threshold_down": 0.5}})
        assert any("must be below rolling_threshold_up" in p for p in probs)

    def test_confirm_above_le_max_flow(self, tmp_path):
        probs = _problems(tmp_path, {"postproc": {"max_flow_lpm": 20, "confirm_above_lpm": 25}})
        assert any(p.startswith("postproc.confirm_above_lpm") for p in probs)

    def test_problems_are_aggregated(self, tmp_path):
        probs = _problems(tmp_path, {"meter": {"type": "x"}, "mqtt": {"port": 0},
                                     "processing": {"interval_sec": -1}})
        assert len(probs) == 3

    def test_unknown_key_with_suggestion(self, tmp_path):
        probs = _problems(tmp_path, {"processing": {"intervall_sec": 20}})
        assert probs == ["processing.intervall_sec: unknown key; did you mean "
                         "'processing.interval_sec'?"]

    def test_unknown_section(self, tmp_path):
        probs = _problems(tmp_path, {"mqttt": {"host": "x"}})
        assert "did you mean 'mqtt'" in probs[0]

    def test_unknown_dial_key(self, tmp_path, full_config_yaml):
        doc = _yaml.safe_load(full_config_yaml.read_text())
        doc["rois"]["dials"][0]["zero_angle"] = -90
        probs = _problems(tmp_path, doc)
        assert any(p.startswith("rois.dials[0].zero_angle: unknown key") for p in probs)

    def test_bad_regex(self, tmp_path):
        probs = _problems(tmp_path, _digital(digital={"total_regex": "^(\\d{6}"}))
        assert any(p.startswith("digital.total_regex: is not a valid regular expression")
                   for p in probs)

    def test_roi_out_of_frame(self, tmp_path):
        probs = _problems(tmp_path, {"rois": {"digits": [0.6, 0.1, 0.5, 0.1]}})
        assert any(p.startswith("rois.digits: extends past the frame") for p in probs)

    def test_mqtt_password_type_error_never_echoes_value(self, tmp_path):
        probs = _problems(tmp_path, {"mqtt": {"password": 123456}})
        assert probs == ["mqtt.password: must be a string or null"]

    def test_digital_without_total(self, tmp_path):
        probs = _problems(tmp_path, {"meter": {"type": "digital"}})
        assert any("needs rois.digital.total" in p for p in probs)

    def test_digital_lone_int_side(self, tmp_path):
        doc = _digital()
        del doc["rois"]["digital"]["total_frac_digits"]
        doc["rois"]["digital"]["total"] = [0.1, 0.1, 0.8, 0.2]
        probs = _problems(tmp_path, doc)
        assert any("total_int is set but total_frac is not" in p for p in probs)

    def test_digital_four_frac_cells_rejected(self, tmp_path):
        probs = _problems(tmp_path, _digital(frac_n=4))
        assert probs == ["rois.digital: needs 6 integer + 3 fractional cells for "
                         "total_regex '^\\\\d{6}\\\\.?\\\\d{3}$', got 6 + 4"]

    def test_digital_digit_count_checked_too(self, tmp_path):
        doc = _digital()
        doc["rois"]["digital"] = {"total_int": [0.1, 0.1, 0.4, 0.2],
                                  "total_frac": [0.5, 0.1, 0.3, 0.2]}
        doc["digital"] = {"total_int_digit_count": 5, "total_frac_digit_count": 3}
        probs = _problems(tmp_path, doc)
        assert any("got 5 + 3" in p for p in probs)

    def test_digital_custom_regex_changes_expected_counts(self, tmp_path):
        doc = _digital(int_n=8, frac_n=2, digital={"total_regex": "^\\d{8}\\.\\d{2}$"})
        load_config(str(_write(tmp_path, doc)))

    def test_holed_digit_list(self, tmp_path):
        doc = _digital()
        doc["rois"]["digital"]["total_int_digits"][2] = None
        probs = _problems(tmp_path, doc)
        assert any(p.startswith("rois.digital.total_int_digits[2]: cell is empty") for p in probs)

    def test_flow_digits_count(self, tmp_path):
        doc = _digital()
        doc["rois"]["digital"]["flow_digits"] = _cells(3, x0=0.1)
        probs = _problems(tmp_path, doc)
        assert any(p.startswith("rois.digital.flow_digits: 3 flow cells") for p in probs)

    def test_view_sync_dwells_must_sum_to_period(self, tmp_path):
        doc = _digital(digital={"view_sync": {"enabled": True, "period_sec": 20.0,
                                              "numeric_dwell_sec": 11.0,
                                              "view_dwells_sec": [11.0, 3.9, 4.0]}})
        probs = _problems(tmp_path, doc)
        assert any(p.startswith("digital.view_sync: view dwells") for p in probs)

    def test_mechanical_dial0_must_be_tenths(self, tmp_path, full_config_yaml):
        doc = _yaml.safe_load(full_config_yaml.read_text())
        doc["rois"]["dials"][0]["factor"] = 0.01
        probs = _problems(tmp_path, doc)
        assert any(p.startswith("rois.dials[0].factor: must be 0.1") for p in probs)

    def test_valid_total_digit_counts_default(self):
        assert valid_total_digit_counts(_W._DEFAULT_DIGITAL_TOTAL_REGEX) == [(6, 3)]
        assert digit_count_problem(6, 3, _W._DEFAULT_DIGITAL_TOTAL_REGEX) is None


class TestCheckConfigCli:
    def test_ok(self, full_config_yaml, capsys):
        with pytest.raises(SystemExit) as ei:
            _W.main(["--config", str(full_config_yaml), "--check-config"])
        assert ei.value.code == 0
        assert "config OK" in capsys.readouterr().out

    def test_invalid_exits_78(self, tmp_path, capsys):
        p = _write(tmp_path, {"meter": {"type": "analog"}})
        with pytest.raises(SystemExit) as ei:
            _W.main(["--config", str(p), "--check-config"])
        assert ei.value.code == _W.EX_CONFIG == 78
        assert "meter.type" in capsys.readouterr().err

    def test_mechanical_without_dials_cannot_run(self, minimal_config_yaml):
        with pytest.raises(SystemExit) as ei:
            _W.main(["--config", str(minimal_config_yaml), "--check-config"])
        assert ei.value.code == 78

    def test_missing_file_exits_78(self, tmp_path):
        with pytest.raises(SystemExit) as ei:
            _W.main(["--config", str(tmp_path / "nope.yaml"), "--check-config"])
        assert ei.value.code == 78

    def test_bad_config_never_reaches_mqtt_or_logging(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must not get this far")
        monkeypatch.setattr(_W, "MqttClient", boom)
        monkeypatch.setattr(_W, "setup_logging", boom)
        p = _write(tmp_path, _digital(frac_n=4))
        with pytest.raises(SystemExit) as ei:
            _W.main(["--config", str(p)])
        assert ei.value.code == 78
