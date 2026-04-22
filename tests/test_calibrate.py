"""Tests for calibrate.py — YAML merge, payload validation, and config round-trip.

Browser UX and live ESP32 fetch are manual-only. Here we exercise the server-
side logic: validate_payload, extract_rois, merge_rois, backup_and_write.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
import yaml

from calibrate import (
    DIAL_COUNT,
    MIN_ROI_DIM,
    backup_and_write,
    extract_rois,
    load_yaml_doc,
    merge_rois,
    validate_payload,
)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _valid_dial(factor=0.1):
    return {
        "name": f"dial_{factor}",
        "roi": [0.1, 0.1, 0.2, 0.2],
        "factor": factor,
        "rotation": "cw",
        "zero_angle_deg": -90.0,
    }


def _valid_payload():
    return {
        "digits": [0.3, 0.2, 0.4, 0.1],
        "dials": [
            _valid_dial(0.1),
            _valid_dial(0.01),
            _valid_dial(0.001),
            _valid_dial(0.0001),
        ],
        "anchors": [[0.1, 0.0, 0.6, 0.25], [0.0, 0.3, 0.3, 0.2], None],
    }


# ---------------------------------------------------------------------------
# validate_payload
# ---------------------------------------------------------------------------

class TestValidatePayload:
    def test_valid(self):
        assert validate_payload(_valid_payload()) is None

    def test_missing_digits(self):
        p = _valid_payload(); p["digits"] = None
        err = validate_payload(p)
        assert err and "digits" in err

    def test_digits_out_of_bounds(self):
        p = _valid_payload(); p["digits"] = [0.5, 0.5, 0.6, 0.6]  # x+w > 1
        err = validate_payload(p)
        assert err and "past image bounds" in err

    def test_digits_wrong_length(self):
        p = _valid_payload(); p["digits"] = [0.1, 0.1, 0.1]
        err = validate_payload(p)
        assert err and "4-element" in err

    def test_roi_negative(self):
        p = _valid_payload(); p["digits"] = [-0.1, 0.1, 0.2, 0.2]
        err = validate_payload(p)
        assert err and "out of [0,1]" in err

    def test_too_few_dials(self):
        p = _valid_payload(); p["dials"] = p["dials"][:3]
        err = validate_payload(p)
        assert err and "4 entries" in err

    def test_too_many_dials(self):
        p = _valid_payload(); p["dials"].append(_valid_dial(0.00001))
        err = validate_payload(p)
        assert err and "4 entries" in err

    def test_dials_zero_factor_must_be_tenths(self):
        p = _valid_payload()
        p["dials"][0]["factor"] = 0.01  # wrong!
        err = validate_payload(p)
        assert err and "tenths" in err

    def test_dial_rotation_invalid(self):
        p = _valid_payload()
        p["dials"][0]["rotation"] = "other"
        err = validate_payload(p)
        assert err and "rotation" in err

    def test_dial_zero_angle_out_of_range(self):
        p = _valid_payload()
        p["dials"][0]["zero_angle_deg"] = 9999
        err = validate_payload(p)
        assert err and "zero_angle_deg" in err

    def test_dial_roi_too_small(self):
        p = _valid_payload()
        p["dials"][0]["roi"] = [0.1, 0.1, MIN_ROI_DIM / 2, 0.2]
        err = validate_payload(p)
        assert err and "too small" in err

    def test_anchors_truncated_to_max(self):
        p = _valid_payload()
        p["anchors"] = [[0.1, 0.1, 0.1, 0.1]] * 5
        err = validate_payload(p)
        assert err and "most" in err

    def test_anchors_can_be_empty(self):
        p = _valid_payload(); p["anchors"] = []
        assert validate_payload(p) is None

    def test_anchors_can_contain_none(self):
        p = _valid_payload(); p["anchors"] = [None, None, None]
        assert validate_payload(p) is None


# ---------------------------------------------------------------------------
# extract_rois
# ---------------------------------------------------------------------------

class TestExtractRois:
    def test_empty_doc_returns_defaults(self):
        r = extract_rois({})
        assert r["digits"] is None
        assert len(r["dials"]) == DIAL_COUNT
        assert all(d["roi"] is None for d in r["dials"])
        # dials[0] should default to tenths wheel
        assert r["dials"][0]["factor"] == pytest.approx(0.1)
        assert len(r["anchors"]) == 3

    def test_partial_doc_preserved(self):
        doc = {
            "rois": {
                "digits": [0.3, 0.2, 0.4, 0.1],
                "dials": [
                    {"name": "d0", "roi": [0.6, 0.5, 0.1, 0.2], "factor": 0.1,
                     "rotation": "ccw", "zero_angle_deg": -80.0},
                ],
            }
        }
        r = extract_rois(doc)
        assert r["digits"] == [0.3, 0.2, 0.4, 0.1]
        assert r["dials"][0]["roi"] == [0.6, 0.5, 0.1, 0.2]
        assert r["dials"][0]["rotation"] == "ccw"
        # dials beyond the first are padded with defaults
        assert r["dials"][1]["roi"] is None
        assert r["dials"][1]["factor"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# merge_rois
# ---------------------------------------------------------------------------

class TestMergeRois:
    def test_preserves_unrelated_keys(self):
        doc = {
            "esp32": {"base_url": "http://1.2.3.4"},
            "mqtt": {"host": "10.0.0.1", "port": 1883, "username": "user"},
            "processing": {"interval_sec": 10},
            "rois": {"digits": [0, 0, 0.1, 0.1]},
        }
        merged = merge_rois(doc, _valid_payload())
        # Unrelated top-level keys untouched
        assert merged["esp32"] == {"base_url": "http://1.2.3.4"}
        assert merged["mqtt"]["host"] == "10.0.0.1"
        assert merged["mqtt"]["username"] == "user"
        assert merged["processing"]["interval_sec"] == 10
        # Target keys updated
        assert merged["rois"]["digits"] == [0.3, 0.2, 0.4, 0.1]
        assert len(merged["rois"]["dials"]) == DIAL_COUNT

    def test_anchors_filter_none(self):
        doc = {}
        payload = _valid_payload()
        payload["anchors"] = [[0.1, 0.1, 0.1, 0.1], None, [0.2, 0.2, 0.1, 0.1]]
        merged = merge_rois(doc, payload)
        # None entries are dropped from the final YAML
        assert merged["alignment"]["anchor_rois"] == [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]]

    def test_dial_fields_typed(self):
        merged = merge_rois({}, _valid_payload())
        d0 = merged["rois"]["dials"][0]
        assert isinstance(d0["factor"], float)
        assert isinstance(d0["zero_angle_deg"], float)
        assert d0["rotation"] in ("cw", "ccw")


# ---------------------------------------------------------------------------
# backup_and_write (round-trip through watermeter.load_config)
# ---------------------------------------------------------------------------

class TestBackupAndWrite:
    def _seed_existing(self, tmp_path):
        """Write a valid starter config, to be overwritten by the test."""
        p = tmp_path / "config.yaml"
        p.write_text(
            "esp32:\n"
            "  base_url: \"http://localhost\"\n"
            "mqtt:\n"
            "  host: \"localhost\"\n"
            "  topic: \"home/watermeter\"\n"
        )
        return p

    def test_writes_fresh_file_no_backup(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        assert not config_path.exists()
        merged = merge_rois({"esp32": {"base_url": "http://localhost"}}, _valid_payload())
        backup = backup_and_write(config_path, merged)
        assert backup is None
        assert config_path.exists()
        # New file is valid YAML containing our ROIs
        written = yaml.safe_load(config_path.read_text())
        assert written["rois"]["digits"] == [0.3, 0.2, 0.4, 0.1]
        assert len(written["rois"]["dials"]) == DIAL_COUNT

    def test_backs_up_existing_file(self, tmp_path):
        config_path = self._seed_existing(tmp_path)
        before = config_path.read_text()
        merged = merge_rois(load_yaml_doc(config_path), _valid_payload())
        backup = backup_and_write(config_path, merged)
        assert backup is not None
        assert backup.exists()
        assert backup.read_text() == before
        # Live config updated
        after = yaml.safe_load(config_path.read_text())
        assert after["rois"]["digits"] == [0.3, 0.2, 0.4, 0.1]

    def test_preserves_non_roi_keys_in_backup(self, tmp_path):
        config_path = self._seed_existing(tmp_path)
        merged = merge_rois(load_yaml_doc(config_path), _valid_payload())
        backup_and_write(config_path, merged)
        after = yaml.safe_load(config_path.read_text())
        # Non-ROI keys round-tripped
        assert after["esp32"]["base_url"] == "http://localhost"
        assert after["mqtt"]["host"] == "localhost"
        assert after["mqtt"]["topic"] == "home/watermeter"

    def test_round_trips_through_watermeter_load_config(self, tmp_path):
        """backup_and_write must validate via watermeter.load_config before commit."""
        import watermeter
        config_path = tmp_path / "config.yaml"
        merged = merge_rois({"esp32": {"base_url": "http://localhost"}}, _valid_payload())
        backup_and_write(config_path, merged)
        cfg = watermeter.load_config(str(config_path))
        assert len(cfg.dials) == DIAL_COUNT
        assert cfg.dials[0].factor == pytest.approx(0.1)

    def test_invalid_merged_yaml_does_not_overwrite(self, tmp_path, monkeypatch):
        """If the merged doc fails to round-trip, the live file is untouched."""
        config_path = self._seed_existing(tmp_path)
        before = config_path.read_text()

        import watermeter
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic load_config failure")
        monkeypatch.setattr("calibrate.watermeter.load_config", boom)

        merged = merge_rois(load_yaml_doc(config_path), _valid_payload())
        with pytest.raises(RuntimeError, match="failed to load"):
            backup_and_write(config_path, merged)
        # Live file unchanged; no backup created
        assert config_path.read_text() == before
        assert len(glob.glob(f"{config_path}.bak.*")) == 0
