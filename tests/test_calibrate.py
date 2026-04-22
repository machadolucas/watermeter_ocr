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
    detect_meter_type,
    extract_digital_rois,
    extract_rois,
    load_yaml_doc,
    merge_digital_rois,
    merge_rois,
    validate_digital_payload,
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
# Digital-mode variants
# ---------------------------------------------------------------------------

def _valid_digital_payload():
    return {
        "total": [0.10, 0.25, 0.80, 0.18],
        "flow":  [0.25, 0.50, 0.55, 0.15],
        "anchors": [[0.1, 0.0, 0.6, 0.25], None, None],
    }


class TestDetectMeterType:
    def test_default_mechanical(self):
        assert detect_meter_type({}) == "mechanical"

    def test_explicit_mechanical(self):
        assert detect_meter_type({"meter": {"type": "mechanical"}}) == "mechanical"

    def test_digital(self):
        assert detect_meter_type({"meter": {"type": "digital"}}) == "digital"

    def test_case_insensitive(self):
        assert detect_meter_type({"meter": {"type": "DIGITAL"}}) == "digital"


class TestValidateDigitalPayload:
    def test_valid(self):
        assert validate_digital_payload(_valid_digital_payload()) is None

    def test_missing_total_when_no_split_mode(self):
        # Single-ROI mode and split-ROI mode are mutually exclusive-OR-both.
        # Missing BOTH is a config error — nothing tells the pipeline where
        # to OCR the total line.
        p = _valid_digital_payload(); p["total"] = None
        err = validate_digital_payload(p)
        assert err and "total" in err

    def test_missing_total_allowed_in_split_mode(self):
        # When the split-ROI pair is set, `total` is no longer needed — the
        # pipeline reads from the split sub-ROIs and concatenates.
        p = _valid_digital_payload()
        p["total"] = None
        p["total_int"]  = [0.10, 0.25, 0.48, 0.18]
        p["total_frac"] = [0.58, 0.30, 0.22, 0.14]
        assert validate_digital_payload(p) is None

    def test_flow_is_optional(self):
        # Flow ROI is optional — some installs can't align both lines cleanly
        # due to flash glare and deliberately capture only the total.
        p = _valid_digital_payload(); p["flow"] = None
        assert validate_digital_payload(p) is None

    def test_flow_missing_key_is_also_ok(self):
        p = _valid_digital_payload(); del p["flow"]
        assert validate_digital_payload(p) is None

    def test_flow_bad_shape_still_rejected(self):
        # But if the user DID draw a flow ROI, its geometry is still validated.
        p = _valid_digital_payload(); p["flow"] = [0.9, 0.9, 0.5, 0.5]  # x+w > 1
        err = validate_digital_payload(p)
        assert err and "flow" in err

    def test_out_of_bounds(self):
        p = _valid_digital_payload(); p["total"] = [0.9, 0.9, 0.5, 0.5]
        err = validate_digital_payload(p)
        assert err and "past image bounds" in err

    def test_rejects_extra_dials(self):
        # Digital payload shape doesn't accept a "dials" field — but the current
        # validator doesn't forbid extra keys, so ensure required keys still lead.
        p = _valid_digital_payload(); p["dials"] = []
        assert validate_digital_payload(p) is None  # extra key is tolerated


class TestExtractDigitalRois:
    def test_empty_doc_returns_none_rois(self):
        r = extract_digital_rois({})
        assert r["total"] is None
        assert r["flow"] is None
        assert len(r["anchors"]) == 3
        assert all(a is None for a in r["anchors"])

    def test_preserves_existing_total_flow(self):
        doc = {
            "rois": {"digital": {
                "total": [0.1, 0.2, 0.8, 0.2],
                "flow":  [0.2, 0.5, 0.6, 0.2],
            }},
            "alignment": {"anchor_rois": [[0.1, 0.1, 0.3, 0.3]]},
        }
        r = extract_digital_rois(doc)
        assert r["total"] == [0.1, 0.2, 0.8, 0.2]
        assert r["flow"] == [0.2, 0.5, 0.6, 0.2]
        assert r["anchors"][0] == [0.1, 0.1, 0.3, 0.3]
        assert r["anchors"][1] is None


class TestMergeDigitalRois:
    def test_writes_digital_subtree_preserves_mechanical(self):
        # A user might flip meter.type back and forth; merge_digital_rois must
        # NOT clobber pre-existing rois.digits / rois.dials in case they want
        # to flip to mechanical later.
        doc = {
            "esp32": {"base_url": "http://1.2.3.4"},
            "rois": {"digits": [0, 0, 0.1, 0.1], "dials": [{"name": "d0"}]},
        }
        merged = merge_digital_rois(doc, _valid_digital_payload())
        assert merged["esp32"] == {"base_url": "http://1.2.3.4"}
        assert merged["rois"]["digits"] == [0, 0, 0.1, 0.1]  # untouched
        assert merged["rois"]["dials"] == [{"name": "d0"}]   # untouched
        assert merged["rois"]["digital"]["total"] == [0.10, 0.25, 0.80, 0.18]
        assert merged["rois"]["digital"]["flow"]  == [0.25, 0.50, 0.55, 0.15]
        # Anchors still funnel through the shared alignment.anchor_rois key.
        assert merged["alignment"]["anchor_rois"] == [[0.1, 0.0, 0.6, 0.25]]

    def test_merge_without_flow_removes_existing_flow_key(self):
        # User previously had a flow ROI, then decided to drop it (e.g. flash
        # glare). The save should clear the stored flow so the runtime sees
        # "flow tracking disabled" instead of reading a stale ROI.
        doc = {"rois": {"digital": {
            "total": [0.1, 0.1, 0.8, 0.2],
            "flow":  [0.1, 0.5, 0.8, 0.2],
        }}}
        payload = {
            "total": [0.2, 0.2, 0.6, 0.2],
            "flow": None,
            "anchors": [],
        }
        merged = merge_digital_rois(doc, payload)
        assert merged["rois"]["digital"]["total"] == [0.2, 0.2, 0.6, 0.2]
        assert "flow" not in merged["rois"]["digital"]

    def test_merge_writes_split_roi_pair(self):
        # Split-ROI mode: total_int + total_frac stored alongside total.
        doc = {"rois": {"digital": {"total": [0.1, 0.1, 0.8, 0.2]}}}
        payload = {
            "total":      [0.10, 0.25, 0.80, 0.18],
            "total_int":  [0.10, 0.25, 0.48, 0.18],
            "total_frac": [0.58, 0.30, 0.22, 0.14],
            "flow":       [0.25, 0.50, 0.55, 0.15],
            "anchors":    [],
        }
        merged = merge_digital_rois(doc, payload)
        assert merged["rois"]["digital"]["total_int"]  == [0.10, 0.25, 0.48, 0.18]
        assert merged["rois"]["digital"]["total_frac"] == [0.58, 0.30, 0.22, 0.14]

    def test_merge_clears_stale_split_keys(self):
        # User had split-ROI mode configured, then decided to revert to single
        # ROI. The save must drop total_int / total_frac so the pipeline reads
        # a clean state (otherwise it would keep reading from stale sub-ROIs).
        doc = {"rois": {"digital": {
            "total":      [0.1, 0.1, 0.8, 0.2],
            "total_int":  [0.1, 0.1, 0.4, 0.2],
            "total_frac": [0.5, 0.1, 0.3, 0.2],
        }}}
        payload = {
            "total": [0.15, 0.20, 0.70, 0.20],
            "total_int": None,
            "total_frac": None,
            "flow": None,
            "anchors": [],
        }
        merged = merge_digital_rois(doc, payload)
        assert "total_int"  not in merged["rois"]["digital"]
        assert "total_frac" not in merged["rois"]["digital"]


class TestValidateDigitalSplitRois:
    def test_both_split_rois_accepted(self):
        p = _valid_digital_payload()
        p["total_int"]  = [0.10, 0.25, 0.48, 0.18]
        p["total_frac"] = [0.58, 0.30, 0.22, 0.14]
        assert validate_digital_payload(p) is None

    def test_only_total_int_rejected(self):
        # Either BOTH split keys or NEITHER — one alone is an incomplete
        # configuration and would crash the pipeline expecting both.
        p = _valid_digital_payload()
        p["total_int"] = [0.10, 0.25, 0.48, 0.18]
        err = validate_digital_payload(p)
        assert err and "total_int and total_frac" in err

    def test_only_total_frac_rejected(self):
        p = _valid_digital_payload()
        p["total_frac"] = [0.58, 0.30, 0.22, 0.14]
        err = validate_digital_payload(p)
        assert err and "total_int and total_frac" in err

    def test_split_roi_bad_shape_rejected(self):
        p = _valid_digital_payload()
        p["total_int"]  = [0.9, 0.9, 0.5, 0.5]  # x+w > 1
        p["total_frac"] = [0.58, 0.30, 0.22, 0.14]
        err = validate_digital_payload(p)
        assert err and "total_int" in err


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

    def test_digital_round_trips_through_watermeter_load_config(self, tmp_path):
        import watermeter
        config_path = tmp_path / "config.yaml"
        # Seed with meter.type: digital so calibrate's dispatch picks up digital mode.
        config_path.write_text(
            "esp32:\n  base_url: \"http://localhost\"\n"
            "meter:\n  type: digital\n"
        )
        payload = {
            "total": [0.10, 0.25, 0.80, 0.18],
            "flow":  [0.25, 0.50, 0.55, 0.15],
            "anchors": [[0.1, 0.0, 0.6, 0.25], None, None],
        }
        merged = merge_digital_rois(load_yaml_doc(config_path), payload)
        backup_and_write(config_path, merged)
        cfg = watermeter.load_config(str(config_path))
        assert cfg.meter_type == "digital"
        assert cfg.digital_total_roi == payload["total"]
        assert cfg.digital_flow_roi == payload["flow"]

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
