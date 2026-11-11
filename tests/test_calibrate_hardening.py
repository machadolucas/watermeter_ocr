"""calibrate.py hardening (2026-11-04 audit follow-up).

Per-digit list validation, the real digital "Test" pipeline, the session
token / body cap / socket timeout, and race-free saves. The HTTP tests run
the real handler on an ephemeral loopback port.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest
import requests
import yaml

import calibrate
import watermeter
from calibrate import (
    FrameCache,
    ThreadingHTTPServer,
    backup_and_write,
    digit_slots_for,
    make_handler_class,
    merge_digital_rois,
    run_digital_test,
    trim_cells,
    validate_digital_payload,
)

LCD = Path(__file__).parent / "fixtures" / "lcd"
SPEC = json.loads((LCD / "lcd_fixtures.json").read_text())
FRAME = (LCD / "numeric_ok.png").read_bytes()
INT_CELLS = SPEC["rois"][:6]
FRAC_CELLS = SPEC["rois"][6:]
TOKEN = "test-token"


def _cell(i):
    return [0.05 + i * 0.07, 0.3, 0.05, 0.2]


def _payload(n_int=6, n_frac=3, **extra):
    p = {
        "total": None, "total_int": None, "total_frac": None, "flow": None,
        "total_int_digits": [_cell(i) for i in range(n_int)],
        "total_frac_digits": [_cell(6 + i) for i in range(n_frac)],
        "flow_digits": [],
        "anchors": [None, None, None],
    }
    p.update(extra)
    return p


def _digital_doc(tmp_path, **rois):
    return {
        "esp32": {"base_url": "http://192.0.2.1"},
        "meter": {"type": "digital"},
        "processing": {"image_path": str(tmp_path / "live_raw.jpg"),
                       "debug_dir": str(tmp_path / "live_debug")},
        "alignment": {"enabled": False, "reference_path": str(tmp_path / "ref.jpg")},
        "paths": {"ocr_bin": str(tmp_path / "no-ocr-binary")},
        "rois": {"digital": rois or {"total_int_digits": INT_CELLS,
                                     "total_frac_digits": FRAC_CELLS}},
    }


# ---------------------------------------------------------------------------
# Per-digit list validation
# ---------------------------------------------------------------------------

class TestDigitLists:
    def test_trailing_empty_cells_are_trimmed(self):
        cells, err = trim_cells("total_frac_digits", [_cell(0), _cell(1), _cell(2), None])
        assert err is None and len(cells) == 3

    def test_gap_before_a_drawn_cell_is_rejected(self):
        _, err = trim_cells("total_frac_digits", [_cell(0), None, _cell(2)])
        assert err == ("total_frac_digits: cell 2 is empty but cell 3 is drawn; "
                       "draw cell 2 or clear the cells after it")

    def test_valid_6_plus_3(self):
        assert validate_digital_payload(_payload()) is None

    def test_trailing_null_from_ui_accepted_and_trimmed(self):
        p = _payload()
        p["total_frac_digits"].append(None)   # the UI's unused 4th slot
        assert validate_digital_payload(p) is None
        assert len(p["total_frac_digits"]) == 3

    def test_holed_int_list_rejected(self):
        p = _payload()
        p["total_int_digits"][1] = None
        err = validate_digital_payload(p)
        assert err and "total_int_digits: cell 2 is empty" in err

    def test_four_fractional_cells_rejected_with_expected_counts(self):
        err = validate_digital_payload(_payload(n_frac=4))
        assert err and "needs 6 integer + 3 fractional cells" in err and "got 6 + 4" in err

    def test_five_integer_cells_rejected(self):
        err = validate_digital_payload(_payload(n_int=5))
        assert err and "got 5 + 3" in err

    def test_int_only_list_rejected_even_with_total(self):
        p = _payload(total=[0.1, 0.1, 0.8, 0.2])
        p["total_frac_digits"] = []
        err = validate_digital_payload(p)
        assert err and "total_int and total_frac must be set together" in err

    def test_custom_regex_changes_expected_counts(self):
        import re
        rx = re.compile(r"^\d{5}\.\d{4}$")
        assert validate_digital_payload(_payload(n_int=5, n_frac=4), rx) is None
        assert validate_digital_payload(_payload(), rx)

    def test_flow_cell_count_checked(self):
        p = _payload(flow_digits=[_cell(i) for i in range(3)])
        err = validate_digital_payload(p)
        assert err and err.startswith("flow_digits: 3 flow cells")

    def test_digit_slots_follow_the_regexes(self):
        assert digit_slots_for({}) == {"total_int": 6, "total_frac": 3, "flow": 8}
        doc = {"digital": {"total_regex": r"^\d{8}\.?\d{2}$"}}
        assert digit_slots_for(doc)["total_int"] == 8
        assert digit_slots_for(doc)["total_frac"] == 2

    def test_digit_slots_fall_back_on_bad_regex(self):
        assert digit_slots_for({"digital": {"total_regex": "("}})["total_frac"] == 3


# ---------------------------------------------------------------------------
# Real digital pipeline behind the Test button
# ---------------------------------------------------------------------------

class TestRunDigitalTest:
    def test_reads_the_real_lcd_fixture(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        doc = _digital_doc(tmp_path)
        doc["alignment"]["enabled"] = True   # but no reference yet
        out = run_digital_test(FRAME, doc, work)
        assert out["success"] is True and out["reason"] == "ok"
        assert out["total"] == pytest.approx(57.308)
        assert out["raw_total"] == "000057.308"
        assert out["cells"]["total_int_digits"] == list("000057")
        assert out["cells"]["total_frac_digits"] == list("308")
        assert "alignment skipped: no reference.jpg yet" in out["notes"]
        # Nothing the live service owns was written.
        assert not (tmp_path / "ref.jpg").exists()
        assert not (tmp_path / "live_raw.jpg").exists()
        assert not (tmp_path / "live_debug").exists()

    def test_wrong_view_is_reported_honestly(self, tmp_path):
        import cv2
        import numpy as np
        blank = cv2.imencode(".png", np.zeros((120, 340, 3), np.uint8))[1].tobytes()
        out = run_digital_test(blank, _digital_doc(tmp_path), tmp_path)
        assert out["success"] is False
        assert out["reason"] == "wrong_view"
        assert "diagnostic screen" in out["hint"]

    def test_undecodable_frame(self, tmp_path):
        out = run_digital_test(b"not an image", _digital_doc(tmp_path), tmp_path)
        assert out["success"] is False and out["reason"] == "no_frame"

    def test_line_mode_without_ocr_helper_is_a_clear_error(self, tmp_path):
        doc = _digital_doc(tmp_path, total=[0.0, 0.0, 1.0, 1.0])
        out = run_digital_test(FRAME, doc, tmp_path)
        assert "OCR helper not found" in out["error"]

    def test_invalid_merged_config_raises_config_error(self, tmp_path):
        doc = _digital_doc(tmp_path, total_int_digits=INT_CELLS,
                           total_frac_digits=FRAC_CELLS + [FRAC_CELLS[-1]])
        with pytest.raises(watermeter.ConfigError):
            run_digital_test(FRAME, doc, tmp_path)


# ---------------------------------------------------------------------------
# Save races, backup names and permissions
# ---------------------------------------------------------------------------

class TestBackupAndWriteHardening:
    def _seed(self, tmp_path, mode=0o640):
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(_digital_doc(tmp_path)))
        os.chmod(p, mode)
        return p

    def test_back_to_back_saves_get_distinct_backups(self, tmp_path):
        p = self._seed(tmp_path)
        doc = calibrate.load_yaml_doc(p)
        b1 = backup_and_write(p, doc)
        b2 = backup_and_write(p, doc)
        assert b1 != b2 and b1.exists() and b2.exists()

    def test_backup_is_private_and_config_keeps_its_mode(self, tmp_path):
        p = self._seed(tmp_path, mode=0o640)
        backup = backup_and_write(p, calibrate.load_yaml_doc(p))
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
        assert stat.S_IMODE(p.stat().st_mode) == 0o640

    def test_new_config_is_private(self, tmp_path):
        p = tmp_path / "config.yaml"
        backup_and_write(p, _digital_doc(tmp_path))
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_failed_save_leaves_no_temp_file(self, tmp_path):
        p = self._seed(tmp_path)
        bad = _digital_doc(tmp_path)
        bad["meter"]["type"] = "analog"
        with pytest.raises(RuntimeError, match="failed to load"):
            backup_and_write(p, bad)
        assert [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")] == []


# ---------------------------------------------------------------------------
# HTTP: token, body cap, socket timeout, concurrent saves, /test
# ---------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_digital_doc(tmp_path)))
    cfg = watermeter.load_config(str(cfg_path))
    cache = FrameCache()
    cache.set(FRAME)
    handler = make_handler_class(cfg_path, cfg, cache, TOKEN, timeout=1.0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, cfg_path, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


HDR = {"X-Calibrate-Token": TOKEN}


class TestHttp:
    def test_every_route_needs_the_token(self, server):
        base, _, _ = server
        for path in ("/", "/config", "/frame.jpg"):
            assert requests.get(base + path, timeout=5).status_code == 403
        assert requests.get(base + "/config", headers={"X-Calibrate-Token": "nope"},
                            timeout=5).status_code == 403
        for path in ("/config", "/test", "/flash"):
            assert requests.post(base + path, json={}, timeout=5).status_code == 403

    def test_token_by_header_or_query(self, server):
        base, _, _ = server
        r = requests.get(base + "/config", headers=HDR, timeout=5)
        assert r.status_code == 200
        assert r.json()["digit_slots"] == {"total_int": 6, "total_frac": 3, "flow": 8}
        assert requests.get(f"{base}/?token={TOKEN}", timeout=5).status_code == 200

    def test_missing_content_length_is_411(self, server):
        _, _, port = server
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", "/config")
        conn.putheader("X-Calibrate-Token", TOKEN)
        conn.endheaders()
        assert conn.getresponse().status == 411

    def test_oversized_body_is_413_without_reading_it(self, server):
        _, _, port = server
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", "/config")
        conn.putheader("X-Calibrate-Token", TOKEN)
        conn.putheader("Content-Length", str(calibrate.MAX_BODY_BYTES + 1))
        conn.endheaders()
        assert conn.getresponse().status == 413

    def test_bad_content_length_and_bad_json_are_400(self, server):
        base, _, port = server
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", "/config")
        conn.putheader("X-Calibrate-Token", TOKEN)
        conn.putheader("Content-Length", "abc")
        conn.endheaders()
        assert conn.getresponse().status == 400
        r = requests.post(base + "/config", data=b"{nope", headers=HDR, timeout=5)
        assert r.status_code == 400 and "invalid JSON" in r.json()["error"]

    def test_stalled_client_is_dropped(self, server):
        _, _, port = server
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(b"POST /config HTTP/1.1\r\n")   # never finishes the headers
        t0 = time.monotonic()
        assert s.recv(1024) == b""               # server closed the socket
        assert time.monotonic() - t0 < 4.0
        s.close()

    def test_holed_list_rejected_over_http(self, server):
        base, cfg_path, _ = server
        before = cfg_path.read_text()
        p = _payload()
        p["total_int_digits"][3] = None
        r = requests.post(base + "/config", json=p, headers=HDR, timeout=5)
        assert r.status_code == 400 and "cell 4 is empty" in r.json()["error"]
        assert cfg_path.read_text() == before

    def test_concurrent_saves(self, server):
        base, cfg_path, _ = server
        results = []

        def save(i):
            p = _payload()
            p["total_int_digits"][0] = [0.01 + i * 0.001, 0.3, 0.05, 0.2]
            results.append(requests.post(base + "/config", json=p, headers=HDR,
                                         timeout=10).status_code)

        threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == [200] * 8
        backups = sorted(cfg_path.parent.glob("config.yaml.bak.*"))
        assert len(backups) == 8
        assert all(stat.S_IMODE(b.stat().st_mode) == 0o600 for b in backups)
        assert [f for f in cfg_path.parent.iterdir() if f.name.endswith(".tmp")] == []
        cfg = watermeter.load_config(str(cfg_path))
        assert len(cfg.digital_total_int_digits) == 6

    def test_digital_test_endpoint_runs_the_pipeline(self, server):
        base, _, _ = server
        p = _payload()
        p["total_int_digits"] = INT_CELLS
        p["total_frac_digits"] = FRAC_CELLS + [None]
        r = requests.post(base + "/test", json=p, headers=HDR, timeout=10)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["success"] is True
        assert data["total"] == pytest.approx(57.308)

    def test_flash_endpoint(self, server):
        base, _, _ = server
        # ai_on_the_edge config: flash is per capture, so this is a no-op.
        r = requests.post(base + "/flash", json={"on": True}, headers=HDR, timeout=5)
        assert r.status_code == 200 and r.json() == {"ok": True}

    def test_save_then_merge_round_trip(self, tmp_path):
        doc = merge_digital_rois(_digital_doc(tmp_path), _payload())
        p = tmp_path / "c.yaml"
        backup_and_write(p, doc)
        assert len(watermeter.load_config(str(p)).digital_total_frac_digits) == 3
