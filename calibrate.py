#!/usr/bin/env python3
"""Browser-based calibration tool for watermeter_ocr.

Runs a local HTTP server on 127.0.0.1:8765 that serves an interactive page
where you drag rectangles over a live ESP32 frame to define digit/dial/anchor
ROIs, then save the result back to ~/watermeter/config.yaml. See
docs/CONFIGURATION.md for the end-to-end meter-swap procedure.

Usage:
    python3 calibrate.py [--config PATH] [--port N] [--no-browser]

No new dependencies — uses only the existing venv (yaml, requests, cv2, numpy).
"""
from __future__ import annotations

import argparse
import copy
import http.server
import json
import os
import socketserver
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import watermeter  # noqa: E402

DEFAULT_CONFIG = str(Path.home() / "watermeter" / "config.yaml")
DEFAULT_PORT = 8765
DIAL_COUNT = 4
ANCHOR_COUNT = 3
MIN_ROI_DIM = 0.005  # reject rectangles smaller than 0.5% in either dimension

# dials[0] MUST be the tenths wheel (factor 0.1) — see AGENTS.md invariants.
DIAL_FACTORS = [0.1, 0.01, 0.001, 0.0001]
DIAL_NAMES = ["dial_0_1", "dial_0_01", "dial_0_001", "dial_0_0001"]

# Digital meter (LCD, e.g. Qalcosonic W1): two line ROIs for the two text lines
# the display shows — total consumption and instantaneous flow.
DIGITAL_ROI_NAMES = ["total", "flow"]


def detect_meter_type(doc: dict) -> str:
    """Return "digital" if the config declares meter.type: digital, else "mechanical"."""
    m = (doc.get("meter") or {}).get("type")
    return "digital" if str(m or "mechanical").lower() == "digital" else "mechanical"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml_doc(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def extract_rois(doc: dict) -> dict:
    """Return the subset of config keys the calibrator cares about, shaped for
    the frontend. Fills missing sections with sensible defaults so a fresh
    install still yields a valid starting state.
    """
    rois = doc.get("rois", {}) or {}
    dials_cfg = rois.get("dials") or []
    dials_out = []
    for i in range(DIAL_COUNT):
        if i < len(dials_cfg):
            d = dials_cfg[i]
            dials_out.append({
                "name": d.get("name", DIAL_NAMES[i]),
                "roi": d.get("roi"),
                "factor": d.get("factor", DIAL_FACTORS[i]),
                "rotation": d.get("rotation", "cw"),
                "zero_angle_deg": d.get("zero_angle_deg", -90.0),
            })
        else:
            dials_out.append({
                "name": DIAL_NAMES[i],
                "roi": None,
                "factor": DIAL_FACTORS[i],
                "rotation": "cw",
                "zero_angle_deg": -90.0,
            })

    anchors = (doc.get("alignment", {}) or {}).get("anchor_rois") or []
    anchors_out = list(anchors[:ANCHOR_COUNT])
    while len(anchors_out) < ANCHOR_COUNT:
        anchors_out.append(None)

    return {
        "digits": rois.get("digits"),
        "dials": dials_out,
        "anchors": anchors_out,
    }


def merge_rois(doc: dict, new_rois: dict) -> dict:
    """Deep-copy doc; replace ROI / dial / anchor sections. Other keys intact."""
    out = copy.deepcopy(doc)
    out.setdefault("rois", {})
    out["rois"]["digits"] = list(new_rois["digits"])
    out["rois"]["dials"] = [
        {
            "name": d["name"],
            "roi": list(d["roi"]),
            "factor": float(d["factor"]),
            "rotation": d["rotation"],
            "zero_angle_deg": float(d["zero_angle_deg"]),
        }
        for d in new_rois["dials"]
    ]
    anchors = [list(a) for a in new_rois["anchors"] if a is not None]
    out.setdefault("alignment", {})
    out["alignment"]["anchor_rois"] = anchors
    return out


# ---------------------------------------------------------------------------
# Digital-meter variants (meter.type: digital)
# ---------------------------------------------------------------------------

def extract_digital_rois(doc: dict) -> dict:
    """Like extract_rois but for a digital LCD meter (total + flow line ROIs)."""
    rois = doc.get("rois", {}) or {}
    digital = rois.get("digital", {}) or {}
    anchors = (doc.get("alignment", {}) or {}).get("anchor_rois") or []
    anchors_out = list(anchors[:ANCHOR_COUNT])
    while len(anchors_out) < ANCHOR_COUNT:
        anchors_out.append(None)
    return {
        "total": digital.get("total"),
        "flow": digital.get("flow"),
        "anchors": anchors_out,
    }


def merge_digital_rois(doc: dict, new_rois: dict) -> dict:
    """Deep-copy doc; replace rois.digital.{total,flow} and alignment.anchor_rois.
    Mechanical ROI keys (rois.digits, rois.dials) are left untouched so a user
    can swap back to mechanical mode without losing their dial calibration.

    `flow` is optional. When the user didn't draw one (null/missing), we clear
    the existing flow entry so the saved config matches the UI state — the
    runtime treats an empty flow ROI as "flow tracking disabled".
    """
    out = copy.deepcopy(doc)
    out.setdefault("rois", {})
    out["rois"].setdefault("digital", {})
    out["rois"]["digital"]["total"] = list(new_rois["total"])
    flow = new_rois.get("flow")
    if flow:
        out["rois"]["digital"]["flow"] = list(flow)
    else:
        out["rois"]["digital"].pop("flow", None)
    anchors = [list(a) for a in new_rois["anchors"] if a is not None]
    out.setdefault("alignment", {})
    out["alignment"]["anchor_rois"] = anchors
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _valid_roi(roi: Any) -> Optional[str]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return "ROI must be a 4-element [x, y, w, h] list"
    x, y, w, h = roi
    for v, name in zip((x, y, w, h), ("x", "y", "w", "h")):
        if not isinstance(v, (int, float)):
            return f"ROI {name} must be a number"
        if not (0.0 <= v <= 1.0):
            return f"ROI {name}={v} out of [0,1]"
    if w < MIN_ROI_DIM or h < MIN_ROI_DIM:
        return f"ROI too small (w={w}, h={h}); min dim is {MIN_ROI_DIM}"
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return f"ROI extends past image bounds (x+w={x+w}, y+h={y+h})"
    return None


def validate_payload(payload: dict) -> Optional[str]:
    if not isinstance(payload, dict):
        return "payload must be a JSON object"

    digits = payload.get("digits")
    if digits is None:
        return "missing digits ROI"
    err = _valid_roi(digits)
    if err:
        return f"digits: {err}"

    dials = payload.get("dials")
    if not isinstance(dials, list) or len(dials) != DIAL_COUNT:
        return f"dials must be a list of exactly {DIAL_COUNT} entries"

    for i, d in enumerate(dials):
        if not isinstance(d, dict):
            return f"dials[{i}] must be an object"
        err = _valid_roi(d.get("roi"))
        if err:
            return f"dials[{i}]: {err}"
        if d.get("rotation") not in ("cw", "ccw"):
            return f"dials[{i}].rotation must be 'cw' or 'ccw'"
        try:
            zad = float(d.get("zero_angle_deg"))
        except (TypeError, ValueError):
            return f"dials[{i}].zero_angle_deg must be a number"
        if not (-360.0 <= zad <= 360.0):
            return f"dials[{i}].zero_angle_deg={zad} outside [-360, 360]"
        try:
            factor = float(d.get("factor"))
        except (TypeError, ValueError):
            return f"dials[{i}].factor must be a number"
        if i == 0 and not (0.09 < factor < 0.11):
            return "dials[0].factor must be 0.1 (the tenths wheel — see AGENTS.md invariants)"

    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list) or len(anchors) > ANCHOR_COUNT:
        return f"anchors must be a list of at most {ANCHOR_COUNT} entries"
    for i, a in enumerate(anchors):
        if a is None:
            continue
        err = _valid_roi(a)
        if err:
            return f"anchors[{i}]: {err}"

    return None


def validate_digital_payload(payload: dict) -> Optional[str]:
    """Digital-mode payload: {total: [x,y,w,h], flow: [x,y,w,h] | null, anchors: [...]}.

    `total` is required. `flow` is optional — some installs sacrifice the flow
    line to keep the total line free of flash glare, so a null/missing flow
    just means the pipeline falls back to delta-computed rate only.
    """
    if not isinstance(payload, dict):
        return "payload must be a JSON object"
    # total is mandatory
    roi = payload.get("total")
    if roi is None:
        return "missing total ROI"
    err = _valid_roi(roi)
    if err:
        return f"total: {err}"
    # flow is optional — validate shape only if the user drew one
    flow = payload.get("flow")
    if flow is not None:
        err = _valid_roi(flow)
        if err:
            return f"flow: {err}"
    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list) or len(anchors) > ANCHOR_COUNT:
        return f"anchors must be a list of at most {ANCHOR_COUNT} entries"
    for i, a in enumerate(anchors):
        if a is None:
            continue
        err = _valid_roi(a)
        if err:
            return f"anchors[{i}]: {err}"
    return None


# ---------------------------------------------------------------------------
# Config write (atomic, with backup + round-trip check)
# ---------------------------------------------------------------------------

def backup_and_write(config_path: Path, new_doc: dict) -> Optional[Path]:
    """Serialize new_doc to config_path. Round-trip through watermeter.load_config
    first to catch schema regressions; back up the old file with a timestamp
    suffix before overwriting. Returns the backup path or None if no prior file.
    """
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        yaml.safe_dump(new_doc, f, sort_keys=False)
    try:
        watermeter.load_config(str(tmp_path))
    except Exception as e:
        tmp_path.unlink()
        raise RuntimeError(f"generated YAML failed to load: {e}") from e

    backup_path = None
    if config_path.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = Path(f"{config_path}.bak.{ts}")
        backup_path.write_bytes(config_path.read_bytes())

    os.replace(tmp_path, config_path)
    return backup_path


# ---------------------------------------------------------------------------
# Frame fetch + dial-reading test
# ---------------------------------------------------------------------------

class FrameCache:
    """Thread-safe single-slot cache of the most recently fetched JPEG."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bytes: Optional[bytes] = None

    def set(self, data: bytes) -> None:
        with self._lock:
            self._bytes = data

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._bytes


def fetch_frame(base_url: str, timeout: float = 8.0) -> bytes:
    url = base_url.rstrip("/") + "/capture_with_flashlight"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def run_dial_tests(frame_bytes: bytes, dials: list[dict]) -> list[dict]:
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return [{"error": "could not decode frame"} for _ in dials]
    H, W = img.shape[:2]
    results = []
    for d in dials:
        roi = d.get("roi")
        if roi is None:
            results.append({"error": "no ROI set"})
            continue
        ax, ay, aw, ah = watermeter.norm_to_abs(roi, W, H)
        sub = img[ay:ay + ah, ax:ax + aw]
        if sub.size == 0:
            results.append({"error": "ROI produces empty crop"})
            continue
        reading, conf, offset = watermeter.read_dial(
            sub,
            zero_angle_deg=float(d.get("zero_angle_deg", -90.0)),
            rotation=d.get("rotation", "cw"),
        )
        results.append({
            "reading": round(float(reading), 3),
            "confidence": round(float(conf), 3),
            "center_offset": [round(float(offset[0]), 2), round(float(offset[1]), 2)],
        })
    return results


# ---------------------------------------------------------------------------
# HTTP server + routes
# ---------------------------------------------------------------------------

def _current_meter_type(config_path: Path) -> str:
    """Re-read config on each request so mode flips (e.g. user edits YAML by
    hand while calibrator is running) take effect without restarting."""
    try:
        doc = load_yaml_doc(config_path)
    except Exception:
        return "mechanical"
    return detect_meter_type(doc)


def make_handler_class(config_path: Path, esp32_url: str, frame_cache: FrameCache):

    class Handler(http.server.BaseHTTPRequestHandler):
        # Quiet default access logs; we only want failures.
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj: Any, status: int = 200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, status: int, content_type: str = "text/plain; charset=utf-8"):
            body = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw.decode() or "{}")

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self._send_text(HTML_PAGE, 200, "text/html; charset=utf-8")
                return

            if self.path == "/frame.jpg" or self.path.startswith("/frame.jpg?"):
                try:
                    data = fetch_frame(esp32_url)
                    frame_cache.set(data)
                except requests.RequestException as e:
                    self._send_text(f"ESP32 fetch failed: {e}", 502)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return

            if self.path == "/config":
                try:
                    doc = load_yaml_doc(config_path)
                    meter_type = detect_meter_type(doc)
                    rois = (extract_digital_rois(doc)
                            if meter_type == "digital"
                            else extract_rois(doc))
                    self._send_json({
                        "config_path": str(config_path),
                        "meter_type": meter_type,
                        "rois": rois,
                        "exists": config_path.exists(),
                    })
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
                return

            self._send_text("not found", 404)

        def do_POST(self):
            if self.path == "/config":
                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as e:
                    self._send_json({"error": f"invalid JSON: {e}"}, 400)
                    return
                meter_type = _current_meter_type(config_path)
                validator = (validate_digital_payload
                             if meter_type == "digital"
                             else validate_payload)
                err = validator(payload)
                if err:
                    self._send_json({"error": err}, 400)
                    return
                try:
                    doc = load_yaml_doc(config_path)
                    merged = (merge_digital_rois(doc, payload)
                              if meter_type == "digital"
                              else merge_rois(doc, payload))
                    backup = backup_and_write(config_path, merged)
                except Exception as e:
                    traceback.print_exc()
                    self._send_json({"error": str(e)}, 500)
                    return
                self._send_json({
                    "ok": True,
                    "backup": str(backup) if backup else None,
                    "config_path": str(config_path),
                })
                return

            if self.path == "/test":
                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as e:
                    self._send_json({"error": f"invalid JSON: {e}"}, 400)
                    return
                meter_type = _current_meter_type(config_path)
                if meter_type == "digital":
                    err = validate_digital_payload(payload)
                    if err:
                        self._send_json({"error": err}, 400)
                        return
                    # Digital test currently reports placeholders: actual line OCR
                    # needs the Swift Vision binary, which is deployment-local.
                    # The calibrator confirms ROI geometry; live verification
                    # happens once the service is restarted.
                    self._send_json({"meter_type": "digital",
                                     "note": "ROI geometry validated; OCR verified on the live service"})
                    return

                err = validate_payload(payload)
                if err:
                    self._send_json({"error": err}, 400)
                    return
                frame = frame_cache.get()
                if frame is None:
                    try:
                        frame = fetch_frame(esp32_url)
                        frame_cache.set(frame)
                    except requests.RequestException as e:
                        self._send_json({"error": f"ESP32 fetch failed: {e}"}, 502)
                        return
                try:
                    dial_results = run_dial_tests(frame, payload["dials"])
                except Exception as e:
                    traceback.print_exc()
                    self._send_json({"error": str(e)}, 500)
                    return
                self._send_json({"meter_type": "mechanical", "dials": dial_results})
                return

            self._send_text("not found", 404)

    return Handler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open the browser")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    # Fall back to the repo template if the live config doesn't exist yet —
    # the typical case during first-time setup.
    source = config_path if config_path.exists() else REPO_ROOT / "config.yaml"
    try:
        esp32_url = watermeter.load_config(str(source)).esp32_base_url
    except Exception as e:
        print(f"error: could not read ESP32 URL from {source}: {e}", file=sys.stderr)
        sys.exit(1)

    frame_cache = FrameCache()
    handler_cls = make_handler_class(config_path, esp32_url, frame_cache)

    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls) as httpd:
        url = f"http://127.0.0.1:{args.port}"
        print(f"[calibrate] listening on {url}")
        print(f"[calibrate] config: {config_path}")
        print(f"[calibrate] ESP32:  {esp32_url}")
        if not args.no_browser:
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[calibrate] shutting down")


# ---------------------------------------------------------------------------
# Embedded frontend (HTML + CSS + JS, no external assets)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>watermeter_ocr — calibrate</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #151924;
    --panel-2: #1c2030;
    --fg: #e6e9ef;
    --muted: #8a93a5;
    --accent: #4aa3ff;
    --ok: #3ccf7b;
    --warn: #ffb547;
    --err: #ff5577;
    --border: #2a3042;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { display: grid; grid-template-columns: 1fr 320px; height: 100vh; }
  header { grid-column: 1 / -1; padding: 10px 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px; background: var(--panel); }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }
  header .spacer { flex: 1; }
  button { cursor: pointer; background: var(--panel-2); color: var(--fg); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 4px; font-size: 13px; }
  button:hover:not(:disabled) { background: #242a3d; border-color: #3b4358; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #06101c; font-weight: 600; }
  button.primary:hover:not(:disabled) { background: #5bb1ff; }
  main { position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center;
    background: #080a0f; }
  #canvas-wrap { position: relative; }
  canvas { display: block; max-width: calc(100vw - 320px); max-height: calc(100vh - 50px); cursor: crosshair; }
  canvas.mode-idle { cursor: default; }
  aside { padding: 12px; overflow-y: auto; border-left: 1px solid var(--border); background: var(--panel); }
  aside h2 { font-size: 12px; letter-spacing: 0.08em; color: var(--muted); font-weight: 600;
    text-transform: uppercase; margin: 12px 0 6px; }
  .roi-list { display: flex; flex-direction: column; gap: 3px; }
  .roi-row { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 4px;
    cursor: pointer; border: 1px solid transparent; background: var(--panel-2); }
  .roi-row:hover { border-color: var(--border); }
  .roi-row.selected { border-color: var(--accent); background: #1d2b41; }
  .roi-row.drawing { border-color: var(--warn); background: #2c2416; }
  .roi-row .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .roi-row .label { flex: 1; font-size: 13px; }
  .roi-row .state { font-size: 11px; color: var(--muted); }
  .roi-row .state.set { color: var(--ok); }
  .field { display: grid; grid-template-columns: 100px 1fr; gap: 8px; margin: 4px 0; font-size: 12px;
    align-items: center; }
  .field label { color: var(--muted); }
  .field input, .field select { background: var(--panel-2); color: var(--fg); border: 1px solid var(--border);
    padding: 4px 6px; border-radius: 3px; font-size: 12px; }
  .field input[readonly] { color: var(--muted); }
  .actions { margin-top: 14px; display: flex; flex-direction: column; gap: 6px; }
  .toast { position: fixed; bottom: 14px; right: 14px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; min-width: 260px;
    max-width: 400px; font-size: 13px; box-shadow: 0 6px 24px rgba(0,0,0,0.4); }
  .toast.ok { border-left: 4px solid var(--ok); }
  .toast.err { border-left: 4px solid var(--err); }
  .toast.warn { border-left: 4px solid var(--warn); }
  .result { font-size: 12px; color: var(--muted); font-family: ui-monospace, monospace; }
  .result .val { color: var(--fg); }
  .result .bad { color: var(--err); }
  .hint { color: var(--muted); font-size: 11px; line-height: 1.5; margin: 8px 0; padding: 8px;
    border-left: 2px solid var(--border); background: var(--panel-2); }
</style>
</head>
<body>

<header>
  <h1>watermeter_ocr · calibrate</h1>
  <span class="meta" id="meta">loading…</span>
  <div class="spacer"></div>
  <button id="refresh-btn">Refresh image</button>
</header>

<main>
  <div id="canvas-wrap">
    <canvas id="canvas" width="640" height="480"></canvas>
  </div>
</main>

<aside>
  <h2>ROIs</h2>
  <div class="roi-list" id="roi-list"></div>

  <div class="hint">
    Click a row, then drag on the image to draw its rectangle. To redraw, just
    click the row again. Coordinates are normalized to [0, 1].
  </div>

  <h2>Selected</h2>
  <div id="selected-panel">
    <div class="hint">No ROI selected.</div>
  </div>

  <div class="actions">
    <button id="test-btn">Test current config</button>
    <button id="save-btn" class="primary">Save to config.yaml</button>
  </div>
</aside>

<div id="toast" class="toast" style="display:none"></div>

<script>
const DIAL_SLOTS = [
  { key: 'dial_0', label: 'Dial 0 (×0.1)', factor: 0.1, color: '#4aa3ff' },
  { key: 'dial_1', label: 'Dial 1 (×0.01)', factor: 0.01, color: '#ffb547' },
  { key: 'dial_2', label: 'Dial 2 (×0.001)', factor: 0.001, color: '#3ccf7b' },
  { key: 'dial_3', label: 'Dial 3 (×0.0001)', factor: 0.0001, color: '#c08eff' },
];
const MECHANICAL_SLOTS = [
  { key: 'digits', label: 'Digits window', color: '#ff5577', kind: 'digits' },
  ...DIAL_SLOTS.map(d => ({ ...d, kind: 'dial' })),
  { key: 'anchor_0', label: 'Anchor 0', color: '#6b7280', kind: 'anchor', idx: 0 },
  { key: 'anchor_1', label: 'Anchor 1', color: '#6b7280', kind: 'anchor', idx: 1 },
  { key: 'anchor_2', label: 'Anchor 2', color: '#6b7280', kind: 'anchor', idx: 2 },
];
const DIGITAL_SLOTS = [
  { key: 'total', label: 'Total line (m³)', color: '#4aa3ff', kind: 'digital' },
  { key: 'flow',  label: 'Flow line (optional)', color: '#3ccf7b', kind: 'digital' },
  { key: 'anchor_0', label: 'Anchor 0', color: '#6b7280', kind: 'anchor', idx: 0 },
  { key: 'anchor_1', label: 'Anchor 1', color: '#6b7280', kind: 'anchor', idx: 1 },
  { key: 'anchor_2', label: 'Anchor 2', color: '#6b7280', kind: 'anchor', idx: 2 },
];
let SLOTS = MECHANICAL_SLOTS;
let METER_TYPE = 'mechanical';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const image = new Image();
let imageLoaded = false;

const state = {
  rois: {},       // key -> {x,y,w,h} normalized
  dialMeta: {},   // dial_N -> {rotation, zero_angle_deg, factor, name}
  selected: null, // slot key currently selected
  drawing: null,  // slot key in draw mode (null = idle)
  dragStart: null,// {x,y} in canvas pixels while dragging
  dragCur: null,  // {x,y} in canvas pixels while dragging
  testResults: null, // key -> {reading, confidence, error}
};

function toast(msg, kind = 'ok') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  el.style.display = 'block';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.style.display = 'none'; }, 5000);
}

function slotFor(key) { return SLOTS.find(s => s.key === key); }

async function loadConfig() {
  const r = await fetch('/config');
  const data = await r.json();
  if (data.error) { toast(data.error, 'err'); return; }
  METER_TYPE = data.meter_type || 'mechanical';
  SLOTS = METER_TYPE === 'digital' ? DIGITAL_SLOTS : MECHANICAL_SLOTS;
  document.getElementById('meta').textContent =
    data.config_path + (data.exists ? '' : ' (new)') + ` · ${METER_TYPE}`;

  const rois = data.rois;
  if (METER_TYPE === 'digital') {
    if (rois.total) state.rois.total = roiArrToObj(rois.total);
    if (rois.flow)  state.rois.flow  = roiArrToObj(rois.flow);
  } else {
    if (rois.digits) state.rois.digits = roiArrToObj(rois.digits);
    rois.dials.forEach((d, i) => {
      const key = 'dial_' + i;
      if (d.roi) state.rois[key] = roiArrToObj(d.roi);
      state.dialMeta[key] = {
        rotation: d.rotation || 'cw',
        zero_angle_deg: d.zero_angle_deg ?? -90,
        factor: d.factor ?? DIAL_SLOTS[i].factor,
        name: d.name,
      };
    });
  }
  rois.anchors.forEach((a, i) => {
    if (a) state.rois['anchor_' + i] = roiArrToObj(a);
  });
  renderSidebar();
  await loadImage();
  draw();
}

function roiArrToObj(a) { return { x: a[0], y: a[1], w: a[2], h: a[3] }; }
function roiObjToArr(o) { return [o.x, o.y, o.w, o.h]; }

async function loadImage() {
  return new Promise((resolve) => {
    image.onload = () => {
      imageLoaded = true;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      resolve();
    };
    image.onerror = () => { toast('Failed to fetch frame from ESP32', 'err'); resolve(); };
    image.src = '/frame.jpg?t=' + Date.now();
  });
}

function renderSidebar() {
  const list = document.getElementById('roi-list');
  list.innerHTML = '';
  SLOTS.forEach(s => {
    const row = document.createElement('div');
    row.className = 'roi-row';
    if (state.selected === s.key) row.classList.add('selected');
    if (state.drawing === s.key) row.classList.add('drawing');
    const set = !!state.rois[s.key];
    row.innerHTML = `
      <span class="dot" style="background:${s.color}"></span>
      <span class="label">${s.label}</span>
      <span class="state${set ? ' set' : ''}">${set ? '◆' : '◌'}</span>
    `;
    row.addEventListener('click', () => {
      state.selected = s.key;
      state.drawing = s.key;  // clicking a row arms draw mode
      renderSidebar();
      renderSelectedPanel();
      draw();
    });
    list.appendChild(row);
  });
}

function renderSelectedPanel() {
  const el = document.getElementById('selected-panel');
  if (!state.selected) {
    el.innerHTML = '<div class="hint">No ROI selected.</div>';
    return;
  }
  const s = slotFor(state.selected);
  const roi = state.rois[s.key];
  const res = (state.testResults || {})[s.key];

  const fmt = v => v.toFixed(4);
  let html = `<div class="field"><label>slot</label><input readonly value="${s.label}"></div>`;
  if (roi) {
    html += `
      <div class="field"><label>x</label><input type="number" min="0" max="1" step="0.001"
        value="${fmt(roi.x)}" data-field="x"></div>
      <div class="field"><label>y</label><input type="number" min="0" max="1" step="0.001"
        value="${fmt(roi.y)}" data-field="y"></div>
      <div class="field"><label>w</label><input type="number" min="0" max="1" step="0.001"
        value="${fmt(roi.w)}" data-field="w"></div>
      <div class="field"><label>h</label><input type="number" min="0" max="1" step="0.001"
        value="${fmt(roi.h)}" data-field="h"></div>
    `;
  } else {
    html += `<div class="hint">Drag on the image to place this ROI.</div>`;
  }

  if (s.kind === 'dial') {
    const meta = state.dialMeta[s.key] || { rotation: 'cw', zero_angle_deg: -90 };
    html += `
      <div class="field"><label>rotation</label>
        <select data-field="rotation">
          <option value="cw"${meta.rotation==='cw'?' selected':''}>cw</option>
          <option value="ccw"${meta.rotation==='ccw'?' selected':''}>ccw</option>
        </select>
      </div>
      <div class="field"><label>zero_angle_deg</label>
        <input type="number" step="1" value="${meta.zero_angle_deg}" data-field="zero_angle_deg">
      </div>
    `;
  }

  if (res) {
    if (res.error) {
      html += `<div class="result"><span class="bad">test: ${res.error}</span></div>`;
    } else {
      html += `<div class="result">test: <span class="val">${res.reading.toFixed(3)}</span>
        · confidence <span class="val">${(res.confidence*100).toFixed(0)}%</span></div>`;
    }
  }

  el.innerHTML = html;
  el.querySelectorAll('input[data-field], select[data-field]').forEach(inp => {
    inp.addEventListener('change', onFieldChange);
    inp.addEventListener('input', onFieldChange);
  });
}

function onFieldChange(e) {
  if (!state.selected) return;
  const s = slotFor(state.selected);
  const field = e.target.dataset.field;
  let v = e.target.value;
  const roi = state.rois[s.key] || {};
  if (['x','y','w','h'].includes(field)) {
    v = parseFloat(v);
    if (isNaN(v)) return;
    roi[field] = v;
    state.rois[s.key] = roi;
  } else if (s.kind === 'dial') {
    state.dialMeta[s.key] = { ...(state.dialMeta[s.key] || {}) };
    if (field === 'rotation') state.dialMeta[s.key].rotation = v;
    else if (field === 'zero_angle_deg') state.dialMeta[s.key].zero_angle_deg = parseFloat(v);
  }
  draw();
  renderSidebar();
}

// ----- drawing loop --------------------------------------------------------

function draw() {
  if (!imageLoaded) return;
  ctx.drawImage(image, 0, 0);
  // Existing ROIs
  SLOTS.forEach(s => {
    const roi = state.rois[s.key];
    if (!roi) return;
    drawRect(roi.x * canvas.width, roi.y * canvas.height,
             roi.w * canvas.width, roi.h * canvas.height,
             s.color, state.selected === s.key, s.label);
  });
  // In-progress drag
  if (state.dragStart && state.dragCur && state.drawing) {
    const s = slotFor(state.drawing);
    const a = state.dragStart, b = state.dragCur;
    const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
    const w = Math.abs(a.x - b.x), h = Math.abs(a.y - b.y);
    drawRect(x, y, w, h, s.color, true, s.label + ' (drawing)');
  }
}

function drawRect(px, py, pw, ph, color, highlighted, label) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = highlighted ? 3 : 2;
  ctx.setLineDash(highlighted ? [] : [6, 4]);
  ctx.strokeRect(px + 0.5, py + 0.5, pw, ph);
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText(label, px + 4, Math.max(12, py - 4));
  ctx.restore();
}

// ----- mouse events --------------------------------------------------------

function canvasPos(ev) {
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width, sy = canvas.height / r.height;
  return { x: (ev.clientX - r.left) * sx, y: (ev.clientY - r.top) * sy };
}

canvas.addEventListener('mousedown', (ev) => {
  if (!state.drawing) return;
  state.dragStart = canvasPos(ev);
  state.dragCur = state.dragStart;
});

canvas.addEventListener('mousemove', (ev) => {
  if (!state.dragStart) return;
  state.dragCur = canvasPos(ev);
  draw();
});

canvas.addEventListener('mouseup', (ev) => {
  if (!state.dragStart || !state.drawing) { state.dragStart = null; return; }
  const a = state.dragStart, b = canvasPos(ev);
  state.dragStart = null;
  state.dragCur = null;
  const xPx = Math.min(a.x, b.x), yPx = Math.min(a.y, b.y);
  const wPx = Math.abs(a.x - b.x), hPx = Math.abs(a.y - b.y);
  if (wPx < 3 || hPx < 3) { state.drawing = null; renderSidebar(); draw(); return; }
  const roi = {
    x: xPx / canvas.width, y: yPx / canvas.height,
    w: wPx / canvas.width, h: hPx / canvas.height,
  };
  state.rois[state.drawing] = roi;
  state.drawing = null;  // finish draw mode; user can click the row again to redraw
  renderSidebar();
  renderSelectedPanel();
  draw();
});

// ----- actions -------------------------------------------------------------

function buildPayload() {
  const anchors = [0, 1, 2].map(i => {
    const r = state.rois['anchor_' + i];
    return r ? roiObjToArr(r) : null;
  });
  if (METER_TYPE === 'digital') {
    return {
      total: state.rois.total ? roiObjToArr(state.rois.total) : null,
      flow:  state.rois.flow  ? roiObjToArr(state.rois.flow)  : null,
      anchors,
    };
  }
  const dials = DIAL_SLOTS.map((d, i) => {
    const key = 'dial_' + i;
    const roi = state.rois[key];
    const meta = state.dialMeta[key] || {};
    return {
      name: meta.name || d.label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+$/, ''),
      roi: roi ? roiObjToArr(roi) : null,
      factor: meta.factor ?? d.factor,
      rotation: meta.rotation || 'cw',
      zero_angle_deg: meta.zero_angle_deg ?? -90,
    };
  });
  return {
    digits: state.rois.digits ? roiObjToArr(state.rois.digits) : null,
    dials,
    anchors,
  };
}

document.getElementById('refresh-btn').addEventListener('click', async () => {
  await loadImage();
  draw();
  toast('Frame refreshed', 'ok');
});

document.getElementById('test-btn').addEventListener('click', async () => {
  const payload = buildPayload();
  const r = await fetch('/test', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  if (!r.ok || data.error) { toast(data.error || 'test failed', 'err'); return; }
  state.testResults = {};
  if (METER_TYPE === 'digital') {
    toast(data.note || 'Digital ROIs validated', 'ok');
  } else {
    (data.dials || []).forEach((res, i) => { state.testResults['dial_' + i] = res; });
    toast('Test complete — see per-dial readings in the sidebar', 'ok');
  }
  renderSelectedPanel();
});

document.getElementById('save-btn').addEventListener('click', async () => {
  const payload = buildPayload();
  const r = await fetch('/config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  if (!r.ok || data.error) { toast(data.error || 'save failed', 'err'); return; }
  const msg = data.backup
    ? `Saved. Backup: ${data.backup.split('/').pop()}`
    : 'Saved (no prior file to back up).';
  toast(msg, 'ok');
});

loadConfig();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()
