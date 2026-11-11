#!/usr/bin/env python3
"""Capture one clean ESP32 frame and save it as the alignment reference.

Usage:
    python3 save_reference.py [CONFIG]      # default ~/watermeter/config.yaml

Captures exactly like the service (watermeter._fetch_jpeg): honours
esp32.capture_mode (ESPHome flash-on → settle → discard → snapshot, with the
flash always switched off again, or AI-on-the-edge /capture_with_flashlight)
and the max_image_bytes cap. The frame must decode, and the reference is
written to a temp file, read back, then moved into place; an existing
reference is first copied to reference.jpg.bak.<timestamp>. On any failure
the old reference is left untouched.

Stop the LaunchAgent first so the service's own captures don't overlap.

Exit codes: 0 saved, 1 capture/decode/write failed, 78 invalid config.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import watermeter  # noqa: E402

DEFAULT_CONFIG = os.path.expanduser("~/watermeter/config.yaml")


class CaptureError(RuntimeError):
    pass


def capture_reference(cfg, session, log) -> "np.ndarray":
    """One decoded frame via the service's capture helper; raises CaptureError."""
    try:
        content = watermeter._fetch_jpeg(cfg, session, log)
    except (requests.RequestException, ValueError) as e:
        raise CaptureError(f"capture from {cfg.esp32_base_url} failed: {e}") from e
    if not content:
        raise CaptureError("camera returned an empty image body")
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise CaptureError(f"camera returned {len(content)} bytes that do not decode as an image")
    return img


def write_reference(img, ref_path: str) -> "str | None":
    """Atomically replace ref_path with img. Returns the backup path, if any."""
    os.makedirs(os.path.dirname(ref_path) or ".", exist_ok=True)
    tmp = ref_path + ".tmp.jpg"   # cv2 picks the encoder from the extension
    try:
        if not cv2.imwrite(tmp, img):
            raise CaptureError(f"could not write {tmp}")
        check = cv2.imread(tmp, cv2.IMREAD_COLOR)
        if check is None or check.shape != img.shape:
            raise CaptureError(f"{tmp} did not read back as the captured frame")
        backup = None
        if os.path.exists(ref_path):
            backup = f"{ref_path}.bak.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            shutil.copy2(ref_path, backup)
        os.replace(tmp, ref_path)
        return backup
    except OSError as e:
        raise CaptureError(f"could not write {ref_path}: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None, session=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg_path = os.path.expanduser(argv[0]) if argv else DEFAULT_CONFIG
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log = logging.getLogger("save_reference")
    try:
        cfg = watermeter.load_config(cfg_path)
    except watermeter.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return watermeter.EX_CONFIG
    except OSError as e:
        print(f"error: cannot read config {cfg_path}: {e}", file=sys.stderr)
        return watermeter.EX_CONFIG
    ref = os.path.expanduser(cfg.align.reference_path)
    print(f"Capturing from {cfg.esp32_base_url} ({cfg.esp32_capture_mode}) ...")
    own_session = session is None
    session = session or requests.Session()
    try:
        img = capture_reference(cfg, session, log)
        backup = write_reference(img, ref)
    except CaptureError as e:
        print(f"error: {e}; the reference was not changed", file=sys.stderr)
        return 1
    finally:
        if own_session:
            session.close()
    h, w = img.shape[:2]
    print(f"Saved reference to {ref} ({w}x{h})")
    if backup:
        print(f"Previous reference kept as {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
