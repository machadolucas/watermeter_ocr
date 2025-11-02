
# Water Meter OCR (Mac) — Alignment Edition

This service aligns each frame to a **reference image** (ORB + RANSAC) so slight camera moves or vibration don't break your ROIs.

- Uses Apple **Vision** for per-digit OCR (full/top/bottom) to resolve rolling digits.
- Uses OpenCV for the 4 analog dials.
- Composes total with **monotonic** clamp and publishes to Home Assistant via MQTT discovery.

## Install
```bash
./install.sh
open -e ~/watermeter/config.yaml   # set ESP32 IP + MQTT
python3 save_reference.py          # (optional) take a clean reference
tail -f ~/watermeter/watermeter.log
```

The first run will also save the first good frame as `~/watermeter/reference.jpg` automatically if none exists.

## Tuning alignment
- `alignment.anchor_rois`: rectangles where features are extracted (keep them on static, printed areas).
- `max_scale_change`, `max_rotation_deg`: reject crazy transforms.
- Debug file: `~/watermeter/aligned_last.jpg` shows the aligned frame used for OCR.
