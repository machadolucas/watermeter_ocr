# Configuration Reference

End-to-end guide to configuring `watermeter_ocr`, including the meter-swap procedure, the interactive calibration tool, and the `--reset-total` flag. Complements the architecture notes in [AGENTS.md](../AGENTS.md) and the user-facing intro in [README.md](../README.md).

## Meter types

`meter.type` selects which reading pipeline runs:

| Value | When to use | Pipeline |
|---|---|---|
| `mechanical` (default) | Traditional meter with 5 odometer digits + 4 red-pointer dials. | Digit OCR + dial-angle reading → `compose_integer` → guards → publish. |
| `digital` | Fully electronic LCD (e.g. Qalcosonic W1). Shows two text lines: total (m³) and instant flow (m³/h). | Line OCR (Apple Vision only) → regex parse → guards → publish. No dials, no composition. |

Everything below that mentions **dials**, **rolling thresholds**, **auto-centering**, or **`compose_integer`** applies only to mechanical meters. See the [Digital meter](#digital-meter) section for the digital-specific keys and procedure.

## Local development environment

The deployed service lives at `~/watermeter/` and uses its own virtualenv at `~/watermeter/venv/` — that's where the Python dependencies (cv2, paho-mqtt, requests, yaml, numpy) actually are. Running `python3 calibrate.py` or `python3 watermeter.py` with your system Python will fail with `ModuleNotFoundError: No module named 'cv2'`.

Three supported ways to run:

```bash
# Option A — point at the deployed venv (recommended; deps are already installed)
~/watermeter/venv/bin/python3 calibrate.py
~/watermeter/venv/bin/python3 watermeter.py --config ~/watermeter/config.yaml

# Option B — activate the deployed venv, then run plainly
source ~/watermeter/venv/bin/activate
python3 calibrate.py

# Option C — fresh local dev venv from the repo (for running pytest, reading code locally)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

`install.sh` creates `~/watermeter/venv/` as a side effect the first time you run it; don't recreate it on each invocation.

## Where config lives

Two copies of `config.yaml`:

| Path | Purpose |
|---|---|
| [`config.yaml`](../config.yaml) (in the repo) | Template shipped with the code. Reflects the author's current meter; use as a starting point when copying to a new machine. |
| `~/watermeter/config.yaml` | **The file the service actually reads.** Created by [`install.sh`](../install.sh) the first time you run it (copied from the repo). All subsequent edits happen here. |

Running [`calibrate.py`](../calibrate.py) edits `~/watermeter/config.yaml` directly and leaves a timestamped backup next to it: `~/watermeter/config.yaml.bak.<YYYYMMDDTHHMMSS>`. If you want to revert, replace the live file with the backup.

## Quick reference

| Key | Default | Purpose |
|---|---|---|
| `esp32.base_url` | — | Base URL of the ESP32 camera. The service fetches `{base_url}/capture_with_flashlight`. |
| `processing.interval_sec` | `10` | Normal sampling interval. |
| `processing.retry_backoff_sec` | `5` | Delay after a failed capture before retrying. |
| `processing.image_path` | `/tmp/water_raw.jpg` | Where the fetched JPEG is cached. |
| `processing.image_timeout` | `8.0` | HTTP timeout for the ESP32 capture. |
| `processing.save_debug_overlays` | `true` | Whether to write annotated overlay JPEGs. |
| `processing.debug_dir` | `~/watermeter/debug` | Overlay destination. |
| `processing.debug_keep_latest_only` | `true` | `true` overwrites a single `overlay_latest.jpg`; `false` timestamps each. |
| `processing.quiet_hours.*` | off | Reduced sampling window (see below). |
| `mqtt.host`, `port`, `username`, `password` | — | Broker connection. |
| `mqtt.topic` | `home/watermeter` | Base topic for published sensor values. |
| `mqtt.client_id` | `water-ocr-mac` | MQTT client id. |
| `mqtt.ha_discovery_prefix` | `homeassistant` | Prefix for Home Assistant MQTT Discovery configs. |
| `overlay.publish_mqtt` | `true` | Publish the debug overlay JPEG to MQTT as a camera entity. |
| `overlay.camera_topic` | `{mqtt.topic}/debug/overlay` | Topic the overlay JPEG is published to. |
| `overlay.font_scale`, `font_thickness`, `outline_thickness`, `line_thickness` | — | Overlay rendering. |
| `overlay.jpeg_quality` | `85` | Re-encode quality before MQTT publish. |
| `digits.count` | `5` | *(mechanical)* Number of odometer digits. MUST match the physical meter. |
| `digits.per_digit_inset` | `0.10` | *(mechanical)* Fraction of each cell's width/height trimmed inward (avoids grabbing the plastic frame). |
| `digits.rolling_threshold_up` | `0.92` | *(mechanical)* Progress level above which the rolling-digit resolver commits to the NEW digit. |
| `digits.rolling_threshold_down` | `0.08` | *(mechanical)* Progress level below which it commits to the OLD digit. |
| `rois.digits` | — | *(mechanical)* Single window enclosing all odometer digits. |
| `rois.dials[]` | — | *(mechanical)* One entry per dial, ordered `×0.1 → ×0.0001`. **dials[0] must be the tenths wheel.** |
| `meter.type` | `mechanical` | Pipeline selector: `mechanical` or `digital`. |
| `rois.digital.total` | — | *(digital)* ROI of the total-consumption LCD line. |
| `rois.digital.flow` | — | *(digital)* ROI of the instant-flow LCD line. |
| `digital.total_regex` | `^\d{6}\.?\d{3}$` | *(digital)* Regex that Vision's raw total output must match. |
| `digital.flow_regex` | `^\d{1,3}\.\d{3}$` | *(digital)* Regex for Vision's flow output. |
| `digital.max_retries` | `2` | *(digital)* Extra captures per cycle if the display is on a diagnostic view. |
| `digital.retry_delay_sec` | `5.5` | *(digital)* Delay between retries. Must exceed longest diag-view dwell (5 s). |
| `digital.min_digits` | `6` | *(digital)* Cheap-fail threshold that rejects diagnostic screens before regex. |
| `postproc.monotonic_epsilon` | `0.0005` | Tolerated backward wobble; anything smaller than this is treated as noise, not a real regression. |
| `postproc.big_jump_guard` | `2.0` | Maximum accepted forward jump in m³ between captures. Anything larger is blocked. |
| `alignment.enabled` | `true` | Run ORB+RANSAC alignment to the reference frame. |
| `alignment.reference_path` | `~/watermeter/reference.jpg` | Reference image for alignment. Auto-captured on first run if missing. |
| `alignment.anchor_rois` | — | Normalized rectangles where ORB is allowed to look for features. Keep over stable printed areas. |
| `alignment.nfeatures`, `ratio_test`, `min_matches`, `ransac_thresh_px`, `max_scale_change`, `max_rotation_deg`, `warp_mode` | — | ORB/RANSAC tunables; defaults are sensible. |
| `auto_centering.enabled` | `true` | Auto-detect each dial center and nudge the ROI inward (helpful against small camera drift). |
| `auto_centering.smoothing_alpha` | `0.3` | EMA weight for ROI adjustments; lower is slower/stabler. |
| `auto_centering.min_confidence_threshold` | `0.4` | Warn when a dial-reading confidence drops below this. |

## The ROI coordinate system

All ROIs in this project are **normalized `[x, y, w, h]` in `[0, 1]`**, with the origin at the top-left of the frame. `x + w` and `y + h` must stay ≤ 1.0.

```
 (0,0) ────────────► x
   │
   │    ┌──────┐
   │    │ ROI  │  ← [x, y, w, h] = [0.25, 0.30, 0.40, 0.20]
   │    └──────┘
   ▼
   y
```

Common mistakes:
- Using `[x1, y1, x2, y2]` (two-corners). This format is not supported anywhere in the config.
- Pixel coordinates. If you measured a ROI in pixels, divide by the image width/height before writing.
- Negative values or coordinates outside `[0, 1]`.

## Dials: order, rotation, zero angle

The dial list in `rois.dials` is **strictly ordered by precision** from most-significant to least-significant:

```yaml
rois:
  dials:
    - name: dial_0_1      # ×0.1 m³   — tenths wheel, the "progress" signal
      factor: 0.1
      ...
    - name: dial_0_01     # ×0.01 m³  — hundredths
      factor: 0.01
      ...
    - name: dial_0_001    # ×0.001 m³ — thousandths
      factor: 0.001
      ...
    - name: dial_0_0001   # ×0.0001 m³ — ten-thousandths (smallest)
      factor: 0.0001
      ...
```

**`dials[0]` MUST be the tenths wheel.** The rolling-digit logic in [`watermeter.py:1090`](../watermeter.py:1090) hardcodes `dial_vals[0]` as the progress signal; putting the wrong dial first silently breaks integer resolution. [`calibrate.py`](../calibrate.py) enforces this before writing.

### Tuning `rotation` and `zero_angle_deg`

Each dial has two discrete tunables:

- `rotation`: either `"cw"` (clockwise) or `"ccw"` (counter-clockwise). This is the direction the dial face increments as water flows. Flip if readings run **backwards** as water is used.
- `zero_angle_deg`: the screen-space angle (degrees, image coords — `0`=right, `90`=down, `-90`=up) at which the "0" marking sits. Default is `-90` (zero at the top). Nudge by small amounts (±5–10°) if the reading is **consistently off by the same fraction** across the full revolution.

Quick test with the calibration tool: after drawing the dial ROIs, click **Test current config**. The sidebar shows each dial's computed reading and confidence. Cross-check with what the physical needle is pointing at.

## Rolling thresholds

`digits.rolling_threshold_up` and `rolling_threshold_down` control when the integer-digit resolver flips:

- When the tenths dial's progress (`0..1`) is **above `rolling_threshold_up`**, the digit rolls forward (e.g. `9 → 0`, ones place also carries).
- When progress is **below `rolling_threshold_down`**, the digit has freshly rolled and is stable at its new value.
- **In between** (the stable zone), the resolver prefers the previous known digit — this is what lets it survive brief OCR misreads.

Defaults (0.92 / 0.08) work for most meters. Tighten (e.g. 0.85 / 0.15) if digits feel slow to flip when the meter actually rolls; widen if OCR jitter is causing spurious flips.

## Guards

Two postprocessing guards in `postproc`:

- **`monotonic_epsilon`**: a backward step of up to this many m³ is treated as noise and the published total is held at the previous value. Keeps tiny jitter from making the Home Assistant total_increasing sensor reset.
- **`big_jump_guard`**: a forward jump of more than this many m³ between captures is blocked. The dial-based `estimate_total_from_dials` fallback kicks in; if it also doesn't fit, the reading is discarded. Prevents an OCR misread ("9" → "8" misread as "9" → "0" at a higher place value) from inflating the total by 100 m³.

Both guards are relative to `prev_total` in `state.json`. After a physical meter swap the new meter's reading will almost certainly trip `big_jump_guard` (the running total drops from ~12k m³ to 0). Use `--reset-total` to clear state — see below.

## Alignment (`anchor_rois`)

The aligner uses ORB features + RANSAC to warp each incoming frame to the saved reference. `anchor_rois` is a list of normalized rectangles that mask WHERE on the frame ORB is allowed to look for keypoints.

Rules of thumb:
- Keep anchors over **stable printed areas** — the meter face labels, company logos, the plastic frame.
- **Avoid the needles and rolling digits.** Moving features confuse feature matching.
- 1–3 anchors is typical. More doesn't help if they're all over the same region.
- If alignment regularly fails (logs show `aligned_ok=False`), widen the anchors or pick different anchor regions.

## Digital meter

For fully-digital LCD meters (e.g. the Qalcosonic W1). Set `meter.type: digital` and calibrate the two text-line ROIs — no dials.

### Config keys

```yaml
meter:
  type: digital                         # default "mechanical"

processing:
  interval_sec: 15                      # recommended — see "Retry budget" below

rois:
  digital:
    total: [x, y, w, h]                 # line with the total consumption (m³)
    flow:  [x, y, w, h]                 # line with the instantaneous flow (m³/h)

digital:
  # Regex the parser uses to validate Vision's raw output. Default accepts both
  # "000100.000" (decimal detected) and "000100000" (decimal missed — parser
  # injects it at position -3). Adjust if your meter shows a different layout.
  total_regex: "^\\d{6}\\.?\\d{3}$"
  flow_regex:  "^\\d{1,3}\\.\\d{3}$"
  max_retries: 2                        # extra captures per cycle on a wrong view
  retry_delay_sec: 5.5                  # must exceed longest diag-view dwell
  min_digits: 6                         # cheap-fail bar for diagnostic screens
```

`alignment.*` still applies — ORB alignment of the whole frame helps keep the LCD ROIs stable under small camera drift, even without dials.

### How the pipeline works

1. Capture a JPEG from `/capture_with_flashlight`.
2. Optionally align against `reference.jpg`.
3. Run the Swift helper in `--mode line` over the two ROIs. It uses Apple Vision's text recognizer; on empty output it retries with `.accurate`. The returned string is digits plus optional `.`.
4. Check `is_valid_digital_view` — cheap pre-validator that rejects frames with too few digit characters (diagnostic screens).
5. Parse each line with the configured regex. `parse_digital_total` injects a decimal at position -3 if Vision didn't see one.
6. Run `validate_digital_reading` — reject non-finite, negative flow, and jumps larger than `big_jump_guard`.
7. If any step fails, retry up to `digital.max_retries` times with `digital.retry_delay_sec` between captures. After that, skip the cycle and hold the previous value.
8. Apply the usual monotonic / big-jump guards to the total and publish.

### The rotating display

The Qalcosonic W1 auto-cycles between three views:

| View | Dwell time |
|---|---|
| Numeric (total + flow) | ~10 s |
| Diagnostic 1 | ~5 s |
| Diagnostic 2 | ~4 s |

Any single capture has a ~53 % chance of landing on the numeric view. `digital.retry_delay_sec = 5.5` guarantees that a retry crosses at least one dwell boundary (> longest diag-view of 5 s), so the retries don't phase-lock onto the same wrong view.

**Retry budget**: 3 captures × ~0.5 s + 2 × 5.5 s delays ≈ 12 s worst case. That's why the recommended `processing.interval_sec: 15` for digital mode — it gives the loop headroom to finish even when the first two captures miss.

### MQTT topics

Digital mode publishes all the mechanical topics plus a new one:

| Topic | Retained | Source |
|---|---|---|
| `{mqtt.topic}/main/value` | yes | Parsed total from the LCD (after guards). |
| `{mqtt.topic}/main/rate` | no | Delta-computed m³/min (same formula as mechanical). |
| `{mqtt.topic}/main/rate_lpm` | no | Delta-computed L/min. |
| `{mqtt.topic}/main/flow_m3h` | no | **New.** Instantaneous flow read directly off the LCD (more responsive than the delta-based rate). |
| `{overlay.camera_topic}` | yes | Debug overlay JPEG. |

Home Assistant discovery payload for `water_flow_m3h` is published **only** when `meter.type: digital`. Mechanical installs don't see a dead sensor.

### HA automations — which rate to use?

- **`water_flow_m3h`** reflects what the meter itself reports right now. No quantization from `1/dt`, no jitter from OCR noise. Use this for leak detectors, shower timers, and anything that needs sub-minute response.
- **`water_rate_lpm`** is the delta from the previous total, same as mechanical. Slightly lagged, but matches historical data from before the meter swap. Keep it if your existing automations depend on it.

## Meter-swap procedure

When your water company replaces the physical meter, the ROIs won't match and the running total will reset to 0 on the new meter. Walk through:

### 1. Stop the service

```bash
launchctl unload ~/Library/LaunchAgents/com.watermeter.ocr.plist
```

### 2. Delete the old alignment reference

```bash
rm ~/watermeter/reference.jpg
```

The service will auto-capture a new reference from the first frame after restart.

### 3. Calibrate ROIs against a live frame

From the repo directory (or anywhere, really — `calibrate.py` uses the deployed venv):

```bash
~/watermeter/venv/bin/python3 calibrate.py
# Or with activation:
source ~/watermeter/venv/bin/activate && python3 calibrate.py
```

A browser tab opens at `http://127.0.0.1:8765` showing the current meter image. The sidebar reflects the `meter.type` you have in config:

- **Mechanical** → 1 digits window + 4 dial ROIs + 3 alignment anchors.
- **Digital** → 2 line ROIs (total, flow) + 3 alignment anchors.

For each ROI in the sidebar:

1. Click the row (turns amber — "draw mode").
2. Drag a rectangle on the image to place it.
3. (Mechanical dials only) set `rotation` and `zero_angle_deg` in the sidebar.

Iterate. Click **Test current config** whenever you want to verify a dial's reading (mechanical only — digital mode's live OCR runs on the service itself, not in the calibrator). When everything looks right:

4. Click **Save to config.yaml**. A toast confirms the backup filename.

The tool validates before writing — invalid ROIs show an error toast and the live file is untouched.

### 4. Reset the accumulated total

The new meter starts at its own reading (usually 0 m³, sometimes the utility pre-loads a number). Tell `watermeter.py` to reset `state.json` so the monotonic and big-jump guards accept the new value:

```bash
# New meter starting at 0
.venv/bin/python3 watermeter.py --config ~/watermeter/config.yaml --reset-total

# Or starting at a specific value
.venv/bin/python3 watermeter.py --config ~/watermeter/config.yaml --reset-total 12.345
```

What this does:
- Backs up the existing `~/watermeter/state.json` to `state.json.bak.<timestamp>`.
- Writes a fresh state with `total=VALUE, ts=now, dial_histories={}`.
- Exits immediately — does NOT enter the capture loop.

### 5. Restart the service

```bash
launchctl load ~/Library/LaunchAgents/com.watermeter.ocr.plist
```

### 6. Watch it come up

```bash
tail -f ~/watermeter/watermeter.log
```

Confirm that Home Assistant's `sensor.water_total` accepts the new reading. If HA is showing it as "unavailable" for longer than ~30 s, check `launch_stderr.log` and the broker.

### Swapping from a mechanical meter to a digital one

The procedure is the same as above, with three differences:

1. **Set `meter.type: digital`** in `~/watermeter/config.yaml` *before* running `calibrate.py` (so the UI shows line ROIs instead of dials).
2. **Recommended**: bump `processing.interval_sec` to `15` — see "Retry budget" under the [Digital meter](#digital-meter) section.
3. **On first start**, `state.json` will contain `meter_type: mechanical` from the old install. The service detects this, logs `state.json meter_type='mechanical' != config meter_type='digital'; discarding prev_total`, and the next reading becomes the new baseline. No user action required — but running `--reset-total` explicitly is cleaner if you know the new meter's starting value.

Example end-to-end after the new meter is installed:

```bash
launchctl unload ~/Library/LaunchAgents/com.watermeter.ocr.plist
# Edit ~/watermeter/config.yaml: meter.type: digital, processing.interval_sec: 15
rm ~/watermeter/reference.jpg                                  # fresh reference for the new face
~/watermeter/venv/bin/python3 calibrate.py                     # draw two line ROIs + anchors
~/watermeter/venv/bin/python3 watermeter.py \
    --config ~/watermeter/config.yaml --reset-total 0          # or whatever the LCD shows
launchctl load ~/Library/LaunchAgents/com.watermeter.ocr.plist
tail -f ~/watermeter/watermeter.log
```

## Manual ROI tuning (without the GUI)

If you're on a headless box or just prefer editing YAML:

1. Grab a current frame: `python3 save_reference.py` — writes to `~/watermeter/reference.jpg`.
2. Open the image in any image viewer that shows pixel coordinates. (macOS Preview: tap `⌘-Shift-i` for inspector.)
3. For each ROI you want to define:
   - Measure top-left (x_px, y_px) and size (w_px, h_px) in pixels.
   - Compute normalized: `x = x_px / image_width`, etc.
4. Write the values into `~/watermeter/config.yaml`.
5. Restart the service and watch `~/watermeter/debug/overlay_latest.jpg` — the overlay shows the computed boxes. Nudge until they hug the correct regions.

The calibration GUI does all of this interactively, but the manual path works fine if you already know the coordinates.

## `state.json` and `--reset-total`

`~/watermeter/state.json` is the service's only persistent state. Schema:

```json
{
  "total": 12345.678,
  "ts": 1700000000.0,
  "dial_histories": {
    "dial_0": [5.1, 5.2, 5.3, ...],
    "dial_1": [...]
  }
}
```

- `total` — last published m³. Compared against each new reading for the monotonic and big-jump guards.
- `ts` — Unix timestamp of that reading. Used to compute flow rate.
- `dial_histories` — last ~20 readings per dial, for temporal validation and prediction. Can be safely cleared if bad readings polluted it.

### When to use `--reset-total` vs hand-editing

- **Use `--reset-total`** after a physical meter swap, a known bad data point, or any situation where the running total no longer reflects reality. It backs up the old state and writes a known-good new one.
- **Hand-edit** if you want to tweak just `dial_histories` (clear one dial's history without nuking the total) or correct a numeric typo. Stop the service first; edit; restart. No backup is made automatically — copy the file before editing.

Safety on `--reset-total`:
- Rejects negative values.
- Rejects `inf` / `nan`.
- Always backs up the old `state.json`.
- Does not touch `config.yaml`, `reference.jpg`, MQTT state, or the LaunchAgent.

## Troubleshooting common misconfigurations

**Every dial reads ~half its true value (off by 5 across the board).**
- `rotation` is inverted. Switch all dials between `cw` and `ccw` and re-test.

**One dial consistently reads ~3 off in the same direction.**
- `zero_angle_deg` is off for that dial. Nudge by `-5°` or `+5°` until the test reading matches the physical needle.

**Integer digits flip early or late around rollover.**
- Adjust `digits.rolling_threshold_up` / `rolling_threshold_down`. Tighter thresholds (0.85 / 0.15) flip sooner; wider thresholds (0.95 / 0.05) flip later.

**Readings lag way behind the physical meter.**
- Usually `big_jump_guard` is clamping every new reading. Check the log for `CLAMP: big jump detected`. After a meter swap, run `--reset-total`. If you legitimately see > 2 m³ per 10 s of flow, raise `big_jump_guard` — rare for residential use.

**Alignment keeps failing (`aligned_ok=False` in logs).**
- Rebuild the reference: `rm ~/watermeter/reference.jpg`. The next capture becomes the new reference.
- Check your `anchor_rois` — they must not overlap the needles or rolling digits. Use [`overlay_example.jpg`](../overlay_example.jpg) or a captured frame to verify.
- Lower `alignment.min_matches` if the camera gives a noisy image (trade-off: more false positives).

**Home Assistant shows `sensor.water_total` as "unavailable".**
- MQTT not connected. Check logs for `MQTT connect rejected` (bad credentials) or `MQTT disconnected`. Also verify the broker is reachable: `mosquitto_sub -h <host> -t 'home/watermeter/#'`.

**HA accepts readings but `sensor.water_total` appears to reset.**
- This usually means you reset the state without informing HA. `total_increasing` sensors expect either strictly monotonic growth or a detectable device-reboot. HA uses the `last_reset` attribute for this, which `watermeter_ocr` currently doesn't publish — a reset may briefly look like a rollback. After a meter swap, HA typically recovers within a few sample periods.
