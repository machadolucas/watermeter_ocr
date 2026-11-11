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

Running [`calibrate.py`](../calibrate.py) edits `~/watermeter/config.yaml` directly and leaves a timestamped backup next to it: `~/watermeter/config.yaml.bak.<YYYYMMDDTHHMMSSffffff>` (mode 0600, since the file holds the MQTT password). If you want to revert, replace the live file with the backup.

### Validation

`load_config` validates the whole file and refuses to start on any problem, listing all of them at once as `dotted.key: message (got X)`. It catches unknown keys (with a "did you mean"), bad enums (`meter.type`, `esp32.capture_mode`, `digital.ocr_preprocess`, `alignment.warp_mode`, dial `rotation`), out-of-range numbers, ROIs outside the frame, regexes that don't compile, and unquoted `HH:MM` quiet-hour times (YAML reads `17:30` as a number). It also enforces the digital-mode rules: a total source is required, the integer and fractional sides go together, and the per-digit cell counts must make a reading `digital.total_regex` parses (6 + 3 with the default regex). Check a config without starting anything:

```bash
~/watermeter/venv/bin/python3 watermeter.py --check-config --config ~/watermeter/config.yaml
# config OK: /Users/.../config.yaml (digital)      exit 0;  invalid -> errors on stderr, exit 78
```

The service exits 78 (EX_CONFIG) before logging, MQTT or any network access when the config is invalid. `install.sh` runs this check on the new binary before it replaces the running one.

## Quick reference

| Key | Default | Purpose |
|---|---|---|
| `esp32.base_url` | — | Base URL of the ESP32 camera. The service fetches `{base_url}/capture_with_flashlight`. |
| `processing.interval_sec` | `10` | Normal sampling interval. |
| `processing.retry_backoff_sec` | `5` | Delay after a failed capture before retrying. |
| `processing.image_path` | `/tmp/water_raw.jpg` | Where the fetched JPEG is cached. |
| `processing.image_timeout` | `8.0` | HTTP timeout for the ESP32 capture. |
| `processing.max_image_bytes` | `2000000` | Empty or larger HTTP bodies are treated as capture failures. |
| `processing.save_debug_overlays` | `true` | Whether to write annotated overlay JPEGs. |
| `processing.debug_dir` | `~/watermeter/debug` | Overlay destination. |
| `processing.debug_keep_latest_only` | `true` | `true` overwrites a single `overlay_latest.jpg`; `false` timestamps each. |
| `processing.quiet_hours.*` | off | Reduced sampling window (see below). |
| `mqtt.host`, `port`, `username`, `password` | — | Broker connection. |
| `mqtt.topic` | `home/watermeter` | Base topic for published sensor values. |
| `mqtt.client_id` | `water-ocr-mac` | MQTT client id. |
| `mqtt.ha_discovery_prefix` | `homeassistant` | Prefix for Home Assistant MQTT Discovery configs. |
| `mqtt.availability_topic` | `{mqtt.topic}/status` | Availability topic (LWT `offline`, retained, QoS 1). |
| `mqtt.stale_after_sec` | `600` | Publish `offline` once no reading has been accepted for this long; `0` disables. |
| `paths.log_max_bytes` | `10000000` | Rotate the log at this size. |
| `paths.log_backup_count` | `5` | Rotated log files kept. |
| `alignment.min_inliers` | `25` | Alignment quality gate: minimum RANSAC inliers (good frames show ≥ 91). |
| `alignment.min_inlier_ratio` | `0.5` | ...minimum inliers / good matches (good frames ≥ 0.83). |
| `alignment.max_translation_frac` | `0.10` | ...maximum shift as a fraction of frame width/height. |
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
| `rois.digital.total` | — | *(digital)* ROI of the total-consumption LCD line. **Required** in digital mode. |
| `rois.digital.flow` | — | *(digital)* ROI of the instant-flow LCD line. **Optional** — omit to skip flow OCR and suppress the `flow_m3h` sensor (use when flash glare makes both lines unreadable in the same capture). |
| `digital.total_regex` | `^\d{6}\.?\d{3}$` | *(digital)* Regex that Vision's raw total output must match. |
| `digital.flow_regex` | `^\d{1,3}\.\d{3}$` | *(digital)* Regex for Vision's flow output. |
| `digital.max_retries` | `2` | *(digital)* Extra captures per cycle if the display is on a diagnostic view. |
| `digital.retry_delay_sec` | `5.5` | *(digital)* Delay between retries. Must exceed longest diag-view dwell (5.0 s). |
| `digital.cycle_budget_sec` | `45` | *(digital)* No new attempts once a cycle has run this long. |
| `digital.view_sync.enabled` | `false` | *(digital)* Time captures to the LCD's numeric view (see "The rotating display"). |
| `digital.view_sync.period_sec` | `20.0` | *(digital)* View-rotation period. |
| `digital.view_sync.numeric_dwell_sec` | `11.0` | *(digital)* Length of the numeric window. |
| `digital.view_sync.margin_sec` | `1.0` | *(digital)* Safety margin kept from the window edges. |
| `digital.view_sync.capture_lead_sec` | `1.5` | *(digital)* Initial start-to-kept-GET-midpoint delay; adapts by EMA, excluding OCR, body transfer and flash-off. Exposure is a host-side proxy; see timing caveat below. |
| `digital.view_sync.view_dwells_sec` | `null` | *(digital)* Ordered view dwells, numeric first. Positive values must sum to `period_sec`; first must equal `numeric_dwell_sec`. `null` splits the remaining time in B:A proportions 3.9:5.1, giving `[11.0, 3.9, 5.1]` with the defaults. Set an explicit list for other rotations (including two views). |
| `digital.view_sync.drift_per_sec` | `0.0003` | *(digital)* Phase uncertainty growth in seconds per elapsed second (0–0.01). Default covers the measured 0.0002 relative period error with 50% headroom. Doubles on contradictory observations, capped at 0.01. Fractional grid bins accumulate without rounding up each capture. |
| `digital.view_sync.drift_decay_observations` | `180` | *(digital)* Halve adapted drift after this many consecutive consistent observations spanning at least one hour; floor at `drift_per_sec`. `0` disables decay. |
| `digital.view_sync.observation_slack_sec` | `0.25` | *(digital)* Nonaccumulating tolerance at each observed view edge, covering measured ±0.2 s edge jitter and sampling uncertainty. Separate from scheduling's `margin_sec`. |
| `digital.min_digits` | `6` | *(digital)* Cheap-fail threshold that rejects diagnostic screens before regex. |
| `postproc.monotonic_epsilon` | `0.0005` | *(mechanical)* Tolerated backward wobble; anything smaller than this is treated as noise, not a real regression. |
| `postproc.big_jump_guard` | `2.0` | *(mechanical)* Base allowance (m³) for forward jumps between captures. The effective allowance is this PLUS `big_jump_rate_per_day` × elapsed-days-since-last-publish. |
| `postproc.big_jump_rate_per_day` | `1.0` | *(mechanical)* Per-day catch-up rate (m³/day) added to the base guard, scaled by elapsed time since the last accepted publish. Lets the service self-heal after long outages instead of getting stuck. Set to `0` for a fixed guard. |
| `postproc.big_jump_elapsed_cap_days` | `7.0` | *(mechanical)* Cap on the elapsed-time component (in days). After this many days offline the rate-per-day stops adding more allowance. Prevents a wildly bogus OCR read from being accepted after a multi-week outage. |
| `postproc.max_flow_lpm` | `40.0` | *(digital, ReadingGate)* Forward step > this × dt + slack is rejected `too_fast`. |
| `postproc.confirm_above_lpm` | `25.0` | *(digital)* Forward steps up to this × min(dt, `immediate_dt_cap_sec`) + slack are accepted at once. |
| `postproc.immediate_dt_cap_sec` | `75.0` | *(digital)* dt cap for the immediate-accept bound. |
| `postproc.flow_slack_l` | `1` | *(digital)* Litres of slack on both bounds and the backward `hold` band. |
| `postproc.confirm_min_gap_sec` | `15.0` | *(digital)* A candidate must be confirmed by a later independent capture at least this far on. |
| `postproc.candidate_ttl_sec` | `600.0` | *(digital)* An unconfirmed candidate expires after this long. |
| `postproc.rebaseline_count` | `3` | *(digital)* Consecutive agreeing backward readings needed to re-baseline. |
| `postproc.rebaseline_min_span_sec` | `60.0` | *(digital)* Those readings must also span at least this long. |
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

Postprocessing guards in `postproc`:

- **`monotonic_epsilon`**: a backward step of up to this many m³ is treated as noise and the published total is held at the previous value. Keeps tiny jitter from showing up as a backward step in the published total.
- **`big_jump_guard`** + **`big_jump_rate_per_day`** + **`big_jump_elapsed_cap_days`**: the **effective** forward-jump allowance is `big_jump_guard + min(elapsed_days, big_jump_elapsed_cap_days) × big_jump_rate_per_day`. Where `elapsed_days` is the time between the last successful publish and the current cycle. In normal operation (cycles seconds apart) the effective guard is essentially the base `big_jump_guard`. After long stretches of failed cycles (LCD stuck on a diagnostic view, network outage, etc.) the allowance grows so the service can resume publishing once OCR works again, instead of staying trapped behind a stale `prev_total`. The cap prevents a wildly bogus OCR read from being accepted after a multi-week outage.
- The two items above apply to the **mechanical** path. The digital path uses `ReadingGate` (see "Digital meter" below).
- For mechanical meters, when the OCR-integer jumps too far, `estimate_total_from_dials` is consulted as a fallback. If neither fits within the effective guard, the reading is held.

All guards are relative to `prev_total` in `state.json`. After a physical meter swap the new meter's reading will almost certainly trip `big_jump_guard` (the running total drops from ~12k m³ to 0). Use `--reset-total` to clear state — see below.

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
    total: [x, y, w, h]                 # REQUIRED — line with the total consumption (m³)
    flow:  [x, y, w, h]                 # OPTIONAL — line with the instant flow (m³/h); omit to skip

digital:
  # Regex the parser uses to validate Vision's raw output. Default accepts both
  # "000100.000" (decimal detected) and "000100000" (decimal missed — parser
  # injects it at position -3). Adjust if your meter shows a different layout.
  total_regex: "^\\d{6}\\.?\\d{3}$"
  flow_regex:  "^\\d{1,3}\\.\\d{3}$"
  max_retries: 2                        # extra captures per cycle on a wrong view
  retry_delay_sec: 5.5                  # must exceed longest diag-view dwell (5.0 s)
  min_digits: 6                         # cheap-fail bar for diagnostic screens
  cycle_budget_sec: 45                  # no new attempts once a cycle has run this long
  view_sync:                            # optional, off by default
    enabled: false
    period_sec: 20.0
    numeric_dwell_sec: 11.0
    margin_sec: 1.0
    capture_lead_sec: 1.5
    view_dwells_sec: null               # defaults to [11.0, 3.9, 5.1]
    drift_per_sec: 0.0003
    drift_decay_observations: 180
    observation_slack_sec: 0.25
```

`alignment.*` still applies — ORB alignment of the whole frame helps keep the LCD ROIs stable under small camera drift, even without dials.

### Flow line is optional

The LCD shows two lines, but in practice camera + flash geometry often makes it hard to get both lines clean in the same capture — flash reflections wash out a band across the glass, and aligning the camera to keep the total line readable sometimes sacrifices the flow line.

If you only calibrate `rois.digital.total` (leave `rois.digital.flow` out of the config), the service:

- Skips the flow OCR call entirely — one less Vision invocation per capture.
- Does NOT publish `main/flow_m3h` or advertise the HA `water_flow_m3h` sensor.
- Still publishes total, rate (delta-computed m³/min), and rate_lpm exactly as in mechanical mode — automations that use `water_rate_lpm` continue to work unchanged.

Calibrating both ROIs is strictly better when you can manage the reflections; calibrating only the total is the safe fallback.

### How the pipeline works

1. Capture a JPEG from `/capture_with_flashlight`.
2. Optionally align against `reference.jpg`. An alignment that fails the quality gate (`alignment.min_inliers`, `min_inlier_ratio`, `max_translation_frac`) ends the attempt as `align_failed` instead of OCR'ing an unaligned frame. OCR reads the aligned image in memory; `aligned_last.jpg` is optional debug output only.
3. Run the Swift helper in `--mode line` over the two ROIs. It uses Apple Vision's text recognizer; on empty output it retries with `.accurate`. The returned string is digits plus optional `.`.
4. Check `is_valid_digital_view` — cheap pre-validator that rejects frames with too few digit characters (diagnostic screens).
5. Parse each line with the configured regex. `parse_digital_total` injects a decimal at position -3 if Vision didn't see one.
6. Run `validate_digital_reading` — reject non-finite or negative flow — and `ReadingGate.preview` on the total. An implausible step is retried with reason `implausible`.
7. If any step fails, retry up to `digital.max_retries` times with `digital.retry_delay_sec` between captures (or until the next numeric window when view sync knows the phase); `digital.cycle_budget_sec` stops new attempts in a slow cycle. After that, skip the cycle and hold the previous value; the give-up is logged at WARNING with digits, litres and the gate detail.
8. `handle_digital_result` runs `ReadingGate.evaluate` on the cycle's final reading. An accepted reading is published (value with 3 decimals, QoS 1, retained) and saved to `state.json`.

### Acceptance policy (`ReadingGate`)

The digital path works in integer litres and does not use `big_jump_guard` or `monotonic_epsilon`. `dt` is the time since the last **accepted** capture. Keys are under `postproc`:

- Forward step ≤ `confirm_above_lpm` (25) × min(dt, `immediate_dt_cap_sec` 75 s) + `flow_slack_l` (1 L): accepted at once.
- Forward step > `max_flow_lpm` (40) × dt + 1 L: rejected `too_fast`.
- In between: `candidate`, accepted only when a later independent capture (≥ `confirm_min_gap_sec` 15 s, within `candidate_ttl_sec` 600 s) confirms it.
- Backward ≤ 1 L: `hold`.
- Larger backward: rejected, unless `rebaseline_count` (3) consecutive backward readings agree (rate-consistent) and span ≥ `rebaseline_min_span_sec` (60 s); the total is then `rebaseline`d to the lower value. The meter is bidirectional, and this frees a total trapped high by an accepted misread.

Sizing: observed rates 2026-10-01..11-04 were p50 6, p90 9, p99 12.9, p99.9 23.3 L/min. The service pipe is ≤ 20 mm with a pressure-reducing valve (~56 L/min at 3 m/s); a DN20 Q3 2.5 meter has Q4 ≈ 52 L/min.

The 7-segment classifier (`classify_7seg`) returns digit `""` (unreadable, retried) for ink between 0.10 and 0.30 and uses a 0.20 threshold; measured on 616 live frames, unlit segments were ≤ 0.061 and lit ≥ 0.375.

### The rotating display

The Qalcosonic W1 auto-cycles between three views:

| View | Dwell time |
|---|---|
| Numeric (total + flow) | 11.0 s |
| Diagnostic B | 3.9 s |
| Diagnostic A | 5.0 s |

Measured 2026-11-04 at 4.7 fps over 12 rotations; the period is 20.0 s (20.004 s by fit). Any single capture has a ~55 % chance of landing on the numeric view. `digital.retry_delay_sec = 5.5` exceeds the longest diagnostic dwell (5.0 s), so a retry crosses at least one dwell boundary and doesn't phase-lock onto the same wrong view.

**View sync.** A `processing.interval_sec` of 20 on a 20.0 s rotation phase-locks: every cycle's first capture lands on the same view. Setting `digital.view_sync.enabled: true` turns on `ViewPhase`. `next_start` schedules a capture safe for every feasible numeric-window origin; `explore_start` staggers probes when uncertainty exceeds that safe window. Wrong-view retries use the next safe window when possible. The default drift allowance is 0.03%: the fitted period error is `(20.004 - 20) / 20 = 0.0002`, with 50% headroom. Contradictions double the allowance up to 1%; a clean hour with at least 180 consistent observations halves it toward the default. Fractional grid-bin growth is carried forward, avoiding the former 0.1 s enlargement on every observation. Edge slack is applied to each observation without accumulating it. Numeric captures can trim the feasible arc at a boundary but captures well inside the window provide no new edge information, so occasional exploration is still expected.

With explicit per-cell integer **and** fractional ROIs, diagnostic identity is learned from the exact unreadable-cell mask, independent of digit values. A separate numeric/not-numeric phase model must place the mask unambiguously in the same ordered diagnostic view on three observations at least 0.8 periods apart before that mask can narrow the main phase. New, mostly blank, conflicting or ambiguous masks, and OCR paths without cell boundaries, retain the ordinary not-numeric constraint. A phase reset discards learned signatures. No identity can validate a total or bypass `ReadingGate`. The measured dwells sum to 19.9 s; the default model assigns the 0.1 s rounding residual to the final view. For another meter, set `view_dwells_sec` to its measured ordered rotation rather than assuming these proportions.

**Diagnostics.** With view sync enabled, the first attempt of a cycle logs one INFO line only for `wrong_view`, `parse_failed` or `align_failed`. It includes `source=synced` (`next_start`) or `source=exploring` (`explore_start`, including startup), predicted exposure `offset` from the midpoint of the pre-update feasible numeric-origin arc, that arc's start/width, lead EMA and drift. Unknown offsets/start are `nan`. These are predictions before observing that frame, not hindsight. Once per elapsed hour, INFO reports cycles, first-attempt numeric parses (including gate rejections), first-attempt wrong views split by source, explore probes, contradictions, projected arc width, lead and drift. Retries remain DEBUG; cycle give-ups remain WARNING. Counters and learned masks reset on process restart, and the summary is emitted on the first completed cycle after the hour.

**Timing.** `stamp["t_frame"]` is the midpoint from starting the kept streamed GET to receiving its headers, not a camera exposure timestamp. The lead EMA measures that proxy minus attempt start. Scheduling adds lead to a proposed start to predict the frame time and chooses its wait accordingly. A constant camera-buffer delay cancels because both observations and predictions use the same proxy; variable buffering and lead variation do not cancel and must fit within `margin_sec` (default 1 s). The 0.25 s edge slack does not establish a bound on camera buffering. No camera timestamps were available to verify that bound. Flash settling and the discarded frame retain their existing behaviour. Processing time is subtracted from the next start-to-start interval; overruns schedule from the current time. Neither ordinary processing time nor the lead EMA adds a cumulative per-cycle delay. Waiting for a safe window or an exploration probe deliberately extends some intervals.

Offline simulation (`tests/view_sync_simulation.py`, 400 cycles, first 20 excluded from miss rates) gives first-attempt misses of 0.26%, 0.26%, 0%, 1.58% at true periods 20.0/20.004/20.01/20.05 s, respectively, versus 4.21%, 4.21%, 3.68%, 2.63% with the previous tracker in the same simulation. Every cycle gets a reading. This is model validation, not a live camera measurement.

**Retry budget**: 3 captures × ~0.5 s + 2 × 5.5 s delays ≈ 12 s worst case. That's why the recommended `processing.interval_sec: 15` for digital mode — it gives the loop headroom to finish even when the first two captures miss.

### MQTT topics

Digital mode publishes all the mechanical topics plus a new one:

| Topic | Retained | Source |
|---|---|---|
| `{mqtt.topic}/main/value` | yes | Parsed total from the LCD (after `ReadingGate`), 3 decimals, QoS 1. |
| `{mqtt.availability_topic}` (default `{mqtt.topic}/status`) | yes | `online` / `offline`; retained LWT `offline`. Goes `offline` after `mqtt.stale_after_sec` without an accepted reading. |
| `{mqtt.topic}/main/rate` | no | Delta-computed m³/min (same formula as mechanical). |
| `{mqtt.topic}/main/rate_lpm` | no | Delta-computed L/min. |
| `{mqtt.topic}/main/flow_m3h` | no | **New.** Instantaneous flow read directly off the LCD. Published only when `rois.digital.flow` is configured. |
| `{overlay.camera_topic}` | yes | Debug overlay JPEG. |

Home Assistant discovery payload for `water_flow_m3h` is published **only** when `meter.type: digital` AND `rois.digital.flow` is set. Mechanical installs and digital-total-only installs don't see a dead sensor.

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

A browser tab opens at `http://127.0.0.1:8765/?token=…` showing the current meter image. Every request needs that per-run token, so open the exact URL printed in the terminal; a URL without it gets 403. This stops other LAN hosts and cross-site pages from rewriting the config. With `--host 0.0.0.0` (to use a phone at the meter), open the printed URL with the Mac's LAN address, and stop the tool when you're done. The sidebar reflects the `meter.type` you have in config:

- **Mechanical** → 1 digits window + 4 dial ROIs + 3 alignment anchors.
- **Digital** → per-digit cells (as many as `digital.total_regex` / `flow_regex` accept: 6 integer + 3 fractional by default), the legacy line ROIs (total, flow) and 3 alignment anchors. Cells are positional: an empty cell before a drawn one is rejected (it would shift every later digit), and the cell counts must match the parser.

For each ROI in the sidebar:

1. Click the row (turns amber — "draw mode").
2. Drag a rectangle on the image to place it.
3. (Mechanical dials only) set `rotation` and `zero_angle_deg` in the sidebar.

Iterate. Click **Test current config** whenever you want to check the result. Mechanical: each dial's reading and confidence. Digital: the service's real pipeline (decode, alignment, per-digit / line OCR, parsing, validation) runs on the last fetched frame and shows the reading, or why it failed (for example a diagnostic view), with each cell's digit in the sidebar. Nothing is published, and the live service's files are untouched. When everything looks right:

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

1. Grab a current frame: `~/watermeter/venv/bin/python3 save_reference.py` — writes to `~/watermeter/reference.jpg`. It captures exactly like the service (`esp32.capture_mode`, including the ESPHome flash), checks the decode and the write, keeps the old reference as `reference.jpg.bak.<timestamp>`, and exits non-zero with the old file intact on any failure. Stop the LaunchAgent first.
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
  "total_l": 12345678,
  "ts": 1700000000.0,
  "dial_histories": {
    "dial_0": [5.1, 5.2, 5.3, ...],
    "dial_1": [...]
  }
}
```

- `total` — last published m³. Compared against each new reading for the monotonic and big-jump guards (mechanical).
- `total_l` — the same total in integer litres, written by the digital path (`ReadingGate`).
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

**Long stretches of `validate_failed` and the service won't recover.**
- Digital path: `ReadingGate` frees a total trapped above the meter by itself (`rebaseline` after `postproc.rebaseline_count` agreeing backward readings spanning `rebaseline_min_span_sec`). Look for `Giving up this cycle` warnings; they carry the digits, litres and gate detail. The notes below describe the mechanical path.
- Pre-`big_jump_rate_per_day` (or with `big_jump_rate_per_day: 0`), if `validate_digital_reading` rejected one cycle, every subsequent successful OCR was compared against the same stale `prev_total` and rejected for the same reason — a permanent trap that required deleting `state.json`. The default `big_jump_rate_per_day: 1.0` self-heals: after `Δt` of failure, the allowance grows by `Δt × 1.0` m³/day. If you've kept it at `0`, raise it (e.g. `1.0` for residential, higher for commercial). If self-healing isn't kicking in, check that `state.json`'s `ts` field is not being refreshed when validation fails (it shouldn't be; only successful publishes update it).

**Alignment keeps failing (`aligned_ok=False` in logs).**
- Rebuild the reference: `rm ~/watermeter/reference.jpg`. The next capture becomes the new reference.
- Check your `anchor_rois` — they must not overlap the needles or rolling digits. Use [`overlay_example.jpg`](../overlay_example.jpg) or a captured frame to verify.
- Lower `alignment.min_matches` if the camera gives a noisy image (trade-off: more false positives).

**Home Assistant shows `sensor.water_total` as "unavailable".**
- MQTT not connected. Check logs for `MQTT connect rejected` (bad credentials) or `MQTT disconnected`. Also verify the broker is reachable: `mosquitto_sub -h <host> -t 'home/watermeter/#'`.

**HA accepts readings but `sensor.water_total` appears to reset.**
- This usually means you reset the state without informing HA. the total sensor's discovery uses `state_class: total` (a net register, so dips are legitimate), which does not need `last_reset`. If it still looks wrong, check the recorded statistics for a one-off jump.
