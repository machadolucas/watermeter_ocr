# AGENTS.md

Guide for AI agents (and humans) working on `watermeter_ocr`. Complements [README.md](README.md) (user/install), [docs/CONFIGURATION.md](docs/CONFIGURATION.md) (config reference + meter-swap procedure), and [VISUAL_GUIDE.md](VISUAL_GUIDE.md) (dial algorithm).

## Orientation

A macOS service that reads a water meter from an ESP32 PoE camera on a fixed interval, resolves the reading, and publishes to Home Assistant over MQTT. Two pipelines are selectable via `meter.type` in config:

- `mechanical` (default) — 5 odometer digits + 4 red-pointer dials, resolved with OpenCV (needle detection) + Apple Vision (per-digit OCR).
- `digital` — fully electronic LCD showing total (m³) and instant flow (m³/h) as two text lines. Apple Vision line OCR only; no dials. Handles the display's rotating numeric/diagnostic views via in-cycle retry.

Entry point: `main()` at [watermeter.py](watermeter.py). The mechanical pipeline and the digital pipeline (`run_digital_cycle`) share the capture helper, alignment, MQTT publish, overlay + state code; they diverge only in how a reading is resolved.

## Repo layout

| Path | Purpose |
|------|---------|
| [watermeter.py](watermeter.py) | Everything — config, OCR wrapper, dial/digit logic, alignment, MQTT, main loop (1190 lines, flat module) |
| [ocr.swift](ocr.swift) | Apple Vision CLI helper; reads one digit slice from a given ROI |
| [config.yaml](config.yaml) | Default config; deploy copy lives at `~/watermeter/config.yaml` |
| [install.sh](install.sh) | Bash installer: venv, Swift build, LaunchAgent, starter config; hands the release to `release.sh` |
| [release.sh](release.sh) | Release transaction sourced by install.sh: stage → `--check-config` gate → swap keeping `*.prev` → restart → stable-pid gate → auto-rollback. `./release.sh rollback` / `status` by hand |
| [save_reference.py](save_reference.py) | One-shot: capture a clean ESP32 frame (via `_fetch_jpeg`, so ESPHome mode works) and atomically save it as the alignment reference |
| [calibrate.py](calibrate.py) | Browser-based ROI calibration tool. Run `python3 calibrate.py` and open the printed `?token=` URL; see [docs/CONFIGURATION.md](docs/CONFIGURATION.md). |
| [tests/](tests) | pytest suite (fixtures, unit + synthetic-image tests) |
| [overlay_example.jpg](overlay_example.jpg) | Sample debug overlay for tuning ROIs visually |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Full config reference, meter-swap procedure, troubleshooting |

## Runtime layout (NOT the repo)

`install.sh` copies files into `$WTM_ROOT` (default `~/watermeter/`). Everything the running service needs lives there:

```
~/watermeter/
  config.yaml                 # user-edited config
  bin/watermeter              # signed PyInstaller binary the service runs (built by install.sh)
  bin/watermeter.prev, bin/ocr.prev   # previous release, kept by release.sh for rollback
  reference.jpg               # alignment reference frame
  state.json                  # persisted total + per-dial history
  watermeter.log              # logs
  aligned_last.jpg            # last aligned frame (optional debug output; not the OCR input)
  debug/overlay_latest.jpg    # overlay image (also published to MQTT)
  bin/ocr                     # compiled Swift helper
  venv/                       # CPython 3.15.2 build/dev venv; service runs bundled binary
  launch_stdout.log, launch_stderr.log
```

Running `python3 watermeter.py` from the repo works only if `~/watermeter/` already has a reference image + OCR binary; otherwise `install.sh` first. The repo directory is not the runtime directory — this trips up local dev.

## Build / run / test cheatsheet

```bash
# Full deploy (runs the LaunchAgent afterwards)
./install.sh

# Rebuild just the Swift helper (fast iteration on OCR)
swiftc -O ocr.swift -o ~/watermeter/bin/ocr

# Foreground run against the deployed config
~/watermeter/venv/bin/python3 watermeter.py --config ~/watermeter/config.yaml

# Validate a config without starting anything (exit 0 OK, 78 invalid)
~/watermeter/venv/bin/python3 watermeter.py --check-config --config ~/watermeter/config.yaml

# Roll the service back to the previous release / check its health
./release.sh rollback
./release.sh status

# Tests (from repo root; create a CPython 3.15.2 arm64 venv first)
python -m pip install --require-hashes --only-binary=:all: -r requirements-build.lock
python -m pip install --require-hashes --no-cache-dir --no-build-isolation -r requirements.lock
pytest -q
pytest -q tests/test_digit_logic.py -k rollover   # selective run

# Interactive ROI calibration (after a meter swap, or first-time setup)
python3 calibrate.py                               # browser opens at 127.0.0.1:8765/?token=...

# Reset the running total (e.g. new physical meter starts at 0)
python3 watermeter.py --config ~/watermeter/config.yaml --reset-total 0

# Service control
launchctl unload ~/Library/LaunchAgents/com.watermeter.ocr.plist
launchctl load   ~/Library/LaunchAgents/com.watermeter.ocr.plist
launchctl list | grep watermeter
tail -f ~/watermeter/watermeter.log
```

## Architecture at a glance

```
ESP32 /capture_with_flashlight (JPEG via HTTP)
        │
        ▼
 cv2.imdecode → raw frame
        │
        ▼
 Aligner.align  (ORB features + RANSAC + similarity warp)    watermeter.py:752
        │                       │
   aligned frame            aligned_ok? (false falls through with raw frame)
        │
        ├─► for each of 4 dials:
        │     read_dial(roi)                                  watermeter.py:463
        │       ├─ detect_dial_center  (needle drop > Hough circles > geometric)
        │       ├─ detect_needle_by_color   (HSV red mask)
        │       ├─ detect_needle_by_lines   (Canny + HoughLinesP)
        │       └─ validate_reading_with_history
        │   → (reading∈[0,10), confidence, center_offset)
        │
        ├─► for each of 5 odometer digits:
        │     VisionOCR.full_top_bottom(roi) via ~/watermeter/bin/ocr
        │   → (full_str, top_str, bottom_str)
        │
        ├─► compose_integer(obs, frac_prog, thr_up, thr_dn, prev_int_str)
        │                                                      watermeter.py:625
        │   ripples dial-progress carry from rightmost → leftmost digit;
        │   each position resolved by decide_digit (watermeter.py:565)
        │
        ├─► total = integer + frac_pub
        │   guards: monotonic (no backward steps), big_jump (<= big_jump_guard m³)
        │   fallback: estimate_total_from_dials  (watermeter.py:919)
        │
        ├─► MQTT publish:  home/watermeter/main/{value,rate,rate_lpm}
        │                  overlay JPEG → camera entity topic
        │
        └─► save ~/watermeter/state.json, sleep (interval or quiet_interval)
```

Two fractions coexist — do not confuse:
- **`frac_pub`** (line ~1087): what we **publish**. Sum of floored dial digits × factor. No double counting.
- **`frac_prog`** (line ~1090): smooth progress `[0..1)` from the **tenths dial only**, used to drive rolling digit logic.

## Digital-meter pipeline (`meter.type: digital`)

Alternative, simpler pipeline for fully-digital LCD meters. Runs instead of the mechanical one when `meter.type: digital` in config. Skips `read_dial`, `decide_digit`, `compose_integer`, `estimate_total_from_dials` entirely.

```
 ESP32 capture  ─►  Aligner (optional)  ─►  VisionOCR.read_line(total_roi)     ─┐
                                            VisionOCR.read_line(flow_roi)      ─┤
                                                     │                         │
                                          is_valid_digital_view?  ── no ─► retry
                                                     │                         │
                                          parse_digital_total / _flow          │
                                                     │                         │
                                          validate_digital_reading ── no ─► retry
                                                     │                         │
                                                     ▼                         │
                                run_digital_cycle returns a dict ───────◄──────┘
                                                     │
                                                     ▼
                    ReadingGate (handle_digital_result) → publish value, rate, rate_lpm, flow_m3h
                                                     │
                                                     ▼
                                        draw_overlays_digital, save state, sleep
```

All orchestrated by `run_digital_cycle(..., gate=, phase=)` ([watermeter.py](watermeter.py)), called from `main()` when `cfg.meter_type == "digital"`. An alignment that fails `alignment_quality_ok` ends the attempt as `align_failed` (no OCR on an unaligned frame); OCR reads the aligned image in memory (path helpers get `<image_path>.aligned.jpg`), not the optional debug `aligned_last.jpg`. `digital.cycle_budget_sec` (45) stops new attempts in a slow cycle. Per attempt `gate.preview` runs without changing state; an implausible step gets reason `implausible` and is retried. `handle_digital_result` runs `gate.evaluate` on the cycle's final reading, publishes (value with 3 decimals, QoS 1, retained), saves `state.json` (`total` in m³ and `total_l` in litres) and logs give-ups at WARNING with digits, litres and the gate detail.

### `ReadingGate` (digital acceptance policy)

Works in integer litres (`to_litres`) and replaces `big_jump_guard` + the elapsed-time growth on the digital path; the mechanical path still uses `monotonic_epsilon` / `big_jump_*`. `dt` is the time since the last ACCEPTED capture. Keys are under `postproc`:

| Step | Rule | Outcome |
|---|---|---|
| forward ≤ `confirm_above_lpm` (25) × min(dt, `immediate_dt_cap_sec` 75 s) + `flow_slack_l` (1 L) | plausible | accepted at once |
| forward > `max_flow_lpm` (40) × dt + 1 L | impossible | rejected `too_fast` |
| in between | suspicious | `candidate`; accepted only when a later independent capture (≥ `confirm_min_gap_sec` 15 s, within `candidate_ttl_sec` 600 s) confirms it |
| backward ≤ 1 L | noise | `hold` |
| larger backward | rejected, unless `rebaseline_count` (3) consecutive backward readings agree (rate-consistent) and span ≥ `rebaseline_min_span_sec` (60 s) | `rebaseline` to the lower value |

The meter is bidirectional, and the re-baseline frees a total trapped high by an accepted misread. Sizing: observed rates 2026-10-01..11-04 were p50 6, p90 9, p99 12.9, p99.9 23.3 L/min; the service pipe is ≤ 20 mm behind a pressure-reducing valve (~56 L/min at 3 m/s); a DN20 Q3 2.5 meter has Q4 ≈ 52 L/min.

### LCD view rotation and `ViewPhase`

Measured 2026-11-04 at 4.7 fps over 12 rotations: numeric view 11.0 s, diagnostic view B 3.9 s, diagnostic view A 5.0 s; period 20.0 s (20.004 s by fit). A 20 s `interval_sec` phase-locks on that rotation. `ViewPhase` (opt-in via `digital.view_sync.enabled`) tracks feasible numeric-window origins; `next_start` schedules inside the safe window, `explore_start` staggers probes while uncertain. Ordered `view_dwells_sec` generalises the rotation (numeric first; defaults to 11/3.9/5.1 s, assigning the 0.1 s rounding residual to A). A separate numeric/not-numeric model learns unreadable-cell masks from three independent rotations before diagnostic identity can narrow the main model. Unknown or conflicting masks fall back to not-numeric; identities never validate readings. Preserve that separation to prevent self-reinforcing signature mistakes. The small derived recording fixture is `tests/fixtures/lcd/view_sequence.json`; no JPEGs are required for identity validation.

Drift defaults to 0.0003 s/s (measured relative period error 0.0002 plus 50% headroom), doubles on contradictions up to 0.01, and halves toward the default after 180 consistent observations spanning at least one hour. **Carry fractional dilation bins**: rounding up every observation silently restores excessive drift even with a lower default. Observation edge slack (0.25 s) does not accumulate. Forecast both arc edges through the prospective exposure, including lead and waiting time. Wrong-view retries wait for the next safe numeric window when possible. Simulated first misses are 0–1.6% across true periods 20.0/20.004/20.01/20.05 s, with every cycle read; jitter/quiet-hour cases are also covered. Treat this as offline evidence only.

`_fetch_jpeg(..., stamp=)` supplies `stamp["t_frame"]`, the kept GET start/response-headers midpoint, an exposure **proxy**, excluding body transfer, flash-off and OCR. Constant buffering bias cancels between phase observations and predictions; variable buffering is unverified and must fit the scheduling margin. The lead EMA uses proxy time minus attempt start. Digital scheduling subtracts work already done and clamps an overrun's earliest start to now. Normal capture/processing time does not accumulate into an added sleep each cycle. With sync enabled, `first_attempt` logs only the first wrong/parse/alignment failure at INFO, with the pre-update phase prediction and scheduling source; `summary` emits hourly counters. Preserve first-attempt attribution across retries, and count parsed numeric frames even when `ReadingGate` subsequently rejects them. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the config keys and timing caveats.

### Availability

`MqttClient` sets an LWT (`will_set(availability_topic, "offline", qos=1, retain=True)`); discovery for total/rate/rate_lpm/flow carries `availability_topic` (default `<mqtt.topic>/status`, key `mqtt.availability_topic`); `set_available()` publishes online/offline on change, and `_on_connect` republishes discovery, then availability. `StaleTracker` goes offline once no reading has been accepted for `mqtt.stale_after_sec` (600; 0 disables). The overlay camera has no availability. The total sensor's discovery is `state_class: total` (net register; dips are legitimate), not `total_increasing`.

### History: the "9" misread (2026-11)

The 7-segment classifier's vertical-segment probes overlapped the horizontal bars, so a "3" read 0.13–0.21 ink in the top-left probe against a 0.18 threshold and came out "9" on ~16 % of frames. Accepted +60 L / +600 L misreads then trapped `prev_total` above the meter and every correct reading failed as "backward" until real use caught up (lost consumption, give-up streaks of hundreds/thousands per day). Fix in `classify_7seg`: vertical probes at rows 20–38 % / 62–80 %, horizontal bands 0–18 % / 42–58 % / 82–100 %, inner x 30–70 %, threshold 0.20, and an ambiguity band (0.10, 0.30) → digit `""` (unreadable, retried). Measured on 616 live frames: unlit ≤ 0.061, lit ≥ 0.375; 616/616 correct after the fix (100 were "9" before). Real-LCD fixtures live in `tests/fixtures/lcd/`.

## Load-bearing invariants

Break these and things go silently wrong.

- **`cfg.dials[0]` MUST be the ×0.1 tenths wheel.** `frac_prog` at [watermeter.py:1090](watermeter.py:1090) hardcodes `dial_vals[0]` and `dial_vals[1]`; the YAML order is load-bearing. `config.yaml` already documents this (`dials[0]` is `dial_0_1`).
- **`decide_digit` keeps `prev_digit` in the stable zone** (`thr_dn < progress < thr_up`). It only flips to a new value at the thresholds. Changing `rolling_threshold_up/down` shifts where OCR flips vs. where the previous value is trusted. See [watermeter.py:565](watermeter.py:565). The function is stateless per call — it does NOT infer a rollover from "low progress + all OCR None" (an earlier heuristic caused persistent OCR failure to creep the digit by 1 every cycle). If OCR is silent, the fallback prefers `prev_digit`.
- **`estimate_total_from_dials` assumes < 1 m³ consumption per capture** — at most one rollover between samples. At 10 s sampling this is comfortably true for residential meters. If you change `interval_sec` or `quiet_interval_sec` to minutes, revisit. See [watermeter.py:919](watermeter.py:919).
- **`VisionOCR._run` silently swallows subprocess errors** and returns `""`. Also truncates multi-char output to the first digit (`out[0] if len(out)>1 else out` at [watermeter.py:163](watermeter.py:163)) — intentional because Vision sometimes returns two digits during rollover. Callers must always handle empty strings.
- **`Aligner.align` returns `(img, None, False)` on failure** and processing continues on the raw frame. On the mechanical path `aligned_ok=False` is a normal steady state, not an error; the digital path ends the attempt as `align_failed` instead (see above).
- **ROI format everywhere is `[x, y, w, h]` normalized to `[0..1]`**, not `[x1, y1, x2, y2]`. `norm_to_abs` converts to integer pixel rects.
- **`read_dial` returns `reading % 10.0`** — a full revolution maps to `0.0`, never `10.0`. A test expecting `10.0` will fail.
- **Dial reading history is persisted in `state.json`** (up to 20 per dial) and restored on startup ([watermeter.py:971](watermeter.py:971)). Temporal validation in `validate_reading_with_history` and prediction in `predict_expected_reading` depend on this history — resetting state.json is a legitimate fix when a dial is consistently misread and has polluted history.
- **The Swift helper's `--half top|bottom` carves 55% of height**, not 50% ([ocr.swift:34-38](ocr.swift:34)). Top and bottom intentionally overlap by ~10% so a digit mid-transition is visible in both crops.
- **MQTT defaults cascade**: if `overlay.camera_topic` is omitted, it defaults to `{mqtt.topic}/debug/overlay`. Moving the `mqtt.topic` key or renaming it silently changes the overlay topic.
- **HA discovery publishes fire from `MqttClient._on_connect`**, not from the main loop. They happen automatically on every (re)connect and are idempotent because the broker retains them. Do NOT add a per-loop `mqttc.discovery()` call — it just burns bandwidth.
- **`MqttClient.connected` is gated on `reason_code == 0`.** A rejected CONNECT (bad credentials, protocol mismatch, access denied) leaves `connected=False` and `publish()` silently no-ops. Watch the logs for `MQTT connect rejected` to spot this. paho's loop thread owns connect/reconnect (`connect_async` + `loop_start` + `reconnect_delay_set(1, 60)`); `ensure_connection` no longer reconnects.
- **Readings in `read_dial` are blended circularly** when `conf < 0.6` and a prediction is available. The helper is `circular_blend(a, b, alpha)` (mod 10). A previous linear blend produced nonsensical midpoints across the 9→0 wrap; do not revert without replacing the helper.
- **Per-dial `reading_history` is only appended when `confidence > 0.2`.** Low-confidence fallback readings (blank frame → `reading=0.0`) were previously polluting the temporal validator; the gate lives in the main loop around [watermeter.py:1052](watermeter.py:1052).
- **Digital pipeline bypasses `compose_integer` and all rolling-threshold logic.** LCD digits transition atomically; there's no in-between frame to arbitrate with top/bottom crops. Do NOT try to graft mechanical composition onto the digital path — it reintroduces the drift bug warned about at [watermeter.py:611-614](watermeter.py:611) (persistent silent OCR + stateless progress heuristics creeping the counter).
- **`digital_total_regex` must accept both dotted and undotted input.** Vision is unreliable about the decimal point on 7-segment LCDs — some frames emit `"000100.000"`, others `"000100000"`. `parse_digital_total` injects a `.` at position `-3` when missing. The default regex `^\d{6}\.?\d{3}$` matches both; a stricter regex will silently drop half your readings.
- **`run_digital_cycle` is the ONLY place that retries within a single capture cycle.** The display auto-cycles between 3 views (measured 2026-11-04: numeric 11.0 s, diagnostic B 3.9 s, diagnostic A 5.0 s; period 20.0 s). Adding ad-hoc retries elsewhere compounds the budget and can overrun `interval_sec`. If you need more retries, tune `digital_max_retries` in config.
- **`digital_retry_delay_sec` must exceed the longest diag-view dwell (5.0 s, diagnostic A).** Shorter delays phase-lock: every retry lands on the same diagnostic screen. The default 5.5 s guarantees a retry crosses at least one dwell boundary. If the meter's firmware changes dwell times, recheck this.
- **`state.json.meter_type` is the cross-contamination guard.** When config flips between `mechanical` and `digital`, the persisted totals aren't comparable. The load path at [watermeter.py](watermeter.py) discards `prev_total` on mismatch and logs a warning; it does NOT exit, because launchd crash-looping is worse than a one-cycle unknown rate.
- **HA discovery for `water_flow_m3h` is gated on `cfg.meter_type == "digital"` AND `cfg.digital_flow_roi` being non-empty.** Mechanical installs must not see a dead sensor in HA. Digital installs that chose to skip the flow ROI (flash-glare workaround) also must not see it. The gate lives in `MqttClient.discovery`.
- **`cfg.digital_flow_roi` is optional.** When it's empty, `run_digital_cycle` skips the flow OCR call, `parse_digital_flow` is not invoked, and `flow` in the result dict is `None`. The `is_valid_digital_view` and `validate_digital_reading` helpers both treat flow as optional when the ROI is unconfigured. Any new code that reads `result["flow_m3h"]` must handle `None`.
- **`big_jump_guard` is time-aware (mechanical path only).** The digital path uses `ReadingGate` instead (see above). The effective forward-jump allowance is `effective_big_jump_guard(cfg, prev_ts, now) = big_jump_guard + min(elapsed_days, big_jump_elapsed_cap_days) × big_jump_rate_per_day`. Used by `validate_digital_reading` (forward delta only — backward deltas are still strictly bounded by `monotonic_epsilon`) AND by both pipelines' main-loop clamps. A previous fixed-guard implementation had a permanent lock-in trap: once a long OCR outage accumulated more consumption than the base guard, every subsequent successful read failed `validate_digital_reading` because `prev_ts` is held while validation fails (only successful publishes refresh it). Manual `state.json` deletion was the only escape. Do not revert to a fixed guard without replacing the self-healing mechanism.

## Config pitfalls

- **`load_config` validates** (since 2026-11-11) and raises `ConfigError` with every problem as `dotted.key: message`. Pass 1 (`_validate_raw`) checks the YAML: unknown keys against `_CONFIG_SCHEMA` (with a did-you-mean), types, enums, ranges, ROI shapes, regexes, quoted `HH:MM`. Pass 2 (`_validate_built`) checks the built Config: digital total source, int/frac sides together, cell counts vs the parser (`digit_count_problem` / `flow_count_problem`), `ViewPhase` dwell rules, `dials[0].factor == 0.1`. **A new config key must be added to `_CONFIG_SCHEMA`**, or every config using it fails to load. `runtime_config_problems` (mechanical needs ≥ 2 dials) is checked only by `main()`, so tools and tests can load a dial-less config. `main()` exits 78 before logging/MQTT/network; `--check-config` validates and exits, and release.sh's gate 1 depends on it.
- **Digit-cell counts come from the parser.** Split mode hands `parse_digital_total` the string `f"{int}.{frac}"`, so a count is valid exactly when that many zeros parse (`valid_total_digit_counts`; 6 + 3 for the default regex). Per-digit flow cells are joined without a dot (`valid_flow_digit_counts`; 5–8). calibrate.py uses the same helpers for its slot counts and payload checks.
- `rois.digits` is a **single window** split `digits.count` ways by `build_digit_rois` at [watermeter.py:944](watermeter.py:944). `digits.count` must match the physical meter; the current meter has 5.
- `per_digit_inset` shrinks each derived digit cell inward. Too high and Vision sees nothing; too low and it includes the plastic frame.
- `dials[].rotation` is `cw` or `ccw`; `zero_angle_deg` is where "0" points relative to screen coordinates (-90 = top). Nudge by ±5–10° if readings are consistently offset.
- `alignment.anchor_rois` is a list of normalized `[x, y, w, h]` rects that mask where ORB looks for features; keep them over stable printed areas, not over moving needles.

- **Alignment quality gate** (`alignment_quality_ok`): `alignment.min_inliers` (25), `min_inlier_ratio` (0.5), `max_translation_frac` (0.10). Good frames show ≥ 91 inliers and a ratio ≥ 0.83.
- **Capture hardening**: empty bodies and bodies over `processing.max_image_bytes` (2 MB) are capture failures; flash-off is checked and retried once.
- **Logging**: rotating log, `paths.log_max_bytes` (10 MB) and `paths.log_backup_count` (5).
- **`load_state` validates `state.json`**; a bad `total` is ignored with a warning. A single `flow_tracked(cfg)` predicate decides whether the flow line is tracked.

## Extension points

- **Add a dial**: append to `rois.dials` with its `factor`. Verify `dials[0]` stays the tenths wheel. `frac_prog` may need updating if the new dial sits between tenths and the existing hundredths.
- **Swap OCR backend**: implement the `VisionOCR.full_top_bottom(img_path, roi) -> (full, top, bottom)` interface at [watermeter.py:155](watermeter.py:155). Each returned value is a one-char digit string or `""`. Keep the silent-failure contract.
- **Additional sanity check on total**: either extend `estimate_total_from_dials` or add inline logic around [watermeter.py:1111](watermeter.py:1111) where `big_jump_guard` is consulted.
- **New Home Assistant sensor**: add a discovery payload in `MqttClient.discovery` at [watermeter.py:709](watermeter.py:709) and publish its value in the main loop.
- **New guard/clamp**: the publish branch at [watermeter.py:1122-1131](watermeter.py:1122) is where `publish_total` is decided separately from the raw `total` — add behavior there if you want the raw reading logged but not published.
- **Changing ROIs**: prefer `python3 calibrate.py` over hand-editing `config.yaml`. The tool validates (ROI bounds, positional per-digit cells with no gaps, counts vs the parser, dial ordering, round-trip through `load_config`) before writing, serializes saves (`_SAVE_LOCK`, unique temp file) and creates a unique 0600 backup. Every route needs the per-run session token (`X-Calibrate-Token` header or `?token=`); bodies are capped at 64 KiB and the socket timeout is 15 s. The digital **Test** button runs `run_digital_cycle` on the cached frame through `_FrameSession` with a temp `image_path`, no retries, no aligner unless `reference.jpg` exists, and no debug writes. It must never publish or touch the live service's files. Hand-editing is the fallback for headless boxes — see [docs/CONFIGURATION.md](docs/CONFIGURATION.md) § "Manual ROI tuning".
- **Physical meter replacement**: the full procedure lives in [docs/CONFIGURATION.md](docs/CONFIGURATION.md) § "Meter-swap procedure". Short version: stop LaunchAgent → delete `reference.jpg` → run `calibrate.py` → `watermeter.py --reset-total 0` → restart LaunchAgent.
- **Swap to a digital LCD meter**: set `meter.type: digital` in config, bump `processing.interval_sec` to 15, calibrate the two line ROIs, then the standard meter-swap procedure. State's `meter_type` field auto-discards `prev_total` on mismatch. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) § "Digital meter".
- **Adapt the digital pipeline to a different LCD layout**: change `digital.total_regex` and `digital.flow_regex` in config to match your meter's display format (e.g. 8 integer digits with no decimal: `^\d{8}$`). If the layout has no decimal at all, adjust `parse_digital_total` so it doesn't inject one. ROI count stays at two — total and flow.

## Testing strategy

pytest suite at `tests/`. Runs with zero external services — no MQTT broker, no ESP32, no Swift binary required. Use the CPython 3.15.2 arm64 build/dev venv and install the generated hash locks. See [tests/conftest.py](tests/conftest.py) for fixtures.

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-build.lock
python -m pip install --require-hashes --no-cache-dir --no-build-isolation -r requirements.lock
pytest -q                                          # full suite
pytest -q tests/test_digit_logic.py                # one module
pytest -q -k "rollover or wraparound"              # keyword filter
pytest -q --tb=long                                # longer tracebacks for debugging
```

Coverage is organized by concern:
- `test_utils.py` — small helpers (`clip01`, `norm_to_abs`, time windows, ROI math)
- `test_config.py` — `load_config` on minimal/full YAML, including digital keys; validation (`ConfigError`) and `--check-config` exit codes
- `test_calibrate.py` / `test_calibrate_hardening.py` — payload validation, merge, backups; the real digital Test on the LCD fixture; token / 411 / 413 / socket timeout / concurrent saves over a live loopback server
- `test_save_reference.py` — capture via `_fetch_jpeg` (ESPHome order, flash-off on failure), decode/write checks, atomic replace
- `test_release.py` — `release.sh` with a stub `launchctl`: gate 1 failure, keep-previous, crash-loop rollback, manual rollback
- `test_digit_logic.py` — `decide_digit` branches, `compose_integer` rollover scenarios (highest-value mechanical module)
- `test_postproc.py` — `estimate_total_from_dials`, temporal validation & prediction
- `test_dial_detection.py` — synthetic-dial tests for `detect_dial_center`, `read_dial`, needle methods (mechanical only)
- `test_alignment.py` — `Aligner` on translated/rotated synthetic pairs (shared)
- `test_ocr_wrapper.py` — `VisionOCR` via a mock shell script; both `full_top_bottom` (digit mode) and `read_line` (line mode) contracts
- `test_digital.py` — `parse_digital_total`/`parse_digital_flow`, `is_valid_digital_view`, `validate_digital_reading`, `run_digital_cycle` retry behaviour, HA `flow_m3h` discovery gating, `draw_overlays_digital` smoke

Tolerances on image tests are deliberately loose. ORB, Hough, and the dial-detection cascade are non-deterministic enough that per-pixel assertions will flake across OpenCV versions. Use `pytest.approx` and statistical (mean-error) assertions, never exact equality, for anything downstream of a cv2 primitive.

## What is NOT tested (and why)

- **Real MQTT broker I/O** (`MqttClient.connect/publish`). Testing needs a broker; the coverage is mostly boilerplate. Manual verification: watch the Home Assistant entities and `mosquitto_sub -t 'home/watermeter/#'`.
- **Real HTTP to ESP32**. 3 lines of `requests.get` + write-to-disk. Not worth the mock.
- **Real Swift OCR binary.** Tests mock it via a shell script. Real Vision output varies by macOS version and is verified manually against `overlay_example.jpg` during ROI tuning.
- **`main()` loop.** Too many side effects (files, MQTT, network, sleep). Manual verification via live run.
- **`draw_overlays`.** Smoke-testable but visually noisy; skipped to keep the suite fast and deterministic. Verify visually by inspecting `~/watermeter/debug/overlay_latest.jpg` after a live run.

## Sharp edges

- **`opencv-python-headless` only.** Installing `opencv-python` in the same venv clobbers GUI-backed symbols and crashes on import. `install.sh` already pins headless.
- **HSV red range is hard-coded** in `detect_needle_center` and `detect_needle_by_color`. A non-red needle breaks both. If you adapt to a different meter, add a second color range or parametrize via config.
- **`detect_needle_center` has a 180° ambiguity on uniform-thickness needles.** It uses `cv2.distanceTransform` on the red mask and picks `max_loc` as the center. For a real meter needle that tapers to a thick pivot "drop", `max_loc` lands right on the pivot — stable. For a uniform-thickness needle (or a synthetic line drawn in test fixtures), the distance transform has a flat plateau along the centerline, and `max_loc` can land anywhere on it — sometimes at the needle tip, which yields a reading rotated by 180° from the true value. Synthetic test fixtures must include a filled pivot circle so the distance-transform peak is unambiguous; real meters have this naturally.
- **ORB is not deterministic across OpenCV versions** for degenerate match sets. Alignment tests assert "close enough," never exact pixel values.
- **`_in_window_local` uses local `datetime`.** Unit tests must construct explicit `datetime(2025, 1, 1, H, M)` — never `datetime.now()` — so they're reproducible regardless of host timezone.
- **macOS is case-insensitive by default.** `CLAUDE.md` and `claude.md` would collide. The repo uses one symlink (`CLAUDE.md → AGENTS.md`) to keep a single source of truth; do not commit both as files.
- **MQTT is best-effort.** `MqttClient.publish` silently no-ops when disconnected; paho's loop thread retries the connection with backoff. Missing data points are expected during broker outages — that's by design, not a bug.
- **`state.json` is the source of truth for `prev_total`.** Deleting it resets monotonic guards; the next reading is accepted unconditionally and becomes the new floor. Treat it as recoverable but not trivially expendable.
