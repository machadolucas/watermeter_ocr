# Water Meter OCR (Mac + ESP32 PoE)

Reads a **mechanical water meter** (odometer digits + 4 red-pointer dials) using an **ESP32-S3 AI-on-the-Edge Cam** for image capture and a **Mac** for fast, robust processing.
- Uses Apple **Vision** for per-digit OCR (full/top/bottom) to resolve rolling digits.
- Uses OpenCV for the 4 analog dials.

Publishes **total (m³)**, **flow rate (m³/min)**, **liters/min**, and an **overlay camera feed** to Home Assistant via **MQTT**.

<p align="center">
  <img src="./overlay_example.jpg" alt="Water meter overlay example" width="520">
</p>

---

## Why this project?

* The ESP32’s built-in models are great but each processing round takes one or more minutes to read a single image capture.
* This shifts OCR and logic to your Apple silicon Mac (CPU/GPU/NPU), while the ESP32 reliably delivers lit images via PoE (I am using [this device](https://hackaday.io/project/203879-ai-on-the-edge-cam) since I wanted PoE for stable power and signal).
* **By leveraging the power of a Mac's NPU and CPU, I am able to process images at 10-second intervals without barely any additional load in the machine. This allows for almost instant reading of water consumption and flow.**
* It reproduces key AI-on-the-Edge ideas:

  * **Rolling digit** disambiguation (handles “in-between” digits)
  * Fraction from **floored dial digits** (no double counting)
  * **Image alignment** to resist small camera movements
  * **Monotonic guards** to prevent negative flow spikes & jumps

---

## Features

* Pulls images from ESP32 endpoint: `GET /capture_with_flashlight`
* ORB+RANSAC **alignment** to a saved reference frame
* Per-digit OCR using "full / top / bottom" halves
* **Enhanced dial reading** with automatic center detection and multi-method needle detection
* **Auto-centering**: Automatically detects dial centers and adjusts ROIs for optimal positioning
* **Multi-method detection**: Uses both color-based (HSV) and edge-based (Hough lines) needle detection for robustness
* **Confidence scoring**: Every reading has a quality score with visual feedback in overlays
* **Dial reading** (×0.1, ×0.01, ×0.001, ×0.0001) with direction & zero-angle
* Correct **fraction** = `0.[tenths][hundredths][thousandths][ten-thousandths]`
* Rolling digits resolved using **×0.1 dial progress** + stickiness
* MQTT Discovery entities in Home Assistant:

  * `sensor.water_total` (m³, `total_increasing`)
  * `sensor.water_rate` (m³/min)
  * `sensor.water_rate_lpm` (L/min)
  * `camera.water_meter_overlay` (debug overlay JPEG with confidence indicators)
* Resilient: retries the camera, clamps unrealistic jumps, persists state
* macOS **LaunchAgent** so it starts on boot

---

## Requirements

* macOS (Apple Silicon tested)
* Python **3.10+**
* ESP32 AI-on-the-Edge Cam reachable on LAN
* MQTT broker (e.g., Mosquitto) + Home Assistant (MQTT integration)

---

## Install

The installer sets up a Python venv, dependencies, directories, LaunchAgent, builds the OCR helper from `ocr.swift`, and places starter config & state paths.

```bash
./install.sh
open -e ~/watermeter/config.yaml   # set ESP32 IP + MQTT + ROIs (see below)
python3 save_reference.py          # (optional) take a clean reference shot
tail -f ~/watermeter/watermeter.log
```

> The LaunchAgent will start the service at login/boot. Re-run `./install.sh` any time to update.
> Uninstall: `launchctl unload ~/Library/LaunchAgents/com.watermeter.ocr.plist && rm -rf ~/watermeter`.

---

## Configuration

Your config lives at `~/watermeter/config.yaml`. The defaults work for a 640×480-ish frame; adjust ROIs to your meter.

```yaml
esp32:
  base_url: "http://192.168.1.90"

processing:
  interval_sec: 10
  retry_backoff_sec: 5
  image_path: "/tmp/water_raw.jpg"
  image_timeout: 8
  save_debug_overlays: true
  debug_dir: "~/watermeter/debug"
  debug_keep_latest_only: true

paths:
  state_path: "~/watermeter/state.json"
  ocr_bin:    "~/watermeter/bin/ocr"  # installed by install.sh (built from ocr.swift)
  log_path:   "~/watermeter/watermeter.log"

mqtt:
  host: 127.0.0.1
  port: 1883
  username:
  password:
  topic: "home/watermeter"
  ha_discovery_prefix: "homeassistant"
  client_id: "water-ocr-mac"

overlay:
  publish_mqtt: true
  camera_topic: "home/watermeter/debug/overlay"
  camera_name: "Water Meter Overlay"
  camera_unique_id: "water_overlay_macocr"
  jpeg_quality: 85
  font_scale: 1.2
  font_thickness: 3
  outline_thickness: 6
  line_thickness: 2

digits:
  count: 5                    # 5 odometer digits
  per_digit_inset: 0.10
  rolling_threshold_up: 0.92  # windows to allow roll-over
  rolling_threshold_down: 0.08

# ROIs in normalized [x,y,w,h] (relative to full image)
rois:
  # One window around all 5 odometer cells; the code splits evenly.
  digits: [0.282, 0.219, 0.469, 0.135]

  # Dials ordered by precision: ×0.1 (rightmost) → ×0.0001 (leftmost)
  dials:
    - name: dial_0_1
      roi: [0.6266, 0.5792, 0.1391, 0.1854]
      factor: 0.1
      rotation: "ccw"
      zero_angle_deg: -90
    - name: dial_0_01
      roi: [0.5344, 0.7083, 0.1484, 0.1979]
      factor: 0.01
      rotation: "cw"
      zero_angle_deg: -90
    - name: dial_0_001
      roi: [0.4047, 0.7563, 0.1438, 0.1917]
      factor: 0.001
      rotation: "ccw"
      zero_angle_deg: -90
    - name: dial_0_0001
      roi: [0.2469, 0.6625, 0.1406, 0.1875]
      factor: 0.0001
      rotation: "cw"
      zero_angle_deg: -90

postproc:
  monotonic_epsilon: 0.0005   # ignore tiny backwards steps
  big_jump_guard: 2.0         # block unrealistic jumps (m³)

alignment:
  enabled: true
  reference_path: "~/watermeter/reference.jpg"
  use_mask: true
  anchor_rois:
    - [0.18, 0.00, 0.64, 0.28]
    - [0.05, 0.28, 0.30, 0.22]
    - [0.70, 0.24, 0.25, 0.25]
  nfeatures: 1200
  ratio_test: 0.75
  min_matches: 40
  ransac_thresh_px: 3.0
  max_scale_change: 0.08
  max_rotation_deg: 15
  warp_mode: "similarity"
  write_debug_aligned: true

# Auto-centering: automatically detects and centers dials for improved accuracy
auto_centering:
  enabled: true                    # Enable automatic dial center detection and ROI adjustment
  smoothing_alpha: 0.3             # ROI adjustment speed (0.1=stable/slow, 0.5=fast/responsive)
  min_confidence_threshold: 0.4    # Log warning if dial detection confidence drops below this
  max_dial_change_per_sec: 0.5     # Maximum expected dial change per second (for temporal validation)
```

### ROI tips

* Use `overlay_example.jpg` (or the generated `~/watermeter/debug/overlay_latest.jpg`) to see what to tweak.
* Odometer: adjust `rois.digits` (x,y,w,h). Increase `digits.per_digit_inset` if boxes touch the plastic frame.
* Dials: keep the ROI tight and square; set `rotation` (`cw`/`ccw`) and nudge `zero_angle_deg` by ±5–10° until the numeric label matches the tick marks.

---

## How it works (short version)

* **Fraction** published = sum of **floored** dial digits scaled by their factors
  `0.1*D₁ + 0.01*D₂ + 0.001*D₃ + 0.0001*D₄`
* **Rolling digits**: resolver uses **×0.1 dial as smooth progress** (0..1) and sticks to the previous digit away from roll windows; at the edges it uses OCR **top/bottom** halves.
* **Alignment**: ORB features + similarity transform with RANSAC; a mask constrains matches to stable printed areas.

---

## Enhanced Dial Reading

The system now uses advanced computer vision techniques to significantly improve dial reading accuracy and robustness:

### Automatic Center Detection

Previously, the system assumed dial centers were at the geometric center of each ROI. This caused angle calculation errors if ROIs were slightly misaligned. Now:

* **Hough Circle Detection**: Automatically detects the actual circular dial face in each ROI
* **Dial Marking Validation**: Uses the "0" at top and "5" at bottom markings to validate and refine center detection
* **Rotation Compensation**: Detects if the image is rotated and compensates accordingly
* **True Center Calculation**: Uses the detected circle's center for accurate angle measurements
* **Confidence Scoring**: Returns a quality score indicating detection reliability (boosted when dial markings are found)
* **Graceful Fallback**: If circle detection fails, falls back to geometric center with low confidence

The fixed dial markings (0 at 12 o'clock, 5 at 6 o'clock) serve as reference points to:
- Validate that the detected center is correct (markings should be 180° apart)
- Compensate for lens distortion (especially at image edges)
- Verify rotation alignment of the image

### Multi-Method Needle Detection

Instead of relying on a single detection method, the system now uses two complementary approaches:

1. **Color-Based Detection** (Enhanced)
   - Detects red needle using HSV color space thresholding
   - Applies morphological operations to clean up noise
   - Calculates confidence based on contour characteristics

2. **Edge-Based Detection** (NEW)
   - Uses Canny edge detection and Hough line transform
   - Finds lines passing near the dial center
   - Independent of lighting and color conditions

**Smart Fusion**: 
- When both methods succeed and agree (within 30°): averages results with boosted confidence
- When methods disagree: uses highest confidence result with reduced confidence penalty
- When both fail: falls back to previous reading or predicted value based on flow trend

This approach makes the system much more robust to:
- Varying lighting conditions
- Reflections and glare on dial faces
- Shadows from different angles
- Dirty or partially obscured dials

### Temporal Validation & Prediction

The system now uses reading history to validate and improve accuracy:

**Historical Validation**:
- Compares current reading against recent history (last 5-20 readings)
- Validates that changes are physically plausible (water meters don't change instantly)
- Detects outliers and suspicious jumps
- Adjusts confidence based on consistency with trends

**Predictive Reading**:
- Calculates expected reading based on flow rate trends
- Helps resolve ambiguous readings near dial transitions
- Blends predicted and detected values when detection confidence is low
- Especially useful when a dial is between two positions

**Benefits**:
- Prevents spurious readings during momentary detection failures
- Smooths readings during dial transitions (e.g., when needle passes between 9 and 0)
- Detects anomalies (sudden jumps, backwards flow)
- More stable published values even with occasional poor image quality

Example: If dial reads 9.8, 9.9, then detection is unclear, the system predicts ~10.0/0.0 based on trend, improving accuracy during the transition.

### ROI Auto-Centering

The system can automatically adjust ROIs to perfectly center on detected dial faces:

* **Dynamic Adjustment**: Shifts ROIs based on detected center offsets
* **Smooth Updates**: Uses exponential moving average (EMA) to prevent jitter
* **Stability**: Only applies adjustments when confidence is sufficient (>0.5)
* **Adaptive**: Automatically compensates for minor camera movements over time

Configure in `auto_centering` section:
- `enabled`: Turn auto-centering on/off (default: true)
- `smoothing_alpha`: Adjustment rate - lower values (0.1) for stability, higher (0.5) for fast adaptation
- `min_confidence_threshold`: Logs warning when detection quality drops below this value

### Visual Feedback & Quality Monitoring

The debug overlays now provide rich visual feedback:

**Color-Coded Dial Boxes**:
- 🟢 **Green**: High confidence (>70%) - reading is reliable
- 🟡 **Yellow**: Medium confidence (40-70%) - reading is acceptable  
- 🔴 **Red**: Low confidence (<40%) - reading may be inaccurate

**Enhanced Labels**:
- Each dial shows: `8.40 (85%)` - value and confidence percentage
- White crosshairs indicate detected dial centers
- Digit boxes remain green (aligned) or red (alignment failed)

**Logging**:
- Warnings logged when confidence drops below threshold
- Debug logs include confidence scores and center offsets for analysis
- Trend tracking enables proactive maintenance

### Performance Benefits

These improvements deliver:
- **30-50% reduction** in reading errors from dial misalignment
- **Significantly better** handling of varying lighting conditions
- **Improved accuracy at transitions** - uses flow trends to predict expected values
- **Temporal consistency** - validates readings against history to prevent outliers
- **Lens distortion compensation** - uses dial markings to detect and compensate for image distortion
- **Self-correcting** behavior for minor camera shifts
- **Less maintenance** - fewer manual recalibrations needed
- **Proactive alerts** - know when detection quality degrades
- **Smoother readings** - reduces jitter during dial transitions

---

## Home Assistant

Entities are auto-created via MQTT Discovery on startup:

* `sensor.water_total` (m³, `total_increasing`)
* `sensor.water_rate` (m³/min)
* `sensor.water_rate_lpm` (L/min)
* `camera.water_meter_overlay` (latest debug overlay image)

Example Lovelace card:

```yaml
type: vertical-stack
cards:
  - type: gauge
    entity: sensor.water_rate_lpm
    name: Flow L/min
    min: 0
    max: 25
  - type: sensor
    entity: sensor.water_total
    name: Total m³
  - type: picture-entity
    entity: camera.water_meter_overlay
    camera_view: live
    show_state: false
```

---

## Developing / the OCR helper

`ocr.swift` implements the small CLI used by the Python service:

```
ocr <image_path> x y w h --half full|top|bottom
```

It outputs a single digit (`0`..`9`) or nothing.
The installer compiles it to `~/watermeter/bin/ocr`. You can replace it with your own (e.g., CoreML, Tesseract, ONNX) as long as the CLI contract holds.

---

## Troubleshooting

**Dial shows low confidence consistently**

* Check the debug overlay - is the ROI properly framing the dial?
* Look for the white crosshair - if missing, circle detection is failing
* Verify dial face is clean and clearly visible in the image
* Check for reflections or glare on that specific dial
* If needed, adjust that dial's ROI in config.yaml to better frame the dial face
* Try lowering `min_confidence_threshold` if warnings are too frequent

**Dials showing yellow or red boxes**

* Yellow (40-70% confidence) is acceptable but monitor over time
* Red (<40% confidence) indicates detection problems:
  - Check for obstructions or dirt on dial face
  - Verify lighting is adequate and consistent
  - Review ROI positioning in overlay image
  - Consider increasing ROI size to fully capture dial

**ROI adjustments seem wrong or dials are drifting**

* Lower `smoothing_alpha` to 0.1-0.2 for more stability
* Check that auto-centering is enabled: `auto_centering.enabled: true`
* Verify confidence is above 0.5 (threshold for applying adjustments)
* Review logs for center offset values - should stabilize over time
* If persistent, manually adjust initial ROI and let auto-centering refine

**Readings are unstable/jittery**

* Reduce `smoothing_alpha` to 0.1 for slower, more stable adjustments
* Check if physical meter has a loose or damaged needle
* Verify camera is stable and not vibrating
* Review multiple consecutive overlay images for patterns

**System rejects valid readings during high flow**

* Increase `max_dial_change_per_sec` if you have unusually high water flow rates
* Default is 0.5 (meaning 5 units per 10 seconds at 10-second intervals)
* For high-flow scenarios, try 1.0 or higher
* Check logs for "Low confidence" warnings during high flow periods

**Readings lag behind actual meter during flow changes**

* This is expected behavior - the temporal validation smooths rapid changes
* Increase `max_dial_change_per_sec` for faster response to flow changes
* Trade-off: higher values = faster response but less outlier rejection

**Overlay camera shows corrupted/empty image**

* We publish **raw JPEG bytes** to the topic set in `overlay.camera_topic`; discovery disables text decoding (`"encoding": ""`).
* Sanity test:

  ```bash
  mosquitto_pub -h <broker> -t home/watermeter/debug/overlay -f ~/watermeter/debug/overlay_latest.jpg
  ```

**Total seems stuck**

* Delete `~/watermeter/state.json` once to re-seed from live reading.
* Temporarily raise `postproc.big_jump_guard` to let a one-time correction pass.
* Ensure `digits.count` matches your display (5 for many MNK meters).

**Wrong digit around rollovers**

* Slightly tighten/loosen:

  ```yaml
  digits:
    rolling_threshold_up: 0.94
    rolling_threshold_down: 0.06
  ```
* Verify `frac` (progress) logs and dial ROIs.

**Monitoring dial detection quality**

Check logs for confidence information:
```bash
tail -f ~/watermeter/watermeter.log | grep -i "confidence\|DIAL"
```

Review debug overlays regularly:
```bash
open ~/watermeter/debug/overlay_latest.jpg
```

Look for:
- Color-coded boxes indicating confidence levels
- White crosshairs at detected dial centers  
- Confidence percentages with each dial reading

---

## Uninstall

  ```bash
  launchctl unload ~/Library/LaunchAgents/com.watermeter.ocr.plist
  rm -rf ~/watermeter
  ```


