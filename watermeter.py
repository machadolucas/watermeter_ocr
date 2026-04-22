
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water meter reader (Mac) with reference-image alignment.
See README for details.
"""
import os, re, sys, time, json, math, subprocess, logging, argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
import requests
import yaml
import cv2
import numpy as np
import paho.mqtt.client as mqtt

_DEFAULT_DIGITAL_TOTAL_REGEX = r"^\d{6}\.?\d{3}$"
_DEFAULT_DIGITAL_FLOW_REGEX = r"^\d{2,5}\.?\d{3}$"

def load_yaml(path): return yaml.safe_load(open(path,"r"))
def save_json(path,obj):
    tmp=path+".tmp"; open(tmp,"w").write(json.dumps(obj,indent=2))
    os.replace(tmp,path)
def read_json(path,default=None):
    try: return json.load(open(path,"r"))
    except (OSError, ValueError): return default
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def norm_to_abs(box,W,H):
    x,y,w,h=box; return (int(x*W),int(y*H),int(w*W),int(h*H))
def clip01(v): return max(0.0, min(1.0, v))

def circular_blend(a, b, alpha, period=10.0):
    """Weighted blend of two values on a circular scale [0, period).

    Avoids the failure mode of a linear blend across the wrap boundary
    (e.g. linear 0.7*9.9 + 0.3*0.1 = 6.96, circular ≈ 9.96).
    alpha is the weight of `a`; (1-alpha) is the weight of `b`.
    """
    ta = 2.0 * math.pi * (a % period) / period
    tb = 2.0 * math.pi * (b % period) / period
    x = alpha * math.cos(ta) + (1.0 - alpha) * math.cos(tb)
    y = alpha * math.sin(ta) + (1.0 - alpha) * math.sin(tb)
    theta = math.atan2(y, x)
    return (theta * period / (2.0 * math.pi)) % period

def circular_dist(a, b, period=10.0):
    """Shortest distance on a circular scale [0, period)."""
    d = abs(a - b) % period
    return min(d, period - d)

@dataclass
class DialConfig:
    name: str; roi: List[float]; factor: float
    rotation: str="cw"; zero_angle_deg: float=-90.0
    # Runtime adjustments (not from config)
    adjusted_roi: Optional[List[float]]=None
    center_offset: Tuple[float, float]=(0.0, 0.0)
    avg_confidence: float=0.0
    reading_history: List[float]=field(default_factory=list)

@dataclass
class AlignConfig:
    enabled: bool=True
    reference_path: str=os.path.expanduser("~/watermeter/reference.jpg")
    use_mask: bool=True
    anchor_rois: List[List[float]]=field(default_factory=lambda:[
        [0.18,0.00,0.64,0.28],
        [0.05,0.28,0.30,0.22],
        [0.70,0.24,0.25,0.25],
    ])
    nfeatures:int=1200; ratio_test:float=0.75; min_matches:int=40
    ransac_thresh_px: float=3.0; max_scale_change: float=0.08; max_rotation_deg: float=15.0
    warp_mode:str="similarity"; write_debug_aligned: bool=True

@dataclass
class Config:
    esp32_base_url:str
    interval_sec:int=10; retry_backoff_sec:int=5
    image_path:str="/tmp/water_raw.jpg"; image_timeout:float=8.0
    save_debug_overlays:bool=True; debug_dir:str=os.path.expanduser("~/watermeter/debug")
    digits_count:int=6; digits_window:List[float]=field(default_factory=lambda:[0.2078,0.0104,0.5984,0.1917])
    per_digit_inset:float=0.10; rolling_threshold_up:float=0.92; rolling_threshold_down:float=0.08
    state_path:str=os.path.expanduser("~/watermeter/state.json")
    monotonic_epsilon:float=0.0005; big_jump_guard:float=2.0
    mqtt_host:str="127.0.0.1"; mqtt_port:int=1883
    mqtt_username:Optional[str]=None; mqtt_password:Optional[str]=None
    mqtt_main_topic:str="home/watermeter"; mqtt_client_id:str="water-ocr-mac"
    ha_discovery_prefix:str="homeassistant"
    dials:List[DialConfig]=field(default_factory=list)
    ocr_bin:str=os.path.expanduser("~/watermeter/bin/ocr")
    log_path:str=os.path.expanduser("~/watermeter/watermeter.log")
    align:AlignConfig=field(default_factory=AlignConfig)
    overlay_font_scale: float = 1.0
    overlay_font_thickness: int = 2
    overlay_outline_thickness: int = 4
    overlay_line_thickness: int = 2
    debug_keep_latest_only: bool = True
    overlay_publish_mqtt: bool = True
    overlay_camera_topic: str = ""
    overlay_camera_name: str = "Water Meter Overlay"
    overlay_camera_unique_id: str = "water_overlay_macocr"
    overlay_jpeg_quality: int = 85
    quiet_hours_enabled: bool = True
    quiet_start: str = "00:00"
    quiet_end: str = "07:00"
    quiet_interval_sec: int = 60
    # Auto-centering options
    auto_center_dials: bool = True
    center_smoothing_alpha: float = 0.3  # EMA smoothing for center adjustments
    min_confidence_threshold: float = 0.4  # Warn if confidence drops below this
    # Digital (LCD) meter options
    meter_type: str = "mechanical"
    digital_total_roi: List[float] = field(default_factory=list)
    # Optional split-ROI mode for LCDs whose fractional digits are a different
    # size from the integer digits (Apple Vision tends to drop the smaller
    # ones). When BOTH _int and _frac are set, the pipeline OCRs them
    # separately and concatenates with a "." before parsing. When both are
    # empty, the pipeline falls back to single-ROI mode using digital_total_roi.
    digital_total_int_roi: List[float] = field(default_factory=list)
    digital_total_frac_roi: List[float] = field(default_factory=list)
    digital_flow_roi: List[float] = field(default_factory=list)
    digital_total_regex: "re.Pattern" = field(
        default_factory=lambda: re.compile(_DEFAULT_DIGITAL_TOTAL_REGEX)
    )
    digital_flow_regex: "re.Pattern" = field(
        default_factory=lambda: re.compile(_DEFAULT_DIGITAL_FLOW_REGEX)
    )
    digital_max_retries: int = 2
    digital_retry_delay_sec: float = 5.5
    digital_min_digits: int = 6
    # Optional OCR preprocessing. "clahe" applies Contrast Limited Adaptive
    # Histogram Equalization to the captured frame before Vision OCR runs —
    # useful for low-contrast 7-segment LCDs where thin glyphs (e.g. a "1"
    # rendered as two right segments) get dropped by the recognizer.
    # "none" (default) skips preprocessing entirely.
    digital_ocr_preprocess: str = "none"
    # Optional per-digit OCR mode for the digital pipeline. When a digit_count
    # is non-zero, the corresponding region's base ROI is subdivided into
    # that many equal-width sub-ROIs and each digit is recognized
    # independently via the Swift digit-mode helper, then concatenated. More
    # robust than line OCR for LCDs where line recognition is unreliable
    # (rotation ambiguity on small glyphs, narrow "1"s dropped as noise,
    # etc.) — at the cost of N extra subprocess calls per region per attempt.
    # 0 (default) means use line OCR for that region (original behaviour).
    digital_total_int_digit_count: int = 0
    digital_total_frac_digit_count: int = 0
    digital_flow_digit_count: int = 0
    digital_per_digit_inset: float = 0.10
    # When True (default) per-digit mode tries to auto-detect each digit's
    # bounding box via vertical projection before falling back to equal
    # subdivision. This self-corrects minor calibration offsets in the
    # outer ROI. Set False to always use equal subdivision — useful when
    # projection is confused by a busy background.
    digital_per_digit_auto_detect: bool = True
    # Per-digit OCR upscale factor. 1 (default) keeps the sub-ROI at native
    # resolution. Values > 1 upscale each sub-ROI with INTER_CUBIC before
    # OCR — helps when Apple Vision's text detector ignores tight
    # single-character crops of small 7-segment glyphs because they're
    # below its minimum-feature-size threshold. Typical useful range: 2–4.
    digital_ocr_upscale_factor: int = 1
    # Debug: save every per-digit OCR input crop to
    # `{debug_dir}/ocr_crops/{region}_a{attempt}_d{idx}.jpg` and log the
    # call's sub-ROI + raw result at INFO level. Off by default — turn on
    # when diagnosing why per-digit OCR is returning empty.
    digital_save_ocr_crops: bool = False

def load_config(path)->Config:
    raw=load_yaml(path)
    dials=[DialConfig(**d) for d in raw.get("rois",{}).get("dials",[])]
    al=raw.get("alignment",{})
    align=AlignConfig(
        enabled=al.get("enabled",True),
        reference_path=os.path.expanduser(al.get("reference_path","~/watermeter/reference.jpg")),
        use_mask=al.get("use_mask",True),
        anchor_rois=al.get("anchor_rois",AlignConfig().anchor_rois),
        nfeatures=al.get("nfeatures",1200), ratio_test=al.get("ratio_test",0.75),
        min_matches=al.get("min_matches",40), ransac_thresh_px=al.get("ransac_thresh_px",3.0),
        max_scale_change=al.get("max_scale_change",0.08), max_rotation_deg=al.get("max_rotation_deg",15.0),
        warp_mode=al.get("warp_mode","similarity"), write_debug_aligned=al.get("write_debug_aligned",True)
    )
    ov = raw.get("overlay", {})
    topic_default = raw.get("mqtt", {}).get("topic", "home/watermeter")
    qh = raw.get("processing", {}).get("quiet_hours", {})
    ac = raw.get("auto_centering", {})
    meter = raw.get("meter", {})
    dig = raw.get("digital", {})
    dig_rois = raw.get("rois", {}).get("digital", {}) or {}
    return Config(
        esp32_base_url=raw.get("esp32",{}).get("base_url","http://192.168.101.190"),
        interval_sec=raw.get("processing",{}).get("interval_sec",10),
        retry_backoff_sec=raw.get("processing",{}).get("retry_backoff_sec",5),
        image_path=raw.get("processing",{}).get("image_path","/tmp/water_raw.jpg"),
        image_timeout=raw.get("processing",{}).get("image_timeout",8.0),
        save_debug_overlays=raw.get("processing",{}).get("save_debug_overlays",True),
        debug_dir=os.path.expanduser(raw.get("processing",{}).get("debug_dir","~/watermeter/debug")),
        digits_count=raw.get("digits",{}).get("count",6),
        digits_window=raw.get("rois",{}).get("digits",[0.2078,0.0104,0.5984,0.1917]),
        per_digit_inset=raw.get("digits",{}).get("per_digit_inset",0.10),
        rolling_threshold_up=raw.get("digits",{}).get("rolling_threshold_up",0.92),
        rolling_threshold_down=raw.get("digits",{}).get("rolling_threshold_down",0.08),
        state_path=os.path.expanduser(raw.get("paths",{}).get("state_path","~/watermeter/state.json")),
        monotonic_epsilon=raw.get("postproc",{}).get("monotonic_epsilon",0.0005),
        big_jump_guard=raw.get("postproc",{}).get("big_jump_guard",2.0),
        mqtt_host=raw.get("mqtt",{}).get("host","127.0.0.1"),
        mqtt_port=raw.get("mqtt",{}).get("port",1883),
        mqtt_username=raw.get("mqtt",{}).get("username",None),
        mqtt_password=raw.get("mqtt",{}).get("password",None),
        mqtt_main_topic=raw.get("mqtt",{}).get("topic","home/watermeter"),
        mqtt_client_id=raw.get("mqtt",{}).get("client_id","water-ocr-mac"),
        ha_discovery_prefix=raw.get("mqtt",{}).get("ha_discovery_prefix","homeassistant"),
        dials=dials,
        ocr_bin=os.path.expanduser(raw.get("paths",{}).get("ocr_bin","~/watermeter/bin/ocr")),
        log_path=os.path.expanduser(raw.get("paths",{}).get("log_path","~/watermeter/watermeter.log")),
        align=align,
        overlay_font_scale = ov.get("font_scale", 1.0),
        overlay_font_thickness = ov.get("font_thickness", 2),
        overlay_outline_thickness = ov.get("outline_thickness", 4),
        overlay_line_thickness = ov.get("line_thickness", 2),
        debug_keep_latest_only = raw.get("processing", {}).get("debug_keep_latest_only", True),
        overlay_publish_mqtt = ov.get("publish_mqtt", True),
        overlay_camera_topic = ov.get("camera_topic", f"{topic_default}/debug/overlay"),
        overlay_camera_name = ov.get("camera_name", "Water Meter Overlay"),
        overlay_camera_unique_id = ov.get("camera_unique_id", "water_overlay_macocr"),
        overlay_jpeg_quality = ov.get("jpeg_quality", 85),
        quiet_hours_enabled = qh.get("enabled", False),
        quiet_start         = qh.get("start",  "00:00"),
        quiet_end           = qh.get("end",    "07:00"),
        quiet_interval_sec  = qh.get("interval_sec", 60),
        auto_center_dials = ac.get("enabled", True),
        center_smoothing_alpha = ac.get("smoothing_alpha", 0.3),
        min_confidence_threshold = ac.get("min_confidence_threshold", 0.4),
        meter_type = str(meter.get("type", "mechanical")).lower(),
        digital_total_roi = list(dig_rois.get("total", []) or []),
        digital_total_int_roi = list(dig_rois.get("total_int", []) or []),
        digital_total_frac_roi = list(dig_rois.get("total_frac", []) or []),
        digital_flow_roi = list(dig_rois.get("flow", []) or []),
        digital_total_regex = re.compile(dig.get("total_regex", _DEFAULT_DIGITAL_TOTAL_REGEX)),
        digital_flow_regex = re.compile(dig.get("flow_regex", _DEFAULT_DIGITAL_FLOW_REGEX)),
        digital_max_retries = int(dig.get("max_retries", 2)),
        digital_retry_delay_sec = float(dig.get("retry_delay_sec", 5.5)),
        digital_min_digits = int(dig.get("min_digits", 6)),
        digital_ocr_preprocess = str(dig.get("ocr_preprocess", "none")).lower(),
        digital_total_int_digit_count = int(dig.get("total_int_digit_count", 0)),
        digital_total_frac_digit_count = int(dig.get("total_frac_digit_count", 0)),
        digital_flow_digit_count = int(dig.get("flow_digit_count", 0)),
        digital_per_digit_inset = float(dig.get("per_digit_inset", 0.10)),
        digital_per_digit_auto_detect = bool(dig.get("per_digit_auto_detect", True)),
        digital_ocr_upscale_factor = int(dig.get("ocr_upscale_factor", 1)),
        digital_save_ocr_crops = bool(dig.get("save_ocr_crops", False)),
    )

class VisionOCR:
    def __init__(self,bin_path): self.bin=bin_path
    def _run(self,img,roi,half="full"):
        x,y,w,h=roi
        cmd=[self.bin,img,str(x),str(y),str(w),str(h),"--half",half]
        try:
            out=subprocess.check_output(cmd,stderr=subprocess.DEVNULL,timeout=5).decode().strip()
            out="".join(c for c in out if c.isdigit())
            return out[0] if len(out)>1 else out
        except (subprocess.SubprocessError, OSError):
            return ""
    def full_top_bottom(self,img,roi):
        return self._run(img,roi,"full"), self._run(img,roi,"top"), self._run(img,roi,"bottom")

    def read_line(self, img, roi):
        """Run Vision in line mode and return the raw recognized text.

        Returns the raw string (digits plus ".") on success, possibly empty if
        Vision found nothing. Returns None on subprocess failure so callers can
        distinguish a hard error from an empty recognition. Timeout is wider
        than _run because line mode may fall back to the .accurate recognizer.
        """
        x, y, w, h = roi
        cmd = [self.bin, img, str(x), str(y), str(w), str(h), "--mode", "line"]
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
        except (subprocess.SubprocessError, OSError):
            return None

def detect_dial_markings(roi_img, cx, cy, radius):
    """
    Detect the 0 and 5 markings on the dial to validate rotation.
    Returns (rotation_offset_deg, marking_confidence) or (0.0, 0.0) if detection fails.

    The markings should sit at -90° (top, 0) and +90° (bottom, 5) from the true
    dial center. If the dial face is rotated relative to the ROI (camera tilt,
    loose sensor mount) the marking centroids fall at slightly different angles;
    the returned `rotation_offset_deg` is the mean deviation, signed.
    """
    hh, ww = roi_img.shape[:2]
    sample_radius = radius * 0.85

    # Expected (digit, expected_angle_deg, sample_px, sample_py)
    positions = [
        ("0", -90.0, cx, cy - sample_radius),  # top
        ("5",  90.0, cx, cy + sample_radius),  # bottom
    ]

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    detections = []  # (expected_angle_deg, actual_angle_deg)

    for _digit, expected_angle, px, py in positions:
        sample_size = int(radius * 0.15)
        x1 = max(0, int(px - sample_size))
        x2 = min(ww, int(px + sample_size))
        y1 = max(0, int(py - sample_size))
        y2 = min(hh, int(py + sample_size))
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        sample = gray[y1:y2, x1:x2]
        _, thresh = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Use the centroid of the largest dark blob as the marking position.
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] <= 0:
            continue
        actual_px = x1 + M['m10'] / M['m00']
        actual_py = y1 + M['m01'] / M['m00']
        actual_angle = math.degrees(math.atan2(actual_py - cy, actual_px - cx))
        detections.append((expected_angle, actual_angle))

    if not detections:
        return (0.0, 0.0)

    # Mean signed angular deviation, wrapped to [-180, 180].
    offsets = [((act - exp + 180.0) % 360.0) - 180.0 for exp, act in detections]
    rotation_offset = sum(offsets) / len(offsets)
    confidence = len(detections) / 2.0  # 0.5 for one mark, 1.0 for both
    return (rotation_offset, confidence)


def detect_needle_center(roi_img):
    """
    Detect center by analyzing the red needle's thick end (drop shape).
    The needle is thickest at the center pivot point.
    Returns (cx, cy, confidence) or None if detection fails.
    """
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # Detect red needle
    m = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    m = cv2.medianBlur(m, 5)
    
    # Use morphological operations to find the thickest part
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    
    # Find contours
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    
    cnt = max(cs, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None
    
    # Find the point on the contour with maximum local thickness
    # Use distance transform to find thickest region
    dist_transform = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    
    # Find the maximum value (thickest point)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    
    if max_val < 2.0:  # Needle too thin
        return None
    
    cx, cy = max_loc[0], max_loc[1]
    
    # Confidence based on thickness and contour size
    area = cv2.contourArea(cnt)
    roi_area = roi_img.shape[0] * roi_img.shape[1]
    area_ratio = area / roi_area
    
    # Good needle should be 1-5% of ROI area
    if 0.01 < area_ratio < 0.10:
        confidence = 0.9
    else:
        confidence = 0.6
    
    return (cx, cy, confidence)


def detect_dial_center(roi_img):
    """
    Detect the actual center of the dial face using multiple methods.
    Priority: needle center > Hough circles > dial markings > geometric center
    Returns (cx, cy, radius, confidence) or None if detection fails.
    """
    hh, ww = roi_img.shape[:2]
    default_radius = min(ww, hh) / 2.0
    
    # Method 1: Detect center from needle's thick end (most reliable for drop-shaped needles)
    needle_result = detect_needle_center(roi_img)
    if needle_result is not None:
        cx, cy, conf = needle_result
        return (cx, cy, default_radius * 0.8, conf)
    
    # Method 2: Try Hough circles (dial face detection)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=int(min(roi_img.shape[:2]) * 0.3),
        maxRadius=int(min(roi_img.shape[:2]) * 0.6)
    )
    
    if circles is not None and len(circles[0]) > 0:
        # Take the most prominent circle
        circle = circles[0][0]
        cx, cy, radius = circle[0], circle[1], circle[2]
        
        # Try to validate/refine using dial markings
        rotation_offset, marking_conf = detect_dial_markings(roi_img, cx, cy, radius)
        
        # Confidence for circle detection
        confidence = 0.6 + 0.2 * marking_conf  # 0.6-0.8 range (lower than needle method)
        return (cx, cy, radius, confidence)
    
    # Fallback: use geometric center with low confidence
    return (ww / 2.0, hh / 2.0, default_radius, 0.3)


def detect_needle_by_color(roi_img, cx, cy):
    """
    Detect red needle using color segmentation.
    Returns (angle, confidence) or None if detection fails.
    """
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    m = cv2.medianBlur(m, 5)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_DILATE, k, iterations=1)
    
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    
    cnt = max(cs, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    
    px, py = M['m10'] / M['m00'], M['m01'] / M['m00']
    ang = math.degrees(math.atan2(py - cy, px - cx))
    
    # Confidence based on contour area relative to expected needle size
    area = cv2.contourArea(cnt)
    roi_area = roi_img.shape[0] * roi_img.shape[1]
    area_ratio = area / roi_area
    confidence = min(1.0, area_ratio * 50.0)  # heuristic: needle should be ~2% of ROI
    
    return (ang, confidence)


def detect_needle_by_lines(roi_img, cx, cy):
    """
    Detect needle using Hough line detection on edges.
    Returns (angle, confidence) or None if detection fails.
    """
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(min(roi_img.shape[:2]) * 0.2),
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return None
    
    # Find line closest to passing through center
    best_angle = None
    best_score = float('inf')
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate distance from line to center
        line_vec = np.array([x2 - x1, y2 - y1])
        line_len = np.linalg.norm(line_vec)
        if line_len < 1:
            continue
        
        # Point-to-line distance
        point_vec = np.array([cx - x1, cy - y1])
        # Use 3D vectors for cross product to avoid NumPy 2.0 deprecation warning
        line_vec_3d = np.array([line_vec[0], line_vec[1], 0])
        point_vec_3d = np.array([point_vec[0], point_vec[1], 0])
        cross = abs(np.cross(line_vec_3d, point_vec_3d)[2])
        dist = cross / line_len
        
        if dist < best_score:
            best_score = dist
            # Calculate angle from center to line midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            best_angle = math.degrees(math.atan2(my - cy, mx - cx))
    
    if best_angle is None:
        return None
    
    # Confidence inversely related to distance from center
    confidence = max(0.0, 1.0 - best_score / (min(roi_img.shape[:2]) * 0.5))
    return (best_angle, confidence)


def validate_reading_with_history(current_reading, history):
    """
    Validate current dial reading against historical readings for temporal consistency.
    Focus on trend consistency rather than absolute rate limits.
    
    Returns (validated_reading, confidence_adjustment)
    """
    if not history or len(history) == 0:
        return (current_reading, 1.0)
    
    recent = history[-5:]  # Last 5 readings
    
    diff = abs(current_reading - recent[-1])
    
    # Dial readings should change smoothly (0-10 range)
    # Small changes are always confident
    if diff < 1.0:
        return (current_reading, 1.0)
    
    # Medium changes (1-3 units) - check if consistent with trend
    if diff < 3.0 and len(recent) >= 3:
        # Calculate average rate of change
        changes = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_change = sum(changes) / len(changes)
        
        # If current change is in same direction and similar magnitude, accept
        current_change = current_reading - recent[-1]
        if abs(current_change - avg_change) < 2.0:
            return (current_reading, 0.9)
    
    # Large sudden change (>3 units) - likely a misread, reduce confidence
    # The higher-level big_jump_guard will handle this at the total reading level
    return (current_reading, 0.5)


def predict_expected_reading(history):
    """
    Predict the next dial reading by linear extrapolation from the most recent
    history window. Useful for biasing ambiguous readings toward the trend.

    Returns predicted reading in [0, 10) or None if insufficient data.
    """
    if len(history) < 3:
        return None
    recent = history[-5:]
    changes = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_change = sum(changes) / len(changes)
    return (recent[-1] + avg_change) % 10.0


def read_dial(roi_img, zero_angle_deg=-90.0, rotation="cw", prev_reading=None, history=None):
    """
    Enhanced dial reading with automatic center detection, multi-method needle detection,
    and temporal validation using reading history.
    
    Returns (reading, confidence, center_offset) where:
    - reading: 0-10 value
    - confidence: 0-1 quality score
    - center_offset: (dx, dy) offset from ROI center to detected dial center
    """
    hh, ww = roi_img.shape[:2]
    
    # Step 1: Detect actual dial center (enhanced with marking detection)
    center_result = detect_dial_center(roi_img)
    if center_result is None:
        cx, cy = ww / 2.0, hh / 2.0
        center_confidence = 0.1
    else:
        cx, cy, radius, center_confidence = center_result
    
    center_offset = (cx - ww / 2.0, cy - hh / 2.0)
    
    # Step 2: Try multiple needle detection methods
    methods = []
    
    # Method 1: Color-based detection (original method)
    color_result = detect_needle_by_color(roi_img, cx, cy)
    if color_result:
        methods.append(('color', color_result[0], color_result[1]))
    
    # Method 2: Line-based detection
    line_result = detect_needle_by_lines(roi_img, cx, cy)
    if line_result:
        methods.append(('lines', line_result[0], line_result[1]))
    
    # Step 3: Predict expected reading from history
    predicted = None
    if history and len(history) >= 3:
        predicted = predict_expected_reading(history)
    
    if not methods:
        # No detection - use previous or predicted reading if available
        if predicted is not None and abs(predicted - (prev_reading or 0)) < 1.0:
            return (predicted, 0.3, center_offset)
        if prev_reading is not None:
            return (prev_reading, 0.2, center_offset)
        return (0.0, 0.0, center_offset)
    
    # Step 4: Combine results with weighted average
    if len(methods) == 1:
        method_name, ang, conf = methods[0]
    else:
        # Weight by confidence and check for agreement
        angles = [m[1] for m in methods]
        confs = [m[2] for m in methods]
        
        # Normalize angles to same quadrant for comparison
        normalized = [(a % 360) for a in angles]
        angle_diff = abs(normalized[0] - normalized[1])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # If methods agree (within 30 degrees), boost confidence
        if angle_diff < 30:
            total_conf = sum(confs)
            ang = sum(a * c for a, c in zip(angles, confs)) / total_conf
            conf = min(1.0, total_conf / len(methods) * 1.2)
        else:
            # Methods disagree - use highest confidence
            best_idx = confs.index(max(confs))
            ang = angles[best_idx]
            conf = confs[best_idx] * 0.7  # reduce confidence due to disagreement
    
    # Convert angle to reading
    a = (ang - zero_angle_deg) % 360.0
    v = (a / 360.0) * 10.0
    if rotation == "ccw":
        v = (10.0 - v) % 10.0
    
    reading = (v + 10.0) % 10.0
    
    # Step 5: Validate with temporal consistency
    if history and len(history) > 0:
        validated_reading, temporal_conf = validate_reading_with_history(reading, history)
        
        # If predicted reading exists and current reading is ambiguous,
        # bias toward prediction using a circular blend so values across the
        # 9→0 wrap (e.g. reading=9.9, predicted=0.1) collapse to the correct
        # side of the ring rather than a linear midpoint.
        if predicted is not None and conf < 0.6:
            if circular_dist(reading, predicted) < 1.5:
                reading = circular_blend(reading, predicted, alpha=0.7)
                conf = conf * 0.8 + 0.2  # small boost for using prediction
        
        # Apply temporal confidence adjustment
        conf = conf * temporal_conf
    
    # Final confidence combines center detection and needle detection
    final_confidence = conf * (0.7 + 0.3 * center_confidence)
    
    return (reading % 10.0, final_confidence, center_offset)

def decide_digit(bottom, top, full, progress, thr_up, thr_dn, prev_digit=None):
    """
    Prefer previous digit when we're not near a rollover (thr_dn < progress < thr_up).
    Use top/bottom rules only at the edges or when both agree strongly.
    
    Progress represents the fractional progress from lower-order positions.
    When progress is high (>thr_up), we expect the digit to have rolled to the next value.
    When progress is low (<thr_dn) after being high, it indicates a rollover just occurred.
    """
    toint = lambda s: int(s) if (isinstance(s, str) and s.isdigit()) else None
    b, t, f = toint(bottom), toint(top), toint(full)

    in_middle = (thr_dn < progress < thr_up)

    # 1) Rolling window logic (top is next of bottom) - handle transitions
    if b is not None and t is not None and t == ((b + 1) % 10):
        # Near the top threshold - digit is rolling to next value
        if progress >= thr_up:
            return t
        # Near the bottom threshold - digit just rolled or is stable at bottom
        if progress <= thr_dn:
            return b
        # In transition zone - use previous if available
        return prev_digit if prev_digit is not None else b

    # 2) If top and bottom agree, trust them
    if b is not None and t is not None and b == t:
        return b

    # 3) Check if we just rolled over (low progress, OCR shows the new digit)
    # This handles the case where digit is cropped/transitioning but dials show rollover happened.
    # NOTE: we deliberately do NOT infer a rollover when all OCR is None. Progress alone is not a
    # transition signal (decide_digit is stateless per call), so "progress low + OCR silent" also
    # matches a long-stable low-progress window. Speculatively incrementing there caused a
    # runaway drift of ~1 digit per cycle under persistent OCR failure.
    if prev_digit is not None and progress <= thr_dn:
        next_digit = (prev_digit + 1) % 10

        # If OCR managed to see the new digit, great! Use it
        if f == next_digit or t == next_digit or b == next_digit:
            return next_digit

        # If OCR clearly shows the previous digit is still there, trust OCR
        # (rollover hasn't happened yet, or low progress is normal for this digit)
        if f == prev_digit or t == prev_digit or b == prev_digit:
            return prev_digit

        # Otherwise fall through to the standard fallback, which prefers prev.
        # Worst case: we lag by one reading for one cycle — far safer than creeping drift.

    # 4) If we have previous digit and we're in stable zone, use it
    if prev_digit is not None and in_middle:
        return prev_digit

    # 5) Otherwise try full, then bottom, then top, then prev
    for v in (f, b, t, prev_digit):
        if v is not None:
            return v

    return 0

def parse_digital_total(raw, regex):
    """Parse the total-consumption line read from the LCD.

    Accepts both dotted (`"000100.000"`) and undotted (`"000100000"`) strings —
    Vision sometimes misses the decimal on 7-segment displays. When the regex
    matches but no "." is present, the decimal is injected three places from
    the right so the meter's 3 fractional digits always map to the same order
    of magnitude regardless of what Vision emitted.

    Returns the parsed float on success, or None if the input is None, fails
    the regex, or float() chokes. The caller is expected to treat None as
    "wrong view / unreadable" — never as 0.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not regex.match(s):
        return None
    if "." not in s:
        s = s[:-3] + "." + s[-3:]
    try:
        return float(s)
    except ValueError:
        return None


def parse_digital_flow(raw, regex):
    """Parse the instantaneous-flow line (m³/h).

    Accepts both dotted (`"00.125"`) and dotless (`"00125"`) forms — mirrors
    parse_digital_total because Vision routinely drops the small baseline
    decimal dot on 7-segment LCDs. When the regex matches but no "." is
    present, the decimal is injected three places from the right (the meter's
    fractional-digit count is a physical property, same as the total line).
    """
    if raw is None:
        return None
    s = raw.strip()
    if not regex.match(s):
        return None
    if "." not in s:
        s = s[:-3] + "." + s[-3:]
    try:
        return float(s)
    except ValueError:
        return None


def is_valid_digital_view(raw_total, raw_flow, cfg):
    """Cheap pre-validator: does this capture *look* like the numeric view?

    Runs before the stricter regex parse so the diagnostic views (no digits, or
    too few) are rejected without paying for a float() / retry-logic round-trip.
    The total line must contain at least `digital_min_digits` digit characters.

    The flow line is optional: some installations have flash glare that forces
    camera alignment to favour a clean total line at the expense of the flow
    line. When `digital_flow_roi` is empty, flow isn't captured or validated
    here, and `raw_flow` is expected to be None.
    """
    if raw_total is None:
        return False
    total_digits = sum(1 for c in raw_total if c.isdigit())
    if total_digits < cfg.digital_min_digits:
        return False
    if cfg.digital_flow_roi:
        if raw_flow is None:
            return False
        flow_digits = sum(1 for c in raw_flow if c.isdigit())
        if flow_digits < 2:  # flow is at minimum "X.YYY" — 4 digits, but accept minor misses
            return False
    return True


def validate_digital_reading(total, flow, prev_total, cfg):
    """Post-parse sanity check.

    Rejects None/NaN/inf totals, negative/non-finite flow (when flow tracking
    is enabled), and totals that drift more than `big_jump_guard` from the
    previous reading. The first read (prev_total is None) is always accepted
    so the service has a baseline on a cold start. When `digital_flow_roi` is
    empty, `flow` is expected to be None and is not evaluated.
    """
    if total is None or not math.isfinite(total):
        return False
    if cfg.digital_flow_roi:
        if flow is None or not math.isfinite(flow) or flow < 0:
            return False
    if prev_total is not None and abs(total - prev_total) > cfg.big_jump_guard:
        return False
    return True


def compose_integer(digit_obs, frac, thr_up, thr_dn, prev_int_str=None):
    """
    digit_obs: list of tuples [(full, top, bottom), ...] left->right
    frac: smooth progress in [0..1) from the ×0.1 dial (not the published fraction)
    prev_int_str: previous integer zero-padded to len(digit_obs)
    """
    n = len(digit_obs)
    prev_digits = (
        [int(c) for c in prev_int_str]
        if prev_int_str and prev_int_str.isdigit() and len(prev_int_str) == n
        else [None] * n
    )

    resolved = [0] * n
    # lower = value from less-significant positions scaled to [0..1)
    # start with the smooth tenths progress
    lower = frac

    for pos in range(n - 1, -1, -1):           # from rightmost to leftmost
        full, top, bot = digit_obs[pos]
        progress = clip01(lower)               # what fraction of carry-in we have for this digit
        prev = prev_digits[pos] if prev_digits[pos] is not None else None
        chosen = decide_digit(bot, top, full, progress, thr_up, thr_dn, prev)
        resolved[pos] = chosen
        # carry ripple for the next (more significant) digit
        lower = (chosen + lower) / 10.0

    return "".join(str(d) for d in resolved), resolved

class MqttClient:
    def __init__(self,cfg,log):
        self.cfg=cfg; self.log=log
        self.client=mqtt.Client(
            client_id=cfg.mqtt_client_id, 
            clean_session=True,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,)
        if cfg.mqtt_username: self.client.username_pw_set(cfg.mqtt_username,cfg.mqtt_password)
        self.connected=False
        self._loop_started=False
        self._last_attempt=0.0
        self._reconnect_delay=5.0
        def _on_connect(client, userdata, flags, reason_code, properties=None):
            # reason_code 0 == Success. Anything else means the broker rejected
            # the CONNECT (bad credentials, protocol mismatch, etc). Publishing
            # in that state silently no-ops because `if self.connected:` passes,
            # so keep the flag False on failure.
            if reason_code == 0:
                self.connected = True
                self.log.info(f"MQTT connected rc={reason_code}")
                # Republish HA discovery payloads on every (re)connect. These are
                # retained on the broker so we do NOT need to republish every loop.
                try:
                    self.discovery()
                except Exception as e:
                    self.log.warning(f"MQTT discovery publish failed: {e}")
            else:
                self.connected = False
                self.log.error(f"MQTT connect rejected rc={reason_code}")

        def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
            self.connected = False
            self.log.warning(f"MQTT disconnected rc={reason_code}")

        self.client.on_connect = _on_connect
        self.client.on_disconnect = _on_disconnect
    def _ensure_loop(self):
        if not self._loop_started:
            try:
                self.client.loop_start()
                self._loop_started = True
            except Exception as e:
                self.log.error(f"MQTT loop_start failed: {e}")
    def connect(self):
        self._last_attempt = time.time()
        try:
            self.client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
        except Exception as e:
            self.log.error(f"MQTT connect failed: {e}")
        self._ensure_loop()
    def publish(self,t,p,retain=False):
        if self.connected: self.client.publish(t,p,retain=retain)
    def ensure_connection(self):
        if self.connected:
            return
        now = time.time()
        if now - self._last_attempt < self._reconnect_delay:
            return
        self._last_attempt = now
        try:
            self.client.reconnect()
            self.log.info("Attempting MQTT reconnect")
        except Exception as e:
            self.log.warning(f"MQTT reconnect failed ({e}); retrying full connect")
            try:
                self.client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
            except Exception as e2:
                self.log.error(f"MQTT connect retry failed: {e2}")
        self._ensure_loop()
    def discovery(self):
        base = self.cfg.mqtt_main_topic
        pre  = self.cfg.ha_discovery_prefix

        total = {
            "name": "Water Total",
            "state_topic": f"{base}/main/value",
            "unit_of_measurement": "m³",
            "state_class": "total_increasing",
            "device_class": "water",
            "unique_id": "water_total_macocr",
            "device": {"identifiers": ["water_cam_mac"], "name": "Water Meter (Mac OCR)"},
        }
        rate = {
            "name": "Water Rate",
            "state_topic": f"{base}/main/rate",
            "unit_of_measurement": "m³/min",
            "state_class": "measurement",
            "unique_id": "water_rate_macocr",
            "device": {"identifiers": ["water_cam_mac"], "name": "Water Meter (Mac OCR)"},
        }
        rate_lpm = {
            "name": "Water Rate L/min",
            "state_topic": f"{base}/main/rate_lpm",
            "unit_of_measurement": "L/min",
            "state_class": "measurement",
            "unique_id": "water_rate_lpm_macocr",
            "icon": "mdi:water-pump",
            "device": {"identifiers": ["water_cam_mac"], "name": "Water Meter (Mac OCR)"},
        }
        self.publish(f"{pre}/sensor/water_total/config",json.dumps(total),retain=True)
        self.publish(f"{pre}/sensor/water_rate/config",json.dumps(rate),retain=True)
        self.publish(f"{pre}/sensor/water_rate_lpm/config", json.dumps(rate_lpm), retain=True)

        # Digital meters publish an on-meter instantaneous flow in m³/h directly
        # (no need to infer it from deltas). Advertise the sensor only when
        # (a) we're in digital mode AND (b) a flow ROI is actually configured —
        # some installs capture only the total line because of flash glare.
        digital_flow_tracked = (
            getattr(self.cfg, "meter_type", "mechanical") == "digital"
            and bool(getattr(self.cfg, "digital_flow_roi", []))
        )
        if digital_flow_tracked:
            flow_m3h = {
                "name": "Water Flow",
                "state_topic": f"{base}/main/flow_m3h",
                "unit_of_measurement": "m³/h",
                "state_class": "measurement",
                "device_class": "water",
                "unique_id": "water_flow_m3h_macocr",
                "icon": "mdi:water-pump",
                "device": {"identifiers": ["water_cam_mac"], "name": "Water Meter (Mac OCR)"},
            }
            self.publish(f"{pre}/sensor/water_flow_m3h/config", json.dumps(flow_m3h), retain=True)

        cam = {
            "name": self.cfg.overlay_camera_name,
            "topic": self.cfg.overlay_camera_topic,
            "encoding": "",    # we publish raw JPEG bytes
            "unique_id": self.cfg.overlay_camera_unique_id,
            "device": {"identifiers": ["water_cam_mac"], "name": "Water Meter (Mac OCR)"},
        }
        self.publish(f"{pre}/camera/water_overlay/config", json.dumps(cam), retain=True)

class Aligner:
    def __init__(self,cfg,log):
        self.cfg=cfg; self.log=log
        self.ref_img=None; self.ref_gray=None; self.mask=None
        self.kpR=None; self.descR=None
        self.orb=cv2.ORB_create(nfeatures=self.cfg.nfeatures)
    def ensure_reference(self,img):
        if self.ref_img is not None: return True
        p=os.path.expanduser(self.cfg.reference_path)
        if os.path.exists(p):
            self.ref_img=cv2.imread(p,cv2.IMREAD_COLOR)
            if self.ref_img is None: self.log.error("Cannot read reference"); return False
        else:
            os.makedirs(os.path.dirname(p),exist_ok=True)
            cv2.imwrite(p,img); self.ref_img=img.copy()
            self.log.info(f"Saved first frame as reference: {p}")
        self.ref_gray=cv2.cvtColor(self.ref_img,cv2.COLOR_BGR2GRAY)
        self._build_mask(self.ref_img.shape[1], self.ref_img.shape[0])
        self.kpR,self.descR=self.orb.detectAndCompute(self.ref_gray,self.mask)
        return True
    def _build_mask(self,W,H):
        if not self.cfg.use_mask: self.mask=None; return
        m=np.zeros((H,W),dtype=np.uint8)
        for x,y,w,h in self.cfg.anchor_rois:
            ax,ay,aw,ah=int(x*W),int(y*H),int(w*W),int(h*H)
            cv2.rectangle(m,(ax,ay),(ax+aw,ay+ah),255,-1)
        self.mask=m
    def align(self,img):
        if not self.ensure_reference(img): return img,None,False
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # The mask is sized for the reference frame. If the incoming frame has
        # a different shape (e.g. ESP32 resolution changed), cv2.detectAndCompute
        # would raise; detect a shape mismatch and fall through without a mask
        # rather than crash.
        mask = self.mask
        if mask is not None and mask.shape[:2] != gray.shape[:2]:
            self.log.warning(
                f"Incoming frame {gray.shape[:2]} != reference {mask.shape[:2]}; "
                "aligning without anchor mask this cycle"
            )
            mask = None
        kpC,descC=self.orb.detectAndCompute(gray,mask)
        if descC is None or self.descR is None or len(kpC)<10 or len(self.kpR)<10: return img,None,False
        bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches=bf.knnMatch(self.descR,descC,k=2)
        good=[m for m,n in matches if m.distance < self.cfg.ratio_test*n.distance]
        if len(good) < self.cfg.min_matches: return img,None,False
        src=np.float32([ self.kpR[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst=np.float32([ kpC[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        if self.cfg.warp_mode=="similarity":
            M,_=cv2.estimateAffinePartial2D(dst,src,method=cv2.RANSAC,ransacReprojThreshold=self.cfg.ransac_thresh_px)
        else:
            M,_=cv2.estimateAffine2D(dst,src,method=cv2.RANSAC,ransacReprojThreshold=self.cfg.ransac_thresh_px)
        if M is None: return img,None,False
        a,b=M[0,0],M[0,1]; scale=(a*a+b*b)**0.5; rot=math.degrees(math.atan2(b,a))
        if abs(scale-1.0)>self.cfg.max_scale_change or abs(rot)>self.cfg.max_rotation_deg:
            return img,None,False
        aligned=cv2.warpAffine(img,M,(self.ref_img.shape[1],self.ref_img.shape[0]),flags=cv2.INTER_LINEAR)
        if self.cfg.write_debug_aligned:
            cv2.imwrite(os.path.expanduser("~/watermeter/aligned_last.jpg"), aligned)
        return aligned,M,True

def _draw_label(img, text, org, scale, color, thickness, outline_thickness):
    # draw a black outline (stroke) for readability, then the colored text
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def adjust_dial_roi(original_roi, center_offset, W, H, smoothing_alpha=0.3):
    """
    Adjust dial ROI to center it on the detected dial center.
    Uses exponential moving average for smooth adjustments.
    """
    x, y, w, h = original_roi
    dx, dy = center_offset
    
    # Convert offset from ROI coordinates to normalized coordinates
    dx_norm = dx * w / W
    dy_norm = dy * h / H
    
    # Apply smoothing to avoid jitter
    new_x = x + dx_norm * smoothing_alpha
    new_y = y + dy_norm * smoothing_alpha
    
    # Ensure ROI stays within image bounds
    new_x = max(0.0, min(1.0 - w, new_x))
    new_y = max(0.0, min(1.0 - h, new_y))
    
    return [new_x, new_y, w, h]

def _overlay_base(img, cfg, aligned_ok):
    """Shared overlay preamble: copy, compute scales, stamp timestamp and
    alignment-tinted status. Returns (ov, fs, th, o_th, box_th, status_color).
    Used by both the mechanical and digital draw_overlays_* helpers.

    Timestamp is drawn at the BOTTOM-LEFT so it doesn't collide with a
    failure banner rendered later in draw_overlays_digital (the banner sits
    at the top of the image)."""
    ov = img.copy()
    H = ov.shape[0]
    fs = cfg.overlay_font_scale * (H / 480.0)
    th = int(cfg.overlay_font_thickness)
    o_th = int(cfg.overlay_outline_thickness)
    box_th = int(cfg.overlay_line_thickness)
    status_color = (0, 255, 0) if aligned_ok else (0, 0, 255)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _draw_label(ov, timestamp, (10, H - int(8 * fs)), fs * 0.6,
                (255, 255, 255), th, o_th)
    return ov, fs, th, o_th, box_th, status_color


def draw_overlays(img, digits_rois_abs, per_digit_vals, dials_abs, dial_vals, dial_confidences,
                  out_path, aligned_ok, cfg, center_offsets=None, resolved_digits=None):
    ov, fs, th, o_th, box_th, color_digits = _overlay_base(img, cfg, aligned_ok)

    # Digit boxes + labels
    for i, (x, y, w, h) in enumerate(digits_rois_abs):
        cv2.rectangle(ov, (x, y), (x + w, y + h), color_digits, box_th)
        if i < len(per_digit_vals):
            # Show raw OCR value, or fall back to resolved digit if OCR failed
            val = per_digit_vals[i]
            if val in (None, ""):
                # Fall back to resolved digit if available
                val = resolved_digits[i] if resolved_digits and i < len(resolved_digits) else None
            if val is not None:
                _draw_label(ov, f"{val}",
                            (x, max(12, y - 4)), fs, color_digits, th, o_th)

    # Dial boxes + labels with confidence-based coloring
    for i, (x, y, w, h) in enumerate(dials_abs):
        # Color based on confidence: green (high) -> yellow (medium) -> red (low)
        if i < len(dial_confidences):
            conf = dial_confidences[i]
            if conf > 0.7:
                color_dials = (0, 255, 0)  # green
            elif conf > 0.4:
                color_dials = (0, 255, 255)  # yellow
            else:
                color_dials = (0, 0, 255)  # red
        else:
            color_dials = (255, 0, 255)  # magenta (no confidence info)
        
        cv2.rectangle(ov, (x, y), (x + w, y + h), color_dials, box_th)
        
        # Draw center crosshair if we have offset info
        if center_offsets and i < len(center_offsets):
            dx, dy = center_offsets[i]
            cx_abs = int(x + w/2 + dx)
            cy_abs = int(y + h/2 + dy)
            cross_size = 5
            cv2.line(ov, (cx_abs - cross_size, cy_abs), (cx_abs + cross_size, cy_abs), (255, 255, 255), 1)
            cv2.line(ov, (cx_abs, cy_abs - cross_size), (cx_abs, cy_abs + cross_size), (255, 255, 255), 1)
        
        # Label with value and confidence
        if i < len(dial_vals):
            label_text = f"{dial_vals[i]:.2f}"
            if i < len(dial_confidences):
                label_text += f" ({dial_confidences[i]:.0%})"
            
            # Position labels to avoid overlap:
            # - Middle-left dial (index 1): bottom-left, outside the dial box
            # - Middle-right dial (index 2): bottom-right, outside the dial box  
            # - Other dials: above the dial (default)
            num_dials = len(dials_abs)
            if num_dials == 4 and (i == 2 or i == 3):
                # Bottom-left of dial: text right-aligned to left edge of box
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, th)[0]
                label_x = x - text_size[0] - 4
                label_y = y + h - 4
                _draw_label(ov, label_text, (label_x, label_y), fs * 0.7, color_dials, th, o_th)
            elif num_dials == 4 and (i == 1 or i == 0):
                # Bottom-right of dial: text left-aligned to right edge of box
                label_x = x + w + 4
                label_y = y + h - 4
                _draw_label(ov, label_text, (label_x, label_y), fs * 0.7, color_dials, th, o_th)
            else:
                # Default: above the dial
                _draw_label(ov, label_text,
                            (x, max(12, y - 4)), fs * 0.7, color_dials, th, o_th)

    if out_path:
        cv2.imwrite(out_path, ov)
    return ov


def draw_overlays_digital(img, total_roi_abs, flow_roi_abs,
                          raw_total_str, raw_flow_str,
                          parsed_total, parsed_flow,
                          aligned_ok, cfg, reason=None,
                          out_path=None, log=None,
                          total_int_roi_abs=None, total_frac_roi_abs=None,
                          raw_total_int_str=None, raw_total_frac_str=None):
    """Overlay for digital LCD meters: rectangles for the total/flow ROIs,
    each labelled with what Vision read raw and what the parser produced.
    `raw_*_str` may be None/empty when the display was on a diag view at the
    moment of capture — we still draw the rectangle so ROI placement is
    visible in the debug image.

    Split-ROI mode: when both `total_int_roi_abs` and `total_frac_roi_abs`
    are provided, those two rectangles are drawn (labelled INT/FRAC with
    their individual raw OCR strings) and `total_roi_abs` is ignored. The
    combined parsed value is still shown on the INT rect. This lets the
    debug JPEG reflect the actual OCR geometry when the fractional digits
    are captured from a separate, smaller ROI.

    `reason` is an optional failure tag ("wrong_view" / "parse_failed" /
    "validate_failed" / "no_frame"); when set (and not "ok"), a coloured
    banner is drawn across the top of the image so the debug JPEG visibly
    distinguishes a rejected cycle from a successful one. Callers pass
    reason="ok" or None on success."""
    ov, fs, th, o_th, box_th, status = _overlay_base(img, cfg, aligned_ok)

    def _draw_line(roi_abs, raw_str, parsed_val, label):
        if roi_abs is None or len(roi_abs) != 4:
            return
        x, y, w, h = roi_abs
        cv2.rectangle(ov, (x, y), (x + w, y + h), status, box_th)
        raw_disp = raw_str if raw_str else "(no read)"
        parsed_disp = f"{parsed_val:.3f}" if parsed_val is not None else "—"
        _draw_label(ov, f"{label}: {parsed_disp}",
                    (x, max(12, y - 6)), fs * 0.7, status, th, o_th)
        _draw_label(ov, f"raw: {raw_disp}",
                    (x, y + h + int(18 * fs)), fs * 0.55, (220, 220, 220), th, o_th)

    split_mode = (total_int_roi_abs is not None and len(total_int_roi_abs) == 4
                  and total_frac_roi_abs is not None and len(total_frac_roi_abs) == 4)
    if split_mode:
        _draw_line(total_int_roi_abs, raw_total_int_str, parsed_total, "TOTAL m³ (int)")
        _draw_line(total_frac_roi_abs, raw_total_frac_str, None, "frac")
    else:
        _draw_line(total_roi_abs, raw_total_str, parsed_total, "TOTAL m³")
    _draw_line(flow_roi_abs, raw_flow_str, parsed_flow, "FLOW m³/h")

    if reason and reason != "ok":
        # BGR. Failure banner helps during debugging — e.g. "all cycles giving
        # up" is visually distinct from "service is up and reading normally".
        banner_color = {
            "wrong_view":      (30, 30, 200),    # red
            "parse_failed":    (30, 140, 230),   # orange
            "validate_failed": (30, 180, 230),   # yellow-orange
            "no_frame":        (100, 100, 100),  # gray
        }.get(reason, (30, 30, 200))
        banner_h = max(20, int(28 * fs))
        cv2.rectangle(ov, (0, 0), (ov.shape[1], banner_h), banner_color, -1)
        _draw_label(ov, f"REJECTED: {reason}", (8, int(20 * fs)),
                    fs * 0.85, (255, 255, 255), th, o_th)

    if out_path:
        try:
            cv2.imwrite(out_path, ov)
        except Exception as e:
            if log is not None:
                log.warning("Failed to save overlay to %s: %s", out_path, e)
    return ov


def estimate_total_from_dials(prev_total, frac_pub, cfg):
    """Use dial fractions plus previous reading to infer a sane total.

    Assumes consumption between captures is <1 m³, so at most one rollover.
    """
    if prev_total is None:
        return None

    prev_int = math.floor(prev_total)
    prev_frac = prev_total - prev_int

    # Use configurable thresholds to detect rollover around zero.
    low = min(0.25, max(0.05, cfg.rolling_threshold_down * 2.0))
    high = 1.0 - low

    expected_int = prev_int

    if prev_frac > high and frac_pub < low:
        expected_int += 1  # rollover happened but OCR missed it
    elif prev_frac < low and frac_pub > high:
        expected_int = max(0, expected_int - 1)  # handle rare counter rollbacks

    return expected_int + frac_pub


def build_digit_rois(cfg,W,H):
    x,y,w,h=cfg.digits_window; dw=w/cfg.digits_count; rois=[]; inset=cfg.per_digit_inset
    for i in range(cfg.digits_count):
        sx=x+i*dw; sy=y; sw=dw; sh=h
        rois.append([sx+sw*inset, sy+sh*inset, sw*(1-2*inset), sh*(1-2*inset)])
    return rois


def _equal_subdivide_roi(base_roi, count, inset):
    """Split base_roi into `count` equal-width horizontal cells with `inset`
    fraction shrinkage on edges SHARED with a neighbour cell (inner edges).
    Outer edges — the base ROI's left edge for cell 0, right edge for cell
    N-1, and both the top + bottom edges for every cell — are NOT shrunk,
    so edge glyphs aren't accidentally clipped when the base ROI hugs the
    first/last digit tightly. Returns normalized [x,y,w,h] sub-ROIs.
    """
    x, y, w, h = base_roi
    dw = w / count
    pad_y = h * inset
    subs = []
    for i in range(count):
        left_inset = dw * inset if i > 0 else 0.0
        right_inset = dw * inset if i < count - 1 else 0.0
        sx = x + i * dw + left_inset
        sw = dw - left_inset - right_inset
        subs.append([sx, y + pad_y, sw, h - 2 * pad_y])
    return subs


def _auto_detect_digit_sub_rois(img_path, base_roi, expected_count, inset):
    """Detect digit bounding rects inside `base_roi` using a vertical-pixel
    projection of the binarized crop. Returns a list of `expected_count`
    normalized sub-ROIs sorted left-to-right, or None if detection can't
    reliably identify exactly that many digits (caller should fall back to
    equal subdivision).

    How it works: crop the ROI, Otsu-threshold to binary (digits = white),
    MORPHOLOGICALLY CLOSE horizontally to fill the hollow interior of
    7-segment glyphs (a "0" would otherwise project as two separate runs —
    its left and right vertical strokes — tripping the count check), sum
    "ink" per column, mark columns above 15% of the peak as "in a digit".
    Contiguous ink runs → one digit each; runs below a minimum width are
    discarded as noise. If we get exactly `expected_count` runs we return
    them (with `inset` padding); otherwise None.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    Hf, Wf = img.shape[:2]
    ax, ay, aw, ah = norm_to_abs(base_roi, Wf, Hf)
    if aw <= 0 or ah <= 0:
        return None
    crop = img[ay:ay + ah, ax:ax + aw]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, thr = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ch, cw = thr.shape
    if cw == 0 or ch == 0:
        return None

    expected_digit_w = max(1, cw // expected_count)
    min_run_w = max(2, cw // (expected_count * 3))

    def _runs_for(binary):
        profile = (binary > 0).sum(axis=0)
        if profile.max() == 0:
            return None
        ink = profile > profile.max() * 0.15
        rs = []
        s = None
        for i, v in enumerate(ink):
            if v and s is None:
                s = i
            elif not v and s is not None:
                rs.append((s, i))
                s = None
        if s is not None:
            rs.append((s, len(ink)))
        return [r for r in rs if r[1] - r[0] >= min_run_w]

    # Pass 1: raw binary. Works when digits are solid rectangles with clear
    # gaps between them (mostly synthetic / cleanly-rendered fonts).
    runs = _runs_for(thr)
    if runs is None:
        return None
    if len(runs) != expected_count:
        # Pass 2: horizontally close the binary so each 7-segment glyph's
        # hollow interior fills in. "0" has two vertical strokes per digit;
        # without closing each "0" projects as TWO runs, inflating the count.
        # Kernel width is 50% of the expected per-digit column — narrower
        # than the gap between adjacent digits so we don't merge them, but
        # wider than a glyph's internal gap.
        kernel_w = max(3, int(expected_digit_w * 0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        runs = _runs_for(closed)
        if runs is None:
            return None
    if len(runs) != expected_count:
        return None

    # Convert crop-pixel rects → full-image normalized sub-ROIs, padded by
    # `inset` on each side to keep some whitespace around each glyph.
    subs = []
    for (x0, x1) in runs:
        w_px = x1 - x0
        pad_x = max(1, int(w_px * inset))
        pad_y = max(1, int(ch * inset))
        px0 = max(0, x0 - pad_x)
        px1 = min(cw, x1 + pad_x)
        py0 = pad_y
        py1 = ch - pad_y
        subs.append([
            (ax + px0) / Wf,
            (ay + py0) / Hf,
            (px1 - px0) / Wf,
            (py1 - py0) / Hf,
        ])
    return subs


def build_digit_sub_rois(img_path, base_roi, expected_count, inset=0.10,
                         auto_detect=True, log=None, label=""):
    """Build normalized sub-ROIs for per-digit OCR.

    When `auto_detect` is True (default), tries to locate each digit's
    bounding box inside `base_roi` via vertical projection so small
    miscalibrations of the outer ROI are self-corrected. Falls back to
    equal subdivision when detection fails or finds the wrong number of
    runs. Returns exactly `expected_count` sub-ROIs either way.

    When `log` is provided, logs which path (auto-detect vs fallback) was
    taken — useful during calibration to tell whether the projection is
    actually helping or silently falling through to equal subdivision.
    """
    if not base_roi or expected_count <= 0:
        return []
    if auto_detect:
        detected = _auto_detect_digit_sub_rois(img_path, base_roi,
                                               expected_count, inset)
        if detected is not None:
            if log is not None:
                log.info("build_digit_sub_rois[%s]: auto-detect OK (%d digits)",
                         label, expected_count)
            return detected
        if log is not None:
            log.info("build_digit_sub_rois[%s]: auto-detect failed → "
                     "equal subdivision", label)
    return _equal_subdivide_roi(base_roi, expected_count, inset)


def _ocr_digit_upscaled(ocr, img_path, sub_roi, factor, save_to=None):
    """Crop `sub_roi` out of `img_path`, upscale it by `factor`× with cubic
    interpolation, write to a temp file, and run single-digit OCR on the
    upscaled crop. Optionally also save the upscaled crop to `save_to`
    for debugging. Returns the single-digit string (or "" on failure).

    Upscaling addresses Vision's minimum-feature-size behaviour: a tight
    7-segment digit crop can be below the recognizer's threshold, so
    enlarging 2–4× with smooth interpolation gives Vision a bigger glyph
    on similarly-blurred edges, which it handles far better than the
    pixel-hinted original.

    When digit mode (single-character) returns nothing, a line-mode
    fallback reruns OCR on the same upscaled crop via Vision's line
    recognizer (which uses both .fast AND .accurate and is more tolerant
    of isolated glyphs) and takes the first digit of the result. This
    catches cases where Vision refuses to emit anything for a lone
    character in digit mode but happily reads it in line mode.
    """
    img = cv2.imread(img_path)
    if img is None:
        return ""
    H, W = img.shape[:2]
    x, y, w, h = norm_to_abs(sub_roi, W, H)
    if w <= 0 or h <= 0:
        return ""
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    up = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    temp_path = img_path + ".digit.jpg"
    cv2.imwrite(temp_path, up)
    if save_to is not None:
        ensure_dir(os.path.dirname(save_to))
        cv2.imwrite(save_to, up)
    result = ocr._run(temp_path, [0.0, 0.0, 1.0, 1.0], "full")
    if not result:
        line = ocr.read_line(temp_path, [0.0, 0.0, 1.0, 1.0]) or ""
        digits = "".join(c for c in line if c.isdigit())
        if digits:
            result = digits[0]
    return result


def read_region_per_digit(ocr, img_path, base_roi, digit_count, inset=0.10,
                          auto_detect=True, upscale=1,
                          crops_dir=None, crops_prefix=None, log=None):
    """OCR a line region digit-by-digit and return the concatenated string.

    Builds sub-ROIs (auto-detected when possible, else equal subdivision),
    calls the single-digit OCR mode on each, and joins the outputs. More
    robust than line OCR for LCDs where Vision's line recognizer drops
    narrow glyphs or flips rotation-ambiguous small digits — every sub-ROI
    is a one-character classification with no line-segmentation involved.

    `upscale` (default 1) scales each sub-ROI crop by that factor before
    OCR — useful when Vision's text detector ignores crops below its
    minimum-feature-size threshold.

    `crops_dir` + `crops_prefix`: when set, write each per-digit input
    crop to `{crops_dir}/{crops_prefix}_d{i}.jpg` for visual debugging.
    `log`: when set, info-log each call's sub-ROI coords and OCR result
    so the text log can be cross-referenced with the saved crops.

    Returns "" (not None) if the ROI is empty or digit_count <= 0, so the
    downstream concat / regex match treats it as a failed read rather than
    a crash. Each digit that OCR couldn't read contributes an empty string
    to the join, which naturally shortens the output and fails the strict
    regex on the total/flow parsers.
    """
    if not base_roi or digit_count <= 0:
        return ""
    subs = build_digit_sub_rois(img_path, base_roi, digit_count, inset,
                                auto_detect, log=log, label=crops_prefix or "")
    digits = []
    for i, sub in enumerate(subs):
        save_to = None
        if crops_dir is not None and crops_prefix is not None:
            save_to = os.path.join(crops_dir, f"{crops_prefix}_d{i}.jpg")
        if upscale > 1:
            digit = _ocr_digit_upscaled(ocr, img_path, sub, upscale,
                                        save_to=save_to)
        else:
            digit = ocr._run(img_path, sub, "full") or ""
            if not digit:
                # Line-mode fallback: Vision's line recognizer sometimes
                # finds glyphs the single-digit recognizer refuses to emit.
                line = ocr.read_line(img_path, sub) or ""
                digits_only = "".join(c for c in line if c.isdigit())
                if digits_only:
                    digit = digits_only[0]
            # Save the raw (un-upscaled) crop when requested, so crops on
            # disk always reflect what Vision saw regardless of upscale.
            if save_to is not None:
                img = cv2.imread(img_path)
                if img is not None:
                    H, W = img.shape[:2]
                    x, y, w, h = norm_to_abs(sub, W, H)
                    if w > 0 and h > 0:
                        crop = img[y:y + h, x:x + w]
                        if crop.size > 0:
                            ensure_dir(os.path.dirname(save_to))
                            cv2.imwrite(save_to, crop)
        if log is not None:
            log.info("per-digit[%s d%d] roi=[%.4f,%.4f,%.4f,%.4f] → %r",
                     crops_prefix or "?", i, sub[0], sub[1], sub[2], sub[3], digit)
        digits.append(digit or "")
    return "".join(digits)


def apply_ocr_preprocess(src_path, dst_path, mode):
    """Apply optional contrast/sharpening preprocessing to an image before
    it's fed to Apple Vision. Returns True if a new file was written at
    dst_path, False if the mode is unrecognized or the source couldn't be
    loaded (caller should fall back to the original image in that case).

    Supported modes:
      - "clahe": Contrast Limited Adaptive Histogram Equalization (grayscale).
        Locally stretches contrast without blowing out highlights. Good for
        7-segment LCDs where a narrow "1" glyph can be lower contrast than
        its taller-stroke neighbours and get dropped by Vision's recognizer.
    """
    if mode == "clahe":
        img = cv2.imread(src_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        out = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(dst_path, out)
        return True
    return False


def capture_frame(cfg, session, log):
    """Fetch + decode one JPEG from the ESP32 camera.

    Returns the decoded BGR ndarray on success, or None on HTTP/decode failure.
    Callers own backoff policy (mechanical retries on the next loop iteration;
    run_digital_cycle retries within the same cycle).
    """
    try:
        url = cfg.esp32_base_url.rstrip("/") + "/capture_with_flashlight"
        r = session.get(url, timeout=cfg.image_timeout)
        r.raise_for_status()
        with open(cfg.image_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        log.warning(f"Capture failed: {e}")
        return None
    raw = cv2.imdecode(np.fromfile(cfg.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        log.warning("Decode failed")
        return None
    return raw


def run_digital_cycle(cfg, ocr, aligner, session, prev_total, log,
                      sleep=time.sleep, on_attempt=None):
    """Capture + OCR one digital-meter reading, retrying on wrong-view frames.

    The LCD auto-cycles between a numeric view and two diagnostic views. A
    single capture has roughly a 53% chance of hitting the numeric view, so we
    take up to (cfg.digital_max_retries + 1) captures per cycle, spaced by
    cfg.digital_retry_delay_sec (which must exceed the longest diag-view dwell
    so successive captures don't phase-lock onto the same wrong view).

    Always returns a result dict, never None. The dict has the shape:
        {"success": bool,
         "reason":  "ok" | "wrong_view" | "parse_failed" | "validate_failed" | "no_frame",
         "frame":   ndarray | None,           # None only when every capture failed
         "aligned_ok": bool,
         "raw_total": str | None,
         "raw_flow":  str | None,
         "total":     float | None,
         "flow_m3h":  float | None}
    On failure the dict mirrors the LAST attempt's state so the caller can
    still draw a debug overlay (showing the ROI box + raw OCR + reject reason)
    — previously the function returned None on failure and the overlay path
    was skipped entirely, which left users without visual diagnostics.

    `on_attempt` is an optional callable invoked after each attempt with the
    current state dict (same shape as the final return, but mutated per
    attempt). The main loop uses it to write the debug overlay every
    attempt rather than once per cycle — so users diagnosing wrong-view /
    parse-failed cycles see the overlay JPEG refreshed every ~retry_delay
    seconds instead of every ~interval seconds. Callback exceptions are
    logged and swallowed to keep the cycle running.

    `sleep` is injected so tests can run this loop without real sleeps.
    """
    def _notify(state):
        if on_attempt is None:
            return
        try:
            on_attempt(dict(state))
        except Exception as e:
            log.warning("on_attempt callback failed: %s", e)

    attempts = cfg.digital_max_retries + 1
    last = {
        "success": False,
        "reason": "no_frame",
        "frame": None,
        "aligned_ok": False,
        "raw_total": None,
        "raw_total_int": None,
        "raw_total_frac": None,
        "raw_flow": None,
        "total": None,
        "flow_m3h": None,
    }
    for attempt in range(attempts):
        frame = capture_frame(cfg, session, log)
        if frame is None:
            # Capture failure is distinct from wrong view; fall back to the loop's
            # outer retry_backoff_sec rather than the digital retry spacing.
            if attempt < attempts - 1:
                sleep(cfg.retry_backoff_sec)
            continue

        if aligner is not None:
            aligned_img, _, aligned_ok = aligner.align(frame)
        else:
            aligned_img, aligned_ok = frame, False

        ocr_path = (
            os.path.expanduser("~/watermeter/aligned_last.jpg")
            if aligned_ok
            else cfg.image_path
        )

        # Optional contrast preprocessing before OCR. When enabled, writes a
        # preprocessed copy and redirects OCR calls to it. Fallthrough on
        # failure — better to OCR the raw frame than to skip the cycle.
        if cfg.digital_ocr_preprocess and cfg.digital_ocr_preprocess != "none":
            preproc_path = ocr_path + ".preproc.jpg"
            if apply_ocr_preprocess(ocr_path, preproc_path, cfg.digital_ocr_preprocess):
                ocr_path = preproc_path

        # Split-ROI mode: when both integer and fractional sub-ROIs are set,
        # OCR them separately and concatenate with "." before parsing. Needed
        # for LCDs whose fractional digits are rendered in a smaller font and
        # get dropped by Vision when included in a single wide ROI.
        #
        # Each region (int, frac, flow) is OCR'd in line mode by default, but
        # switches to per-digit mode when the corresponding digit_count is
        # configured. Per-digit is slower (N subprocess calls per region)
        # but immune to line-recognition failure modes on small 7-segment
        # glyphs — every sub-ROI is a one-character classification.
        crops_dir = None
        if cfg.digital_save_ocr_crops:
            crops_dir = os.path.join(cfg.debug_dir, "ocr_crops")
            ensure_dir(crops_dir)

        def _read_region(roi, digit_count, region_label):
            if not roi:
                return None
            if digit_count > 0:
                prefix = f"{region_label}_a{attempt + 1}"
                return read_region_per_digit(
                    ocr, ocr_path, roi, digit_count,
                    cfg.digital_per_digit_inset,
                    auto_detect=cfg.digital_per_digit_auto_detect,
                    upscale=max(1, cfg.digital_ocr_upscale_factor),
                    crops_dir=crops_dir, crops_prefix=prefix,
                    log=log,
                )
            line_result = ocr.read_line(ocr_path, roi)
            log.info("line[%s a%d] roi=[%.4f,%.4f,%.4f,%.4f] → %r",
                     region_label, attempt + 1,
                     roi[0], roi[1], roi[2], roi[3], line_result)
            return line_result

        raw_total_int = None
        raw_total_frac = None
        if cfg.digital_total_int_roi and cfg.digital_total_frac_roi:
            raw_total_int = _read_region(cfg.digital_total_int_roi,
                                         cfg.digital_total_int_digit_count,
                                         "total_int") or ""
            raw_total_frac = _read_region(cfg.digital_total_frac_roi,
                                          cfg.digital_total_frac_digit_count,
                                          "total_frac") or ""
            raw_total = f"{raw_total_int}.{raw_total_frac}"
        else:
            raw_total = _read_region(cfg.digital_total_roi, 0, "total")
        # Flow line is optional. If the user configured only rois.digital.total
        # (typical when flash glare forces alignment to favour the total line),
        # skip the flow OCR call entirely and publish only the delta-based rate.
        raw_flow = _read_region(cfg.digital_flow_roi,
                                cfg.digital_flow_digit_count,
                                "flow") if cfg.digital_flow_roi else None

        last.update({
            "frame": aligned_img,
            "aligned_ok": aligned_ok,
            "raw_total": raw_total,
            "raw_total_int": raw_total_int,
            "raw_total_frac": raw_total_frac,
            "raw_flow": raw_flow,
            "total": None,
            "flow_m3h": None,
        })

        if not is_valid_digital_view(raw_total, raw_flow, cfg):
            log.info(
                "Wrong display view (attempt %d/%d): total=%r flow=%r",
                attempt + 1, attempts, raw_total, raw_flow,
            )
            last["reason"] = "wrong_view"
            _notify(last)
            if attempt < attempts - 1:
                sleep(cfg.digital_retry_delay_sec)
            continue

        total = parse_digital_total(raw_total, cfg.digital_total_regex)
        flow = (parse_digital_flow(raw_flow, cfg.digital_flow_regex)
                if cfg.digital_flow_roi else None)
        last["total"] = total
        last["flow_m3h"] = flow

        parse_ok = total is not None and (flow is not None or not cfg.digital_flow_roi)
        if not parse_ok:
            log.warning(
                "Parse failed (attempt %d/%d): raw_total=%r raw_flow=%r parsed=(%s, %s)",
                attempt + 1, attempts, raw_total, raw_flow, total, flow,
            )
            last["reason"] = "parse_failed"
            _notify(last)
            if attempt < attempts - 1:
                sleep(cfg.digital_retry_delay_sec)
            continue

        if not validate_digital_reading(total, flow, prev_total, cfg):
            log.warning(
                "Validate failed (attempt %d/%d): raw_total=%r raw_flow=%r "
                "parsed=(%s, %s) prev=%s",
                attempt + 1, attempts, raw_total, raw_flow, total, flow, prev_total,
            )
            last["reason"] = "validate_failed"
            _notify(last)
            if attempt < attempts - 1:
                sleep(cfg.digital_retry_delay_sec)
            continue

        result = {
            "success": True,
            "reason": "ok",
            "frame": aligned_img,
            "aligned_ok": aligned_ok,
            "total": total,
            "flow_m3h": flow,
            "raw_total": raw_total,
            "raw_total_int": raw_total_int,
            "raw_total_frac": raw_total_frac,
            "raw_flow": raw_flow,
        }
        _notify(result)
        return result

    return last

def _hhmm_to_min(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h*60 + m

def _in_window_local(now: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    cur = now.hour*60 + now.minute
    start = _hhmm_to_min(start_hhmm)
    end   = _hhmm_to_min(end_hhmm)
    return (start <= cur < end) if start < end else (cur >= start or cur < end)


def reset_state(cfg, new_total, log=None):
    """Overwrite state.json with a fresh starting total, preserving a timestamped
    backup of the previous state. Used after a physical meter swap where the new
    meter's counter reads below the old running total and would otherwise trip
    the monotonic / big_jump_guard protections.

    Returns (backup_path_or_None, new_state_dict). Raises ValueError on invalid input.
    """
    if not math.isfinite(new_total):
        raise ValueError(f"reset total must be finite, got {new_total!r}")
    if new_total < 0:
        raise ValueError(f"reset total must be >= 0, got {new_total!r}")

    ensure_dir(os.path.dirname(cfg.state_path))

    backup_path = None
    if os.path.exists(cfg.state_path):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = f"{cfg.state_path}.bak.{ts}"
        # Atomic enough for our purposes — state.json is ~200 bytes.
        with open(cfg.state_path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())

    new_state = {
        "total": float(new_total),
        "ts": time.time(),
        "meter_type": cfg.meter_type,
        "dial_histories": {},
    }
    save_json(cfg.state_path, new_state)

    msg = (
        f"Reset state.json: total={new_total} at ts={new_state['ts']}. "
        f"Previous state backed up to {backup_path}." if backup_path
        else f"Reset state.json: total={new_total} at ts={new_state['ts']}. "
             f"No previous state to back up."
    )
    if log is not None:
        log.info(msg)
    print(msg, file=sys.stderr)
    return backup_path, new_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log")
    parser.add_argument(
        "--reset-total", type=float, nargs="?", const=0.0, default=None,
        metavar="VALUE",
        help="Reset state.json total to VALUE (default 0.0) and exit. "
             "Use after physical meter replacement; pair with a restart of the LaunchAgent.",
    )
    a = parser.parse_args(); cfg = load_config(a.config)
    ensure_dir(os.path.dirname(cfg.log_path))
    logging.basicConfig(filename=a.log or cfg.log_path, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("watermeter")

    # Reset branch — runs before any network/MQTT/image processing setup.
    if a.reset_total is not None:
        try:
            reset_state(cfg, a.reset_total, log)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        sys.exit(0)

    log.info("Starting watermeter (alignment)")
    if cfg.meter_type == "digital":
        log.info(
            "Digital config: total_int_roi=%s total_frac_roi=%s total_roi=%s "
            "flow_roi=%s total_regex=%r flow_regex=%r "
            "total_int_digit_count=%d total_frac_digit_count=%d flow_digit_count=%d "
            "per_digit_inset=%.2f per_digit_auto_detect=%s "
            "ocr_upscale_factor=%d save_ocr_crops=%s ocr_preprocess=%r "
            "max_retries=%d retry_delay_sec=%.1f min_digits=%d",
            bool(cfg.digital_total_int_roi), bool(cfg.digital_total_frac_roi),
            bool(cfg.digital_total_roi), bool(cfg.digital_flow_roi),
            cfg.digital_total_regex.pattern, cfg.digital_flow_regex.pattern,
            cfg.digital_total_int_digit_count, cfg.digital_total_frac_digit_count,
            cfg.digital_flow_digit_count,
            cfg.digital_per_digit_inset, cfg.digital_per_digit_auto_detect,
            cfg.digital_ocr_upscale_factor, cfg.digital_save_ocr_crops,
            cfg.digital_ocr_preprocess,
            cfg.digital_max_retries, cfg.digital_retry_delay_sec,
            cfg.digital_min_digits,
        )
    state=read_json(cfg.state_path,{"total":None,"ts":None,"dial_histories":{}})
    prev_total=state.get("total"); prev_ts=state.get("ts")

    # State-schema guard: if the persisted state was produced by a different
    # meter_type, the running totals aren't comparable (mechanical builds totals
    # out of dial fractions; digital reads them directly). Discard prev_total so
    # the first new reading becomes the baseline — the service keeps running
    # rather than refusing to start, since we're under launchd and a crash loop
    # is worse than a one-cycle unknown rate.
    stored_meter_type = state.get("meter_type")
    if stored_meter_type is not None and stored_meter_type != cfg.meter_type and prev_total is not None:
        log.warning(
            "state.json meter_type=%r != config meter_type=%r; discarding prev_total=%r "
            "to avoid cross-contamination. The next reading becomes the new baseline.",
            stored_meter_type, cfg.meter_type, prev_total,
        )
        prev_total = None
        prev_ts = None

    # Restore dial reading histories from state
    dial_histories = state.get("dial_histories", {})
    for i, d in enumerate(cfg.dials):
        dial_key = f"dial_{i}"
        if dial_key in dial_histories:
            d.reading_history = dial_histories[dial_key]
            log.info(f"Restored {len(d.reading_history)} historical readings for {d.name}")
    
    # discovery() fires automatically from on_connect (see MqttClient._on_connect);
    # no need to invoke it explicitly here or in the loop.
    ocr=VisionOCR(cfg.ocr_bin); mqttc=MqttClient(cfg,log); mqttc.connect()
    aligner = Aligner(cfg.align, log) if cfg.align.enabled else None
    session=requests.Session()
    mode_prev = None

    while True:
        t0=time.time()
        
        # Determine target interval based on local time (needed for temporal validation)
        if cfg.quiet_hours_enabled and _in_window_local(datetime.now(), cfg.quiet_start, cfg.quiet_end):
            target_interval = cfg.quiet_interval_sec
            mode = "quiet"
        else:
            target_interval = cfg.interval_sec
            mode = "normal"

        if mode != mode_prev:
            log.info(f"Sampling mode -> {mode} (interval={target_interval}s)")
            mode_prev = mode
        
        mqttc.ensure_connection()

        if cfg.meter_type == "digital":
            # Per-attempt overlay callback — called inside run_digital_cycle
            # after every OCR attempt (success or failure). We redraw +
            # overwrite ~/watermeter/debug/overlay_latest.jpg every attempt
            # so the debug image reflects the most recent capture instead of
            # only the final attempt of the cycle. Also re-publishes the
            # overlay on MQTT so HA's camera entity updates in near-real-time.
            def _overlay_for_attempt(state):
                sub_img = state.get("frame")
                if sub_img is None:
                    return
                should_save = cfg.save_debug_overlays
                should_publish = getattr(cfg, "overlay_publish_mqtt", True)
                if not (should_save or should_publish):
                    return
                out_path = None
                if should_save:
                    ensure_dir(cfg.debug_dir)
                    out_path = (os.path.join(cfg.debug_dir, "overlay_latest.jpg")
                                if getattr(cfg, "debug_keep_latest_only", True)
                                else os.path.join(cfg.debug_dir,
                                                  f"overlay_{int(time.time())}.jpg"))
                Hi, Wi = sub_img.shape[:2]
                t_abs = norm_to_abs(cfg.digital_total_roi, Wi, Hi) if cfg.digital_total_roi else None
                f_abs = norm_to_abs(cfg.digital_flow_roi, Wi, Hi) if cfg.digital_flow_roi else None
                split = bool(cfg.digital_total_int_roi) and bool(cfg.digital_total_frac_roi)
                ti_abs = norm_to_abs(cfg.digital_total_int_roi, Wi, Hi) if split else None
                tf_abs = norm_to_abs(cfg.digital_total_frac_roi, Wi, Hi) if split else None
                ov_img = draw_overlays_digital(
                    sub_img, t_abs, f_abs,
                    state.get("raw_total"), state.get("raw_flow"),
                    state.get("total"), state.get("flow_m3h"),
                    state.get("aligned_ok", False), cfg,
                    reason=state.get("reason"),
                    out_path=out_path, log=log,
                    total_int_roi_abs=ti_abs, total_frac_roi_abs=tf_abs,
                    raw_total_int_str=state.get("raw_total_int"),
                    raw_total_frac_str=state.get("raw_total_frac"),
                )
                if should_publish:
                    topic = getattr(cfg, "overlay_camera_topic",
                                    f"{cfg.mqtt_main_topic}/debug/overlay")
                    jpeg_q = int(getattr(cfg, "overlay_jpeg_quality", 85))
                    ok, buf = cv2.imencode(".jpg", ov_img,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                    if ok:
                        mqttc.publish(topic, buf.tobytes(), retain=True)

            result = run_digital_cycle(cfg, ocr, aligner, session, prev_total, log,
                                       on_attempt=_overlay_for_attempt)

            img = result["frame"]
            aligned_ok = result["aligned_ok"]
            total = result["total"]
            flow_m3h = result["flow_m3h"]
            raw_total_str = result["raw_total"]
            raw_flow_str = result["raw_flow"]
            success = result["success"]
            reason = result["reason"]

            now = time.time()
            publish_total = None

            if success:
                # Guards mirror the mechanical clamp so HA always sees a monotonic total.
                publish_total = total
                if prev_total is not None:
                    if total + cfg.monotonic_epsilon < prev_total:
                        log.warning(
                            f"CLAMP: negative step detected "
                            f"(total={total:.6f} < prev={prev_total:.6f} - eps). Holding previous."
                        )
                        publish_total = prev_total
                    elif total - prev_total > cfg.big_jump_guard:
                        log.warning(
                            f"CLAMP: big jump detected "
                            f"(Δ={total - prev_total:.6f} > guard={cfg.big_jump_guard}). Holding previous."
                        )
                        publish_total = prev_total

                rate = 0.0
                if prev_total is not None and prev_ts is not None and now > prev_ts:
                    dv = max(0.0, publish_total - prev_total)
                    dt = (now - prev_ts) / 60.0
                    rate = dv / dt if dt > 1e-6 else 0.0
                rate_lpm = rate * 1000.0

                base = cfg.mqtt_main_topic
                mqttc.publish(f"{base}/main/value", f"{publish_total:.6f}", retain=True)
                mqttc.publish(f"{base}/main/rate", f"{rate:.6f}", retain=False)
                mqttc.publish(f"{base}/main/rate_lpm", f"{rate_lpm:.3f}", retain=False)
                if flow_m3h is not None:
                    mqttc.publish(f"{base}/main/flow_m3h", f"{flow_m3h:.3f}", retain=False)

                prev_total = publish_total
                prev_ts = now
                save_json(cfg.state_path, {
                    "total": prev_total,
                    "ts": prev_ts,
                    "meter_type": cfg.meter_type,
                    "dial_histories": {},
                })
            else:
                log.warning("Giving up this cycle (reason=%s) — holding previous total", reason)

            # Overlay drawing moved into the _overlay_for_attempt callback so
            # each retry attempt refreshes overlay_latest.jpg + MQTT camera.

            elapsed = time.time() - t0
            time.sleep(max(0.0, target_interval - elapsed))
            continue

        raw = capture_frame(cfg, session, log)
        if raw is None:
            time.sleep(cfg.retry_backoff_sec); continue

        work=raw; aligned_ok=False
        if aligner is not None:
            work, M, aligned_ok = aligner.align(raw)

        img=work; H,W=img.shape[:2]

        # Dials -> digitized fraction (for PUBLISH) and smooth progress (for ROLLING)
        dial_vals = []
        dial_confidences = []
        center_offsets = []
        dials_abs  = []
        
        for i, d in enumerate(cfg.dials):
            # Use adjusted ROI if available and auto-centering is enabled
            if cfg.auto_center_dials and d.adjusted_roi is not None:
                roi_to_use = d.adjusted_roi
            else:
                roi_to_use = d.roi
            
            ax, ay, aw, ah = norm_to_abs(roi_to_use, W, H)
            dials_abs.append((ax, ay, aw, ah))
            sub = img[ay:ay+ah, ax:ax+aw].copy()
            
            # Get previous reading and history for temporal validation
            prev_reading = d.reading_history[-1] if d.reading_history else None
            history = d.reading_history[-10:] if d.reading_history else []
            
            # Enhanced dial reading with confidence and history
            reading, confidence, center_offset = read_dial(
                sub, 
                zero_angle_deg=float(d.zero_angle_deg), 
                rotation=d.rotation,
                prev_reading=prev_reading,
                history=history
            )
            
            dial_vals.append(reading)
            dial_confidences.append(confidence)
            center_offsets.append(center_offset)
            
            # Update dial tracking data. Gate on confidence so that zero-confidence
            # fallback readings (blank image, total detection failure) don't pollute the
            # history that validate_reading_with_history and predict_expected_reading rely on.
            if confidence > 0.2:
                d.reading_history.append(reading)
                if len(d.reading_history) > 20:  # Keep last 20 readings for trend analysis
                    d.reading_history.pop(0)
            
            # Update average confidence
            if d.avg_confidence == 0.0:
                d.avg_confidence = confidence
            else:
                d.avg_confidence = 0.9 * d.avg_confidence + 0.1 * confidence
            
            # Update center offset with smoothing
            d.center_offset = (
                d.center_offset[0] * 0.7 + center_offset[0] * 0.3,
                d.center_offset[1] * 0.7 + center_offset[1] * 0.3
            )
            
            # Auto-adjust ROI if enabled and we have good confidence
            if cfg.auto_center_dials and confidence > 0.5:
                d.adjusted_roi = adjust_dial_roi(
                    d.roi, 
                    d.center_offset, 
                    W, H, 
                    cfg.center_smoothing_alpha
                )
            
            # Log warnings for low confidence
            if confidence < cfg.min_confidence_threshold:
                log.warning(f"Low confidence ({confidence:.2f}) for dial {d.name} - reading may be inaccurate")
            
            logging.debug(f"DIAL {d.name}: reading={reading:.3f} conf={confidence:.2f} offset={center_offset}")

        # Integer digits from each dial (0..9)
        dial_digits = [int(v) % 10 for v in dial_vals]

        # Fraction to PUBLISH (no double counting)
        frac_pub = sum(d * cfg.dials[i].factor for i, d in enumerate(dial_digits))

        # Smooth progress for integer rolling (0..1) = tenths dial only (robust)
        frac_prog = max(0.0, min(0.9999, (dial_vals[0] % 10)/10.0 + 0.02*((dial_vals[1] % 10)/10.0)))

        # Per-digit OCR
        digit_rois = build_digit_rois(cfg, W, H)
        ocr_img_path = cfg.image_path if not aligned_ok else os.path.expanduser("~/watermeter/aligned_last.jpg")
        obs=[ocr.full_top_bottom(ocr_img_path, r) for r in digit_rois]
        for i, (full, top, bot) in enumerate(obs):
            logging.debug(f"DIGIT{i}: full={full} top={top} bottom={bot}")

        prev_int_str = (f"{int(prev_total):0{cfg.digits_count}d}" if prev_total is not None else None)
        resolved_str, per_digits = compose_integer(obs, frac_prog, cfg.rolling_threshold_up, cfg.rolling_threshold_down, prev_int_str)

        # Raw OCR values for overlay (show what was actually read, not the resolved values)
        raw_ocr_digits = [full for (full, top, bot) in obs]

        try:
            integer = int(resolved_str)
        except (TypeError, ValueError):
            integer = int(prev_int_str) if prev_int_str else 0
        total = float(integer) + float(frac_pub)

        if prev_total is not None:
            diff_candidate = total - prev_total
            if abs(diff_candidate) > cfg.big_jump_guard:
                fallback_total = estimate_total_from_dials(prev_total, frac_pub, cfg)
                if fallback_total is not None and abs(fallback_total - prev_total) <= cfg.big_jump_guard:
                    log.warning(
                        "OCR integer jump (Δ=%.3f) exceeds guard; trusting dial fractions",
                        diff_candidate,
                    )
                    total = fallback_total

        publish_total=total; now=time.time()
        if prev_total is not None:
            if total + cfg.monotonic_epsilon < prev_total:
                log.warning(f"CLAMP: negative step detected "
                            f"(total={total:.6f} < prev={prev_total:.6f} - eps). Holding previous.")
                publish_total = prev_total
            elif total - prev_total > cfg.big_jump_guard:
                log.warning(f"CLAMP: big jump detected "
                            f"(Δ={total - prev_total:.6f} > guard={cfg.big_jump_guard}). Holding previous.")
                publish_total = prev_total

        rate=0.0
        if prev_total is not None and prev_ts is not None and now>prev_ts:
            dv=max(0.0, publish_total - prev_total); dt=(now-prev_ts)/60.0; rate=dv/dt if dt>1e-6 else 0.0

        base=cfg.mqtt_main_topic
        # Discovery payloads are retained on the broker; republishing every cycle
        # is pure waste. Initial discovery happens once at startup; if we
        # reconnect, ensure_connection will bring us back and the broker will
        # replay retained config to late subscribers.
        mqttc.publish(f"{base}/main/value", f"{publish_total:.6f}", retain=True)
        mqttc.publish(f"{base}/main/rate", f"{rate:.6f}", retain=False)
        rate_lpm = rate * 1000.0
        mqttc.publish(f"{base}/main/rate_lpm", f"{rate_lpm:.3f}", retain=False)

        should_save_overlay = cfg.save_debug_overlays
        should_publish_overlay = getattr(cfg, "overlay_publish_mqtt", True)
        if should_save_overlay or should_publish_overlay:
            out = None
            if should_save_overlay:
                ensure_dir(cfg.debug_dir)
                out = os.path.join(cfg.debug_dir, "overlay_latest.jpg") \
                    if getattr(cfg, "debug_keep_latest_only", True) \
                    else os.path.join(cfg.debug_dir, f"overlay_{int(now)}.jpg")

            digits_abs = [norm_to_abs(r, W, H) for r in digit_rois]
            ov_img = draw_overlays(
                img,
                digits_abs,
                raw_ocr_digits,
                dials_abs,
                dial_vals,
                dial_confidences,
                out,
                aligned_ok,
                cfg,
                center_offsets,
                resolved_digits=per_digits,
            )

            if should_publish_overlay:
                # Publish overlay to the MQTT camera (raw JPEG bytes)
                topic = getattr(cfg, "overlay_camera_topic", f"{cfg.mqtt_main_topic}/debug/overlay")
                jpeg_q = int(getattr(cfg, "overlay_jpeg_quality", 85))
                ok, buf = cv2.imencode(".jpg", ov_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                if ok:
                    mqttc.publish(topic, buf.tobytes(), retain=True)

        prev_total=publish_total; prev_ts=now

        # Save state including dial histories for persistence across restarts
        dial_histories = {f"dial_{i}": d.reading_history for i, d in enumerate(cfg.dials)}
        save_json(cfg.state_path, {
            "total": prev_total,
            "ts": prev_ts,
            "meter_type": cfg.meter_type,
            "dial_histories": dial_histories
        })

        elapsed = time.time() - t0
        time.sleep(max(0.0, target_interval - elapsed))

if __name__=="__main__": main()
