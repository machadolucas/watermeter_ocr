
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water meter reader (Mac) with reference-image alignment.
See README for details.
"""
import os, sys, time, json, math, subprocess, logging, argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
import requests
import yaml
import cv2
import numpy as np
import paho.mqtt.client as mqtt

def load_yaml(path): return yaml.safe_load(open(path,"r"))
def save_json(path,obj):
    tmp=path+".tmp"; open(tmp,"w").write(json.dumps(obj,indent=2))
    os.replace(tmp,path)
def read_json(path,default=None):
    try: return json.load(open(path,"r"))
    except: return default
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def norm_to_abs(box,W,H):
    x,y,w,h=box; return (int(x*W),int(y*H),int(w*W),int(h*H))
def clip01(v): return max(0.0, min(1.0, v))

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
        except: return ""
    def full_top_bottom(self,img,roi):
        return self._run(img,roi,"full"), self._run(img,roi,"top"), self._run(img,roi,"bottom")

def detect_dial_markings(roi_img, cx, cy, radius):
    """
    Detect the 0 and 5 markings on the dial to refine center and validate rotation.
    Returns (rotation_offset_deg, marking_confidence) or (0.0, 0.0) if detection fails.
    
    The markings should be at -90° (top, 0) and +90° (bottom, 5) from the true center.
    """
    hh, ww = roi_img.shape[:2]
    
    # Sample regions where we expect 0 (top) and 5 (bottom) markings
    # These are typically near the outer edge of the dial
    sample_radius = radius * 0.85
    
    # Expected positions for 0 (top, -90°) and 5 (bottom, +90°)
    positions = [
        ("0", -90, cx, cy - sample_radius),  # top
        ("5", 90, cx, cy + sample_radius),    # bottom
    ]
    
    detected_angles = []
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    for digit, expected_angle, px, py in positions:
        # Sample a small region around expected position
        sample_size = int(radius * 0.15)
        x1 = max(0, int(px - sample_size))
        x2 = min(ww, int(px + sample_size))
        y1 = max(0, int(py - sample_size))
        y2 = min(hh, int(py + sample_size))
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
            
        sample = gray[y1:y2, x1:x2]
        
        # Use simple threshold to find dark text on light background
        _, thresh = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Look for significant features (text-like regions)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # If we found features in expected location, mark as detected
            detected_angles.append((digit, expected_angle))
    
    if len(detected_angles) >= 1:
        # We detected at least one marking - assume rotation is correct
        confidence = len(detected_angles) / 2.0  # 0.5 for one, 1.0 for both
        return (0.0, confidence)
    
    return (0.0, 0.0)


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


def predict_expected_reading(history, factor):
    """
    Predict expected reading based on history and flow rate.
    Useful for resolving ambiguous readings near transitions.
    
    Returns predicted reading or None if insufficient data.
    """
    if len(history) < 3:
        return None
    
    # Linear extrapolation from last few points
    recent = history[-5:]
    
    # Calculate average rate of change
    changes = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    avg_change = sum(changes) / len(changes)
    
    # Predict next value
    predicted = recent[-1] + avg_change
    
    # Keep in 0-10 range
    return predicted % 10.0


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
        predicted = predict_expected_reading(history, 1.0)
    
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
        # bias toward prediction
        if predicted is not None and conf < 0.6:
            # Check if predicted is closer to current than previous
            if abs(reading - predicted) < 1.5:
                # Blend current and predicted
                reading = 0.7 * reading + 0.3 * predicted
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

    # 3) Check if we just rolled over (low progress after high, OCR may fail during transition)
    # This handles the case where digit is cropped/transitioning but dials show rollover happened
    if prev_digit is not None and progress <= thr_dn:
        # Low progress means dials show values near 0, indicating a rollover just occurred
        # Trust the dial reading and increment the digit, even if OCR doesn't see it yet
        next_digit = (prev_digit + 1) % 10
        
        # If OCR managed to see the new digit, great! Use it
        if f == next_digit or t == next_digit or b == next_digit:
            return next_digit
        
        # If OCR still shows old digit, that's expected during transition
        if f == prev_digit or t == prev_digit or b == prev_digit:
            # But dials say rollover happened, so trust dials over OCR
            return next_digit
        
        # OCR shows something else entirely - might be misread, still trust dials
        # This is the critical case: OCR fails, but dial reading at ~0.001 proves rollover
        return next_digit

    # 4) If we have previous digit and we're in stable zone, use it
    if prev_digit is not None and in_middle:
        return prev_digit

    # 5) Otherwise try full, then bottom, then top, then prev
    for v in (f, b, t, prev_digit):
        if v is not None:
            return v

    return 0

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
        def _on_connect(client, userdata, flags, reason_code, properties=None):
            self.connected = True
            self.log.info(f"MQTT connected rc={reason_code}")

        def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
            self.connected = False
            self.log.warning(f"MQTT disconnected rc={reason_code}")

        self.client.on_connect = _on_connect
        self.client.on_disconnect = _on_disconnect
    def connect(self):
        try: self.client.connect(self.cfg.mqtt_host,self.cfg.mqtt_port,keepalive=30); self.client.loop_start()
        except Exception as e: self.log.error(f"MQTT connect failed: {e}")
    def publish(self,t,p,retain=False):
        if self.connected: self.client.publish(t,p,retain=retain)
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
        kpC,descC=self.orb.detectAndCompute(gray,self.mask)
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

def draw_overlays(img, digits_rois_abs, per_digit_vals, dials_abs, dial_vals, dial_confidences, 
                  out_path, aligned_ok, cfg, center_offsets=None):
    ov = img.copy()
    H, W = ov.shape[:2]

    # scale roughly with image height so it looks similar at other resolutions
    fs = cfg.overlay_font_scale * (H / 480.0)
    th = int(cfg.overlay_font_thickness)
    o_th = int(cfg.overlay_outline_thickness)
    box_th = int(cfg.overlay_line_thickness)

    color_digits = (0, 255, 0) if aligned_ok else (0, 0, 255)  # green or red

    # Digit boxes + labels
    for i, (x, y, w, h) in enumerate(digits_rois_abs):
        cv2.rectangle(ov, (x, y), (x + w, y + h), color_digits, box_th)
        if i < len(per_digit_vals) and per_digit_vals[i] is not None:
            _draw_label(ov, f"{per_digit_vals[i]}",
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
            _draw_label(ov, label_text,
                        (x, max(12, y - 4)), fs * 0.7, color_dials, th, o_th)

    cv2.imwrite(out_path, ov)
    return ov


def build_digit_rois(cfg,W,H):
    x,y,w,h=cfg.digits_window; dw=w/cfg.digits_count; rois=[]; inset=cfg.per_digit_inset
    for i in range(cfg.digits_count):
        sx=x+i*dw; sy=y; sw=dw; sh=h
        rois.append([sx+sw*inset, sy+sh*inset, sw*(1-2*inset), sh*(1-2*inset)])
    return rois

def _hhmm_to_min(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h*60 + m

def _in_window_local(now: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    cur = now.hour*60 + now.minute
    start = _hhmm_to_min(start_hhmm)
    end   = _hhmm_to_min(end_hhmm)
    return (start <= cur < end) if start < end else (cur >= start or cur < end)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--log")
    a=parser.parse_args(); cfg=load_config(a.config)
    ensure_dir(os.path.dirname(cfg.log_path))
    logging.basicConfig(filename=a.log or cfg.log_path, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log=logging.getLogger("watermeter")
    log.info("Starting watermeter (alignment)")
    state=read_json(cfg.state_path,{"total":None,"ts":None,"dial_histories":{}})
    prev_total=state.get("total"); prev_ts=state.get("ts")
    
    # Restore dial reading histories from state
    dial_histories = state.get("dial_histories", {})
    for i, d in enumerate(cfg.dials):
        dial_key = f"dial_{i}"
        if dial_key in dial_histories:
            d.reading_history = dial_histories[dial_key]
            log.info(f"Restored {len(d.reading_history)} historical readings for {d.name}")
    
    ocr=VisionOCR(cfg.ocr_bin); mqttc=MqttClient(cfg,log); mqttc.connect(); mqttc.discovery()
    aligner=AlignConfig() and Aligner(cfg.align,log) if cfg.align.enabled else None
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
        
        try:
            url=cfg.esp32_base_url.rstrip("/")+"/capture_with_flashlight"
            r=session.get(url,timeout=cfg.image_timeout); r.raise_for_status()
            open(cfg.image_path,"wb").write(r.content)
        except Exception as e:
            log.warning(f"Capture failed: {e}"); time.sleep(cfg.retry_backoff_sec); continue

        raw=cv2.imdecode(np.fromfile(cfg.image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if raw is None: log.warning("Decode failed"); time.sleep(cfg.retry_backoff_sec); continue

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
            
            # Update dial tracking data
            d.reading_history.append(reading)
            if len(d.reading_history) > 20:  # Keep last 20 readings for better trend analysis
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

        try:
            integer = int(resolved_str)
        except:
            integer = int(prev_int_str) if prev_int_str else 0
        total = float(integer) + float(frac_pub)

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
        mqttc.discovery()
        mqttc.publish(f"{base}/main/value", f"{publish_total:.6f}", retain=True)
        mqttc.publish(f"{base}/main/rate", f"{rate:.6f}", retain=False)
        rate_lpm = rate * 1000.0
        mqttc.publish(f"{base}/main/rate_lpm", f"{rate_lpm:.3f}", retain=False)

        if cfg.save_debug_overlays:
            ensure_dir(cfg.debug_dir)
            out = os.path.join(cfg.debug_dir, "overlay_latest.jpg") \
                if getattr(cfg, "debug_keep_latest_only", True) \
                else os.path.join(cfg.debug_dir, f"overlay_{int(now)}.jpg")
            digits_abs=[norm_to_abs(r,W,H) for r in digit_rois]
            ov_img = draw_overlays(img, digits_abs, per_digits, dials_abs, dial_vals, 
                                   dial_confidences, out, aligned_ok, cfg, center_offsets)
            # Publish overlay to the MQTT camera (raw JPEG bytes)
            topic  = getattr(cfg, "overlay_camera_topic", f"{cfg.mqtt_main_topic}/debug/overlay")
            jpeg_q = int(getattr(cfg, "overlay_jpeg_quality", 85))
            if getattr(cfg, "overlay_publish_mqtt", True):
                ok, buf = cv2.imencode(".jpg", ov_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                if ok:
                    mqttc.publish(topic, buf.tobytes(), retain=True)

        prev_total=publish_total; prev_ts=now
        
        # Save state including dial histories for persistence across restarts
        dial_histories = {f"dial_{i}": d.reading_history for i, d in enumerate(cfg.dials)}
        save_json(cfg.state_path, {
            "total": prev_total, 
            "ts": prev_ts,
            "dial_histories": dial_histories
        })

        elapsed = time.time() - t0
        time.sleep(max(0.0, target_interval - elapsed))

if __name__=="__main__": main()
