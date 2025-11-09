#!/usr/bin/env python3
"""
Test script for dial detection improvements.
Tests the enhanced dial reading functions on sample dial images.
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import watermeter functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watermeter import detect_dial_center, detect_needle_by_color, detect_needle_by_lines, read_dial

def create_synthetic_dial(size=200, needle_angle=45, add_noise=False):
    """
    Create a synthetic dial image for testing.
    needle_angle: angle in degrees (0=right, 90=up, etc.)
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 240  # light gray background
    center = (size // 2, size // 2)
    radius = size // 3
    
    # Draw dial circle (black)
    cv2.circle(img, center, radius, (50, 50, 50), 2)
    
    # Draw dial markings (0-9)
    for i in range(10):
        angle = np.radians(i * 36 - 90)  # 0 at top
        x1 = int(center[0] + radius * 0.85 * np.cos(angle))
        y1 = int(center[1] + radius * 0.85 * np.sin(angle))
        x2 = int(center[0] + radius * 0.95 * np.cos(angle))
        y2 = int(center[1] + radius * 0.95 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # Draw red needle
    needle_rad = np.radians(needle_angle)
    needle_length = radius * 0.8
    end_x = int(center[0] + needle_length * np.cos(needle_rad))
    end_y = int(center[1] + needle_length * np.sin(needle_rad))
    cv2.line(img, center, (end_x, end_y), (0, 0, 200), 4)  # red in BGR
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    return img

def test_dial_detection():
    """Test dial detection with various configurations."""
    print("Testing Dial Detection Functions")
    print("=" * 50)
    
    test_cases = [
        ("Needle at 0° (right)", 0, 2.5),
        ("Needle at 90° (top)", 90, 0.0),
        ("Needle at 180° (left)", 180, 5.0),
        ("Needle at 270° (bottom)", 270, 7.5),
        ("Needle at 45°", 45, 1.25),
        ("Needle at 135°", 135, 3.75),
    ]
    
    for name, angle, expected_reading in test_cases:
        print(f"\nTest: {name}")
        img = create_synthetic_dial(size=200, needle_angle=angle)
        
        # Test center detection
        center_result = detect_dial_center(img)
        if center_result:
            cx, cy, radius, conf = center_result
            print(f"  Center: ({cx:.1f}, {cy:.1f}), Radius: {radius:.1f}, Conf: {conf:.2f}")
        else:
            print("  Center: Detection failed")
        
        # Test full dial reading
        reading, confidence, offset = read_dial(img, zero_angle_deg=-90, rotation="cw")
        error = abs(reading - expected_reading)
        
        print(f"  Expected reading: {expected_reading:.2f}")
        print(f"  Actual reading: {reading:.2f}")
        print(f"  Error: {error:.2f}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Center offset: ({offset[0]:.2f}, {offset[1]:.2f})")
        
        if error < 0.2:  # Allow small tolerance
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL - Large error!")
    
    print("\n" + "=" * 50)
    print("Testing complete!")

def test_with_noise():
    """Test robustness to noise."""
    print("\nTesting Robustness to Noise")
    print("=" * 50)
    
    angle = 45
    expected = 1.25
    
    results_clean = []
    results_noisy = []
    
    for trial in range(10):
        # Clean image
        img_clean = create_synthetic_dial(size=200, needle_angle=angle, add_noise=False)
        reading, conf, _ = read_dial(img_clean, zero_angle_deg=-90, rotation="cw")
        results_clean.append((reading, conf))
        
        # Noisy image
        img_noisy = create_synthetic_dial(size=200, needle_angle=angle, add_noise=True)
        reading, conf, _ = read_dial(img_noisy, zero_angle_deg=-90, rotation="cw")
        results_noisy.append((reading, conf))
    
    # Statistics
    clean_readings = [r[0] for r in results_clean]
    clean_confs = [r[1] for r in results_clean]
    noisy_readings = [r[0] for r in results_noisy]
    noisy_confs = [r[1] for r in results_noisy]
    
    print(f"\nClean images (n=10):")
    print(f"  Mean reading: {np.mean(clean_readings):.3f} (expected: {expected:.3f})")
    print(f"  Std dev: {np.std(clean_readings):.3f}")
    print(f"  Mean confidence: {np.mean(clean_confs):.3f}")
    
    print(f"\nNoisy images (n=10):")
    print(f"  Mean reading: {np.mean(noisy_readings):.3f} (expected: {expected:.3f})")
    print(f"  Std dev: {np.std(noisy_readings):.3f}")
    print(f"  Mean confidence: {np.mean(noisy_confs):.3f}")
    
    print("\n" + "=" * 50)

def test_off_center():
    """Test detection when dial is not centered in ROI."""
    print("\nTesting Off-Center Dial Detection")
    print("=" * 50)
    
    # Create a larger canvas and place dial off-center
    canvas = np.ones((300, 300, 3), dtype=np.uint8) * 240
    dial = create_synthetic_dial(size=150, needle_angle=45)
    
    offsets = [(0, 0), (30, 0), (0, 30), (-30, 0), (0, -30), (20, 20)]
    
    for dx, dy in offsets:
        test_canvas = canvas.copy()
        center_x = 150 + dx
        center_y = 150 + dy
        
        # Place dial
        y1 = center_y - 75
        y2 = center_y + 75
        x1 = center_x - 75
        x2 = center_x + 75
        
        if y1 >= 0 and y2 < 300 and x1 >= 0 and x2 < 300:
            test_canvas[y1:y2, x1:x2] = dial
            
            reading, conf, offset = read_dial(test_canvas, zero_angle_deg=-90, rotation="cw")
            
            print(f"\nOffset: ({dx:3d}, {dy:3d}) pixels")
            print(f"  Reading: {reading:.2f}")
            print(f"  Confidence: {conf:.2f}")
            print(f"  Detected offset: ({offset[0]:.1f}, {offset[1]:.1f})")
            print(f"  Expected offset: (~{dx:.1f}, ~{dy:.1f})")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    try:
        test_dial_detection()
        test_with_noise()
        test_off_center()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
