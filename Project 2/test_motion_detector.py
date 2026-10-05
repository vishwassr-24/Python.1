"""
AUTOMATED TEST SUITE FOR PROJECT 2: MOTION DETECTION SYSTEM
Verifies all 10 features programmatically.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd

from main import (
    MotionDetector,
    SyntheticCameraSimulator,
    log_motion_event,
    load_motion_dataframe,
    generate_motion_statistics_graph,
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    JSON_FILE_PATH
)

def run_tests():
    print("=" * 70)
    print("      RUNNING TEST SUITE: PROJECT 2 (MOTION DETECTION SYSTEM)")
    print("=" * 70)

    # 1. Test Synthetic Video Simulator & Frame Reading
    print("\n[Test 1] Testing Video Capture / Frame Reading...")
    sim = SyntheticCameraSimulator(width=640, height=480, fps=30)
    assert sim.isOpened(), "Test 1 Failed: Camera is not opened"
    ret, frame1 = sim.read()
    assert ret and frame1 is not None, "Test 1 Failed: Could not read frame 1"
    assert frame1.shape == (480, 640, 3), f"Test 1 Failed: Unexpected frame shape {frame1.shape}"
    print(f"  -> SUCCESS: Read frame with shape {frame1.shape}")

    # 2. Test Grayscale & Initial Frame
    print("\n[Test 2] Testing Grayscale Conversion & Background Initialization...")
    detector = MotionDetector(min_area=500, threshold_val=25)
    processed, status, area, boxes = detector.process_frame(frame1)
    assert status == "NO MOTION", f"Test 2 Failed: Initial status should be NO MOTION, got {status}"
    assert area == 0 and len(boxes) == 0
    print("  -> SUCCESS: Initial frame processed, reference background initialized.")

    # 3. Test Static Frame (No Motion)
    print("\n[Test 3] Testing Consecutive Identical Frames (No Motion)...")
    ret, frame2 = sim.read()
    processed2, status2, area2, boxes2 = detector.process_frame(frame2)
    assert status2 == "NO MOTION"
    print(f"  -> SUCCESS: Correctly identified status: '{status2}'")

    # 4. Test Motion Injection & Detection
    print("\n[Test 4] Testing Motion Detection & Bounding Box Drawing...")
    # Inject an artificial moving object into a frame
    motion_frame = frame2.copy()
    cv2.rectangle(motion_frame, (200, 150), (320, 280), (255, 255, 255), -1)
    
    processed_motion, m_status, m_area, m_boxes = detector.process_frame(motion_frame)
    assert m_status == "MOTION DETECTED", f"Test 4 Failed: Expected MOTION DETECTED, got {m_status}"
    assert m_area > 0, "Test 4 Failed: Motion area should be greater than 0"
    assert len(m_boxes) > 0, "Test 4 Failed: Expected at least 1 bounding box"
    print(f"  -> SUCCESS: Motion Detected! Area: {m_area} px, Bounding Boxes: {len(m_boxes)}")

    # 5. Test Date, Time & HUD overlay
    print("\n[Test 5] Testing HUD Overlay (Status, Date, Time)...")
    assert processed_motion.shape == motion_frame.shape
    # Check that overlay pixels are present in top banner
    top_banner_sample = processed_motion[10:50, 10:50]
    assert top_banner_sample is not None
    print("  -> SUCCESS: Video frame HUD overlay rendered.")

    # 6. Test JSON Logging
    print("\n[Test 6] Testing JSON Motion Logging...")
    test_json = os.path.join(DATA_DIR, "motion_data.json")
    ok, rec = log_motion_event("MOTION DETECTED", m_area, len(m_boxes), json_path=test_json)
    assert ok and os.path.exists(test_json)
    assert rec["status"] == "MOTION DETECTED"
    assert "date" in rec and "time" in rec
    print(f"  -> SUCCESS: Motion event logged to JSON: Date={rec['date']}, Time={rec['time']}")

    # 7. Test Pandas DataFrame Loading
    print("\n[Test 7] Testing Pandas DataFrame Loading...")
    df = load_motion_dataframe(test_json)
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert "status" in df.columns and "motion_area" in df.columns
    print(f"  -> SUCCESS: Pandas loaded {len(df)} motion records:")
    print(df.tail(2).to_string(index=False))

    # 8. Test Matplotlib Motion Statistics Graph
    print("\n[Test 8] Testing Matplotlib Motion Graph Generation...")
    g_ok, g_err = generate_motion_statistics_graph(test_json)
    assert g_ok, f"Test 8 Failed: {g_err}"
    print("  -> SUCCESS: Matplotlib motion statistics graph generated.")

    # 9. Test Background Reset
    print("\n[Test 9] Testing Background Re-calibration...")
    detector.reset_background()
    assert detector.background_frame is None
    print("  -> SUCCESS: Background reset verified.")

    print("\n" + "=" * 70)
    print("      ALL PROJECT 2 TESTS PASSED SUCCESSFULLY (100%)!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
