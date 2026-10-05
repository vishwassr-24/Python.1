"""
================================================================================
PROJECT 2: Simple Motion Detection System using Python and OpenCV
COLLEGE PROJECT: Hands-On 2 Project

TOPICS COVERED:
  - Unit I  : OpenCV, image representation, pixels, image properties
  - Unit II : Video capture, frames, camera, video properties, motion detection
  - Unit III: JSON, Python modules, file handling, exception handling
  - Unit IV : NumPy, Pandas, Matplotlib
================================================================================
"""

import os
import sys
import time
import json
import datetime
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Base project paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
JSON_FILE_PATH = os.path.join(DATA_DIR, "motion_data.json")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ==============================================================================
# SECTION 1: MOTION DETECTION CORE (Unit I & Unit II)
# ==============================================================================

class MotionDetector:
    """
    OpenCV Motion Detector utilizing Frame Differencing and Contour Analysis.
    - Grayscale conversion (Unit I)
    - Gaussian Blurring to eliminate sensor noise (Unit I)
    - Consecutive frame differencing (Unit II)
    - Thresholding and Dilation (Unit II)
    - Contour extraction and bounding box drawing (Unit II)
    """
    def __init__(self, min_area=800, threshold_val=25, blur_size=(21, 21)):
        self.min_area = min_area
        self.threshold_val = threshold_val
        self.blur_size = blur_size
        self.background_frame = None

    def reset_background(self):
        """Reset the reference background frame to the next incoming frame."""
        self.background_frame = None

    def process_frame(self, frame):
        """
        Process an input video frame for motion detection.
        
        Returns:
            processed_frame: Frame with bounding boxes & HUD overlay
            motion_status  : 'MOTION DETECTED' or 'NO MOTION'
            motion_area    : Total pixel area of motion detected
            bounding_boxes : List of (x, y, w, h) boxes
        """
        if frame is None:
            return None, "NO MOTION", 0, []

        output_frame = frame.copy()
        h, w = frame.shape[:2]

        # 1. Convert frame to Grayscale (Unit I)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Apply Gaussian Blur to smooth noise (Unit I)
        blurred = cv2.GaussianBlur(gray, self.blur_size, 0)

        # 3. Establish / update background reference (Unit II)
        if self.background_frame is None:
            self.background_frame = blurred.copy().astype("float")
            return output_frame, "NO MOTION", 0, []

        # Accumulate weighted running average to adapt to lighting changes
        cv2.accumulateWeighted(blurred, self.background_frame, 0.05)
        background_model = cv2.convertScaleAbs(self.background_frame)

        # 4. Compare consecutive frames / background differencing (Unit II)
        frame_delta = cv2.absdiff(background_model, blurred)

        # 5. Thresholding & Dilation to isolate motion blobs (Unit II)
        thresh = cv2.threshold(frame_delta, self.threshold_val, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 6. Find Contours of moving regions (Unit II)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        total_motion_area = 0
        bounding_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            # Filter out tiny noise contours
            if area < self.min_area:
                continue

            motion_detected = True
            total_motion_area += int(area)
            (x, y, bw, bh) = cv2.boundingRect(c)
            bounding_boxes.append((x, y, bw, bh))

            # 7. Draw rectangle around the moving object (Unit II)
            cv2.rectangle(output_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        status_text = "MOTION DETECTED" if motion_detected else "NO MOTION"
        status_color = (0, 0, 255) if motion_detected else (0, 200, 0) # Red for Motion, Green for No Motion

        # 8. Overlay Date, Time and Status on Frame (Unit II)
        now = datetime.datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")

        # Top Information Banner
        cv2.rectangle(output_frame, (0, 0), (w, 65), (20, 20, 20), -1)
        
        cv2.putText(output_frame, f"STATUS: {status_text}", (15, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        
        cv2.putText(output_frame, f"Date: {date_str}   Time: {time_str}", (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Bottom Instructions Banner
        cv2.rectangle(output_frame, (0, h - 28), (w, h), (15, 15, 15), -1)
        cv2.putText(output_frame, "Press 'q' or 'ESC' to Stop  |  'r' to Reset Background  |  's' to Save Snapshot",
                    (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        return output_frame, status_text, total_motion_area, bounding_boxes


# ==============================================================================
# SECTION 2: SYNTHETIC VIDEO SIMULATOR (Fallback for Environments without Webcam)
# ==============================================================================

class SyntheticCameraSimulator:
    """
    Simulates a live webcam feed generating synthetic motion (moving shapes).
    Guarantees that the project runs and demonstrates 100% of features even on
    machines or automated runners where a physical camera is not connected.
    """
    def __init__(self, width=640, height=480, fps=25):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.is_opened = True

    def isOpened(self):
        return self.is_opened

    def read(self):
        self.frame_count += 1
        h, w = self.height, self.width
        
        # Base background scene (simulated room with wall and desk)
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # Wall texture
        frame[:int(h*0.75), :] = [60, 50, 45]
        # Floor
        frame[int(h*0.75):, :] = [30, 70, 90]
        # Desk
        cv2.rectangle(frame, (80, int(h*0.6)), (w-80, int(h*0.75)), (80, 110, 140), -1)
        # Window
        cv2.rectangle(frame, (60, 40), (180, 180), (180, 160, 130), -1)
        cv2.line(frame, (120, 40), (120, 180), (40, 40, 40), 2)
        cv2.line(frame, (60, 110), (180, 110), (40, 40, 40), 2)

        # Periodic motion simulation:
        # Frames 0-40: No motion
        # Frames 41-110: Moving object crosses the camera view
        cycle = self.frame_count % 130
        if 40 <= cycle <= 110:
            progress = (cycle - 40) / 70.0
            cx = int(80 + progress * (w - 180))
            cy = int(h * 0.55 + np.sin(progress * np.pi) * -40)
            
            # Moving object (person / box silhouette)
            cv2.circle(frame, (cx, cy - 35), 25, (120, 220, 255), -1)
            cv2.rectangle(frame, (cx - 25, cy - 10), (cx + 25, cy + 50), (100, 180, 240), -1)

        time.sleep(1.0 / self.fps)
        return True, frame

    def release(self):
        self.is_opened = False


def open_camera_source(cam_index=0, force_simulation=False):
    """
    Attempt to open the physical webcam at cam_index.
    If camera is not found or fails to open, safely fallback to SyntheticCameraSimulator.
    """
    if force_simulation:
        print("[Info] Using Synthetic Camera Simulator.")
        return SyntheticCameraSimulator(), "SIMULATION MODE"

    try:
        cap = cv2.VideoCapture(cam_index)
        if cap is not None and cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                return cap, "WEBCAM (Device 0)"
            cap.release()
    except Exception:
        pass

    print("[Notice] Physical webcam not available or busy. Launching Synthetic Camera Simulation mode.")
    return SyntheticCameraSimulator(), "SIMULATION MODE"


# ==============================================================================
# SECTION 3: JSON LOGGING & PANDAS / MATPLOTLIB ANALYTICS (Unit III & Unit IV)
# ==============================================================================

def log_motion_event(status, motion_area, num_objects, json_path=JSON_FILE_PATH):
    """
    Unit III: Record motion event in JSON file with date, time, status, and metrics.
    """
    now = datetime.datetime.now()
    record = {
        "date": now.strftime("%d-%m-%Y"),
        "time": now.strftime("%H:%M:%S"),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "status": str(status),
        "motion_area": int(motion_area),
        "num_objects": int(num_objects)
    }

    records = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    if isinstance(loaded, list):
                        records = loaded
                    elif isinstance(loaded, dict):
                        records = [loaded]
        except Exception:
            records = []

    records.append(record)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        return True, record
    except Exception as e:
        return False, str(e)


def load_motion_dataframe(json_path=JSON_FILE_PATH):
    """
    Unit IV: Read JSON motion log into a Pandas DataFrame.
    """
    if not os.path.exists(json_path):
        return pd.DataFrame(columns=["date", "time", "timestamp", "status", "motion_area", "num_objects"])

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return pd.DataFrame()
            data = json.loads(content)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"[Error] Failed to read JSON with Pandas: {e}")
        return pd.DataFrame()


def generate_motion_statistics_graph(json_path=JSON_FILE_PATH):
    """
    Unit IV: Generate a simple motion statistics graph using Pandas and Matplotlib.
    Displays:
      1. Motion Status Distribution (Bar Chart: Motion Detected vs No Motion)
      2. Motion Intensity / Area Over Time
    """
    df = load_motion_dataframe(json_path)

    if df.empty:
        print("[Warning] No motion data found to plot. Run motion detection first!")
        return False, "No data available."

    try:
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.canvas.manager.set_window_title("Motion Detection Statistics - Matplotlib")
        fig.patch.set_facecolor('#ffffff')

        # Chart 1: Motion Status Counts
        status_counts = df['status'].value_counts()
        labels = status_counts.index.tolist()
        counts = status_counts.values.tolist()
        
        colors = ['#ef4444' if 'DETECTED' in lbl else '#10b981' for lbl in labels]
        bars = ax1.bar(labels, counts, color=colors, width=0.45, edgecolor='#0f172a', linewidth=1.2)
        
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{int(h)} events",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#0f172a')

        ax1.set_title("Motion Events Summary", fontsize=12, fontweight='bold', color='#0f172a')
        ax1.set_ylabel("Event Count", fontsize=10, fontweight='bold')
        ax1.set_ylim(0, max(counts) * 1.25 if counts else 5)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # Chart 2: Motion Area / Intensity Timeline
        detected_df = df[df['status'] == 'MOTION DETECTED'].tail(15)
        if not detected_df.empty:
            times = detected_df['time'].tolist()
            areas = detected_df['motion_area'].tolist()
            ax2.plot(times, areas, marker='o', color='#2563eb', linewidth=2, markersize=6)
            ax2.fill_between(range(len(times)), areas, color='#3b82f6', alpha=0.2)
            ax2.set_title("Motion Area (Recent 15 Events)", fontsize=12, fontweight='bold', color='#0f172a')
            ax2.set_ylabel("Motion Area (px)", fontsize=10, fontweight='bold')
            ax2.set_xticks(range(len(times)))
            ax2.set_xticklabels(times, rotation=45, ha='right', fontsize=8)
            ax2.grid(True, linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, "No 'Motion Detected' events recorded yet",
                     ha='center', va='center', fontsize=11, color='#64748b')
            ax2.set_title("Motion Area Timeline", fontsize=12, fontweight='bold')

        plt.suptitle("Motion Detection System Analytics • Pandas & Matplotlib", fontsize=14, fontweight='bold', color='#0f172a')
        plt.tight_layout()
        plt.show()
        return True, None
    except Exception as e:
        return False, str(e)


# ==============================================================================
# SECTION 4: LIVE CAMERA RUNNER & INTERACTIVE MENU (Unit II)
# ==============================================================================

def run_live_motion_detector(force_sim=False):
    """
    Launch the live motion detection video window.
    Matches the required Sample Output:
    ================================ 
        MOTION DETECTION SYSTEM 
    ================================ 
    Camera Status : ON 
    Date          : DD-MM-YYYY 
    Time          : HH:MM:SS 
    Motion Status : MOTION DETECTED 
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    print("\n" + "=" * 32)
    print("\n    MOTION DETECTION SYSTEM\n")
    print("=" * 32 + "\n")
    
    cap, cam_type = open_camera_source(cam_index=0, force_simulation=force_sim)
    
    print(f"Camera Status : ON ({cam_type})")
    print(f"Date          : {date_str}")
    print(f"Time          : {time_str}")
    print("\n[ LIVE CAMERA WINDOW OPENING... ]")
    print("Instructions:")
    print("  - Press 'q' or 'ESC' on the video window to stop.")
    print("  - Press 'r' to re-calibrate / reset background reference.")
    print("  - Press 's' to save a motion snapshot into output/ folder.\n")

    detector = MotionDetector(min_area=1000, threshold_val=25)
    last_log_time = 0
    last_status = "NO MOTION"
    snapshot_count = 0

    window_name = "Motion Detection System - Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            processed_frame, status, area, boxes = detector.process_frame(frame)

            # Log events to JSON throttled to avoid file spam (every 1.5s or on state change)
            current_time = time.time()
            if status != last_status or (status == "MOTION DETECTED" and (current_time - last_log_time) > 1.5):
                log_motion_event(status, area, len(boxes))
                last_log_time = current_time
                if status != last_status:
                    print(f"Motion Status : {status}")
                last_status = status

            cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nStopping camera stream...")
                break
            elif key == ord('r'):
                detector.reset_background()
                print("[Info] Background reference frame reset.")
            elif key == ord('s'):
                snapshot_count += 1
                snap_path = os.path.join(OUTPUT_DIR, f"motion_snapshot_{snapshot_count}.jpg")
                cv2.imwrite(snap_path, processed_frame)
                print(f"[Snapshot] Saved to {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Camera Status : OFF]")


def print_menu():
    print("\n" + "=" * 32)
    print("\n    MOTION DETECTION SYSTEM\n")
    print("=" * 32 + "\n")
    print("1. Start Live Motion Detection (Webcam / Live Stream)")
    print("2. Run Motion Detection Simulation (Synthetic Feed)")
    print("3. View Motion Log Data (Pandas DataFrame)")
    print("4. Generate Motion Statistics Graph (Matplotlib)")
    print("5. Exit\n")


def run_menu():
    """Main interactive terminal loop for Project 2."""
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            run_live_motion_detector(force_sim=False)
        elif choice == "2":
            run_live_motion_detector(force_sim=True)
        elif choice == "3":
            print("\n--- MOTION LOG DATA (PANDAS DATAFRAME) ---")
            df = load_motion_dataframe()
            if df.empty:
                print("No motion records found in motion_data.json yet.")
            else:
                print("\n" + df.to_string(index=False))
        elif choice == "4":
            print("\nGenerating motion statistics graph via Matplotlib...")
            ok, err = generate_motion_statistics_graph()
            if not ok:
                print(f"[Notice] {err}")
        elif choice == "5":
            print("\nExiting Motion Detection System. Goodbye!")
            break
        else:
            print("\n[Invalid Choice] Please select a number between 1 and 5.")


def main():
    # If user ran with direct flag --live or --sim
    if len(sys.argv) > 1:
        flag = sys.argv[1].lower()
        if flag in ["--live", "-l"]:
            run_live_motion_detector(force_sim=False)
            return
        elif flag in ["--sim", "-s"]:
            run_live_motion_detector(force_sim=True)
            return
        elif flag in ["--graph", "-g"]:
            generate_motion_statistics_graph()
            return

    run_menu()


if __name__ == "__main__":
    main()
