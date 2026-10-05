# Simple Motion Detection System using Python and OpenCV

> **College Hands-On 2 Project (Project 2)**  
> Developed for academic demonstration, practical evaluation, and viva-voce examination.

---

## 1. Project Title & Objective

**Project Title**: Simple Motion Detection System using Python and OpenCV  
**Objective**: To develop a basic computer vision application that captures live video from a webcam (or simulation stream) and detects whether there is movement in front of the camera in real time.

---

## 2. Features

1. **Webcam Video Capture**: Captures video frames using OpenCV `cv2.VideoCapture(0)`.
2. **Video Frame Reading**: Continuously reads and processes individual video stream frames.
3. **Grayscale Conversion**: Converts each color BGR frame to Grayscale via `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`.
4. **Consecutive Frame Differencing**: Employs Gaussian blurring (`cv2.GaussianBlur`) and adaptive running background subtraction (`cv2.absdiff`, `cv2.accumulateWeighted`) to isolate motion differences from stationary background.
5. **Motion Detection**: Detects significant movements by thresholding and contour area filtering.
6. **Bounding Box Visualization**: Draws green bounding rectangles (`cv2.rectangle`) around detected moving objects.
7. **Real-Time Status Display**: Displays `MOTION DETECTED` (in Red) or `NO MOTION` (in Green) overlaid on the video and printed to console.
8. **Live Date and Time**: Displays real-time timestamp on the camera HUD overlay and console.
9. **JSON Data Logging**: Automatically logs motion events with timestamps, motion area, and object counts into `data/motion_data.json`.
10. **Pandas & Matplotlib Statistics**: Reads stored motion events into a Pandas `DataFrame` and generates Matplotlib graphs comparing event frequencies and motion area over time.
11. **Simulation Fallback**: If a webcam is unavailable or permissions are restricted, it automatically switches to an animated synthetic camera simulator so the project never crashes during evaluation.

---

## 3. Topics Covered & Academic Syllabus Mapping

| Unit | Topic | Implementation in Project |
|---|---|---|
| **Unit I** | OpenCV, image representation, pixels, image properties | Frame representation as NumPy arrays, pixel manipulation, grayscale conversion |
| **Unit II** | Video capture, frames, camera, video properties, motion detection | `cv2.VideoCapture`, frame rate handling, background differencing (`cv2.absdiff`), contour extraction (`cv2.findContours`), bounding boxes |
| **Unit III** | JSON, Python modules, file handling, exception handling | Built-in `json` module (`json.dump`, `json.load`), file logging, robust `try-finally` cleanup |
| **Unit IV** | NumPy, Pandas, Matplotlib | NumPy array processing, Pandas tabular DataFrame analysis, Matplotlib visualization graphs |

---

## 4. Project Structure

```text
Project 2/
│
├── main.py                    # Main Video Motion Detection & Menu application
├── app.py                     # Entry point launcher
├── test_motion_detector.py    # Automated test verification suite
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive Documentation & Viva guide
│
├── data/
│   └── motion_data.json       # JSON storage for logged motion events
│
├── output/
│   └── (Stores captured motion snapshots)
│
└── assets/
```

---

## 5. Installation & Setup

Install required packages:
```bash
pip install -r requirements.txt
```

---

## 6. How to Run the Project

Navigate into the `Project 2` folder:
```bash
cd "Project 2"
```

### Option A: Interactive Menu
```bash
python main.py
```
*(or `python app.py`)*

### Option B: Quick Flags
- Start Live Webcam directly: `python main.py --live`
- Start Simulation Mode directly: `python main.py --sim`
- View Analytics Graph directly: `python main.py --graph`

### Run Automated Tests:
```bash
python test_motion_detector.py
```

---

## 7. Controls in Live Camera Window

- **`q` or `ESC`**: Stop the camera and return to menu / exit.
- **`r`**: Re-calibrate and reset background reference frame.
- **`s`**: Save a high-resolution snapshot of the current frame to `output/`.

---

## 8. Sample Output

```text
================================ 

    MOTION DETECTION SYSTEM 

================================ 

Camera Status : ON (WEBCAM / SIMULATION) 
Date          : 05-10-2026 
Time          : 15:15:30 

Motion Status : MOTION DETECTED 

[ LIVE CAMERA WINDOW ] 
When there is no movement: 
Motion Status : NO MOTION 
```

---

## 9. Viva Voce Questions & Answers

1. **How does motion detection work in OpenCV?**  
   It calculates the absolute difference between a reference background frame and the current frame using `cv2.absdiff()`. A threshold is applied to convert pixel differences into a binary mask, and contours are extracted to locate movement.

2. **Why is Gaussian Blur applied before frame differencing?**  
   Webcam sensors have high-frequency noise and slight pixel fluctuations. `cv2.GaussianBlur()` smoothes out pixel vibrations so false motion triggers are eliminated.

3. **How are moving objects outlined?**  
   Using `cv2.findContours()` to find connected motion components, followed by `cv2.boundingRect()` to compute `(x, y, w, h)`, and `cv2.rectangle()` to render the bounding box.

4. **How are motion statistics plotted?**  
   Logged motion events stored in `data/motion_data.json` are loaded into a Pandas `DataFrame` and visualized with Matplotlib bar charts and area timelines.
