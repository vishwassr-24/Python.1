# Simple Image Analyzer using Python and OpenCV

> **HANDS ON-1: Project 1**  
> Simple Image Analyzer using Python and OpenCV

---

## 1. Project Title & Objective

**Project Title**: Image Analyzer using Python and OpenCV  
**Objective**: To develop a simple Python application that reads an image, displays it, shows its properties, performs basic image manipulation, and stores the image information in a file.

---

## 2. Features

1. **Select & Read Image**: Select and read images from disk using OpenCV (`JPG`, `PNG`, `BMP`, `TIFF`).
2. **Display Image**: Display the loaded image using OpenCV `cv2.imshow()`.
3. **Display Properties**:
   - `Width` (pixels)
   - `Height` (pixels)
   - `Channels` (3 for BGR, 1 for Grayscale)
   - `Data Type` (`uint8`)
4. **Crop Image**: Safe coordinate-based cropping using NumPy array slicing (`img[y1:y2, x1:x2]`).
5. **Resize Image**: Resizing using OpenCV `cv2.resize()` with interpolation.
6. **Convert to Grayscale**: Converts BGR image to grayscale via `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
7. **Save Image**: Exports processed images to disk (`output/`).
8. **JSON Data Storage**: Logs image details, operations, and timestamps to `data/image_data.json`.
9. **Pandas Display**: Reads the JSON file into a Pandas `DataFrame` and displays a clean table.
10. **Matplotlib Graph**: Generates a bar chart comparing image Width and Height with numerical labels.

---

## 3. Sample Output

```text
================================ 

       IMAGE ANALYZER 

================================ 

1. Read Image 
2. Display Image Properties 
3. Crop Image 
4. Resize Image 
5. Convert to Grayscale 
6. Save Image 
7. View Image Data 
8. Generate Graph 
9. Exit 

Enter your choice: 2 

Width     : 1280 
Height    : 720 
Channels  : 3 
Data Type : uint8 
```

---

## 4. Topics Covered & Academic Syllabus Mapping

| Unit | Topic | Implementation in Project 1 |
|---|---|---|
| **Unit I** | OpenCV, image representation, pixels, image properties | Reading (`cv2.imread`), displaying (`cv2.imshow`), shape, resize (`cv2.resize`), cropping, grayscale (`cv2.cvtColor`), saving (`cv2.imwrite`) |
| **Unit III** | JSON, Python modules, file handling, exception handling | `json.dump`, `json.load`, file paths, robust validation preventing crashes |
| **Unit IV** | NumPy, Pandas, Matplotlib | NumPy array properties (`shape`, `dtype`), Pandas `DataFrame` table, Matplotlib dimension bar chart |

---

## 5. How to Run

Navigate into `Project 1`:
```bash
cd "Project 1"
```

Run the interactive menu:
```bash
python main.py
```
*(or `python app.py`)*

To launch with Graphical Interface:
```bash
python gui.py
```

Run unit tests:
```bash
python test_analyzer.py
```
