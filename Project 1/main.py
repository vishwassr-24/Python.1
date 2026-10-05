"""
================================================================================
PROJECT 1: Simple Image Analyzer using Python and OpenCV
COLLEGE PROJECT: Hands-On 1 Project

TOPICS COVERED:
  - Unit I  : OpenCV, image representation, pixels, image properties,
              crop, resize, grayscale conversion
  - Unit III: JSON, Python modules, file handling, exception handling
  - Unit IV : NumPy, Pandas, Matplotlib
================================================================================
"""

import os
import sys
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
JSON_FILE_PATH = os.path.join(DATA_DIR, "image_data.json")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ==============================================================================
# SECTION 1: OPENCV & NUMPY CORE PROCESSING FUNCTIONS (Unit I & Unit IV)
# ==============================================================================

def load_image(filepath):
    """
    Unit I: Select and read an image from disk using OpenCV.
    Supports JPG, JPEG, PNG, BMP, and TIFF formats.
    Handles Unicode paths on Windows via np.fromfile and cv2.imdecode.
    """
    if not filepath or not os.path.exists(filepath):
        return None, f"File does not exist: {filepath}"
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    if not filepath.lower().endswith(valid_extensions):
        return None, f"Unsupported file format. Please select one of: {', '.join(valid_extensions)}"

    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
        if img is None:
            return None, "OpenCV failed to decode the selected image file."

        # If 4-channel BGRA, convert to 3-channel BGR
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img, None
    except Exception as e:
        return None, f"Error reading image: {str(e)}"


def get_image_properties(img):
    """
    Unit I & Unit IV: Extract image properties using OpenCV and NumPy.
    Returns:
      Width     : int
      Height    : int
      Channels  : int (1 for grayscale, 3 for BGR)
      Data Type : str (e.g., uint8)
    """
    if img is None:
        return None
    
    shape = img.shape
    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2]) if len(shape) > 2 else 1
    data_type = str(img.dtype)
    
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "data_type": data_type
    }


def display_image_opencv(img, window_title="Image Preview"):
    """
    Unit I: Display image using OpenCV cv2.imshow.
    Shows notice and safely handles closing with key press.
    """
    if img is None:
        print("[Warning] No image to display.")
        return
    
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_title, img)
        print(f"[OpenCV] Displaying '{window_title}'. Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_title)
    except Exception as e:
        print(f"[Notice] Could not display window via cv2.imshow: {e}")


def crop_image(img, x1, y1, x2, y2):
    """
    Unit I: Crop image using NumPy array slicing: img[y1:y2, x1:x2].
    Includes rigorous coordinate validation to prevent crashes.
    """
    if img is None:
        return None, "No image loaded to crop."
    
    h, w = img.shape[:2]
    
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    except (ValueError, TypeError):
        return None, "Coordinates must be valid integers."
    
    if x1 < 0 or y1 < 0:
        return None, "Coordinates cannot be negative numbers."
    if x1 >= x2:
        return None, f"x1 ({x1}) must be strictly less than x2 ({x2})."
    if y1 >= y2:
        return None, f"y1 ({y1}) must be strictly less than y2 ({y2})."
    if x2 > w:
        return None, f"x2 ({x2}) exceeds image width ({w})."
    if y2 > h:
        return None, f"y2 ({y2}) exceeds image height ({h})."
    
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, "Cropped region is empty."
    
    return cropped, None


def resize_image(img, new_width, new_height):
    """
    Unit I: Resize image using cv2.resize with appropriate interpolation.
    """
    if img is None:
        return None, "No image loaded to resize."
    
    try:
        new_width = int(new_width)
        new_height = int(new_height)
    except (ValueError, TypeError):
        return None, "Width and Height must be valid positive integers."
    
    if new_width <= 0 or new_height <= 0:
        return None, "Invalid dimensions. Please enter positive numbers greater than zero."
    
    if new_width > 10000 or new_height > 10000:
        return None, "Requested dimensions are excessively large (max 10000px)."
    
    curr_h, curr_w = img.shape[:2]
    interp = cv2.INTER_AREA if (new_width < curr_w or new_height < curr_h) else cv2.INTER_LINEAR
    
    try:
        resized = cv2.resize(img, (new_width, new_height), interpolation=interp)
        return resized, None
    except Exception as e:
        return None, f"Error resizing image: {str(e)}"


def convert_to_grayscale(img):
    """
    Unit I: Convert image to Grayscale using cv2.cvtColor.
    """
    if img is None:
        return None, "No image loaded."
    
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return img, "Image is already in grayscale."
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, None
    except Exception as e:
        return None, f"Error converting to grayscale: {str(e)}"


def save_image(img, filepath):
    """
    Unit I & Unit III: Save processed image to disk.
    Supports JPG, PNG, BMP, TIFF formats.
    """
    if img is None:
        return False, "No image available to save."
    
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            ext = ".jpg"
            filepath += ext
            
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        if ext not in valid_exts:
            return False, f"Unsupported file extension '{ext}'. Use JPG, PNG, BMP, or TIFF."
        
        target_dir = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(target_dir, exist_ok=True)
        
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(filepath)
            return True, None
        else:
            return False, "OpenCV failed to encode image for saving."
    except Exception as e:
        return False, f"Error saving image: {str(e)}"


# ==============================================================================
# SECTION 2: JSON STORAGE & PANDAS INTEGRATION (Unit III & Unit IV)
# ==============================================================================

def store_image_data_to_json(filename, width, height, channels, data_type, operation="Read", output_filename="", json_path=JSON_FILE_PATH):
    """
    Unit III: Store image details in a structured JSON file.
    Appends the record to an array of records in image_data.json.
    """
    record = {
        "filename": os.path.basename(filename),
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "data_type": str(data_type),
        "operation": str(operation),
        "output_filename": os.path.basename(output_filename) if output_filename else os.path.basename(filename),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        except Exception as e:
            records = []
            
    records.append(record)
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        return True, record
    except Exception as e:
        return False, f"Failed to write JSON: {str(e)}"


def load_pandas_dataframe(json_path=JSON_FILE_PATH):
    """
    Unit IV: Read stored JSON information and convert it into a Pandas DataFrame.
    """
    if not os.path.exists(json_path):
        return pd.DataFrame(columns=["filename", "width", "height", "channels", "data_type", "operation", "output_filename", "timestamp"])
    
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
        print(f"Error reading JSON with Pandas: {e}")
        return pd.DataFrame()


# ==============================================================================
# SECTION 3: MATPLOTLIB GRAPH GENERATION (Unit IV)
# ==============================================================================

def generate_dimension_graph(width, height, filename="Current Image"):
    """
    Unit IV: Generate a simple bar chart comparing Width and Height using Matplotlib.
    Displays numerical values clearly above the bars.
    """
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.canvas.manager.set_window_title(f"Image Dimensions Graph - {os.path.basename(filename)}")
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        categories = ['Width', 'Height']
        values = [int(width), int(height)]
        bar_colors = ['#2563eb', '#10b981']
        
        bars = ax.bar(categories, values, color=bar_colors, width=0.42, edgecolor='#0f172a', linewidth=1.2)
        
        for bar in bars:
            val = bar.get_height()
            ax.annotate(f"{int(val)} px",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color='#0f172a')
        
        ax.set_title(f"Image Dimensions Comparison\nFile: {os.path.basename(filename)}", 
                     fontsize=13, fontweight='bold', pad=15, color='#0f172a')
        ax.set_ylabel("Dimension (Pixels)", fontsize=11, fontweight='bold', color='#334155')
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis='y', linestyle='--', alpha=0.6, color='#cbd5e1')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94a3b8')
        ax.spines['bottom'].set_color('#94a3b8')
        
        plt.tight_layout()
        plt.show()
        return True, None
    except Exception as e:
        return False, f"Failed to generate Matplotlib graph: {str(e)}"


# ==============================================================================
# SECTION 4: INTERACTIVE MENU-DRIVEN INTERFACE
# Exactly matches the required Sample Output from the College Project Brief!
# ==============================================================================

def print_menu():
    print("\n" + "=" * 32)
    print("\n       IMAGE ANALYZER\n")
    print("=" * 32 + "\n")
    print("1. Read Image")
    print("2. Display Image Properties")
    print("3. Crop Image")
    print("4. Resize Image")
    print("5. Convert to Grayscale")
    print("6. Save Image")
    print("7. View Image Data")
    print("8. Generate Graph")
    print("9. Exit\n")


def run_menu():
    """
    Main interactive loop for Project 1.
    """
    current_image = None
    original_image = None
    current_filename = "sample.jpg"
    
    # Pre-load sample image if present so user can immediately test option 2
    default_sample = os.path.join(ASSETS_DIR, "sample.jpg")
    if os.path.exists(default_sample):
        img, _ = load_image(default_sample)
        if img is not None:
            current_image = img.copy()
            original_image = img.copy()
            current_filename = "sample.jpg"

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            print("\n--- READ IMAGE ---")
            default_path = os.path.join(ASSETS_DIR, "sample.jpg")
            user_path = input(f"Enter image file path [Press Enter for '{default_path}']: ").strip()
            
            filepath = user_path if user_path else default_path
            img, err = load_image(filepath)
            
            if err:
                print(f"[Error] {err}")
            else:
                current_image = img.copy()
                original_image = img.copy()
                current_filename = os.path.basename(filepath)
                props = get_image_properties(current_image)
                
                # Store in JSON
                store_image_data_to_json(
                    filename=current_filename,
                    width=props["width"],
                    height=props["height"],
                    channels=props["channels"],
                    data_type=props["data_type"],
                    operation="Read Image",
                    output_filename=current_filename
                )
                
                print(f"[Success] Image '{current_filename}' loaded successfully!")
                
                # Prompt to display
                disp = input("Display loaded image in OpenCV window? (y/n) [default: y]: ").strip().lower()
                if disp != 'n':
                    display_image_opencv(current_image, f"Image: {current_filename}")

        elif choice == "2":
            if current_image is None:
                print("\n[Warning] No image loaded. Please select option 1 to read an image first.")
            else:
                props = get_image_properties(current_image)
                print()
                print(f"Width     : {props['width']}")
                print(f"Height    : {props['height']}")
                print(f"Channels  : {props['channels']}")
                print(f"Data Type : {props['data_type']}")

        elif choice == "3":
            if current_image is None:
                print("\n[Warning] No image loaded. Please select option 1 first.")
            else:
                h, w = current_image.shape[:2]
                print(f"\n--- CROP IMAGE (Current Dimensions: Width={w}, Height={h}) ---")
                try:
                    x1 = input(f"Enter x1 (0 to {w-1}) [default: 100]: ").strip()
                    x1 = int(x1) if x1 else 100
                    
                    y1 = input(f"Enter y1 (0 to {h-1}) [default: 100]: ").strip()
                    y1 = int(y1) if y1 else 100
                    
                    x2 = input(f"Enter x2 ({x1+1} to {w}) [default: {w-100}]: ").strip()
                    x2 = int(x2) if x2 else (w - 100)
                    
                    y2 = input(f"Enter y2 ({y1+1} to {h}) [default: {h-100}]: ").strip()
                    y2 = int(y2) if y2 else (h - 100)
                    
                    cropped, err = crop_image(current_image, x1, y1, x2, y2)
                    if err:
                        print(f"[Crop Error] {err}")
                    else:
                        current_image = cropped
                        ch, cw = current_image.shape[:2]
                        print(f"[Success] Cropped to: Width={cw}, Height={ch}")
                        display_image_opencv(current_image, "Cropped Image")
                except ValueError:
                    print("[Error] Invalid input. Please enter integer values.")

        elif choice == "4":
            if current_image is None:
                print("\n[Warning] No image loaded. Please select option 1 first.")
            else:
                h, w = current_image.shape[:2]
                print(f"\n--- RESIZE IMAGE (Current Size: {w} x {h}) ---")
                try:
                    nw = input("Enter new Width (px): ").strip()
                    nh = input("Enter new Height (px): ").strip()
                    
                    resized, err = resize_image(current_image, nw, nh)
                    if err:
                        print(f"[Resize Error] {err}")
                    else:
                        current_image = resized
                        print(f"[Success] Resized image to {current_image.shape[1]} x {current_image.shape[0]}")
                        display_image_opencv(current_image, "Resized Image")
                except ValueError:
                    print("[Error] Please enter positive integer dimensions.")

        elif choice == "5":
            if current_image is None:
                print("\n[Warning] No image loaded. Please select option 1 first.")
            else:
                gray, err = convert_to_grayscale(current_image)
                if err:
                    print(f"[Notice] {err}")
                else:
                    current_image = gray
                    print("[Success] Converted image to Grayscale. Number of channels is now 1.")
                    display_image_opencv(current_image, "Grayscale Image")

        elif choice == "6":
            if current_image is None:
                print("\n[Warning] No image loaded. Please select option 1 first.")
            else:
                default_out = os.path.join(OUTPUT_DIR, f"processed_{current_filename}")
                out_path = input(f"Enter destination path [default: '{default_out}']: ").strip()
                dest = out_path if out_path else default_out
                
                ok, err = save_image(current_image, dest)
                if ok:
                    props = get_image_properties(current_image)
                    store_image_data_to_json(
                        filename=current_filename,
                        width=props["width"],
                        height=props["height"],
                        channels=props["channels"],
                        data_type=props["data_type"],
                        operation="Saved Processed",
                        output_filename=os.path.basename(dest)
                    )
                    print(f"[Success] Image saved successfully to: {dest}")
                else:
                    print(f"[Save Error] {err}")

        elif choice == "7":
            print("\n--- VIEW IMAGE DATA (PANDAS DATAFRAME) ---")
            df = load_pandas_dataframe()
            if df.empty:
                print("No image data found in image_data.json yet.")
            else:
                print("\n" + df.to_string(index=False))

        elif choice == "8":
            if current_image is None:
                df = load_pandas_dataframe()
                if not df.empty:
                    last = df.iloc[-1]
                    w, h, fn = last["width"], last["height"], last["filename"]
                    print(f"\n[Info] Generating graph from last logged record: {fn} ({w} x {h})")
                    generate_dimension_graph(w, h, fn)
                else:
                    print("\n[Warning] No image loaded and no records found. Please read an image first.")
            else:
                props = get_image_properties(current_image)
                print(f"\n[Info] Generating graph for current image: {current_filename} ({props['width']} x {props['height']})")
                generate_dimension_graph(props["width"], props["height"], current_filename)

        elif choice == "9":
            print("\nExiting Image Analyzer. Thank you!")
            break
        else:
            print("\n[Invalid Choice] Please enter a number between 1 and 9.")


def main():
    # If user ran with --gui flag, launch Tkinter GUI
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--gui", "-g"]:
        from gui import launch_gui
        launch_gui()
    else:
        run_menu()


if __name__ == "__main__":
    main()
