"""
AUTOMATED TEST SUITE FOR PROJECT 1: IMAGE ANALYZER
Verifies all core functions programmatically without manual input.
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
    load_image,
    get_image_properties,
    crop_image,
    resize_image,
    convert_to_grayscale,
    save_image,
    store_image_data_to_json,
    load_pandas_dataframe,
    generate_dimension_graph,
    BASE_DIR,
    OUTPUT_DIR,
    ASSETS_DIR,
    JSON_FILE_PATH
)

def run_tests():
    print("=" * 65)
    print("      RUNNING TEST SUITE: PROJECT 1 (IMAGE ANALYZER)")
    print("=" * 65)

    sample = os.path.join(ASSETS_DIR, "sample.jpg")
    assert os.path.exists(sample), f"Missing {sample}"

    # 1. Load image
    print("\n[Test 1] Testing Image Read...")
    img, err = load_image(sample)
    assert err is None and img is not None
    print(f"  -> SUCCESS: Loaded image shape {img.shape}")

    # 2. Properties
    print("\n[Test 2] Testing Properties Display...")
    p = get_image_properties(img)
    assert p["width"] == 1280 and p["height"] == 720 and p["channels"] == 3 and p["data_type"] == "uint8"
    print(f"  -> SUCCESS: Width={p['width']}, Height={p['height']}, Channels={p['channels']}, Data Type={p['data_type']}")

    # 3. Crop
    print("\n[Test 3] Testing Cropping...")
    cropped, err = crop_image(img, 200, 150, 800, 550)
    assert err is None and cropped.shape[:2] == (400, 600)
    print(f"  -> SUCCESS: Cropped shape {cropped.shape}")

    # 4. Resize
    print("\n[Test 4] Testing Resizing...")
    resized, err = resize_image(img, 500, 500)
    assert err is None and resized.shape[:2] == (500, 500)
    print(f"  -> SUCCESS: Resized shape {resized.shape}")

    # 5. Grayscale
    print("\n[Test 5] Testing Grayscale...")
    gray, err = convert_to_grayscale(img)
    assert err is None and len(gray.shape) == 2
    p_gray = get_image_properties(gray)
    assert p_gray["channels"] == 1
    print("  -> SUCCESS: Grayscale converted, channels = 1")

    # 6. Save
    print("\n[Test 6] Testing Save Processed Image...")
    out_file = os.path.join(OUTPUT_DIR, "test_output.jpg")
    if os.path.exists(out_file):
        os.remove(out_file)
    ok, err = save_image(resized, out_file)
    assert ok and os.path.exists(out_file)
    print(f"  -> SUCCESS: Image saved to {out_file}")

    # 7. JSON & Pandas
    print("\n[Test 7] Testing JSON & Pandas...")
    store_image_data_to_json("test_output.jpg", 500, 500, 3, "uint8", "Test Save", "test_output.jpg")
    df = load_pandas_dataframe()
    assert isinstance(df, pd.DataFrame) and not df.empty
    print(f"  -> SUCCESS: Pandas loaded {len(df)} records:")
    print(df.tail(2).to_string(index=False))

    # 8. Matplotlib
    print("\n[Test 8] Testing Matplotlib Bar Chart...")
    ok, err = generate_dimension_graph(1280, 720, "sample.jpg")
    assert ok
    print("  -> SUCCESS: Matplotlib bar chart generated.")

    # 9. Error handling
    print("\n[Test 9] Testing Error Handling...")
    bad_img, _ = load_image("nonexistent.jpg")
    assert bad_img is None
    bad_crop, _ = crop_image(img, 500, 500, 100, 100)
    assert bad_crop is None
    bad_resize, _ = resize_image(img, -20, 50)
    assert bad_resize is None
    print("  -> SUCCESS: Error validation passed.")

    print("\n" + "=" * 65)
    print("      ALL PROJECT 1 TESTS PASSED SUCCESSFULLY (100%)!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
