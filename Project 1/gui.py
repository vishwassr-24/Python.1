"""
GUI Interface for Project 1: Simple Image Analyzer
Provides full Tkinter light-themed window with real-time preview and properties.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

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
    ASSETS_DIR
)

class ImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IMAGE ANALYZER - Hands-On 1")
        self.root.geometry("1000x740")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f8fafc")

        self.original_image = None
        self.current_image = None
        self.image_filename = "No image selected"
        self.current_props = None
        self.tk_preview = None

        self._build_ui()

        # Auto-load sample image if available
        sample = os.path.join(ASSETS_DIR, "sample.jpg")
        if os.path.exists(sample):
            self.load_file(sample, show_success=False)

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#f8fafc", padx=16, pady=10)
        header.pack(fill="x")
        tk.Label(header, text="IMAGE ANALYZER", font=("Segoe UI", 18, "bold"), bg="#f8fafc", fg="#0f172a").pack(anchor="w")
        tk.Label(header, text="OpenCV Image Processing & Analysis • Hands-On 1", font=("Segoe UI", 10), bg="#f8fafc", fg="#64748b").pack(anchor="w")

        # Main Body
        body = tk.Frame(self.root, bg="#f8fafc", padx=16, pady=6)
        body.pack(fill="both", expand=True)

        # Left Canvas
        left = tk.Frame(body, bg="#ffffff", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        tk.Label(left, text="IMAGE PREVIEW", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#1e293b", padx=10, pady=8).pack(anchor="w")

        self.canvas = tk.Canvas(left, bg="#f1f5f9", bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda e: self._render())

        # Right Control Panel
        right = tk.Frame(body, bg="#f8fafc", width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # 1. Select
        f1 = tk.Frame(right, bg="#ffffff", bd=1, relief="solid", padx=12, pady=10)
        f1.pack(fill="x", pady=(0, 8))
        tk.Button(f1, text="📂  Select Image", font=("Segoe UI", 10, "bold"), bg="#2563eb", fg="#ffffff", relief="flat", cursor="hand2", command=self.select_file).pack(fill="x", pady=(0, 6))
        self.lbl_fn = tk.Label(f1, text="Filename: No image", bg="#ffffff", fg="#475569", font=("Segoe UI", 9, "italic"), wraplength=330, anchor="w")
        self.lbl_fn.pack(fill="x")

        # 2. Properties
        f2 = tk.Frame(right, bg="#ffffff", bd=1, relief="solid", padx=12, pady=10)
        f2.pack(fill="x", pady=(0, 8))
        tk.Label(f2, text="IMAGE PROPERTIES", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 4))
        
        self.prop_labels = {}
        for prop in ["Width", "Height", "Channels", "Data Type"]:
            row = tk.Frame(f2, bg="#ffffff")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{prop:<10}:", bg="#ffffff", fg="#64748b", font=("Consolas", 10, "bold"), width=12, anchor="w").pack(side="left")
            lbl_v = tk.Label(row, text="--", bg="#ffffff", fg="#0f172a", font=("Consolas", 10, "bold"), anchor="w")
            lbl_v.pack(side="left")
            self.prop_labels[prop] = lbl_v

        # 3. Operations
        f3 = tk.Frame(right, bg="#ffffff", bd=1, relief="solid", padx=12, pady=10)
        f3.pack(fill="x", pady=(0, 8))
        tk.Label(f3, text="IMAGE OPERATIONS", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 6))
        
        r1 = tk.Frame(f3, bg="#ffffff")
        r1.pack(fill="x", pady=2)
        tk.Button(r1, text="✂ Crop", bg="#f1f5f9", font=("Segoe UI", 9), relief="flat", command=self.crop_dialog).pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(r1, text="📐 Resize", bg="#f1f5f9", font=("Segoe UI", 9), relief="flat", command=self.resize_dialog).pack(side="right", fill="x", expand=True, padx=(2, 0))

        r2 = tk.Frame(f3, bg="#ffffff")
        r2.pack(fill="x", pady=2)
        tk.Button(r2, text="🔘 Grayscale", bg="#f1f5f9", font=("Segoe UI", 9), relief="flat", command=self.make_gray).pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(r2, text="↺ Reset", bg="#f1f5f9", font=("Segoe UI", 9), relief="flat", command=self.reset_img).pack(side="right", fill="x", expand=True, padx=(2, 0))

        tk.Button(f3, text="💾  Save Processed Image", bg="#16a34a", fg="#ffffff", font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", command=self.save_proc_image).pack(fill="x", pady=(6, 0))

        # 4. Analytics
        f4 = tk.Frame(right, bg="#ffffff", bd=1, relief="solid", padx=12, pady=10)
        f4.pack(fill="x")
        tk.Label(f4, text="DATA & VISUALIZATION", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 6))
        tk.Button(f4, text="📋  View Image Data (Pandas)", bg="#f1f5f9", font=("Segoe UI", 9), relief="flat", command=self.view_pandas).pack(fill="x", pady=2)
        tk.Button(f4, text="📊  Generate Graph (Matplotlib)", bg="#f1f5f9", font=("Segoe UI", 9), relief="flat", command=self.make_graph).pack(fill="x", pady=2)

        # Status
        self.status = tk.Label(self.root, text="Status: Ready", bg="#e2e8f0", fg="#1e293b", font=("Segoe UI", 9), anchor="w", padx=10, height=2)
        self.status.pack(side="bottom", fill="x")

    def select_file(self):
        f = filedialog.askopenfilename(initialdir=ASSETS_DIR, filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
        if f:
            self.load_file(f, show_success=True)

    def load_file(self, filepath, show_success=True):
        img, err = load_image(filepath)
        if err:
            messagebox.showerror("Error", err)
            return
        self.original_image = img.copy()
        self.current_image = img.copy()
        self.image_filename = os.path.basename(filepath)
        self.lbl_fn.config(text=f"Filename: {self.image_filename}")
        self._update_props()
        self._render()
        
        # Log to JSON
        p = self.current_props
        store_image_data_to_json(self.image_filename, p["width"], p["height"], p["channels"], p["data_type"], "Read Image", self.image_filename)
        
        if show_success:
            messagebox.showinfo("Success", f"Loaded '{self.image_filename}' successfully!")
        self.status.config(text=f"Status: Loaded '{self.image_filename}'")

    def _update_props(self):
        if self.current_image is None:
            return
        p = get_image_properties(self.current_image)
        self.current_props = p
        self.prop_labels["Width"].config(text=f"{p['width']} px")
        self.prop_labels["Height"].config(text=f"{p['height']} px")
        self.prop_labels["Channels"].config(text=f"{p['channels']}")
        self.prop_labels["Data Type"].config(text=f"{p['data_type']}")

    def _render(self):
        if self.current_image is None:
            return
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 350)

        if len(self.current_image.shape) == 2:
            pil_img = Image.fromarray(self.current_image)
        else:
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

        iw, ih = pil_img.size
        ratio = min(cw / iw, ch / ih, 1.0)
        nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
        resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_preview = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_preview, anchor="center")

    def crop_dialog(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        h, w = self.current_image.shape[:2]
        d = tk.Toplevel(self.root)
        d.title("Crop Image")
        d.geometry("340x260")
        d.resizable(False, False)

        tk.Label(d, text=f"Image Dimensions: {w} x {h}", font=("Segoe UI", 9, "bold")).pack(pady=8)
        f = tk.Frame(d); f.pack(pady=4)

        entries = {}
        for idx, (label, val) in enumerate([("x1", "0"), ("y1", "0"), ("x2", str(w)), ("y2", str(h))]):
            tk.Label(f, text=label+":", width=6, anchor="w").grid(row=idx, column=0, pady=2)
            e = tk.Entry(f, width=12); e.insert(0, val); e.grid(row=idx, column=1, pady=2)
            entries[label] = e

        def apply():
            try:
                x1, y1, x2, y2 = [int(entries[k].get()) for k in ["x1", "y1", "x2", "y2"]]
                res, err = crop_image(self.current_image, x1, y1, x2, y2)
                if err:
                    messagebox.showerror("Error", err, parent=d)
                else:
                    self.current_image = res
                    self._update_props()
                    self._render()
                    d.destroy()
            except ValueError:
                messagebox.showerror("Error", "Enter valid integers", parent=d)

        tk.Button(d, text="Apply Crop", bg="#2563eb", fg="#ffffff", command=apply).pack(pady=10)

    def resize_dialog(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        h, w = self.current_image.shape[:2]
        d = tk.Toplevel(self.root)
        d.title("Resize Image")
        d.geometry("320x220")
        d.resizable(False, False)

        tk.Label(d, text=f"Current Size: {w} x {h}", font=("Segoe UI", 9, "bold")).pack(pady=8)
        f = tk.Frame(d); f.pack(pady=4)

        tk.Label(f, text="Width:", width=8, anchor="w").grid(row=0, column=0, pady=4)
        ew = tk.Entry(f, width=12); ew.insert(0, str(w)); ew.grid(row=0, column=1, pady=4)

        tk.Label(f, text="Height:", width=8, anchor="w").grid(row=1, column=0, pady=4)
        eh = tk.Entry(f, width=12); eh.insert(0, str(h)); eh.grid(row=1, column=1, pady=4)

        def apply():
            try:
                nw, nh = int(ew.get()), int(eh.get())
                res, err = resize_image(self.current_image, nw, nh)
                if err:
                    messagebox.showerror("Error", err, parent=d)
                else:
                    self.current_image = res
                    self._update_props()
                    self._render()
                    d.destroy()
            except ValueError:
                messagebox.showerror("Error", "Enter valid positive numbers", parent=d)

        tk.Button(d, text="Apply Resize", bg="#2563eb", fg="#ffffff", command=apply).pack(pady=10)

    def make_gray(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        res, err = convert_to_grayscale(self.current_image)
        if err:
            messagebox.showinfo("Notice", err)
        else:
            self.current_image = res
            self._update_props()
            self._render()

    def reset_img(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self._update_props()
            self._render()

    def save_proc_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        dest = filedialog.asksaveasfilename(initialdir=OUTPUT_DIR, initialfile=f"processed_{self.image_filename}", defaultextension=".jpg")
        if dest:
            ok, err = save_image(self.current_image, dest)
            if ok:
                p = self.current_props
                store_image_data_to_json(self.image_filename, p["width"], p["height"], p["channels"], p["data_type"], "Saved Processed", os.path.basename(dest))
                messagebox.showinfo("Success", f"Saved to {dest}")
            else:
                messagebox.showerror("Error", err)

    def view_pandas(self):
        df = load_pandas_dataframe()
        if df.empty:
            messagebox.showinfo("Info", "No records found.")
            return
        d = tk.Toplevel(self.root)
        d.title("Stored Image Data (Pandas DataFrame)")
        d.geometry("750x380")
        
        tree = ttk.Treeview(d, columns=list(df.columns), show="headings")
        for col in df.columns:
            tree.heading(col, text=col.title())
            tree.column(col, width=90)
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(fill="both", expand=True)

    def make_graph(self):
        if self.current_props:
            generate_dimension_graph(self.current_props["width"], self.current_props["height"], self.image_filename)
        else:
            messagebox.showwarning("Warning", "Load an image first.")

def launch_gui():
    root = tk.Tk()
    app = ImageAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
