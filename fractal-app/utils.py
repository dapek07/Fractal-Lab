# utils.py
import os
import uuid
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploaded")
RECON_FOLDER = os.path.join(BASE_DIR, "static", "recon")

def ensure_directories():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RECON_FOLDER, exist_ok=True)

def generate_filename(prefix="img", ext=".png"):
    return f"{prefix}_{uuid.uuid4().hex}{ext}"

def save_image_gray(img, path):
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)
