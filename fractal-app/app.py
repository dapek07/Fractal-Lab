from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import time
import cv2
import pickle
import numpy as np

# Import modul lokal logika fraktal
from fractal_codec import (
    preprocess_image_cv2,
    encode_fractal_improved,
    decode_fractal_improved,
    compute_metrics
)

# Import helper functions
from utils import (
    ensure_directories, 
    UPLOAD_FOLDER, 
    RECON_FOLDER, 
    save_image_gray, 
    generate_filename
)

# Pastikan folder static/uploaded dan static/recon tersedia
ensure_directories()

app = Flask(__name__)
app.secret_key = "fractal_secret"
ALLOWED = {"jpg", "jpeg", "png", "bmp"}

# --- Helper Functions ---

def get_file_size_str(size_in_bytes):
    """Mengubah ukuran bytes ke format teks (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} GB"

def allowed_file(fname):
    """Cek ekstensi file yang diizinkan."""
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compress", methods=["POST"])
def compress():
    files = request.files.getlist("images")
    mode = request.form.get("mode")

    if not files or len(files) == 0:
        flash("Upload minimal 1 file.")
        return redirect(url_for("index"))

    results = []
    for f in files:
        # Validasi File
        if not f or f.filename == '' or not allowed_file(f.filename):
            continue

        fname = secure_filename(f.filename)
        savepath = os.path.join(UPLOAD_FOLDER, fname)
        f.save(savepath)

        # 1. Hitung Ukuran File Asli (Bytes) untuk perbandingan
        original_size_bytes = os.path.getsize(savepath)

        # === Load Image ===
        img_bgr = cv2.imread(savepath)
        
        # Preprocess: to_gray=False agar gambar tetap berwarna (RGB)
        img = preprocess_image_cv2(img_bgr, to_gray=False)

        # === Mode Settings ===
        if mode == "fast":
            block_size = 8
            domain_size = 24
            sample_ratio = 0.3
            iterations = 20
            step = domain_size
        else:
            # Mode Accurate
            block_size = 8
            domain_size = 32
            sample_ratio = 0.5
            iterations = 30
            step = domain_size // 2

        # === ENCODE (Kompresi) ===
        # Karena input RGB, codes akan berisi list dari 3 channel: [RedCodes, GreenCodes, BlueCodes]
        codes, shape, bs, ds, enc_time = encode_fractal_improved(
            img,
            block_size=block_size,
            domain_size=domain_size,
            sample_ratio=sample_ratio,
            step=step,
            transforms=True
        )

        # === HITUNG STATISTIK UKURAN FILE ===
        
        # A. Ukuran Kompresi RGB (Total 3 Channel)
        bytes_rgb = len(pickle.dumps(codes))
        
        # B. Ukuran Kompresi Grayscale (Estimasi: Ambil channel pertama saja)
        # Karena struktur data per channel sama, ukuran 1 channel merepresentasikan ukuran versi Grayscale
        bytes_gray = len(pickle.dumps(codes[0]))

        # Hitung Persentase (Rasio) Hemat
        if original_size_bytes > 0:
            ratio_rgb = (1 - (bytes_rgb / original_size_bytes)) * 100
            ratio_gray = (1 - (bytes_gray / original_size_bytes)) * 100
        else:
            ratio_rgb = 0
            ratio_gray = 0

        # === DECODE (Rekonstruksi) ===
        t0 = time.time()
        rec_rgb = decode_fractal_improved(
            codes,
            shape,
            block_size,
            domain_size,
            iterations=iterations
        )
        dec_time = time.time() - t0

        # === SIMPAN HASIL (2 Versi) ===
        
        # 1. Simpan Versi RGB
        outname_rgb = generate_filename("recon_rgb", ".png")
        outpath_rgb = os.path.join(RECON_FOLDER, outname_rgb)
        save_image_gray(rec_rgb, outpath_rgb) # Fungsi ini aman untuk RGB

        # 2. Simpan Versi Grayscale (Konversi dari hasil RGB)
        rec_gray = cv2.cvtColor(rec_rgb, cv2.COLOR_BGR2GRAY)
        outname_gray = generate_filename("recon_gray", ".png")
        outpath_gray = os.path.join(RECON_FOLDER, outname_gray)
        save_image_gray(rec_gray, outpath_gray)

        # === METRICS ===
        # Hitung kualitas berdasarkan gambar RGB agar adil dengan preview utama
        psnr_v, ssim_v = compute_metrics(img, rec_rgb)

        # === DATA UNTUK TEMPLATE ===
        results.append({
            "orig": savepath.replace("\\", "/"),
            "recon_rgb": outpath_rgb.replace("\\", "/"),
            "recon_gray": outpath_gray.replace("\\", "/"),
            
            "encode_time": round(enc_time, 2),
            "decode_time": round(dec_time, 2),
            "psnr": round(psnr_v, 2),
            "ssim": round(ssim_v if ssim_v else 0, 3),
            
            "orig_size": get_file_size_str(original_size_bytes),
            
            # Statistik RGB
            "size_rgb": get_file_size_str(bytes_rgb),
            "ratio_rgb": round(ratio_rgb, 2),
            
            # Statistik Grayscale
            "size_gray": get_file_size_str(bytes_gray),
            "ratio_gray": round(ratio_gray, 2)
        })

    return render_template("result.html", results=results)

@app.route("/clear")
def clear_data():
    folders = [UPLOAD_FOLDER, RECON_FOLDER]
    deleted_count = 0
    for folder in folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        deleted_count += 1
                except Exception as e:
                    print(f'Gagal menghapus {file_path}. Error: {e}')

    flash(f"Cache dibersihkan! {deleted_count} file sampah telah dihapus.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)