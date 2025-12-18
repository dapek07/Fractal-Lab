# fractal_codec.py
import numpy as np
import cv2
import random
import time
from math import log10

try:
    from skimage.metrics import structural_similarity as ssim_metric
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

# ============================
# PREPROCESS
# ============================
def preprocess_image_cv2(img_bgr, target_size=256, to_gray=False):
    # Jika to_gray False, kita biarkan dalam format BGR (standar OpenCV)
    if to_gray:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = img_bgr

    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

# ... (Fungsi all_transforms, split_range_blocks, build_domain_blocks TETAP SAMA) ...
# ... (Pastikan Anda menyalin fungsi transformasi dan blok dari kode lama Anda di sini) ...

def all_transforms(block):
    return [
        block,
        np.rot90(block, 1),
        np.rot90(block, 2),
        np.rot90(block, 3),
        np.fliplr(block),
        np.flipud(block),
        np.rot90(np.fliplr(block), 1),
        np.rot90(np.flipud(block), 1),
    ]

def split_range_blocks(img, block_size):
    h, w = img.shape
    blocks, coords = [], []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if y + block_size <= h and x + block_size <= w:
                blocks.append(img[y:y+block_size, x:x+block_size])
                coords.append((y, x))
    return blocks, coords

def build_domain_blocks(img, domain_size, step=None):
    if step is None:
        step = domain_size
    h, w = img.shape
    blocks, coords = [], []
    for y in range(0, h - domain_size + 1, step):
        for x in range(0, w - domain_size + 1, step):
            blocks.append(img[y:y+domain_size, x:x+domain_size])
            coords.append((y, x))
    return blocks, coords

# ============================
# ENCODE (RGB SUPPORT)
# ============================
def encode_fractal_improved(img, block_size=8, domain_size=24, sample_ratio=0.3, transforms=True, step=None, verbose=False):
    t0 = time.time()
    
    # Cek apakah gambar berwarna (3 channels) atau grayscale (2 dimensions)
    if len(img.shape) == 3:
        # Loop untuk setiap channel (B, G, R)
        channels = cv2.split(img)
        all_codes = []
        for ch in channels:
            # Panggil fungsi ini secara rekursif untuk setiap channel tunggal
            # (Kita set verbose=False agar tidak spam log)
            codes, _, _, _, _ = encode_fractal_improved(ch, block_size, domain_size, sample_ratio, transforms, step)
            all_codes.append(codes)
        
        enc_time = time.time() - t0
        # Return list of codes per channel
        return all_codes, img.shape, block_size, domain_size, enc_time

    # --- LOGIKA UNTUK SINGLE CHANNEL (SAMA SEPERTI SEBELUMNYA) ---
    ranges, range_coords = split_range_blocks(img, block_size)
    domains, domain_coords = build_domain_blocks(img, domain_size, step=step)

    if sample_ratio < 1.0:
        n_samples = max(1, int(len(domains) * sample_ratio))
        idx = random.sample(range(len(domains)), n_samples)
        domains = [domains[i] for i in idx]
        domain_coords = [domain_coords[i] for i in idx]

    codes = []
    for i, R in enumerate(ranges):
        best_err = float("inf")
        best_code = None
        Rf = R.flatten()
        R_mean = Rf.mean()
        Rc = Rf - R_mean

        # Optimasi: Pre-resize semua domain blocks
        # Dalam implementasi nyata, ini sebaiknya dilakukan di luar loop
        domain_blocks_small = [cv2.resize(D, (block_size, block_size), interpolation=cv2.INTER_AREA) for D in domains]

        for j, D_small in enumerate(domain_blocks_small):
            variants = all_transforms(D_small) if transforms else [D_small]
            for t_idx, Dv in enumerate(variants):
                Df = Dv.flatten()
                Df_mean = Df.mean()
                Dc = Df - Df_mean
                denom = Dc.dot(Dc) + 1e-8
                
                a = (Dc.dot(Rc)) / denom
                b = R_mean - a * Df_mean

                # Prediksi
                pred = a * Df + b
                err = np.mean((Rf - pred)**2)

                if err < best_err:
                    best_err = err
                    # Simpan data minimal untuk rekonstruksi
                    best_code = (range_coords[i], domain_coords[j], t_idx, a, b)
        
        # Jika tidak ada yang cocok (seharusnya tidak terjadi), pakai default
        if best_code is None:
            best_code = (range_coords[i], domain_coords[0], 0, 0.0, R_mean)
            
        codes.append(best_code)

    enc_time = time.time() - t0
    return codes, img.shape, block_size, domain_size, enc_time


# ============================
# DECODE (RGB SUPPORT)
# ============================
def decode_channel(codes, shape, block_size, domain_size, iterations):
    h, w = shape
    rec = np.zeros((h, w), dtype=np.float32)
    # Start with random noise
    rec = np.clip(0.5 + 0.1*np.random.randn(h, w), 0, 1).astype(np.float32)

    for _ in range(iterations):
        new_rec = np.zeros_like(rec)
        for (ry, rx), (dy, dx), t_idx, a, b in codes:
            domain_block = rec[dy:dy+domain_size, dx:dx+domain_size]
            # Handle boundary cases if resizing might fail (optional safety)
            if domain_block.shape[0] != domain_size or domain_block.shape[1] != domain_size:
                 continue
                 
            small = cv2.resize(domain_block, (block_size, block_size), interpolation=cv2.INTER_AREA)
            trans = all_transforms(small)[t_idx]
            new_rec[ry:ry+block_size, rx:rx+block_size] = a * trans + b
        rec = np.clip(new_rec, 0, 1)
    return rec

def decode_fractal_improved(codes, image_shape, block_size, domain_size, iterations=20):
    # Cek apakah ini kode untuk RGB (list of lists) atau Single Channel
    # Asumsi: Jika codes[0] adalah list, berarti itu RGB (3 channel)
    is_rgb = isinstance(codes[0], list) and len(codes) == 3

    if is_rgb:
        h, w, c = image_shape
        channels_rec = []
        for i in range(3): # B, G, R
            rec_ch = decode_channel(codes[i], (h, w), block_size, domain_size, iterations)
            channels_rec.append(rec_ch)
        return cv2.merge(channels_rec)
    else:
        return decode_channel(codes, image_shape, block_size, domain_size, iterations)


# ============================
# METRICS
# ============================
def psnr(o, r):
    mse = np.mean((o - r)**2)
    if mse == 0:
        return float('inf')
    return 10 * log10(1 / mse)

def compute_metrics(orig, rec):
    # PSNR works fine directly on 3D arrays (mean of all pixels)
    p = psnr(orig, rec)
    
    s = 0
    if HAS_SSIM:
        # Tentukan channel_axis untuk library skimage terbaru
        # Gambar kita formatnya (H, W, C) -> axis 2 adalah channel
        # Untuk gambar grayscale (H, W), channel_axis=None
        
        is_multichannel = (len(orig.shape) == 3)
        try:
            if is_multichannel:
                # Coba parameter versi baru (scikit-image >= 0.19)
                s = ssim_metric(orig, rec, data_range=1.0, channel_axis=2)
            else:
                s = ssim_metric(orig, rec, data_range=1.0)
        except TypeError:
            # Fallback untuk versi lama
            if is_multichannel:
                s = ssim_metric(orig, rec, data_range=1.0, multichannel=True)
            else:
                s = ssim_metric(orig, rec, data_range=1.0)
    
    return p, s