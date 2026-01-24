import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# ===============================
# 1. CONFIG (SUDAH DISESUAIKAN)
# ===============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", DEVICE)

TARGET_FRAMES = 150  # Video panjang (10 detik) aman
IMG_H, IMG_W = 50, 100
PADDING = 10

# Path Output (Tempat simpan hasil .npy)
OUTPUT_ROOT = r"D:\vsr\data\processed"
HISTOGRAM_DIR = os.path.join(OUTPUT_ROOT, "debug_histograms")

# Path Input (Video Mentah)
# Sesuai struktur folder kamu:
PRIVATE_VIDEO_DIR = r"D:\vsr\data\raw\private"
PUBLIC_VIDEO_DIR = r"D:\vsr\data\raw\public"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(HISTOGRAM_DIR, exist_ok=True)

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ===============================
# 2. HELPER FUNCTIONS
# ===============================

def extract_lip_roi(image, face_mesh):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks: return None

    lm = res.multi_face_landmarks[0].landmark
    xs, ys = [], []
    for i in LIP_LANDMARKS:
        xs.append(int(lm[i].x * w))
        ys.append(int(lm[i].y * h))

    x1 = max(min(xs) - PADDING, 0)
    x2 = min(max(xs) + PADDING, w)
    y1 = max(min(ys) - PADDING, 0)
    y2 = min(max(ys) + PADDING, h)

    return image[y1:y2, x1:x2]

def augment_image(img):
    """Augmentasi: Rotasi & Brightness (Khusus Private)"""
    rows, cols = img.shape[:2]
    # Rotasi (-8 s.d 8 derajat)
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    # Brightness
    factor = random.uniform(0.7, 1.3)
    img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return img

def analyze_pixel_intensity(video_array, filename):
    plt.figure(figsize=(10, 5))
    if video_array.ndim == 3: # Grayscale (T, H, W)
        plt.hist(video_array.flatten(), bins=50, color='gray', alpha=0.7, density=True)
        title = "Grayscale"
    else: return
    plt.title(f"Histogram {title}: {filename}")
    plt.savefig(os.path.join(HISTOGRAM_DIR, filename.replace(".npy", ".png")))
    plt.close()

# ===============================
# 3. CORE PROCESSOR
# ===============================

def process_video_frames(video_path, face_mesh, apply_aug=False):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        roi = extract_lip_roi(frame, face_mesh)
        if roi is None: continue

        # A. Augmentasi dulu (sebelum grayscale)
        if apply_aug:
            roi = augment_image(roi)

        # B. Resize & Grayscale
        roi = cv2.resize(roi, (IMG_W, IMG_H))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        frames.append(roi)

    cap.release()
    if len(frames) == 0: return None

    # --- PADDING & SAMPLING (Logic Baru 150 Frames) ---
    input_arr = np.array(frames) 
    T = len(input_arr)
    
    # Normalisasi Pixel (0-1)
    final_data = input_arr / 255.0

    if T > TARGET_FRAMES:
        # Uniform Sampling (Video Panjang -> Dipadatkan)
        indices = np.linspace(0, T-1, TARGET_FRAMES, dtype=int)
        final_data = final_data[indices]
    elif T < TARGET_FRAMES:
        # Zero Padding (Video Pendek -> Ditambah nol)
        padding = np.zeros((TARGET_FRAMES - T, IMG_H, IMG_W), dtype=np.float32)
        final_data = np.concatenate((final_data, padding), axis=0)

    return final_data.astype(np.float32)

# ===============================
# 4. DATASET RUNNER (AUTO SCAN)
# ===============================

def run_processing():
    # ---------------------------------------------------------
    # PART 1: PRIVATE DATASET
    # ---------------------------------------------------------
    print("\n🔹 Processing PRIVATE dataset...")
    out_dir = os.path.join(OUTPUT_ROOT, "private")
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(PRIVATE_VIDEO_DIR):
        files = [f for f in os.listdir(PRIVATE_VIDEO_DIR) if f.endswith('.mp4')]
        print(f"   Found {len(files)} private videos.")
        
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            for vid in tqdm(files):
                vid_path = os.path.join(PRIVATE_VIDEO_DIR, vid)
                base_name = os.path.splitext(vid)[0]
                
                # 1. Asli (Original)
                tensor = process_video_frames(vid_path, face_mesh, apply_aug=False)
                if tensor is not None:
                    np.save(os.path.join(out_dir, f"{base_name}_orig.npy"), tensor)
                    if vid == files[0]: analyze_pixel_intensity(tensor, f"{base_name}_orig.npy")

                # 2. Augmentasi (5 Variasi)
                for i in range(5):
                    tensor_aug = process_video_frames(vid_path, face_mesh, apply_aug=True)
                    if tensor_aug is not None:
                        np.save(os.path.join(out_dir, f"{base_name}_aug{i}.npy"), tensor_aug)
                
                # Auto-delete video asli (Opsional: Aktifkan jika disk penuh)
                # try: os.remove(vid_path)
                # except: pass
    else:
        print(f"❌ Folder Private tidak ditemukan: {PRIVATE_VIDEO_DIR}")

    # ---------------------------------------------------------
    # PART 2: PUBLIC DATASET (Male & Female)
    # ---------------------------------------------------------
    print("\n🔹 Processing PUBLIC dataset...")
    out_dir_pub = os.path.join(OUTPUT_ROOT, "public")
    os.makedirs(out_dir_pub, exist_ok=True)

    # List folder yang mungkin ada di dalam public
    possible_folders = ["female", "male"] 
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        for subfolder in possible_folders:
            # Cari path: raw/public/female  ATAU raw/public/female/video
            current_path = os.path.join(PUBLIC_VIDEO_DIR, subfolder)
            
            # Cek apakah ada subfolder 'video' lagi di dalamnya (kasus umum dataset LUMINA)
            if os.path.exists(os.path.join(current_path, "video")):
                current_path = os.path.join(current_path, "video")
            
            if not os.path.exists(current_path):
                print(f"⚠️ Folder tidak ditemukan, skip: {current_path}")
                continue

            files = [f for f in os.listdir(current_path) if f.endswith('.mp4')]
            print(f"   >> Processing {subfolder.upper()} ({len(files)} videos)...")
            
            for vid in tqdm(files):
                vid_path = os.path.join(current_path, vid)
                base_name = os.path.splitext(vid)[0]
                # Simpan dengan prefix gender biar gak ketukar (misal: female_001.npy)
                save_path = os.path.join(out_dir_pub, f"{subfolder}_{base_name}.npy")
                
                # Skip jika sudah ada
                if os.path.exists(save_path): continue
                
                tensor = process_video_frames(vid_path, face_mesh, apply_aug=False)
                
                if tensor is not None:
                    np.save(save_path, tensor)
                    # Hapus video asli public untuk hemat space (Karena bisa didownload lagi)
                    try: os.remove(vid_path)
                    except: pass

if __name__ == "__main__":
    run_processing()