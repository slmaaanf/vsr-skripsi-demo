import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===============================
# CONFIG KHUSUS PUBLIC
# ===============================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TARGET_FRAMES = 150
IMG_H, IMG_W = 50, 100
PADDING = 10

# Arahkan ke folder RAW Public kamu
PUBLIC_VIDEO_DIR = r"D:\vsr\data\raw\public"
OUTPUT_ROOT = r"D:\vsr\data\processed"

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ===============================
# FUNGSI PEMROSESAN
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

def process_video_frames(video_path, face_mesh):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        roi = extract_lip_roi(frame, face_mesh)
        if roi is None: continue
        
        # Resize & Grayscale (Tanpa Augmentasi untuk Public)
        roi = cv2.resize(roi, (IMG_W, IMG_H))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frames.append(roi)
    cap.release()
    
    if len(frames) == 0: return None

    # Logic 150 Frame (YANG SUDAH DIPERBAIKI)
    input_arr = np.array(frames) / 255.0
    T = len(input_arr)
    
    if T > TARGET_FRAMES:
        # Case 1: Video Kepanjangan -> Sampling
        indices = np.linspace(0, T-1, TARGET_FRAMES, dtype=int)
        final_data = input_arr[indices]
        
    elif T < TARGET_FRAMES:
        # Case 2: Video Kependekan -> Padding
        padding = np.zeros((TARGET_FRAMES - T, IMG_H, IMG_W), dtype=np.float32)
        # BUG FIXED: Dulu 'final_data', sekarang 'input_arr'
        final_data = np.concatenate((input_arr, padding), axis=0) 
        
    else:
        # Case 3: Pas 150 Frame
        final_data = input_arr
        
    return final_data.astype(np.float32)

# ===============================
# EXECUTION (RECURSIVE SEARCH)
# ===============================
def run_public_only():
    print("\n🔹 Processing PUBLIC dataset ONLY (Recursive)...")
    out_dir_pub = os.path.join(OUTPUT_ROOT, "public")
    os.makedirs(out_dir_pub, exist_ok=True)

    possible_folders = ["female", "male"]
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        for subfolder in possible_folders:
            search_path = os.path.join(PUBLIC_VIDEO_DIR, subfolder)
            
            if not os.path.exists(search_path):
                print(f"⚠️ Folder {subfolder} tidak ditemukan di: {search_path}")
                continue

            # 🔥 CARI FILE SAMPAI KE LUBANG SEMUT (Recursive)
            found_files = []
            for root, dirs, files in os.walk(search_path):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov')):
                        found_files.append(os.path.join(root, f))
            
            print(f"   >> Found {len(found_files)} videos in {subfolder.upper()}...")

            for vid_path in tqdm(found_files):
                base_name = os.path.splitext(os.path.basename(vid_path))[0]
                save_path = os.path.join(out_dir_pub, f"{subfolder}_{base_name}.npy")
                
                # Skip jika sudah ada
                if os.path.exists(save_path): continue
                
                try:
                    tensor = process_video_frames(vid_path, face_mesh)
                    if tensor is not None:
                        np.save(save_path, tensor)
                        # Opsional: Hapus video asli untuk hemat space
                        # os.remove(vid_path)
                except Exception as e:
                    print(f"Error {base_name}: {e}")

if __name__ == "__main__":
    run_public_only()