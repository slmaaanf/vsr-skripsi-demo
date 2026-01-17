import os
# --- CONFIG WAJIB ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import pandas as pd
import gdown
from model import VSRModel

# ==========================================
# 1. KONFIGURASI
# ==========================================
st.set_page_config(page_title="VSR Smart Lock", page_icon="🔐")

# ID Google Drive (File .pth)
GDRIVE_FILE_ID = '1qSCv7cQfqLP5Jznt_SfVNh6coK2Id8DI' 

FILES = {
    "model": "vsr_deployed.pth",
    "labels": "labels.csv" # File CSV yang kamu copy tadi
}

DEVICE = torch.device("cpu")
IMG_H, IMG_W, MAX_FRAMES = 50, 100, 75

mp_face_mesh = mp.solutions.face_mesh
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ==========================================
# 2. LOGIKA UTAMA
# ==========================================
@st.cache_resource
def load_engine():
    # A. Download Model
    if not os.path.exists(FILES["model"]):
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        with st.spinner("Mengunduh Model..."):
            gdown.download(url, FILES["model"], quiet=False)

    # B. Load Model
    if not os.path.exists(FILES["model"]): return None, None, None
    
    try:
        ckpt = torch.load(FILES["model"], map_location=DEVICE)
        class_to_idx = ckpt.get('class_to_idx')
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        model = VSRModel(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # C. LOAD MASKING (JURUS RAHASIA)
        # Ini yang bikin confidence tinggi & akurat
        private_indices = []
        if os.path.exists(FILES["labels"]):
            try:
                # Baca CSV pembatas '|' sesuai format lokal kamu
                df = pd.read_csv(FILES["labels"], sep='|', header=None, names=['fn', 'text'])
                private_texts = df['text'].astype(str).str.strip().unique()
                
                for text in private_texts:
                    if text in class_to_idx:
                        private_indices.append(class_to_idx[text])
                print(f"✅ Filter Aktif: {len(private_indices)} kalimat diizinkan.")
            except Exception as e:
                st.error(f"Gagal baca CSV label: {e}")
        
        return model, idx_to_class, private_indices

    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None

# ==========================================
# 3. PREPROCESSING (SAMA PERSIS LOKAL)
# ==========================================
def get_crop(frame, landmarks):
    h, w, _ = frame.shape
    lips_x = [int(landmarks.landmark[i].x * w) for i in LIPS_INDICES]
    lips_y = [int(landmarks.landmark[i].y * h) for i in LIPS_INDICES]
    
    min_x, max_x = min(lips_x), max(lips_x)
    min_y, max_y = min(lips_y), max(lips_y)
    
    center_x, center_y = (min_x+max_x)//2, (min_y+max_y)//2
    target_w = int((max_x - min_x) * 2.2)
    target_h = int(target_w / 2)
    
    x1, y1 = center_x - target_w//2, center_y - target_h//2
    return frame[max(0,y1):min(h,y1+target_h), max(0,x1):min(w,x1+target_w)]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # HAPUS LOGIKA CLAHE & FPS SKIP AGAR SAMA DENGAN LOKAL
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            if res.multi_face_landmarks:
                crop = get_crop(frame, res.multi_face_landmarks[0]) # Crop dari frame asli (BGR)
                if crop.size > 0:
                    # PROSES PERSIS KODE LOKAL:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (IMG_W, IMG_H))
                    frames.append(resized/255.0)
                    
    cap.release()
    if not frames: return None
    
    # Padding Logic
    arr = np.array(frames)
    T = len(arr)
    if T > MAX_FRAMES:
        indices = np.linspace(0, T-1, MAX_FRAMES, dtype=int)
        final = arr[indices]
    else:
        padding = np.zeros((MAX_FRAMES - T, IMG_H, IMG_W), dtype=np.float32)
        final = np.concatenate((arr, padding), axis=0)
    return final

# ==========================================
# 4. TAMPILAN UI
# ==========================================
st.title("🔐 VSR Smart Lock")

model, idx_to_class, private_indices = load_engine()

if model:
    if not private_indices:
        st.warning("⚠️ Warning: File 'labels.csv' tidak ditemukan. Prediksi mungkin tidak akurat (semua kalimat diizinkan).")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    
    if uploaded_file:
        tfile = "temp_input.mp4"
        with open(tfile, "wb") as f: f.write(uploaded_file.getbuffer())
        st.video(uploaded_file)
        
        if st.button("🔍 PROSES"):
            inp = process_video(tfile)
            
            if inp is not None:
                tensor = torch.FloatTensor(inp).unsqueeze(0).unsqueeze(0).to(DEVICE).permute(0,1,2,3,4)
                
                with torch.no_grad():
                    out = model(tensor)
                    
                    # --- MASKING MAGIC (KUNCI SUKSES) ---
                    if private_indices:
                        # Buat mask: Set semua ke -infinity
                        mask = torch.ones_like(out) * float('-inf')
                        # Set yang boleh dipilih ke 0
                        mask[:, private_indices] = 0
                        # Gabungkan
                        final_out = out + mask
                    else:
                        final_out = out
                    
                    probs = torch.nn.functional.softmax(final_out, dim=1)
                    conf, idx = torch.max(probs, 1)
                    
                    p_text = idx_to_class.get(idx.item(), "Unknown")
                    p_conf = conf.item() * 100
                
                # Hasil
                st.info(f"Prediksi: **{p_text}**")
                st.success(f"Confidence: **{p_conf:.2f}%**")
                
                if "buka kunci" in p_text.lower():
                    st.balloons()
                    st.success("🔓 PINTU TERBUKA")
                else:
                    st.error("🔒 AKSES DITOLAK")
            else:
                st.error("Wajah tidak terdeteksi")