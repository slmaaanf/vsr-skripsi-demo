import os
# --- CONFIG WAJIB ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import gdown  # Library untuk download dari GDrive
from model import VSRModel

# ==========================================
# 1. KONFIGURASI PATH & CLOUD
# ==========================================
st.set_page_config(page_title="VSR Smart Lock", page_icon="🔐")

# ID Google Drive (File .pth kamu)
GDRIVE_FILE_ID = '1qSCv7cQfqLP5Jznt_SfVNh6coK2Id8DI' 

# --- PERBAIKAN PATH (PENTING!) ---
# Jangan pakai D:\vsr... Pakai nama file saja.
CHECKPOINT_PATH = "vsr_final_hard.pth" 
# Kita tidak butuh CSV label lagi, karena label sudah tersimpan di dalam .pth
DEVICE = torch.device("cpu") 

IMG_H, IMG_W, MAX_FRAMES = 50, 100, 75
mp_face_mesh = mp.solutions.face_mesh
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ==========================================
# 2. LOGIKA DOWNLOAD & LOAD ENGINE
# ==========================================
@st.cache_resource
def load_engine():
    # A. Download Model Otomatis dari Google Drive
    if not os.path.exists(CHECKPOINT_PATH):
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        with st.spinner("Sedang mengunduh Model AI (±100MB)... Harap tunggu sebentar."):
            try:
                gdown.download(url, CHECKPOINT_PATH, quiet=False)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal download model: {e}")
                return None, None

    # B. Load Model ke Memory
    if not os.path.exists(CHECKPOINT_PATH): return None, None
    
    try:
        # Load CPU Only (Cloud jarang ada GPU)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Ambil daftar kata (label) langsung dari dalam otak model
        class_to_idx = ckpt.get('class_to_idx') 
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        # Inisialisasi Arsitektur
        model = VSRModel(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        return model, idx_to_class
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ==========================================
# 3. PREPROCESSING (Updated: Fix Warna & FPS)
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
    # Pastikan crop tidak keluar batas gambar
    return frame[max(0,y1):min(h,y1+target_h), max(0,x1):min(w,x1+target_w)]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames, debug_crop = [], None
    
    # Auto-Fix FPS (Tangani video 60fps dari HP)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = 2 if fps > 50 else 1
    
    # Auto-Contrast (CLAHE) - Agar bibir jelas meski gelap
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    count = 0
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            count += 1
            if count % skip_rate != 0: continue 

            # Fix Warna: BGR ke RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            if res.multi_face_landmarks:
                crop = get_crop(rgb, res.multi_face_landmarks[0]) # Crop dari RGB
                if crop.size > 0:
                    # Grayscale
                    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                    # Pertajam
                    enhanced = clahe.apply(gray)
                    # Resize
                    resized = cv2.resize(enhanced, (IMG_W, IMG_H))
                    
                    frames.append(resized/255.0)
                    if debug_crop is None: debug_crop = crop
                    
    cap.release()
    if not frames: return None, None
    
    # Padding / Truncating
    arr = np.array(frames)
    T = len(arr)
    if T > MAX_FRAMES:
        indices = np.linspace(0, T-1, MAX_FRAMES, dtype=int)
        final = arr[indices]
    else:
        padding = np.zeros((MAX_FRAMES - T, IMG_H, IMG_W), dtype=np.float32)
        final = np.concatenate((arr, padding), axis=0)
    return final, debug_crop

# ==========================================
# 4. TAMPILAN UTAMA (UI)
# ==========================================
st.title("🔐 VSR Smart Lock System")
st.markdown("### Demo Skripsi: Visual Speech Recognition")

# Load Model
model, idx_to_class = load_engine()

if model:
    uploaded_file = st.file_uploader("Upload Video (Max 3 Detik)", type=["mp4"])
    
    if uploaded_file:
        # Simpan sementara
        tfile = "temp_input.mp4"
        with open(tfile, "wb") as f: f.write(uploaded_file.getbuffer())
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.video(uploaded_file)
        
        if st.button("🔍 ANALISIS AKSES", use_container_width=True):
            with st.spinner("Membaca gerak bibir..."):
                inp, viz = process_video(tfile)
                
                if inp is not None:
                    # Prediksi
                    tensor = torch.FloatTensor(inp).unsqueeze(0).unsqueeze(0).to(DEVICE).permute(0,1,2,3,4)
                    with torch.no_grad():
                        out = model(tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        conf, idx = torch.max(probs, 1)
                        
                        p_text = idx_to_class.get(idx.item(), "Unknown")
                        p_conf = conf.item() * 100
                    
                    # Tampilkan Hasil di Kolom Kanan
                    with c2:
                        st.image(viz, width=150, caption="Mata AI (Enhanced)")
                        st.markdown(f"**Prediksi:** `{p_text}`")
                        st.caption(f"Confidence: {p_conf:.2f}%")
                        
                        # LOGIKA KUNCI PINTU
                        if "buka kunci" in p_text.lower():
                            st.success("🔓 **AKSES DITERIMA**\nPintu Terbuka.")
                        else:
                            st.error("🔒 **AKSES DITOLAK**\nKalimat salah.")
                else:
                    st.warning("Wajah tidak terdeteksi dalam video.")
else:
    st.info("Menunggu model siap...")