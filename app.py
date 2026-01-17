import os
# --- FIX WAJIB WINDOWS (Harus paling atas) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import pandas as pd
import time
from datetime import datetime
from model import VSRModel

# ==========================================
# KONFIGURASI
# ==========================================
st.set_page_config(page_title="VSR Smart Lock", page_icon="🔐", layout="wide")

GDRIVE_FILE_ID = '1qSCv7cQfqLP5Jznt_SfVNh6coK2Id8DI'

# Path (Sesuaikan jika perlu)
CHECKPOINT_PATH = r"D:\vsr\checkpoints\vsr_final_hard.pth"
LABEL_PRIVATE = r"D:\vsr\data\labels\private_final_hard.csv"
LOG_FILE = "maintenance_log.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_H, IMG_W, MAX_FRAMES = 50, 100, 75
mp_face_mesh = mp.solutions.face_mesh
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ==========================================
# ENGINE (CACHED)
# ==========================================
@st.cache_resource
def load_engine():
    if not os.path.exists(CHECKPOINT_PATH): return None, None, None

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        class_to_idx = ckpt.get('class_to_idx')
        if not class_to_idx: return None, None, None
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        model = VSRModel(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # Filter Private
        private_indices = []
        if os.path.exists(LABEL_PRIVATE):
            try: df = pd.read_csv(LABEL_PRIVATE, sep='|', header=None, names=['fn', 'text'])
            except: df = pd.read_csv(LABEL_PRIVATE)
            texts = df['text'].astype(str).str.strip().unique()
            for t in texts:
                if t in class_to_idx: private_indices.append(class_to_idx[t])
                
        return model, idx_to_class, private_indices
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

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
    
    frames, debug_crop = [], None
    
    # Deteksi FPS video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Jika video 60fps, kita skip frame biar jadi mirip 30fps
    skip_rate = 2 if fps > 50 else 1
    
    # CLAHE: Alat ajaib penajam kontras (Bibir gelap jadi terang)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    count = 0
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            count += 1
            if count % skip_rate != 0: continue # Skip frame jika perlu

            # 1. Konversi BGR ke RGB (FIX BIBIR BIRU)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb_frame)
            
            if res.multi_face_landmarks:
                # Crop bibir
                crop = get_crop(rgb_frame, res.multi_face_landmarks[0]) # Crop dari RGB
                
                if crop.size > 0:
                    # 2. Proses Grayscale yang BENAR dari RGB
                    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                    
                    # 3. PERTAJAM KONTRAS
                    enhanced = clahe.apply(gray)
                    
                    # Resize
                    resized = cv2.resize(enhanced, (IMG_W, IMG_H))
                    frames.append(resized/255.0)
                    
                    # Simpan crop berwarna untuk visualisasi (sudah RGB)
                    if debug_crop is None: debug_crop = crop
                    
    cap.release()
    if not frames: return None, None
    
    # Padding Logic (Sama)
    arr = np.array(frames)
    T = len(arr)
    if T > MAX_FRAMES:
        indices = np.linspace(0, T-1, MAX_FRAMES, dtype=int)
        final = arr[indices]
    else:
        padding = np.zeros((MAX_FRAMES - T, IMG_H, IMG_W), dtype=np.float32)
        final = np.concatenate((arr, padding), axis=0)
        
    return final, debug_crop

def log_feedback(filename, text, conf, status):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if header: f.write("Timestamp,File,Prediction,Confidence,Status\n")
        f.write(f"{ts},{filename},{text},{conf:.2f},{status}\n")

# ==========================================
# USER INTERFACE
# ==========================================
st.sidebar.title("⚙️ Panel Kontrol")
st.sidebar.info("Model: ResNet50 + BiGRU")
conf_threshold = st.sidebar.slider("Batas Minimum Confidence", 0, 100, 10)

st.title("🔐 VSR Access Control")
st.markdown("### *Visual Speech Recognition System*")

model, idx_to_class, private_indices = load_engine()

if model:
    col_upload, col_result = st.columns([1, 1])
    
    with col_upload:
        st.subheader("1. Input Video")
        uploaded_file = st.file_uploader("Upload MP4 (Max 3 detik)", type=["mp4"])
        
        if uploaded_file:
            tfile = os.path.join("temp.mp4")
            with open(tfile, "wb") as f: f.write(uploaded_file.getbuffer())
            st.video(uploaded_file)
            
            if st.button("🚀 PROSES PREDIKSI", use_container_width=True):
                with st.spinner("Membaca gerak bibir..."):
                    inp, viz = process_video(tfile)
                    
                    if inp is not None:
                        # Inference
                        tensor = torch.FloatTensor(inp).unsqueeze(0).unsqueeze(0).to(DEVICE).permute(0,1,2,3,4)
                        with torch.no_grad():
                            out = model(tensor)
                            mask = torch.ones_like(out) * float('-inf')
                            if private_indices: mask[:, private_indices] = 0
                            probs = torch.nn.functional.softmax(out+mask, dim=1)
                            conf, idx = torch.max(probs, 1)
                            
                            p_text = idx_to_class.get(idx.item(), "Unknown")
                            p_conf = conf.item() * 100
                            
                        # Simpan hasil ke session state biar gak hilang pas klik feedback
                        st.session_state['result'] = (p_text, p_conf, viz, uploaded_file.name)
                    else:
                        st.error("Wajah tidak terdeteksi!")

    with col_result:
        st.subheader("2. Hasil Analisis")
        if 'result' in st.session_state:
            res_text, res_conf, res_viz, fname = st.session_state['result']
            
            # Tampilkan Mata AI
            st.image(res_viz, caption="Input Visual Model", width=250)
            
            # Tampilkan Hasil
            if res_conf >= conf_threshold:
                if "buka kunci" in res_text.lower():
                    st.success(f"🔓 AKSES DITERIMA")
                else:
                    st.warning(f"🔒 AKSES DITOLAK (Kalimat salah)")
            else:
                st.error(f"⚠️ TIDAK YAKIN (Di bawah {conf_threshold}%)")
                
            st.markdown(f"**Prediksi:** `{res_text}`")
            st.progress(min(int(res_conf), 100))
            st.caption(f"Confidence: {res_conf:.2f}%")
            
            # Feedback
            st.markdown("---")
            st.write("**Apakah prediksi ini benar?** (Data untuk Maintenance)")
            c1, c2 = st.columns(2)
            if c1.button("✅ BENAR"):
                log_feedback(fname, res_text, res_conf, "Correct")
                st.toast("Feedback tersimpan!")
            if c2.button("❌ SALAH"):
                log_feedback(fname, res_text, res_conf, "Wrong")
                st.toast("Laporan tersimpan untuk training ulang.")

else:
    st.error("Gagal memuat model. Pastikan file .pth dan .csv ada.")