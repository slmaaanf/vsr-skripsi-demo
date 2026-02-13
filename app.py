import streamlit as st
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import tempfile
import time
import pandas as pd
import re
import gdown 

# ===============================
# 1. KONFIGURASI HALAMAN & KONSTANTA
# ===============================
st.set_page_config(
    page_title="VSR System Demo",
    page_icon="👄",
    layout="wide"
)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vsr_final_hard.pth" # Nama file model yang disimpan
IMG_H, IMG_W = 50, 100
TARGET_FRAMES = 150
PADDING = 10

# MediaPipe Configuration
mp_face_mesh = mp.solutions.face_mesh
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ===============================
# 2. DEFINISI LABEL (112 KELAS - CLOSED SET)
# ===============================
def get_fixed_labels():
    raw_labels = [
        'P01', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11', 'P12', 'P13', 'P14',
        'aktifkanmodepesawatsekarang', 'apakahkamusudahmengerjakanpr', 'bagaimanacaramasaknasigoreng', 
        'bajuwarnabiruitubagus', 'bebekbekubapakbibibau', 'berapasuhuudarahariini', 'beratbadansayaenampuluhkilo', 
        'berhentimemutarvideoini', 'buayamembawabungkusanbaru', 'bukaaplikasikameradepan', 'bukakuncipintudepan', 
        'bukuinisangatbagusisinya', 'carirestoranpadangterdekat', 'cicakcicakdidindingdiam', 'cobajelaskansekalilagipelanpelan', 
        'cuacahariinisangatpanasya', 'diasedangbelajarbahasapemrogramanpythondasar', 'dimanakamusimpankuncimotor', 
        'enamtujuhdelapansembilansepuluh', 'fififotovasbungaungu', 'filmbioskopterbaruitusangatserusekali', 
        'geserkeslideberikutnya', 'gigigusigigigusigeraham', 'haloapakabarkamuhariini', 'hapusfileyangtidakperlu', 
        'hargabahanpokoksedangnaikdipasartradisional', 'harganyalimapuluhriburupiah', 'hatihatidijalanpulangya', 
        'hewanharimauharushidupdihutan', 'hujanderasmenyebabkanbanjirdibeberapadaerah', 'iniadalahkalimatterakhirdaridatasetsaya', 
        'ituideyangsangatcemerlang', 'janganbuangsampahsembarangandong', 'janganlupamakansiangnanti', 
        'janganlupamematikankomporsebelumpergikeluar', 'jaraknyasekitarduaratusmeter', 'jaringaninternetdisinisedangsangatlambat', 
        'jauhjalanjangankanjatuh', 'kapankitabisapergiliburan', 'kembalikehalamanutama', 'kenapakamudiamsajadaritadi', 
        'keretaapiberangkattepatpukuldelapanpagi', 'kirimpesankegrupkantor', 'kitabertemujamtujuhmalam', 'kitaharusmenjagakesehatantubuh', 
        'kitaperludiskusikelompokuntuktugasbesar', 'kopiinirasanyaterlalumanisuntuksaya', 'kucingitutidurpulasdiatassofaempuk', 
        'kukukakikakekkakukaku', 'laptopsayakehabisanbateraisaatsedangrapat', 'maafsayadatangterlambattadi', 
        'mahasiswasedangmelakukanpenelitiandilaboratorium', 'mamamakanmanggamanismalammalam', 'masukkankodeverifikasiempatdigit', 
        'matikantelevisisekarang', 'mungkinnantisorehujanturun', 'naikkanvolumesuarasedikit', 'nanyanyonyanyanyinyaringnian', 
        'nomorantriankamuadalahduabelas', 'nomorteleponnyaacaksekali', 'nyalakanlampukamarmandi', 'olahragapagisangatbaikuntukkesehatanjantung', 
        'papapulangbawapepayapipih', 'pasangalarmjamlimapagi', 'pemandangandipantaiitusungguhmempesonamata', 
        'pemerintahmengumumkanliburnasionalbesoklusa', 'perkembanganteknologikecerdasanbuatansangatpesat', 'putarmusikfavoritsaya', 
        'raralariluruslalulupa', 'sampaijumpabesokdikampus', 'satuduatigaempatlima', 'sayabelumpernahkesanasebelumnya', 
        'sayabutuhtigalembarkertas', 'sayaharusmenyelesaikanskripsibulandepanpasti', 'sayainginminumkopidingin', 
        'sayalahirtahunsembilanpuluhan', 'sayamerasasangatlelahsekali', 'sayasedangmengerjakantugasakhir', 
        'sayasenangbertemudengankamu', 'sayasetujudenganpendapatkamu', 'sayasukasatesapisurabaya', 'sayatidakmengertimaksudkamu', 
        'selamatpagisemuanyasemogasukses', 'selamatulangtahunsemogapanjangumursehatselalu', 'siapayangmengambilkuesaya', 
        'simpandokumeninisegera', 'sudahjamduabelassianglewat', 'tangkaplayarhalamanini', 'teleponibusekarangjuga', 
        'terimakasihbanyakatasbantuannya', 'tinggalsepuluhdetiklagi', 'tokoitututuptujuhtahun', 'tolongambilkanminumdimeja', 
        'tolongbeliduabotolair', 'tolongsampaikansalamsayakepadaorangtuamu', 'tolongtutuppintunyapelanpelan', 
        'tunjukkanjalankerumahsakit', 'turunkankecerahanlayarhp', 'ulangtahunsayatanggaltigamaret', 'ularmelingkardipagarbundar'
    ]
    le = LabelEncoder()
    le.fit(raw_labels)
    return le

@st.cache_data
def load_sentence_map():
    # Fungsi opsional jika ingin memetakan kode (misal 'P01') ke kalimat panjang
    # Untuk skripsi ini, kita anggap label raw sudah cukup deskriptif
    # atau bisa di-hardcode dictionary-nya di sini jika perlu.
    mapping = {}
    return mapping

# ===============================
# 3. ARSITEKTUR MODEL (ResNet50V2-BiGRU)
# ===============================
class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(ResNetBiGRU, self).__init__()
        # Backbone ResNet50
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-2] 
        self.resnet_feature_extractor = nn.Sequential(*modules)
        self.resnet_out_features = 2048
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal Bi-GRU
        self.gru = nn.GRU(
            input_size=self.resnet_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        # Classification Head
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        b, t, h, w = x.size()
        x = x.view(b * t, 1, h, w)
        x = x.repeat(1, 3, 1, 1) # Grayscale to RGB trick
        features = self.resnet_feature_extractor(x)
        features = self.global_avg_pool(features)
        features = features.view(b, t, -1)
        gru_out, _ = self.gru(features)
        last_timestep = gru_out[:, -1, :] 
        out = self.fc(last_timestep)
        return out

# ===============================
# 4. LOAD MODEL & UTILS
# ===============================
@st.cache_resource
def load_model():
    # Cek file lokal
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Sedang mendownload model dari Google Drive... (Harap tunggu ±1 menit)")
        
        # Link Public Google Drive
        url = 'https://drive.google.com/file/d/1t4Z9NwM-LCRAYw5aM9L1jRS0AgNsV2sY/view?usp=drive_link'
        
        try:
            output = gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            if output and os.path.exists(MODEL_PATH):
                st.success(f"✅ Download selesai! (Ukuran: {os.path.getsize(MODEL_PATH)/1e6:.2f} MB)")
            else:
                st.error("❌ Gagal mendownload model.")
                st.stop()
        except Exception as e:
            st.error(f"Terjadi error saat download: {e}")
            st.stop()

    le = get_fixed_labels()
    num_classes = len(le.classes_)
    
    model = ResNetBiGRU(num_classes=num_classes).to(DEVICE)
    
    try:
        # Load Weights (CPU/GPU safe)
        if torch.cuda.is_available():
            checkpoint = torch.load(MODEL_PATH)
        else:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        model.load_state_dict(checkpoint)
        model.eval()
        return model, le
    except Exception as e:
        st.error(f"❌ Error saat meload model: {e}")
        st.stop()

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

# ===============================
# 5. CORE LOGIC (DENGAN TIMER)
# ===============================
def preprocess_and_predict(video_path, model, le):
    # --- MULAI HITUNG WAKTU PREPROCESSING ---
    start_prep = time.time()
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # ROI Extraction
            roi = extract_lip_roi(frame, face_mesh)
            if roi is None: continue
            
            # Grayscale & Resize
            roi = cv2.resize(roi, (IMG_W, IMG_H))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            frames.append(roi)
    cap.release()
    
    # Handle jika wajah tidak ketemu
    if len(frames) == 0:
        return "❌ Wajah tidak terdeteksi!", 0.0, 0.0, 0.0

    # Normalization & Padding/Truncating
    input_arr = np.array(frames) / 255.0
    T = len(input_arr)
    
    if T > TARGET_FRAMES:
        indices = np.linspace(0, T-1, TARGET_FRAMES, dtype=int)
        final_data = input_arr[indices]
    elif T < TARGET_FRAMES:
        padding = np.zeros((TARGET_FRAMES - T, IMG_H, IMG_W), dtype=np.float32)
        final_data = np.concatenate((input_arr, padding), axis=0)
    else:
        final_data = input_arr
        
    tensor = torch.FloatTensor(final_data).unsqueeze(0).to(DEVICE)
    
    end_prep = time.time() # Selesai Preprocessing
    # ----------------------------------------

    # --- MULAI HITUNG WAKTU INFERENSI MODEL ---
    start_inf = time.time()
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probabilities, 1)
        pred_label = le.inverse_transform([pred_idx.item()])[0]
        confidence = max_prob.item() * 100
        
    end_inf = time.time() # Selesai Inferensi
    # ------------------------------------------

    # Hitung durasi
    durasi_prep = end_prep - start_prep
    durasi_inf = end_inf - start_inf
        
    return pred_label, confidence, durasi_prep, durasi_inf

# ===============================
# 6. USER INTERFACE (MAIN)
# ===============================
def main():
    st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.sidebar.title("VSR Skripsi")
    st.sidebar.markdown("**Visual Speech Recognition**")
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Stats")
    st.sidebar.metric("Akurasi (Val)", "95.49%", "+2.5%")
    st.sidebar.metric("WER (Word Error)", "11.18%", "-1.2%")
    st.sidebar.metric("CER (Char Error)", "8.74%", "-0.8%")
    st.sidebar.markdown("---")
    st.sidebar.info("Model: ResNet50V2 + BiGRU\nBackbone: Frozen ResNet\nDataset: Private & LUMINA")

    st.title("🗣️ Sistem Visual Speech Recognition")
    st.markdown("### Demo Identifikasi Kalimat Berbasis Bibir (Lip Reading)")
    
    # Load Model
    with st.spinner("Sedang memuat sistem..."):
        model, le = load_model()
    
    st.success("✅ Model AI Siap Digunakan!")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Video")
        uploaded_file = st.file_uploader("Pilih file video (.mp4)", type=["mp4"])
        video_path = None
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            # Simpan file sementara agar bisa dibaca cv2
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name
            predict_btn = st.button("🔍 Analisis Gerakan Bibir", type="primary")

    with col2:
        st.subheader("2. Hasil Prediksi")
        
        if uploaded_file is not None and predict_btn and video_path:
            with st.spinner("Sedang memproses video (Preprocessing + Inference)..."):
                try:
                    # PANGGIL FUNGSI PREDIKSI DAN TERIMA 4 NILAI
                    result_key, conf, t_prep, t_inf = preprocess_and_predict(video_path, model, le)
                    
                    if "❌" in str(result_key):
                        st.error(result_key)
                    else:
                        st.balloons()
                        st.markdown("##### Model Memprediksi Kalimat:")
                        
                        st.markdown(
                            f"""
                            <div style="background-color:#d4edda;padding:20px;border-radius:10px;border:2px solid #28a745;text-align:center;">
                                <h2 style="color:#155724;margin:0;letter-spacing:1px;">{result_key}</h2>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(f"**Tingkat Keyakinan (Confidence):** `{conf:.2f}%`")

                        # === TAMPILKAN RINCIAN WAKTU (LATENCY) UNTUK SKRIPSI ===
                        total_waktu = t_prep + t_inf
                        st.markdown("---")
                        with st.expander("📊 Detail Waktu Komputasi (Scientific Data)", expanded=True):
                            st.write(f"⏱️ **Total Waktu (End-to-End):** `{total_waktu:.4f} detik`")
                            st.progress(min(1.0, total_waktu/5.0)) # Visual bar
                            
                            c_a, c_b = st.columns(2)
                            with c_a:
                                st.info(f"**Preprocessing:**\n\n`{t_prep:.4f} s`\n\n(MediaPipe + Crop)")
                            with c_b:
                                st.success(f"**Model Inference:**\n\n`{t_inf:.4f} s`\n\n(ResNet50V2-BiGRU)")
                            
                            st.caption("ℹ️ *Catat angka ini untuk Bab 4 (Analisis Waktu Inferensi).*")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                
                finally:
                    # Bersihkan file temp
                    if os.path.exists(video_path):
                        try:
                            time.sleep(1.0) 
                            os.remove(video_path)
                        except:
                            pass
        
        elif uploaded_file is None:
            st.info("👈 Silakan upload video di panel kiri untuk memulai.")

if __name__ == "__main__":
    main()
