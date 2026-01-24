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

# ===============================
# UPDATE BAGIAN LOAD MODEL INI
# ===============================
@st.cache_resource
def load_model():
    # Cek apakah file ada? Jika tidak, download dari Drive
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Sedang mendownload model dari Cloud (Google Drive)... Mohon tunggu ±1 menit.")
        
        # ID File dari Link Drive kamu
        file_id = '1t4Z9NwM-LCRAYw5aM9L1jRS0AgNsV2sY'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            output = gdown.download(url, MODEL_PATH, quiet=False)
            if output:
                st.success("✅ Download selesai!")
            else:
                st.error("❌ Gagal download. Cek permission Google Drive.")
                st.stop()
        except Exception as e:
            st.error(f"Error saat download: {e}")
            st.stop()

    # Load Label
    le = get_fixed_labels()
    num_classes = len(le.classes_)
    
    # Load Arsitektur
    model = ResNetBiGRU(num_classes=num_classes).to(DEVICE)
    
    # Load Weights
    # Di Cloud biasanya pakai CPU, jadi map_location='cpu' wajib
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    model.load_state_dict(checkpoint)
    model.eval()
    return model, le

# ===============================
# 1. KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="VSR System Demo",
    page_icon="👄",
    layout="wide"
)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vsr_final_hard.pth"
CSV_PATH = "private.csv"  # Pastikan file ini ada di folder yang sama
IMG_H, IMG_W = 50, 100
TARGET_FRAMES = 150
PADDING = 10

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ===============================
# 2. LOGIC MAPPING KALIMAT (BARU)
# ===============================
@st.cache_data
def load_sentence_map():
    """Membaca CSV untuk mengubah 'bukapintu' menjadi 'Buka pintu'"""
    mapping = {}
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, sep='|', header=None, names=['filename', 'label'])
            for _, row in df.iterrows():
                # Bersihkan label agar cocok dengan output model (tanpa spasi/simbol)
                # Contoh: "Buka pintu." -> "bukapintu"
                clean_key = re.sub(r'[^a-zA-Z0-9]', '', str(row['label']).lower())
                
                # Simpan kalimat aslinya
                original_text = str(row['label']).strip()
                mapping[clean_key] = original_text
        except Exception as e:
            st.error(f"Gagal membaca CSV mapping: {e}")
    return mapping

# ===============================
# 3. ARSITEKTUR MODEL
# ===============================
class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(ResNetBiGRU, self).__init__()
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-2] 
        self.resnet_feature_extractor = nn.Sequential(*modules)
        self.resnet_out_features = 2048
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gru = nn.GRU(
            input_size=self.resnet_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        b, t, h, w = x.size()
        x = x.view(b * t, 1, h, w)
        x = x.repeat(1, 3, 1, 1)
        features = self.resnet_feature_extractor(x)
        features = self.global_avg_pool(features)
        features = features.view(b, t, -1)
        gru_out, _ = self.gru(features)
        last_timestep = gru_out[:, -1, :] 
        out = self.fc(last_timestep)
        return out

def get_fixed_labels():
    # Daftar 112 Kelas (Wajib sama dengan training)
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

@st.cache_resource
def load_model():
    le = get_fixed_labels()
    num_classes = len(le.classes_)
    
    model = ResNetBiGRU(num_classes=num_classes).to(DEVICE)
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    model.load_state_dict(checkpoint)
    model.eval()
    return model, le

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

def preprocess_and_predict(video_path, model, le):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret: break
            roi = extract_lip_roi(frame, face_mesh)
            if roi is None: continue
            
            roi = cv2.resize(roi, (IMG_W, IMG_H))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            frames.append(roi)
    cap.release()
    
    if len(frames) == 0:
        return "❌ Wajah tidak terdeteksi dalam video!", 0.0

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
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probabilities, 1)
        pred_label = le.inverse_transform([pred_idx.item()])[0]
        confidence = max_prob.item() * 100
        
    return pred_label, confidence

# ===============================
# 4. USER INTERFACE
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
    st.sidebar.info("Model: ResNet50V2 + BiGRU\nDataset: Private & LUMINA")

    st.title("🗣️ Sistem Visual Speech Recognition")
    st.markdown("### Demo Identifikasi Kalimat Berbasis Bibir (Lip Reading)")
    
    # Load Model & Sentence Map
    with st.spinner("Sedang memuat model AI..."):
        model, le = load_model()
        sentence_map = load_sentence_map() # Load Kamus
    
    st.success("✅ Model AI Siap Digunakan!")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Video")
        uploaded_file = st.file_uploader("Pilih file video (.mp4)", type=["mp4"])
        video_path = None
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name
            predict_btn = st.button("🔍 Analisis Gerakan Bibir", type="primary")

    with col2:
        st.subheader("2. Hasil Prediksi")
        
        if uploaded_file is not None and predict_btn and video_path:
            with st.spinner("Sedang memproses video..."):
                try:
                    result_key, conf = preprocess_and_predict(video_path, model, le)
                    
                    if "❌" in str(result_key):
                        st.error(result_key)
                    else:
                        # --- MODIFIKASI TAMPILAN ---
                        # Ambil kalimat asli dari map. Jika tidak ada, pakai hasil raw.
                        final_sentence = sentence_map.get(result_key, result_key)
                        
                        st.balloons()
                        st.markdown("##### Model Memprediksi Kalimat:")
                        
                        # Tampilan HTML dipercantik dengan Letter Spacing
                        st.markdown(
                            f"""
                            <div style="background-color:#d4edda;padding:20px;border-radius:10px;border:2px solid #28a745;text-align:center;">
                                <h2 style="color:#155724;margin:0;letter-spacing:1px;">{final_sentence}</h2>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(f"**Tingkat Keyakinan (Confidence):** `{conf:.2f}%`")
                        
                        # Debugging (Opsional, biar tau ID nya apa)
                        with st.expander("Lihat Detail Teknis"):
                            st.text(f"Raw Class ID: {result_key}")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                
                finally:
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