import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from torchvision import models
from sklearn.preprocessing import LabelEncoder

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vsr_final_hard.pth"
IMG_H, IMG_W = 50, 100
TARGET_FRAMES = 150
PADDING = 10

mp_face_mesh = mp.solutions.face_mesh
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# ===============================
# 1. ARSITEKTUR MODEL (SAMA PERSIS)
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

# ===============================
# 2. FUNGSI PREPROCESSING
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

def preprocess_video(video_path):
    print(f"🔄 Memproses video: {os.path.basename(video_path)}")
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
        print("❌ Wajah tidak terdeteksi!")
        return None

    # Normalisasi & Padding
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
        
    return torch.FloatTensor(final_data).unsqueeze(0).to(DEVICE)

# ===============================
# 3. DAFTAR KELAS (HARDCODED DARI LOG TRAINING)
# ===============================
def get_fixed_labels():
    # INI DAFTAR KELAS ASLI SAAT TRAINING (112 Kelas)
    # Termasuk 'P01', 'P03' dkk agar urutan index cocok dengan bobot model
    raw_labels = [
        'P01', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11', 'P12', 'P13', 'P14',
        'aktifkanmodepesawatsekarang', 'apakahkamusudahmengerjakanpr',
        'bagaimanacaramasaknasigoreng', 'bajuwarnabiruitubagus',
        'bebekbekubapakbibibau', 'berapasuhuudarahariini',
        'beratbadansayaenampuluhkilo', 'berhentimemutarvideoini',
        'buayamembawabungkusanbaru', 'bukaaplikasikameradepan',
        'bukakuncipintudepan', 'bukuinisangatbagusisinya',
        'carirestoranpadangterdekat', 'cicakcicakdidindingdiam',
        'cobajelaskansekalilagipelanpelan', 'cuacahariinisangatpanasya',
        'diasedangbelajarbahasapemrogramanpythondasar',
        'dimanakamusimpankuncimotor', 'enamtujuhdelapansembilansepuluh',
        'fififotovasbungaungu', 'filmbioskopterbaruitusangatserusekali',
        'geserkeslideberikutnya', 'gigigusigigigusigeraham',
        'haloapakabarkamuhariini', 'hapusfileyangtidakperlu',
        'hargabahanpokoksedangnaikdipasartradisional',
        'harganyalimapuluhriburupiah', 'hatihatidijalanpulangya',
        'hewanharimauharushidupdihutan',
        'hujanderasmenyebabkanbanjirdibeberapadaerah',
        'iniadalahkalimatterakhirdaridatasetsaya', 'ituideyangsangatcemerlang',
        'janganbuangsampahsembarangandong', 'janganlupamakansiangnanti',
        'janganlupamematikankomporsebelumpergikeluar',
        'jaraknyasekitarduaratusmeter', 'jaringaninternetdisinisedangsangatlambat',
        'jauhjalanjangankanjatuh', 'kapankitabisapergiliburan',
        'kembalikehalamanutama', 'kenapakamudiamsajadaritadi',
        'keretaapiberangkattepatpukuldelapanpagi', 'kirimpesankegrupkantor',
        'kitabertemujamtujuhmalam', 'kitaharusmenjagakesehatantubuh',
        'kitaperludiskusikelompokuntuktugasbesar',
        'kopiinirasanyaterlalumanisuntuksaya',
        'kucingitutidurpulasdiatassofaempuk', 'kukukakikakekkakukaku',
        'laptopsayakehabisanbateraisaatsedangrapat', 'maafsayadatangterlambattadi',
        'mahasiswasedangmelakukanpenelitiandilaboratorium',
        'mamamakanmanggamanismalammalam', 'masukkankodeverifikasiempatdigit',
        'matikantelevisisekarang', 'mungkinnantisorehujanturun',
        'naikkanvolumesuarasedikit', 'nanyanyonyanyanyinyaringnian',
        'nomorantriankamuadalahduabelas', 'nomorteleponnyaacaksekali',
        'nyalakanlampukamarmandi', 'olahragapagisangatbaikuntukkesehatanjantung',
        'papapulangbawapepayapipih', 'pasangalarmjamlimapagi',
        'pemandangandipantaiitusungguhmempesonamata',
        'pemerintahmengumumkanliburnasionalbesoklusa',
        'perkembanganteknologikecerdasanbuatansangatpesat',
        'putarmusikfavoritsaya', 'raralariluruslalulupa',
        'sampaijumpabesokdikampus', 'satuduatigaempatlima',
        'sayabelumpernahkesanasebelumnya', 'sayabutuhtigalembarkertas',
        'sayaharusmenyelesaikanskripsibulandepanpasti', 'sayainginminumkopidingin',
        'sayalahirtahunsembilanpuluhan', 'sayamerasasangatlelahsekali',
        'sayasedangmengerjakantugasakhir', 'sayasenangbertemudengankamu',
        'sayasetujudenganpendapatkamu', 'sayasukasatesapisurabaya',
        'sayatidakmengertimaksudkamu', 'selamatpagisemuanyasemogasukses',
        'selamatulangtahunsemogapanjangumursehatselalu',
        'siapayangmengambilkuesaya', 'simpandokumeninisegera',
        'sudahjamduabelassianglewat', 'tangkaplayarhalamanini',
        'teleponibusekarangjuga', 'terimakasihbanyakatasbantuannya',
        'tinggalsepuluhdetiklagi', 'tokoitututuptujuhtahun',
        'tolongambilkanminumdimeja', 'tolongbeliduabotolair',
        'tolongsampaikansalamsayakepadaorangtuamu',
        'tolongtutuppintunyapelanpelan', 'tunjukkanjalankerumahsakit',
        'turunkankecerahanlayarhp', 'ulangtahunsayatanggaltigamaret',
        'ularmelingkardipagarbundar'
    ]
    
    le = LabelEncoder()
    le.fit(raw_labels) # Fit dengan daftar manual ini
    return le

# ===============================
# 4. MAIN
# ===============================
if __name__ == "__main__":
    # --- GANTI PATH VIDEO DI SINI ---
    target_video = r"D:\vsr\data\raw\private\005.mp4"
    # --------------------------------

    if not os.path.exists(target_video):
        print(f"❌ File tidak ditemukan: {target_video}")
        exit()

    # 1. Load Labels (Fixed)
    le = get_fixed_labels()
    num_classes = len(le.classes_) # Harus 112
    print(f"✅ Menggunakan {num_classes} Kelas (Sesuai Log Training).")
    
    # 2. Load Model
    print("🏗️ Memuat Model...")
    model = ResNetBiGRU(num_classes=num_classes).to(DEVICE)
    
    # Load weights (Safe Mode)
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Preprocess MP4
    input_tensor = preprocess_video(target_video)
    
    if input_tensor is not None:
        # 4. Predict
        print("🧠 Sedang memprediksi...")
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            pred_label = le.inverse_transform([pred_idx.item()])[0]
            
        print("\n" + "="*50)
        print(f"🎬 VIDEO: {os.path.basename(target_video)}")
        print(f"🗣️ HASIL PREDIKSI: {pred_label.upper()}")
        print("="*50)