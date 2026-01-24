import os
import torch
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vsr_final_hard.pth"
DATA_ROOT = r"D:\vsr\data\processed"
CSV_PATH = "private.csv"

# ===============================
# 1. RUMUS WER & CER (Levenshtein)
# ===============================
def levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): d[i][0] = i
    for j in range(n + 1): d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,      # deletion
                          d[i][j - 1] + 1,      # insertion
                          d[i - 1][j - 1] + cost) # substitution
    return d[m][n]

def calculate_wer(reference, hypothesis):
    # Hitung error per kata
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words) if len(ref_words) > 0 else 0.0

def calculate_cer(reference, hypothesis):
    # Hitung error per karakter (tanpa spasi)
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars) if len(ref_chars) > 0 else 0.0

# ===============================
# 2. LOAD DATA & MAPPING TEKS
# ===============================
def load_text_mapping():
    print("📖 Membaca Private.csv untuk Ground Truth teks...")
    mapping = {}
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, sep='|', header=None, names=['filename', 'label'])
        for _, row in df.iterrows():
            # Kunci: Label bersih (tanpa spasi/simbol) -> dipakai model
            # Nilai: Kalimat asli (dengan spasi) -> dipakai WER
            clean = re.sub(r'[^a-zA-Z0-9]', '', str(row['label']).lower())
            original = str(row['label']).strip()
            mapping[clean] = original
    return mapping

# ===============================
# 3. MODEL (Arsitektur Tetap)
# ===============================
class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(ResNetBiGRU, self).__init__()
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-2] 
        self.resnet_feature_extractor = nn.Sequential(*modules)
        self.resnet_out_features = 2048
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(input_size=self.resnet_out_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
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
# 4. MAIN EVALUATION
# ===============================
def evaluate():
    # A. Setup Labels & Data Split
    # Kita gunakan Logic Hardcoded 112 Kelas agar cocok dengan model
    raw_labels = [
        'P01', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11', 'P12', 'P13', 'P14',
        'aktifkanmodepesawatsekarang', 'apakahkamusudahmengerjakanpr', 'bagaimanacaramasaknasigoreng', 'bajuwarnabiruitubagus', 'bebekbekubapakbibibau', 'berapasuhuudarahariini', 'beratbadansayaenampuluhkilo', 'berhentimemutarvideoini', 'buayamembawabungkusanbaru', 'bukaaplikasikameradepan', 'bukakuncipintudepan', 'bukuinisangatbagusisinya', 'carirestoranpadangterdekat', 'cicakcicakdidindingdiam', 'cobajelaskansekalilagipelanpelan', 'cuacahariinisangatpanasya', 'diasedangbelajarbahasapemrogramanpythondasar', 'dimanakamusimpankuncimotor', 'enamtujuhdelapansembilansepuluh', 'fififotovasbungaungu', 'filmbioskopterbaruitusangatserusekali', 'geserkeslideberikutnya', 'gigigusigigigusigeraham', 'haloapakabarkamuhariini', 'hapusfileyangtidakperlu', 'hargabahanpokoksedangnaikdipasartradisional', 'harganyalimapuluhriburupiah', 'hatihatidijalanpulangya', 'hewanharimauharushidupdihutan', 'hujanderasmenyebabkanbanjirdibeberapadaerah', 'iniadalahkalimatterakhirdaridatasetsaya', 'ituideyangsangatcemerlang', 'janganbuangsampahsembarangandong', 'janganlupamakansiangnanti', 'janganlupamematikankomporsebelumpergikeluar', 'jaraknyasekitarduaratusmeter', 'jaringaninternetdisinisedangsangatlambat', 'jauhjalanjangankanjatuh', 'kapankitabisapergiliburan', 'kembalikehalamanutama', 'kenapakamudiamsajadaritadi', 'keretaapiberangkattepatpukuldelapanpagi', 'kirimpesankegrupkantor', 'kitabertemujamtujuhmalam', 'kitaharusmenjagakesehatantubuh', 'kitaperludiskusikelompokuntuktugasbesar', 'kopiinirasanyaterlalumanisuntuksaya', 'kucingitutidurpulasdiatassofaempuk', 'kukukakikakekkakukaku', 'laptopsayakehabisanbateraisaatsedangrapat', 'maafsayadatangterlambattadi', 'mahasiswasedangmelakukanpenelitiandilaboratorium', 'mamamakanmanggamanismalammalam', 'masukkankodeverifikasiempatdigit', 'matikantelevisisekarang', 'mungkinnantisorehujanturun', 'naikkanvolumesuarasedikit', 'nanyanyonyanyanyinyaringnian', 'nomorantriankamuadalahduabelas', 'nomorteleponnyaacaksekali', 'nyalakanlampukamarmandi', 'olahragapagisangatbaikuntukkesehatanjantung', 'papapulangbawapepayapipih', 'pasangalarmjamlimapagi', 'pemandangandipantaiitusungguhmempesonamata', 'pemerintahmengumumkanliburnasionalbesoklusa', 'perkembanganteknologikecerdasanbuatansangatpesat', 'putarmusikfavoritsaya', 'raralariluruslalulupa', 'sampaijumpabesokdikampus', 'satuduatigaempatlima', 'sayabelumpernahkesanasebelumnya', 'sayabutuhtigalembarkertas', 'sayaharusmenyelesaikanskripsibulandepanpasti', 'sayainginminumkopidingin', 'sayalahirtahunsembilanpuluhan', 'sayamerasasangatlelahsekali', 'sayasedangmengerjakantugasakhir', 'sayasenangbertemudengankamu', 'sayasetujudenganpendapatkamu', 'sayasukasatesapisurabaya', 'sayatidakmengertimaksudkamu', 'selamatpagisemuanyasemogasukses', 'selamatulangtahunsemogapanjangumursehatselalu', 'siapayangmengambilkuesaya', 'simpandokumeninisegera', 'sudahjamduabelassianglewat', 'tangkaplayarhalamanini', 'teleponibusekarangjuga', 'terimakasihbanyakatasbantuannya', 'tinggalsepuluhdetiklagi', 'tokoitututuptujuhtahun', 'tolongambilkanminumdimeja', 'tolongbeliduabotolair', 'tolongsampaikansalamsayakepadaorangtuamu', 'tolongtutuppintunyapelanpelan', 'tunjukkanjalankerumahsakit', 'turunkankecerahanlayarhp', 'ulangtahunsayatanggaltigamaret', 'ularmelingkardipagarbundar'
    ]
    le = LabelEncoder()
    le.fit(raw_labels)

    # Load All Files
    all_files = []
    labels = []
    priv_dir = os.path.join(DATA_ROOT, "private")
    for f in os.listdir(priv_dir):
        if f.endswith('.npy'):
            parts = f.replace('.npy', '').split('_')
            cleaned = [p for p in parts if p not in ['female', 'male', 'orig'] and not p.startswith('aug')]
            if cleaned: 
                all_files.append(os.path.join(priv_dir, f))
                labels.append(cleaned[0])

    # Gunakan split yang sama dengan training (random_state=42)
    # Kita evaluasi pada X_val (Data yang tidak dilihat saat training)
    _, X_test, _, y_test = train_test_split(all_files, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"📊 Mengevaluasi pada {len(X_test)} sampel data validasi/test...")

    # B. Load Model
    model = ResNetBiGRU(num_classes=len(le.classes_)).to(DEVICE)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # C. Load Text Mapping (buayamembawa... -> Buaya membawa...)
    text_map = load_text_mapping()

    # D. Loop Prediction
    total_wer = 0.0
    total_cer = 0.0
    count = 0
    
    print("\n" + "="*90)
    print(f"{'PREDIKSI (RAW)':<30} | {'GROUND TRUTH (FULL)':<40} | {'WER':<5} | {'CER':<5}")
    print("="*90)

    with torch.no_grad():
        for i, fpath in enumerate(X_test):
            # Load Data
            data = np.load(fpath)
            tensor = torch.FloatTensor(data).unsqueeze(0).to(DEVICE)
            
            # Predict
            outputs = model(tensor)
            _, pred_idx = torch.max(outputs, 1)
            pred_class_raw = le.inverse_transform([pred_idx.item()])[0] # ex: bukakuncipintudepan
            true_class_raw = y_test[i] # ex: bukakuncipintudepan
            
            # Convert to Full Sentence
            # Jika tidak ada di CSV, pakai raw string saja
            pred_text = text_map.get(pred_class_raw, pred_class_raw)
            true_text = text_map.get(true_class_raw, true_class_raw)
            
            # Calculate Metrics
            wer = calculate_wer(true_text, pred_text)
            cer = calculate_cer(true_text, pred_text)
            
            total_wer += wer
            total_cer += cer
            count += 1
            
            # Print Sampel (10 Pertama saja biar gak nyampah)
            if i < 10:
                print(f"{pred_class_raw[:30]:<30} | {true_text[:40]:<40} | {wer:.2f}  | {cer:.2f}")

    # E. Final Results
    avg_wer = (total_wer / count) * 100
    avg_cer = (total_cer / count) * 100
    
    print("="*90)
    print(f"📈 HASIL AKHIR (Rata-rata dari {count} sampel):")
    print(f"   🎯 Average WER (Word Error Rate):      {avg_wer:.2f}%")
    print(f"   🎯 Average CER (Character Error Rate): {avg_cer:.2f}%")
    print("="*90)
    print("Analisis:")
    print("- WER/CER mendekati 0% semakin bagus.")
    print("- Karena akurasi klasifikasi tinggi (~95%), WER/CER harusnya sangat rendah.")

if __name__ == "__main__":
    evaluate()