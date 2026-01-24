import os
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import models
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. KONFIGURASI
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"D:\vsr\data\processed"
MODEL_PATH = "vsr_final_hard.pth"
IMG_H, IMG_W = 50, 100

# ===============================
# 2. DEFINISI ULANG ARSITEKTUR (WAJIB SAMA PERSIS)
# ===============================
class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(ResNetBiGRU, self).__init__()
        resnet = models.resnet50(weights=None) # Tidak perlu download weight lagi
        
        # Bekukan bobot tidak diperlukan saat testing, tapi struktur harus sama
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
# 3. FUNGSI LOAD LABEL & MODEL
# ===============================
def get_label_encoder():
    # Kita harus scan ulang folder untuk mendapatkan urutan kelas yang SAMA
    labels = []
    print("📂 Sedang menyusun daftar kelas (Label Encoder)...")
    
    # Scan Private
    priv_dir = os.path.join(DATA_ROOT, "private")
    if os.path.exists(priv_dir):
        for f in os.listdir(priv_dir):
            if f.endswith('.npy'):
                # Logika nama file: "bukapintu_001_aug0.npy"
                parts = f.replace('.npy', '').split('_')
                # Filter kata kunci (logic sama dengan training)
                cleaned = [p for p in parts if p not in ['female', 'male', 'orig'] and not p.startswith('aug')]
                if cleaned: labels.append(cleaned[0])

    # Scan Public (Jika ada)
    pub_dir = os.path.join(DATA_ROOT, "public")
    if os.path.exists(pub_dir):
         for f in os.listdir(pub_dir):
            if f.endswith('.npy'):
                parts = f.replace('.npy', '').split('_')
                cleaned = [p for p in parts if p not in ['female', 'male', 'orig'] and not p.startswith('aug')]
                if cleaned: labels.append(cleaned[0])
    
    le = LabelEncoder()
    le.fit(labels)
    return le

def load_inference_model(num_classes):
    print(f"🏗️ Memuat model dari {MODEL_PATH}...")
    model = ResNetBiGRU(num_classes=num_classes)
    
    # Load Weights
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval() # Mode Evaluasi (Penting!)
    return model

# ===============================
# 4. JALANKAN TEST
# ===============================
def run_test():
    # 1. Siapkan Label
    le = get_label_encoder()
    num_classes = len(le.classes_)
    print(f"✅ Terdeteksi {num_classes} Kelas.")
    
    # 2. Siapkan Model
    model = load_inference_model(num_classes)
    
    # 3. Ambil Sampel Acak dari Folder Private
    priv_dir = os.path.join(DATA_ROOT, "private")
    all_files = [f for f in os.listdir(priv_dir) if f.endswith('.npy')]
    
    # Ambil 10 file acak
    test_samples = random.sample(all_files, 10)
    
    print("\n" + "="*80)
    print(f"{'FILE (SAMPEL ACAK)':<40} | {'PREDIKSI MODEL':<30} | {'STATUS':<10}")
    print("="*80)
    
    correct_count = 0
    
    with torch.no_grad(): # Matikan gradien biar hemat memori
        for fname in test_samples:
            # A. Load Data
            path = os.path.join(priv_dir, fname)
            data = np.load(path) # (150, 50, 100)
            tensor = torch.FloatTensor(data).unsqueeze(0).to(DEVICE) # Tambah batch dim -> (1, 150, 50, 100)
            
            # B. Prediksi
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = le.inverse_transform([predicted_idx.item()])[0]
            
            # C. Cek Jawaban Benar (Dari nama file)
            parts = fname.replace('.npy', '').split('_')
            cleaned = [p for p in parts if p not in ['female', 'male', 'orig'] and not p.startswith('aug')]
            actual_label = cleaned[0] if cleaned else "unknown"
            
            # D. Tampilkan
            status = "✅ BENAR" if predicted_label == actual_label else "❌ SALAH"
            if predicted_label == actual_label: correct_count += 1
            
            # Potong nama file biar muat di tabel
            disp_name = (fname[:35] + '..') if len(fname) > 35 else fname
            
            print(f"{disp_name:<40} | {predicted_label:<30} | {status}")
            
    print("="*80)
    print(f"📊 SKOR TESTING SEMENTARA: {correct_count}/10 ({correct_count*10}%)")
    
    if correct_count >= 8:
        print("🎉 HASIL LUAR BIASA! Model siap didemokan.")
    elif correct_count >= 5:
        print("🙂 HASIL CUKUP BAIK. Mungkin perlu tuning lagi.")
    else:
        print("⚠️ PERLU CEK ULANG. Akurasi rendah pada data tes.")

if __name__ == "__main__":
    run_test()