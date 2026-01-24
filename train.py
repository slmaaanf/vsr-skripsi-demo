import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===============================
# 1. KONFIGURASI (CONFIG)
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"D:\vsr\data\processed"
MODEL_SAVE_PATH = "vsr_final_hard.pth"

BATCH_SIZE = 8       # Kurangi jadi 4 jika VRAM GPU penuh
EPOCHS = 100         # Kita set 100, nanti ada Early Stopping
LEARNING_RATE = 1e-4 # Learning rate kecil karena Fine Tuning
IMG_H, IMG_W = 50, 100
SEQ_LEN = 150        # Frame

# ===============================
# 2. DATASET LOADER
# ===============================
class VSRDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load NPY (Sequence x H x W)
        path = self.file_paths[idx]
        data = np.load(path) # Shape: (150, 50, 100)
        
        # Konversi ke Tensor PyTorch
        # Input perlu (Channel, Sequence, H, W) -> PyTorch convention agak beda
        # Tapi nanti di model kita handle. Kirim raw dulu.
        tensor_data = torch.FloatTensor(data)
        
        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label

def load_data_and_labels():
    print("📂 Sedang memuat dataset...")
    files = []
    labels = []
    
    # 1. Load Private & Public
    for folder in ["private", "public"]:
        dir_path = os.path.join(DATA_ROOT, folder)
        if not os.path.exists(dir_path): continue
        
        for f in os.listdir(dir_path):
            if f.endswith('.npy'):
                files.append(os.path.join(dir_path, f))
                
                # --- LOGIKA LABEL OTOMATIS ---
                # Asumsi nama file: "kata_001.npy" atau "female_kata_001.npy"
                # Kita harus ambil "kata"-nya saja sebagai label.
                # Sesuaikan logika ini dengan nama file aslimu!
                parts = f.replace('.npy', '').split('_')
                
                # Contoh filter sederhana:
                # Jika format: "female_buka_01.npy" -> label "buka"
                # Jika format: "buka_001.npy" -> label "buka"
                
                # (LOGIKA SEMENTARA: Ambil kata pertama yang bukan 'female'/'male'/'aug')
                cleaned_parts = [p for p in parts if p not in ['female', 'male', 'orig'] and not p.startswith('aug')]
                
                if cleaned_parts:
                    label_name = cleaned_parts[0] # Ambil kata kuncinya
                else:
                    label_name = "unknown"
                    
                labels.append(label_name)
    
    # Encode Label (String -> Angka)
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    
    print(f"✅ Total Data: {len(files)}")
    print(f"✅ Jumlah Kelas: {len(le.classes_)}")
    print(f"   Kelas: {le.classes_}")
    
    return files, labels_enc, len(le.classes_), le

# ===============================
# 3. ARSITEKTUR MODEL (ResNet50 + BiGRU)
# ===============================
class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(ResNetBiGRU, self).__init__()
        
        # A. BACKBONE: ResNet50 (Pre-trained)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Bekukan bobot (Frozen)
        for param in resnet.parameters():
            param.requires_grad = False
            
        # Kita buang layer klasifikasi terakhir (fc) dan pooling terakhir
        modules = list(resnet.children())[:-2] 
        self.resnet_feature_extractor = nn.Sequential(*modules)
        
        # ResNet output channels = 2048
        self.resnet_out_features = 2048
        
        # Global Average Pooling (Spatial -> 1D Vector per frame)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # B. TEMPORAL: Bi-GRU
        # Input size = 2048 (fitur dari ResNet)
        self.gru = nn.GRU(
            input_size=self.resnet_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, # BiGRU
            dropout=0.3
        )
        
        # C. CLASSIFICATION HEAD
        # Hidden size * 2 karena Bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Sequence, Height, Width) -> (B, 150, 50, 100)
        b, t, h, w = x.size()
        
        # 1. TimeDistributed Wrapper Logic
        # Gabungkan Batch dan Sequence biar bisa masuk CNN
        x = x.view(b * t, 1, h, w) # (B*T, 1, H, W)
        
        # Trik: ResNet butuh 3 channel (RGB), data kita Grayscale (1 channel)
        # Kita duplikasi channel 1 jadi 3
        x = x.repeat(1, 3, 1, 1) # (B*T, 3, H, W)
        
        # 2. Spatial Feature Extraction (ResNet)
        features = self.resnet_feature_extractor(x) # (B*T, 2048, H', W')
        features = self.global_avg_pool(features)   # (B*T, 2048, 1, 1)
        features = features.view(b, t, -1)          # (B, T, 2048) -> Kembali ke Sequence
        
        # 3. Temporal Sequence Modelling (BiGRU)
        # output shape: (Batch, Sequence, Hidden*2)
        # hidden shape: (NumLayers*2, Batch, Hidden)
        gru_out, _ = self.gru(features)
        
        # Ambil output dari timestep terakhir saja untuk klasifikasi
        # Atau bisa pakai Global Average Pooling di dimensi waktu
        # Di sini kita pakai output terakhir (umum untuk sequence classification)
        last_timestep = gru_out[:, -1, :] 
        
        # 4. Classification Head
        out = self.fc(last_timestep)
        return out

# ===============================
# 4. MAIN TRAINING LOOP
# ===============================
def train_model():
    print(f"🚀 Device: {DEVICE}")
    
    # 1. Load Data
    files, labels, num_classes, le = load_data_and_labels()
    
    # Split Train/Val (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
    
    train_dataset = VSRDataset(X_train, y_train)
    val_dataset = VSRDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Inisialisasi Model
    print("🏗️ Membangun Model ResNet50V2 + BiGRU...")
    model = ResNetBiGRU(num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 3. Training Loop
    best_acc = 0.0
    patience = 10  # Early Stopping Patience
    patience_counter = 0
    
    print(f"🔥 Mulai Training ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- TRAIN ---
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
        
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        
        print(f"   📝 Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   📝 Val Loss:   {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # --- EARLY STOPPING & SAVE BEST ---
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 Model disimpan! (New Best Acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print("🛑 Early Stopping triggered! Training selesai.")
            break
            
    print(f"🎉 Selesai! Model tersimpan di: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()