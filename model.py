import torch
import torch.nn as nn
import torchvision.models as models

class VSRModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, dropout_rate=0.5):
        super(VSRModel, self).__init__()

        # ====================================================
        # 1. LOAD PRE-TRAINED RESNET50
        # ====================================================
        print("🧠 Loading ResNet50 Backbone (Pre-trained)...")
        # Menggunakan bobot yang sudah terlatih pada jutaan gambar
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        # ====================================================
        # 2. MODIFIKASI INPUT LAYER (PENTING!)
        # ====================================================
        # ResNet asli: Input 3 Channel (RGB)
        # Data Kita  : Input 1 Channel (Grayscale)
        # Kita harus mengoperasi layer pertamanya.
        
        original_weight = resnet.conv1.weight.data # Shape: [64, 3, 7, 7]
        
        # Buat layer baru dengan input channel = 1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # TEKNIK PINTAR: Rata-rata bobot RGB
        # Kita ambil rata-rata bobot RGB agar pengetahuan pre-trained tidak hilang.
        # Shape berubah dari [64, 3, 7, 7] -> [64, 1, 7, 7]
        new_conv1.weight.data = torch.mean(original_weight, dim=1, keepdim=True)
        
        # Pasang layer baru ke ResNet
        resnet.conv1 = new_conv1

        # ====================================================
        # 3. FREEZE & UNFREEZE STRATEGY
        # ====================================================
        # Freeze semua layer dulu (biar hemat memori & waktu)
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze Layer Terakhir (Layer 4)
        # Bagian ini yang akan belajar fitur spesifik "Gerakan Bibir"
        for param in resnet.layer4.parameters():
            param.requires_grad = True
            
        # Hapus layer classification bawaan ResNet (FC layer)
        # Kita cuma butuh fitur ekstraktornya
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Pooling layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ====================================================
        # 4. TEMPORAL MODELING (RNN - BiGRU)
        # ====================================================
        # ResNet50 output size = 2048
        self.gru = nn.GRU(
            input_size=2048, 
            hidden_size=hidden_size, 
            num_layers=2,           
            batch_first=True, 
            bidirectional=True,     
            dropout=0.2 if dropout_rate > 0 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)

        # ====================================================
        # 5. CLASSIFIER HEAD
        # ====================================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes) 
        )

    def forward(self, x):
        # Input: (Batch, Channel=1, Time, Height, Width)
        b, c, t, h, w = x.size()
        
        # --- 1. TimeDistributed Wrapper ---
        # Kita gabungkan Batch dan Time agar bisa diproses CNN 2D
        # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.transpose(1, 2).contiguous() 
        x = x.view(b * t, c, h, w)         

        # --- 2. CNN Backbone (ResNet) ---
        x = self.backbone(x)      # Output: (B*T, 2048, H_kecil, W_kecil)
        x = self.gap(x)           # Output: (B*T, 2048, 1, 1)
        x = x.view(x.size(0), -1) # Flatten -> (B*T, 2048)

        # --- 3. RNN Processing ---
        # Kembalikan dimensi waktu untuk masuk ke GRU
        # (B*T, 2048) -> (B, T, 2048)
        x = x.view(b, t, -1)
        
        gru_out, _ = self.gru(x)
        
        # Ambil output timestep terakhir
        last_output = gru_out[:, -1, :] 
        last_output = self.dropout(last_output)

        # --- 4. Classification ---
        logits = self.classifier(last_output)
        
        return logits

# Debugging (Opsional)
if __name__ == "__main__":
    print("Testing Model Shape...")
    model = VSRModel(num_classes=10)
    # Dummy input: Batch=2, Channel=1, Frames=5, H=50, W=100
    dummy = torch.randn(2, 1, 5, 50, 100) 
    out = model(dummy)
    print(f"Output Shape: {out.shape} (Harus [2, 10])")
    print("✅ Model Ready!")