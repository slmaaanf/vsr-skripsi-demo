import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DATA ASLI DARI LOG TRAINING KAMU
# ==========================================
# Saya salin manual dari chat history kamu tadi biar valid
epochs = list(range(1, 39)) # Epoch 1 sampai 38

# Data Akurasi (Dalam Persen)
train_acc = [
    58.93, 89.62, 90.49, 91.24, 91.70, 92.61, 93.36, 94.11, 94.86, 95.50, # 1-10
    95.87, 96.45, 96.92, 97.43, 97.73, 98.42, 98.50, 98.79, 99.09, 99.20, # 11-20
    99.48, 99.30, 99.61, 99.54, 99.53, 99.62, 99.85, 99.90, 99.87, 99.86, # 21-30
    99.84, 99.93, 99.91, 99.90, 99.83, 99.92, 99.87, 99.92                # 31-38
]

val_acc = [
    90.04, 90.56, 90.66, 90.89, 91.21, 91.31, 91.87, 91.97, 92.62, 92.55, # 1-10
    92.95, 92.91, 93.60, 93.79, 93.86, 93.53, 94.35, 93.34, 93.83, 95.04, # 11-20
    93.66, 94.61, 94.48, 94.55, 94.68, 94.38, 95.00, 95.49, 94.32, 94.74, # 21-30
    95.17, 94.64, 95.49, 95.07, 95.07, 94.77, 95.49, 95.17                # 31-38
]

# Data Loss (Error Rate)
train_loss = [
    1.6795, 0.4394, 0.3805, 0.3490, 0.3202, 0.2923, 0.2659, 0.2378, 0.2157, 0.1856,
    0.1655, 0.1394, 0.1217, 0.0987, 0.0872, 0.0666, 0.0587, 0.0496, 0.0403, 0.0356,
    0.0234, 0.0288, 0.0192, 0.0189, 0.0197, 0.0169, 0.0086, 0.0056, 0.0074, 0.0061,
    0.0079, 0.0039, 0.0051, 0.0050, 0.0075, 0.0034, 0.0043, 0.0046
]

val_loss = [
    0.4453, 0.3821, 0.3561, 0.3411, 0.3250, 0.3218, 0.3022, 0.2968, 0.2675, 0.2620,
    0.2480, 0.2407, 0.2300, 0.2105, 0.2127, 0.2371, 0.2202, 0.2472, 0.2289, 0.2013,
    0.2424, 0.2091, 0.2206, 0.2173, 0.2146, 0.2459, 0.2001, 0.1964, 0.2215, 0.2329,
    0.1983, 0.2339, 0.1962, 0.2074, 0.2146, 0.2279, 0.1946, 0.2241
]

# ==========================================
# 2. PLOT GRAFIK AKURASI
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Training Accuracy', color='blue', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='red', linewidth=2, linestyle='--')

# Tandai titik tertinggi (Best Model)
best_epoch = 28
best_val = 95.49
plt.scatter(best_epoch, best_val, color='green', s=100, zorder=5)
plt.annotate(f'Best Model\nEpoch {best_epoch}\n{best_val}%', 
             xy=(best_epoch, best_val), xytext=(best_epoch-5, best_val-5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Grafik Peningkatan Akurasi (Training vs Validation)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Akurasi (%)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('grafik_akurasi_final.png', dpi=300)
print("✅ Grafik Akurasi tersimpan: grafik_akurasi_final.png")

# ==========================================
# 3. PLOT GRAFIK LOSS
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='red', linewidth=2, linestyle='--')

plt.title('Grafik Penurunan Loss (Training vs Validation)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('grafik_loss_final.png', dpi=300)
print("✅ Grafik Loss tersimpan: grafik_loss_final.png")