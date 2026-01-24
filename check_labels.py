import os
import collections

DATA_ROOT = r"D:\vsr\data\processed"

def check_labels():
    print("🕵️‍♂️ Memeriksa Label Dataset...")
    
    labels = []
    sample_filenames = []
    
    # Cek folder Private
    private_dir = os.path.join(DATA_ROOT, "private")
    if os.path.exists(private_dir):
        files = [f for f in os.listdir(private_dir) if f.endswith('.npy')]
        for f in files:
            # Logika ekstraksi label dari nama file
            parts = f.replace('.npy', '').split('_')
            # Filter kata 'orig', 'aug', 'female', 'male'
            cleaned = [p for p in parts if p not in ['female', 'male', 'orig'] and not p.startswith('aug')]
            
            if cleaned:
                label = cleaned[0] # Mengambil kata pertama sebagai label
                labels.append(label)
                if len(sample_filenames) < 5: sample_filenames.append((f, label))
    
    # Hitung Statistik
    counter = collections.Counter(labels)
    unique_labels = list(counter.keys())
    
    print(f"\n✅ Total File Terbaca: {len(labels)}")
    print(f"📊 Jumlah Kelas (Kata Unik): {len(unique_labels)}")
    
    print("\n--- 5 Sampel Deteksi Label ---")
    for fname, lbl in sample_filenames:
        print(f"📄 File: {fname}  -->  🏷️ Label: {lbl}")
        
    print("\n--- 10 Label Terbanyak ---")
    print(counter.most_common(10))
    
    print("\n⚠️ KESIMPULAN:")
    if len(unique_labels) > 50 and len(labels) < 2000:
        print("❌ BAHAYA: Terlalu banyak kelas unik! Sepertinya nama file masih berupa angka (001, 002).")
        print("   JANGAN TRAINING DULU. Model tidak akan belajar apapun.")
    else:
        print("✅ AMAN: Label terlihat masuk akal (berupa kata). Silakan lanjut training!")

if __name__ == "__main__":
    check_labels()