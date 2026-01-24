import os
import pandas as pd
import re

# ===============================
# KONFIGURASI
# ===============================
CSV_PATH = "private.csv"
PROCESSED_DIR = r"D:\vsr\data\processed\private"

def clean_text(text):
    # Hapus semua karakter SELAIN huruf dan angka
    # Contoh: "Halo?" -> "halo"
    return re.sub(r'[^a-zA-Z0-9]', '', str(text).lower())

def run_fix_rename():
    print("📂 Membaca file CSV...")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ File CSV tidak ditemukan: {CSV_PATH}")
        print("   Pastikan file 'private.csv' ada di folder yang sama dengan script ini.")
        return

    # 1. Baca CSV
    try:
        df = pd.read_csv(CSV_PATH, sep='|', header=None, names=['filename', 'label'])
    except Exception as e:
        print(f"❌ Gagal baca CSV: {e}")
        return

    # Mapping ID -> Label Bersih
    mapping = {}
    for index, row in df.iterrows():
        # Ambil ID: "001.mp4" -> "001"
        fname = str(row['filename'])
        if '.mp4' in fname:
            vid_id = fname.replace('.mp4', '').strip()
        else:
            vid_id = fname.strip()
            
        # Bersihkan Label (Hapus ? ! dll)
        label_bersih = clean_text(row['label'])
        mapping[vid_id] = label_bersih

    print(f"✅ Berhasil memuat {len(mapping)} label dari CSV.")
    
    # 2. Rename File
    print("\n🚀 Mulai memperbaiki nama file...")
    
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ Folder data tidak ditemukan: {PROCESSED_DIR}")
        return

    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npy')]
    renamed_count = 0
    missing_ids = set()
    
    for filename in files:
        # Cari ID angka 3 digit di nama file (misal 105 di "105_aug0.npy")
        parts = filename.split('_')
        video_id = None
        
        for p in parts:
            if p.isdigit() and len(p) == 3:
                video_id = p
                break
        
        if video_id:
            if video_id in mapping:
                label_baru = mapping[video_id]
                
                # Cek apakah SUDAH direname (mengandung label baru)
                if label_baru in filename:
                    continue # Skip yang sudah benar
                
                # Cek apakah GAGAL direname sebelumnya (masih angka)
                # Rename: "105_aug0.npy" -> "berapasuhuudarahariini_105_aug0.npy"
                new_name = f"{label_baru}_{filename}"
                
                old_path = os.path.join(PROCESSED_DIR, filename)
                new_path = os.path.join(PROCESSED_DIR, new_name)
                
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except OSError as e:
                    print(f"⚠️ Gagal rename {filename}: {e}")
            else:
                missing_ids.add(video_id)
    
    print(f"\n🎉 SELESAI! {renamed_count} file berhasil diperbaiki.")
    
    if missing_ids:
        print("\n⚠️ PERINGATAN: ID berikut ada di folder tapi TIDAK ADA di CSV:")
        print(f"   {sorted(list(missing_ids))}")
        print("   (File-file ini tidak direname. Harap cek CSV kamu.)")
    else:
        print("\n✅ Semua file berhasil dikenali dan direname.")

if __name__ == "__main__":
    run_fix_rename()