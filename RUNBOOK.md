# Runbook — menjalankan pipeline di mesin compute

Kode ditulis di mesin lain dan **belum pernah dieksekusi pada dataset asli**. Yang sudah
diuji di mesin penulis: self-check setiap modul `src/`, dan *dry run* seluruh sel
`notebooks/pipeline.ipynb` pada dataset sintetis kecil (240 dokumen, epoch 2).

Semua catatan hasil Fase 2 di `vault/` berstatus `belum-dijalankan` sampai diisi dari
keluaran notebook. Aturan 3 di `CLAUDE.md` melarang mengisi angka yang belum dihitung.

---

## Jalur singkat

1. Clone repo, buat venv Python **3.12**, `pip install -r requirements.txt`
2. Buka `notebooks/pipeline.ipynb` (Jupyter Lab / VS Code / Anaconda)
3. Atur sel `RUN` paling atas (cukup `quick` saja, sisanya punya bawaan)
4. **Run All**

Atau lewat terminal:

```bash
python run_all.py --periksa   # prasyarat + 14 self-check, tidak menjalankan apa pun
python run_all.py --cepat     # 1 seed, epoch dipangkas: uji pipeline dulu
python run_all.py             # setelan penuh
```

Notebook **bisa dilanjutkan**. Halaman OCR, sekuens teks, dan model yang sudah ada di
disk dipakai ulang; kalau kernel mati atau listrik padam, cukup Run All (atau perintah
yang sama) lagi. `RUN["overwrite"] = True` / `--timpa` memaksa semuanya dihitung ulang.

Mode cepat dan penuh menulis ke file yang sama. Setelah uji cepat, hapus
`data/processed/fase2/outputs/` sebelum run penuh, atau pakai `--timpa`, supaya model
1-seed hasil uji tidak terbawa.

## Urutan di dalam notebook

| Bagian | Isi | Perkiraan (RTX 3050 4 GB) |
|---|---|---|
| 1 | pip, cek CUDA, token Kaggle, self-check | menit |
| 2 | unduh citra Kaggle (3,3 GB) + QS-OCR, audit, `manifest.csv` | 15-30 menit |
| 3 | split 800/2.682, tiga seed | detik |
| 4 | EDA: distribusi, statistik citra/teks, kurasi, duplikat, jalan pintas kode ID | 5-10 menit |
| 5 | OCR Tesseract, EasyOCR, PaddleOCR + proksi kualitas | **± 1-3 jam per mesin** (uji 20 halaman dulu, perkiraannya dicetak) |
| 6 | spaCy + FastText (unduh 7 GB sekali), sekuens per kondisi, probe + t-SNE | 30-90 menit |
| 7 | training IMAGE ×3 seed, TEXT ×4 kondisi ×3 seed | **jam-an**; sel estimasi mencetak batas atasnya |
| 8 | LATE, CONCAT, ORACLE, missing modality | menit |
| 9 | McNemar H1/H3, pemakaian modalitas, F1 per kelas | detik |
| 10 | blok ringkasan untuk vault | detik |

Hasil: `reports/fase2/*.csv`, gambar `reports/figures/fase2_*.png`, model
`models/fase2/`, keluaran unimodal `data/processed/fase2/outputs/`. Semua terpisah dari
artefak Fase 1, jadi run ini tidak menimpa hasil lama.

---

## Yang perlu disiapkan

| Kebutuhan | Ukuran | Catatan |
|---|---|---|
| Disk | **± 25 GB** | citra 3,3 + FastText 7 + 4 sekuens teks 4 + venv PaddleOCR 1 + model |
| RAM | **16 GB minimum** | FastText ± 15 GB saat dimuat (bagian 6 saja) |
| GPU | sangat disarankan | tanpa GPU, bagian 7 bisa berhari-hari |
| Python | **3.12** | 3.13/3.14 belum punya wheel `torch`/`spacy`/`fasttext` |
| Kaggle | token API | kaggle.com → Settings → API → Create New Token |
| Tesseract | biner | Windows: dipasang otomatis lewat winget (jendela izin admin → *Yes*); macOS `brew install tesseract`; Linux `apt install tesseract-ocr` |

Token Kaggle: notebook mencari `kaggle.json` di Downloads/Unduhan/folder proyek dan
memindahkannya ke `~/.kaggle/`; kalau tidak ada, ia meminta username dan key lewat
`getpass` (tidak tersimpan di output). Lewat `run_all.py` tidak ada prompt, jadi token
harus sudah ada; `--periksa` memberitahu.

**PyTorch dengan CUDA.** `requirements.txt` memakai indeks `cu121`. Kalau GPU NVIDIA ada
tapi `torch.cuda.is_available()` `False`, notebook memasang ulang dari indeks `cu124`
lalu meminta restart kernel. Sesuaikan versi CUDA dengan `nvidia-smi` bila perlu.

**Pustaka OCR.** `pytesseract` dan `easyocr` dipasang di bagian 5 dengan *constraints*
yang mengunci numpy/torch/torchvision. PaddleOCR dipasang di `.venv-paddle/`
(± 1 GB) karena pin numpy-nya bentrok dengan torch. Keras-OCR tidak dipakai: butuh TF
dengan API Keras 2 yang tidak tersedia untuk Python 3.12.

---

## VRAM 4 GB dan micro-batch

Batch 40 MobileNetV2 384×384 tidak muat di 4 GB. `src/config.py` membaca VRAM lalu
memilih micro-batch sendiri (8 untuk < 5 GB, 16 untuk 6-8 GB); gradien diakumulasi
sehingga optimizer tetap melangkah dari 40 sampel. **Yang tidak setara, dan dicetak
sebagai deviasi:** BatchNorm menormalisasi per micro-batch. AMP menyala otomatis di
CUDA. Notebook mencetak daftar deviasi runtime di bagian 1 dan 10; salin ke catatan.

Kalau masih OOM: tambahkan `CONFIG["micro_batch_size"] = 4` setelah sel import,
restart kernel, Run All (yang sudah selesai tidak diulang).

---

## Masalah yang sudah diantisipasi

**DataLoader hang di Windows.** `CONFIG["num_workers"]` otomatis 0 di Windows.

**Symlink gagal di Windows.** `download_images` menangkapnya dan menulis lokasi cache
ke `data/raw/LOKASI_CITRA.txt`. Tidak ada yang rusak.

**`prepare_dataset` gagal "hanya N kelas terpetakan".** Nama folder kelas tidak
tercakup `CLASS_ALIASES`. Tambahkan aliasnya di `src/data.py`, jangan diakali di
notebook.

**Assert "halaman belum di-OCR".** Satu mesin belum tuntas. Jalankan ulang; OCR
melanjutkan dari halaman terakhir. Halaman yang gagal dicatat di
`data/interim/ocr/<mesin>/_errors.txt` dan dicoba lagi pada run berikutnya.

**Bagian 6 kehabisan RAM.** Tutup aplikasi lain dan perbesar swap/pagefile. Jangan
pakai `fasttext.util.reduce_model`: ia memuat model penuh lebih dulu.

**Jangan menyalin `data/processed/manifest.csv` dari mesin lain.** Isinya path absolut;
notebook membuatnya ulang.

---

## Setelah selesai: isi catatan vault

Sel terakhir mencetak blok siap-salin (field Dataview `oa::`, `macro_f1::`, dan
ringkasan H1/H3). Isi catatan Fase 2 di `vault/30-eksperimen/` (13 sampai 16) dan
`vault/50-hasil/`, lalu ubah `status: belum-dijalankan` menjadi `selesai`.
Hipotesis yang ditulis sebelum eksekusi dibiarkan apa adanya meski meleset.
