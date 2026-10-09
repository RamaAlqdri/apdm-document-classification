# Implementasi Jurnal Multimodal (Teks + Citra)

Replikasi Audebert et al., *Multimodal deep networks for text and image-based
document classification* (arXiv:1907.06370) untuk matakuliah **Analisis dan
Pemrosesan Data Multimodal**, plus ablasi tambahan yang tidak ada di paper.

- Paper: `references/1907.06370v1.pdf`
- Rencana kerja bertahap: `vault/00-index/rencana-prompt.md`
- Aturan kerja untuk agen: `CLAUDE.md`

## Dataset

| Modalitas | Sumber | Catatan |
|---|---|---|
| Citra | Kaggle `patrickaudriaz/tobacco3482jpg` | 3482 JPG, 10 kelas |
| Teks | QS-OCR-Small, GitHub `QuickSign/ocrized-text-dataset` tab **Releases** tag `v1.0` | 3482 `.txt`, hasil Tesseract 4.0.0-beta.1 |

RVL-CDIP dan QS-OCR-Large tidak dipakai. Teks di-OCR dari TIF asli sedangkan
citra kita dari JPG re-encode — sumber kedua modalitas tidak identik, ini
limitasi yang dicatat, bukan bug.

## Cara kerja: dua mesin

Kode ditulis di satu mesin, dijalankan di mesin lain. Mesin penulisan tidak
memegang dataset dan tidak menjalankan model, jadi notebook di repo ini **belum
pernah dieksekusi** — sel kodenya diperiksa sintaksnya saja.

**Mau menjalankannya?** Buka satu notebook dan tekan Run All:

    notebooks/00_setup_dan_jalankan.ipynb

Notebook itu memasang dependency, memeriksa GPU dan memasang ulang PyTorch versi
CUDA kalau perlu, menerima token Kaggle langsung di dalam notebook, menjalankan
sebelas self-check modul, lalu mengeksekusi notebook 01-06 berurutan. Model spaCy dan
FastText diunduh sendiri. Micro-batch dipilih dari kapasitas VRAM, mixed precision
dari jenis device, jumlah worker dari sistem operasi.

Tidak ada perintah terminal dan tidak ada sel yang perlu disunting. Satu-satunya
pilihan: `CEPAT = True` di sel terakhir (± 1 jam, untuk menguji pipeline) atau
`False` (setelan paper penuh, 12-18 jam di RTX 3050).

Lebih suka terminal? `python run_all.py` melakukan hal yang sama. Rincian dan
penanganan masalah ada di [RUNBOOK.md](RUNBOOK.md).

Artinya: catatan eksperimen di `vault/30-eksperimen/` berstatus
`belum-dijalankan` sampai dijalankan di mesin compute, dan angkanya hanya diisi
dari keluaran eksekusi nyata.

## Setup

Gunakan Python 3.12 — pada 3.14 wheel `torch`, `spacy`, dan `fasttext` belum
tersedia.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # wajib, model spaCy tidak ikut pip
```

Kredensial Kaggle (`~/.kaggle/kaggle.json` atau env `KAGGLE_USERNAME`/`KAGGLE_KEY`)
harus disiapkan sebelum Tahap 3. Citra dari Kaggle berukuran **3,29 GB**; teks
QS-OCR-small hanya 2,5 MB. Model FastText `cc.en.300.bin` (~7 GB di disk, ~15 GB
saat dimuat) diperlukan di Tahap 4.

Pemeriksaan mandiri tanpa dataset:

```bash
python -m src.config && python -m src.data
```

Cek konfigurasi:

```bash
python3 src/config.py
```

## Peta folder

```
├── CLAUDE.md              # aturan kerja agen
├── requirements.txt
├── references/            # PDF paper
├── data/{raw,interim,processed}/   # gitignored
├── notebooks/             # 01_data_audit, 02_text_features, 03..05
├── src/                   # config.py, utils.py, + modul per tahap
├── models/                # checkpoint, gitignored
├── reports/figures/
└── vault/                 # Obsidian vault
    ├── 00-index/          # MOC: peta-proyek, peta-tag, peta-metode, ...
    ├── 10-jurnal/         # ringkasan + spesifikasi implementasi paper
    ├── 20-metode/         # catatan konsep atomik
    ├── 30-eksperimen/     # satu catatan per run, dengan field Dataview
    ├── 40-log/            # log harian
    ├── 50-hasil/          # tabel utama, sintesis akhir
    ├── 90-referensi/      # catatan referensi yang dikutip paper
    ├── templates/
    └── attachments/
```

Semua hyperparameter ada di satu `CONFIG` dict di `src/config.py`, seed global
lewat `set_seed()`. Angka hasil di catatan vault **selalu** berasal dari
eksekusi nyata; yang belum dijalankan ditandai `status: belum-dijalankan`.

## Hasil

Seluruh pipeline dijalankan penuh pada 2026-09-30 (laptop Windows, RTX 3050 4 GB, CUDA).

| Model | Kita | Paper |
|---|---|---|
| TEXT (CNN1D) | 0,7266 ± 0,0059 | 0,738 |
| IMAGE (MobileNetV2) | 0,8086 ± 0,0210 | 0,845 |
| FUSION concat | 0,8342 ± 0,0105 | 0,878 |
| FUSION sum | 0,8275 ± 0,0164 | tidak dilaporkan |
| Oracle | 0,8977 | 0,921 |

Urutan TEXT < IMAGE < FUSION < Oracle terpenuhi, semuanya 1-4% di bawah paper.

**Tapi replikasi angka bukan temuan utamanya.** Ablasi menunjukkan cabang teks pada model
fusion praktis **tidak terpakai**: menolkan seluruh masukan teks menurunkan akurasi hanya
0,0015. Penyebabnya ditelusuri sampai mekanisme — cabang teks kolaps (varians 38× lebih
kecil dari CNN1D standalone, 67 dari 128 unit mati) dan skalanya timpang 3,7× terhadap
cabang citra — lalu **diperbaiki** dengan LayerNorm per cabang plus inisialisasi dari
checkpoint CNN1D. Setelah diperbaiki, probe linear cabang teks naik 0,4236 → 0,7468
(acuan CNN1D 0,7479) dan kelas Resume 0,803 → 0,968 (paper 0,96).

Pelajaran metodologisnya: **OA fusion yang tinggi bukan bukti bahwa fusion memakai kedua
modalitas.**

Cerita lengkapnya di `vault/50-hasil/sintesis-akhir.md`, limitasinya di
`vault/50-hasil/limitasi-dan-lanjutan.md`, tabel per kelas di
`vault/50-hasil/tabel-utama.md`.

## Progress

Lihat `vault/00-index/papan-progress.md` (butuh plugin Dataview di Obsidian).
