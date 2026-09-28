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

**Mau menjalankannya? Ikuti [RUNBOOK.md](RUNBOOK.md)** — prosedur lengkap dari
clone sampai seluruh catatan eksperimen terisi angka, termasuk masalah khas
Windows dan cara memotong budget training dengan benar.

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

## Progress

Lihat `vault/00-index/papan-progress.md` (butuh plugin Dataview di Obsidian).
