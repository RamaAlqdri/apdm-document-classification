# Klasifikasi Dokumen Multimodal (Teks OCR + Citra): Mesin OCR sebagai Variabel

Proyek matakuliah **Analisis dan Pemrosesan Data Multimodal**, dua fase:

- **Fase 1 (selesai):** replikasi Audebert et al., *Multimodal deep networks for text
  and image-based document classification* (ECML PKDD 2019 Workshops, Springer 2020).
  Hasilnya di bagian [Hasil Fase 1](#hasil-fase-1).
- **Fase 2 (aktif):** mesin OCR dijadikan variabel. Referensi utama Kashyap et al.
  (2026), *Int. J. Intelligent Systems*, doi:10.1155/int/4241437. Review tiga paper
  rujukan ada di `review-1/`.

Rencana kerja bertahap: `vault/00-index/rencana-prompt.md`. Aturan kerja agen: `CLAUDE.md`.

## Rancangan Fase 2

| Komponen | Isi |
|---|---|
| Data | Tobacco3482 saja (tanpa RVL-CDIP); 800 latih terstratifikasi (80 val), 2.682 uji; seed 42/43/44 |
| Variabel bebas 1 | Mesin OCR: QS-OCR (acuan, Tesseract 4 dari TIF), Tesseract (JPG), EasyOCR, PaddleOCR |
| Variabel bebas 2 | Aturan fusi: CONCAT (fitur, gaya Audebert) vs LATE (rata-rata logit, gaya Kashyap) |
| Dikontrol | MobileNetV2, FastText subword + CNN 1D, hiperparameter, teks mentah tanpa praproses |
| Terikat | OA, macro-F1, F1 per kelas, akurasi tiap unimodal, penurunan saat satu modalitas dikosongkan |
| Analisis | rerata ± std 3 seed, McNemar berpasangan (Holm per seed), oracle, uji missing modality |

Hipotesis: **H1** mesin OCR mengubah akurasi cabang teks secara nyata; **H3** cabang teks
yang lebih kuat lebih dipakai fusi, dan rata-rata logit lebih tahan cabang timpang.
H2 (BERT) adalah *future work*. Diagram: `review-1/pipeline-usulan/main.pdf`.

## Dataset

| Modalitas | Sumber | Catatan |
|---|---|---|
| Citra | Kaggle `patrickaudriaz/tobacco3482jpg` | 3482 JPG, 10 kelas, 3,3 GB |
| Teks acuan | QS-OCR-Small, GitHub `QuickSign/ocrized-text-dataset` tab **Releases** tag `v1.0` | 3482 `.txt`, Tesseract 4.0.0-beta.1 dari TIF |
| Teks Fase 2 | dibuat sendiri dari citra JPG oleh `src/ocr.py` | Tesseract, EasyOCR, PaddleOCR |

RVL-CDIP dan QS-OCR-Large tidak dipakai. QS-OCR di-OCR dari TIF asli sedangkan citra
kita JPG re-encode; limitasi yang dicatat, bukan bug.

## Menjalankan

Kode ditulis di satu mesin dan dijalankan di mesin compute. **Satu notebook menjalankan
semuanya dari awal sampai akhir:**

    notebooks/pipeline.ipynb      atur sel RUN, lalu Run All

Isinya: setup → unduh + audit data → split → EDA → OCR multi-mesin → fitur teks → training
unimodal → dua aturan fusi → uji hipotesis → ringkasan untuk vault. Semua logika ada di
`src/` (notebook hanya memanggil). Bagian mahal bisa dilanjutkan setelah terputus.

Lewat terminal:

```bash
python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python run_all.py --periksa   # prasyarat + self-check
python run_all.py --cepat     # uji pipeline: 1 seed, epoch dipangkas
python run_all.py             # setelan penuh
```

Prosedur lengkap, kebutuhan disk/RAM/VRAM, dan penanganan masalah: [RUNBOOK.md](RUNBOOK.md).
Notebook per tahap yang lama (Fase 1, tahap 10-11) disimpan dengan outputnya di
`notebooks/arsip/` sebagai riwayat.

| Modul `src/` | Isi |
|---|---|
| `config.py` | satu `CONFIG` (path, hiperparameter, kondisi OCR) + `set_seed` |
| `setup_env.py` | cek torch/CUDA, token Kaggle, self-check semua modul |
| `data.py` | unduh, audit, pairing, `manifest.csv` |
| `splits.py` | split 800/2.682 tiga seed, dibekukan ke disk |
| `eda.py` | statistik citra/teks, kurasi, duplikat, kebocoran, probe, t-SNE |
| `ocr.py` | tiga mesin OCR, urutan baca, proksi kualitas, runner subprocess |
| `text_features.py` | spaCy + FastText subword, sekuens per kondisi OCR |
| `datasets.py`, `train.py` | Dataset/DataLoader, loop training, oracle, metrik |
| `models/` | CNN1D, MobileNetV2, `ConcatHead` (Fase 2), `FusionModel` (Fase 1) |
| `experiment.py` | training unimodal + cache keluaran, LATE, CONCAT, missing modality, McNemar |
| `plots.py` | gambar untuk notebook |
| `ablation.py` | ablasi Fase 1 (dipakai notebook arsip) |

## Peta folder

```
├── CLAUDE.md              # aturan kerja agen
├── requirements.txt
├── references/            # PDF paper
├── data/{raw,interim,processed}/   # gitignored
├── notebooks/             # pipeline.ipynb (aktif) + arsip/ (riwayat)
├── src/                   # semua logika, lihat tabel di atas
├── review-1/              # review 3 paper + diagram pipeline usulan (LaTeX)
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

Semua hyperparameter ada di satu `CONFIG` dict di `src/config.py` (sel `RUN` di notebook hanya memilih mode), seed global
lewat `set_seed()`. Angka hasil di catatan vault **selalu** berasal dari
eksekusi nyata; yang belum dijalankan ditandai `status: belum-dijalankan`.

## Hasil Fase 1

Pipeline Fase 1 dijalankan penuh pada 2026-09-30 (laptop Windows, RTX 3050 4 GB, CUDA).

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

## Hasil Fase 2

Run penuh `notebooks/pipeline.ipynb` pada 2026-10-24 (laptop Windows, RTX 3050 4 GB). OA
rerata ± std 3 seed pada 2.682 dokumen uji:

| Kondisi OCR | TEXT | LATE | CONCAT | ORACLE |
|---|---|---|---|---|
| qs-ocr (Tesseract 4, TIF) | 0,7266 ± 0,0059 | 0,8456 ± 0,0221 | 0,8316 ± 0,0119 | 0,9023 |
| tesseract 5 (JPG) | 0,6970 ± 0,0047 | 0,8376 ± 0,0174 | 0,8349 ± 0,0102 | 0,8950 |
| easyocr | 0,7155 ± 0,0061 | 0,8538 ± 0,0158 | 0,8567 ± 0,0034 | 0,8972 |
| paddleocr | **0,7514 ± 0,0043** | 0,8653 ± 0,0122 | **0,8674 ± 0,0014** | 0,9079 |

IMAGE 0,8169 ± 0,0257 (sama untuk semua kondisi). Mesin OCR saja menggeser cabang teks
5,4 poin (H1 didukung), dan kenaikannya terbawa ke kedua aturan fusi. Cabang TEXT qs-ocr
mereproduksi Fase 1 bit-identik. Analisis lengkap: `vault/50-hasil/sintesis-fase-2.md`.
