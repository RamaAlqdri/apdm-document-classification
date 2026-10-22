---
judul: Log 2026-10-22 — Pipeline Fase 2 dalam satu notebook + audit proyek
tipe: log
tahap: "12"
tanggal: 2026-10-22
status: selesai
tags:
  - tipe/log
  - tahap/10
  - tahap/11
  - tahap/12
  - status/selesai
terkait:
  - "[[rencana-prompt]]"
  - "[[peta-proyek]]"
  - "[[15-grid-teks-ocr]]"
  - "[[16-fusi-per-ocr]]"
---

# Log 2026-10-22 — Pipeline Fase 2 dalam satu notebook

Permintaan pengguna: implementasikan rancangan penelitian (review-1), audit proyek supaya
sinkron, logika di `.py`, dan satu notebook yang menjalankan semuanya dari awal sampai
akhir termasuk EDA. Pengguna berencana menjalankan ulang semuanya dari nol.

## Rancangan yang diimplementasikan

Tobacco3482 saja (RVL-CDIP diputuskan tidak dipakai setelah sempat dipertimbangkan).
Variabel bebas: mesin OCR (QS-OCR, Tesseract-JPG, EasyOCR, PaddleOCR) dan aturan fusi
(CONCAT di atas fitur beku vs LATE rata-rata logit). FastText + CNN1D tetap; BERT ditunda.
Rinciannya dan alasannya di bagian "Implementasi (2026-10-22)" pada [[rencana-prompt]].

## Yang dibuat atau diubah

- `src/experiment.py` (baru): training unimodal + cache keluaran semua dokumen, LATE,
  CONCAT, ORACLE, missing modality, McNemar + Holm, ringkasan.
- `src/models/fusion.py`: `ConcatHead`.
- `src/ocr.py`: runner subprocess, cari Tesseract, venv PaddleOCR, teks per kondisi,
  proksi kualitas, kesepakatan antar-mesin (dipindah dari notebook 10).
- `src/eda.py`: loop EDA dari notebook 11 (statistik citra/teks, probe, t-SNE, kata khas,
  kurasi, jalan pintas kode ID).
- `src/data.py: prepare_dataset()`, `src/text_features.py: build_condition_sequences()`,
  `src/plots.py` (baru), `src/setup_env.py` (baru).
- `src/config.py`: kondisi OCR, folder `fase2`, epoch *head*, α, `APDM_TIMPA`.
- `notebooks/pipeline.ipynb` (43 sel): setup → data → split → EDA → OCR → fitur → training
  → fusi → uji H1/H3 → ringkasan. Bagian mahal bisa dilanjutkan.
- `run_all.py` ditulis ulang untuk menjalankan notebook itu; `RUNBOOK.md`, `README.md`,
  `CLAUDE.md`, `requirements.txt` (scipy, wordfreq) disesuaikan.
- Notebook 00-11 dipindah ke `notebooks/arsip/` beserta outputnya.

## Verifikasi di mesin penulis

- 14 self-check modul lolos (`python run_all.py --periksa`; spaCy/FastText memang tidak
  ada di mesin ini).
- *Dry run* seluruh sel kode notebook pada dataset sintetis (240 dokumen, 10 kelas, OCR
  palsu, FastText palsu, citra 64 px, 2 epoch) lolos sampai sel ringkasan dan menulis 23
  berkas laporan. Yang dilewati hanya sel yang butuh jaringan atau interaksi (cek torch,
  token Kaggle, self-check, pemasangan mesin OCR). Angka dari *dry run* tidak bermakna dan
  tidak dicatat di mana pun.

## Belum

Belum dijalankan pada data asli. Daftar mesin OCR masih perlu dikonfirmasi ke dosen
(PaddleOCR disebut dua kali di catatan tanggapan).
