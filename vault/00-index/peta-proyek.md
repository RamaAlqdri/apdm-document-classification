---
judul: Peta Proyek
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: wip
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-tag]]"
  - "[[peta-metode]]"
  - "[[peta-eksperimen]]"
  - "[[papan-progress]]"
  - "[[rencana-prompt]]"
---

# Peta Proyek

Pintu masuk vault. MOC induk.

## Tujuan

Mereplikasi Audebert et al. (arXiv:1907.06370): klasifikasi dokumen dengan
menggabungkan modalitas **citra** (MobileNetV2) dan **teks hasil OCR**
(FastText + CNN1D), lalu menguji sendiri klaim bahwa fusion memberi nilai
tambah. Di atas replikasi, proyek ini menambah **ablasi degradasi modalitas**
yang tidak dikerjakan penulis, untuk menjawab kapan multimodal benar-benar
menolong.

Target relatif yang harus terlihat: TEXT < IMAGE < FUSION < Oracle.
Angka paper pada Tobacco3482: 73,8% / 84,5% / 87,8% / 92,1%.

## Tahap

- [ ] **00** — Brief proyek (`CLAUDE.md`) → lihat [[rencana-prompt]]
- [x] **01** — Scaffolding + vault + template + MOC
- [x] **02** — Sintesis jurnal → [[1907.06370-ringkasan]],
      [[1907.06370-spesifikasi-implementasi]], 10 catatan konsep, 8 catatan pustaka
- [~] **03** — Audit data, pairing citra↔teks, EDA — kode siap
      ([[00-audit-data]] masih `belum-dijalankan`, eksekusi di mesin compute)
- [~] **04** — Split + representasi teks — kode siap
      ([[01-ekstraksi-fitur-teks]] masih `belum-dijalankan`)
- [~] **05** — Baseline unimodal TEXT dan IMAGE — kode siap
      ([[02-baseline-mlp-sif]], [[03-baseline-cnn1d]], [[04-baseline-mobilenetv2]]
      masih `belum-dijalankan`)
- [~] **06** — FUSION (concat vs penjumlahan adaptif) + Oracle — kode siap
      ([[05-fusion-concat]], [[06-fusion-sum]], [[tabel-utama]] masih
      `belum-dijalankan`)
- [~] **07** — Ablasi: degradasi citra, missing modality, degradasi teks — kode
      siap ([[07-ablasi-degradasi-citra]], [[08-ablasi-missing-modality]],
      [[09-ablasi-degradasi-teks]] masih `belum-dijalankan`; ablasi 4 dilewati)
- [ ] **08** — Sintesis akhir + limitasi

## MOC lain

- [[peta-tag]] — daftar resmi tag dan kapan dipakai
- [[peta-metode]] — peta konsep metode
- [[peta-eksperimen]] — tabel agregat seluruh run
- [[papan-progress]] — status per tahap
- [[rencana-prompt]] — prompt bertahap + basis rujukan tiap klaim

## Cara kerja: dua mesin

Mesin penulisan kode **tidak memegang dataset dan tidak menjalankan model**. Semua
akuisisi data, training, dan evaluasi dikerjakan di mesin compute terpisah.

Konsekuensinya untuk seluruh vault: notebook dan modul `src/` ditulis lengkap dan
diperiksa sintaksnya di sini, tapi catatan eksperimennya berstatus
`belum-dijalankan` sampai dieksekusi di sana. Angka hanya masuk vault setelah
benar-benar dihitung — aturan 3 `CLAUDE.md` berlaku tanpa pengecualian.

## Lingkungan eksekusi

Mesin compute memakai **Python 3.12.10** lewat `.venv` proyek, kernel Jupyter
`apdam-py312`. Dua batasan yang tidak terlihat dari kode:

- **Plafonnya 3.12, bukan versi terbaru.** `fasttext-wheel` 0.9.2 hanya punya wheel
  Windows sampai cp312; di 3.13 `fasttext` harus dibangun dari source dan butuh MSVC.
- **torch wajib dari index cu121.** Wheel PyPI untuk Windows adalah build CPU-only,
  jadi `pip install torch` menghasilkan lingkungan yang jalan tapi tanpa GPU — gagal
  senyap. `requirements.txt` memaksa sumbernya lewat `--extra-index-url` dan pin
  `torch==2.5.1+cu121`.

Migrasi dari 3.10 ke 3.12 diverifikasi setara: split dan metrik checkpoint identik
di kedua lingkungan. Rinciannya di
[[2026-09-30-migrasi-lingkungan-py312]].

## Deviasi dari paper yang sudah pasti

Dicatat sejak awal supaya tidak tersalahartikan sebagai gap replikasi:

1. Paper pakai TensorFlow 1.12 + Keras, kita PyTorch. Bobot pretrained
   MobileNetV2 tidak identik.
2. Paper pakai k-fold cross-validation; kita 3 random split terstratifikasi
   (800 train) dengan validation dipotong dari train untuk early stopping.
3. Citra kita JPG re-encode dari Kaggle, sedangkan teks QS-OCR di-OCR dari TIF
   asli. Sumber kedua modalitas tidak identik.
4. Budget epoch kemungkinan dipotong dari 200 karena keterbatasan compute —
   dicatat per eksperimen kalau terjadi.

Daftar lengkap 11 ambiguitas paper beserta asumsi yang kita ambil ada di
[[1907.06370-spesifikasi-implementasi]].
