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
- [ ] **03** — Audit data, pairing citra↔teks, EDA
- [ ] **04** — Split + representasi teks (FastText, SIF, sekuens 500×300)
- [ ] **05** — Baseline unimodal TEXT dan IMAGE
- [ ] **06** — FUSION (concat vs penjumlahan adaptif) + Oracle
- [ ] **07** — Ablasi: degradasi citra, missing modality, degradasi teks
- [ ] **08** — Sintesis akhir + limitasi

## MOC lain

- [[peta-tag]] — daftar resmi tag dan kapan dipakai
- [[peta-metode]] — peta konsep metode
- [[peta-eksperimen]] — tabel agregat seluruh run
- [[papan-progress]] — status per tahap
- [[rencana-prompt]] — prompt bertahap + basis rujukan tiap klaim

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
