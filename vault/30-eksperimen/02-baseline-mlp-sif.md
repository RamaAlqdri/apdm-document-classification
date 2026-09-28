---
judul: Baseline TEXT — MLP atas embedding SIF
tipe: eksperimen
tahap: "05"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/05
  - modal/teks
  - komponen/embedding
terkait:
  - "[[sif-document-embedding]]"
  - "[[01-ekstraksi-fitur-teks]]"
  - "[[03-baseline-cnn1d]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
---

# Baseline TEXT — MLP atas embedding SIF

model:: MLP
dataset:: Tobacco3482 (QS-OCR-small, representasi SIF)
split:: 800 train (termasuk 10% val) / 2682 test, stratified
seed:: 42, 43, 44
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Arsitektur dan loop training sudah ditulis dan
> lolos self-check bentuk tensor, tapi **belum pernah dilatih**: mesin penulis kode
> tidak memegang dataset. Semua field angka kosong sampai dijalankan di mesin
> compute. Lihat aturan 3 di `CLAUDE.md`.

## Arsitektur

Paper §4.2: ReLU, Dropout, dan BatchNorm setelah setiap layer; lebar 2048 untuk
semua layer kecuali yang terakhir, yang menghasilkan vektor fitur 128; He
initialization.

| Item | Nilai | Sumber |
|---|---|---|
| Lebar layer tersembunyi | 2048 | paper |
| Jumlah layer tersembunyi | 2 | **asumsi kita** — paper tidak menyebutnya |
| Dimensi fitur | 128 | paper |
| Dropout | 0,5 | **asumsi kita** |
| Inisialisasi | He (kaiming normal, ReLU) | paper |
| Parameter | 5.084.810 | terhitung dari self-check |

Masukannya vektor SIF per seed (`sif_seed{42,43,44}.npy`), karena arah komponen
utama dipasang pada baris train saja.

## Hyperparameter

SGD momentum 0,9, lr 0,01, batch 40, 100 epoch (paper §4.2). Early stopping
`patience=15` adalah **deviasi kita**; paper melatih epoch tetap.

## Hasil

Diisi setelah eksekusi.

| Seed | OA | Macro F1 | Epoch terbaik | Durasi |
|---|---|---|---|---|
| 42 | | | | |
| 43 | | | | |
| 44 | | | | |
| **rata-rata** | | | | |

Target paper (Tabel 1a): **OA 70,8% / F1 0,69**.

## Perbandingan dengan paper

Diisi setelah eksekusi. Yang penting bukan angka persisnya, tapi bahwa MLP tetap
**di bawah** CNN1D — lihat [[03-baseline-cnn1d]]. Kalau urutannya terbalik, itu
temuan negatif terhadap paper dan harus diperiksa dulu penyebabnya: apakah PCA SIF
benar-benar dipasang pada train saja, dan apakah sekuens CNN1D ter-truncate terlalu
agresif.

## Temuan

Diisi setelah eksekusi.
