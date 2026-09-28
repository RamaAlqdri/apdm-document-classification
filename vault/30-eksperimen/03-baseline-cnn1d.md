---
judul: Baseline TEXT — CNN1D atas sekuens kata
tipe: eksperimen
tahap: "05"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/05
  - modal/teks
  - komponen/cnn
terkait:
  - "[[cnn-1d-teks]]"
  - "[[02-baseline-mlp-sif]]"
  - "[[04-baseline-mobilenetv2]]"
  - "[[01-ekstraksi-fitur-teks]]"
---

# Baseline TEXT — CNN1D atas sekuens kata

model:: CNN1D
dataset:: Tobacco3482 (QS-OCR-small, sekuens 500x300)
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

Paper §4.2: 4 layer, konvolusi 1D window 12, 512 channel per layer, ReLU,
maxpooling stride 2 diselipkan antar layer, lalu max-pooling-through-time, Dropout,
fully connected, softmax.

| Item | Nilai | Sumber |
|---|---|---|
| Kedalaman | 4 layer konvolusi | paper |
| Window | 12 | paper |
| Channel | 512 | paper |
| Maxpool | stride 2, **hanya antar layer** | posisi = asumsi kita |
| Agregasi | max-pooling-through-time | paper |
| Dimensi fitur | 128 | paper (via bagian fusion) |
| Dropout | 0,5 | **asumsi kita** |
| Parameter | 11.349.386 | terhitung dari self-check |

Masukan (batch, 300, 500) — konvensi channel-first Conv1d, jadi sekuens 500x300
di-transpose. Self-check memverifikasi transpose ini benar, karena sumbu yang
tertukar akan tetap berjalan tanpa error sambil melatih model yang salah.

## Hyperparameter

SGD momentum 0,9, lr 0,01, batch 40, 100 epoch (paper §4.2). Early stopping
`patience=15` adalah deviasi kita.

## Hasil

Diisi setelah eksekusi.

| Seed | OA | Macro F1 | Epoch terbaik | Durasi |
|---|---|---|---|---|
| 42 | | | | |
| 43 | | | | |
| 44 | | | | |
| **rata-rata** | | | | |

Target paper: **Tabel 1a 73,9% / 0,71**, dan baris TEXT Tabel 3 **73,8% / 0,71**.

## F1 per kelas

Diisi setelah eksekusi. Kolom pembanding adalah baris TEXT Tabel 3 paper:

| Kelas | Kita | Paper |
|---|---|---|
| Advertisement | | 0,60 |
| Email | | 0,96 |
| Form | | 0,76 |
| Letter | | 0,71 |
| Memo | | 0,79 |
| News | | 0,67 |
| Note | | 0,62 |
| Report | | 0,43 |
| Resume | | 0,97 |
| Scientific | | 0,57 |

Dua kelas yang paling menentukan: **Resume** (paper 0,97, jauh di atas citra 0,80 —
inilah sumber komplementaritas terkuat) dan **Report** (paper 0,43, terlemah).

Confusion matrix: `reports/figures/05_confusion_text_cnn1d_seed42.png`.

## Catatan untuk Tahap 6

Prediksi per-sampel seed 42 disimpan ke
`data/processed/pred_text_seed42.npy`. Oracle di Tahap 6 **wajib** dihitung dari
prediksi baseline standalone ini, bukan dari cabang internal model fusion. Lihat
[[oracle-sebagai-batas-atas]].

## Temuan

Diisi setelah eksekusi.
