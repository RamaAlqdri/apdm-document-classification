---
judul: Baseline TEXT — CNN1D atas sekuens kata
tipe: eksperimen
tahap: "05"
tanggal: 2026-09-28
status: selesai
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
oa:: 0.7266
macro_f1:: 0.7017
durasi:: 4.7 menit

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050, CUDA). Angka di bawah dari
> eksekusi nyata `notebooks/03_baseline_teks.ipynb`.

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

| Seed | OA | Macro F1 | Epoch terbaik | Epoch dijalankan | Durasi |
|---|---|---|---|---|---|
| 42 | 0,7345 | 0,7090 | 28 | 43 | 92,2s |
| 43 | 0,7248 | 0,6994 | 25 | 40 | 88,9s |
| 44 | 0,7204 | 0,6967 | 31 | 46 | 103,3s |
| **rata-rata ± std** | **0,7266 ± 0,0059** | **0,7017** | 28 | 43 | 4,7 menit total |

Target paper: **Tabel 1a 73,9% / 0,71**, dan baris TEXT Tabel 3 **73,8% / 0,71**.
Selisih kita **−0,011 OA** — **baseline yang paling dekat dengan paper di seluruh
proyek.**
## Deviasi runtime yang benar-benar terjadi

Dicetak otomatis oleh notebook, disalin apa adanya:

1. **Batch 40 dipecah jadi micro-batch 8 dengan akumulasi gradien.** Update
   optimizer tetap dihitung dari 40 sampel seperti paper, tapi BatchNorm
   menormalisasi per 8 sampel. VRAM 4 GB tidak bisa menampung batch 40 pada
   384x384.
2. **Mixed precision (AMP) aktif**, tidak dipakai paper.
3. **Early stopping menghentikan run jauh sebelum 200 epoch.** Lihat kolom "epoch
   dijalankan" di tabel hasil — ini ternyata masalah serius, didiagnosis di
   [[tabel-utama]].


## F1 per kelas

Kolom pembanding adalah baris TEXT Tabel 3 paper:

| Kelas | Kita | Paper | Selisih |
|---|---|---|---|
| Advertisement | 0,537 | 0,60 | −0,063 |
| Email | 0,953 | 0,96 | −0,007 |
| Form | 0,718 | 0,76 | −0,042 |
| Letter | 0,671 | 0,71 | −0,039 |
| Memo | 0,758 | 0,79 | −0,032 |
| News | 0,709 | 0,67 | **+0,039** |
| Note | 0,621 | 0,62 | +0,001 |
| Report | 0,495 | 0,43 | **+0,065** |
| **Resume** | **0,969** | 0,97 | −0,001 |
| Scientific | 0,587 | 0,57 | +0,017 |

**Resume 0,969 vs 0,97 — cocok sampai satu angka desimal.** Itu penting karena
Resume adalah kelas dengan keunggulan teks terbesar (IMAGE hanya 0,784), jadi ia
sumber komplementaritas utama. Cabang teks kita bekerja persis seperti di paper di
kelas yang paling menentukan.

Kita **lebih baik** dari paper di Report (+0,065) dan News (+0,039), lebih buruk di
Advertisement (−0,063). Advertisement masuk akal: 18,3% dokumennya punya < 20 kata
(lihat [[00-audit-data]]), jadi cabang teks kekurangan bahan.

Kelemahan terbesar: **Scientific recall hanya 0,408** pada seed 42 meski
precision-nya 0,845. Model mengenali Scientific dengan yakin ketika mengenalinya,
tapi melewatkan 59% di antaranya.

Confusion matrix: `reports/figures/05_confusion_text_cnn1d_seed42.png`.

Estimasi durasi yang dicetak notebook sebelum training: 2,57 s/epoch → 4,28 menit
untuk 100 epoch. Aktual: 43 epoch karena early stopping, 92s.

## Catatan untuk Tahap 6

Prediksi per-sampel seed 42 disimpan ke
`data/processed/pred_text_seed42.npy`. Oracle di Tahap 6 **wajib** dihitung dari
prediksi baseline standalone ini, bukan dari cabang internal model fusion. Lihat
[[oracle-sebagai-batas-atas]].

## Temuan

**#temuan/positif — replikasi terbaik di seluruh proyek.** OA −0,011 dari paper, dan
F1 Resume cocok sampai satu desimal (0,969 vs 0,97).

**#temuan/positif — cabang teks jelas berfungsi sebagai model mandiri.** 0,7266 dari
teks OCR yang 56,8% tokennya di luar kosakata rujukan. Ini penting sebagai pembanding
untuk Tahap 6: cabang teks yang sama, di dalam model fusion, ternyata **tidak
terpakai** — lihat [[08-ablasi-missing-modality]]. Jadi masalahnya bukan cabang
teksnya, melainkan cara fusion melatihnya.

**#temuan/anomali — Scientific recall 0,408 dengan precision 0,845.** Ketimpangan
sebesar itu menandakan model hanya memprediksi Scientific ketika sangat yakin.
Menariknya cabang citra juga lemah di kelas ini (F1 0,554), dan fusion tidak
memperbaikinya (0,583) — satu-satunya kelas di mana ketiga model sama-sama gagal.
