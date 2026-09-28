---
judul: Baseline IMAGE — MobileNetV2 384x384
tipe: eksperimen
tahap: "05"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/05
  - modal/citra
  - komponen/cnn
terkait:
  - "[[mobilenetv2]]"
  - "[[inverted-residual-block]]"
  - "[[03-baseline-cnn1d]]"
  - "[[oracle-sebagai-batas-atas]]"
---

# Baseline IMAGE — MobileNetV2 384x384

model:: MobileNetV2
dataset:: Tobacco3482 (Tobacco3482-jpg dari Kaggle)
split:: 800 train (termasuk 10% val) / 2682 test, stratified
seed:: 42, 43, 44
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Arsitektur dan loop training sudah ditulis dan
> lolos self-check bentuk tensor, tapi **belum pernah dilatih**: mesin penulis kode
> tidak memegang dataset. Semua field angka kosong sampai dijalankan di mesin
> compute. Lihat aturan 3 di `CLAUDE.md`.

## Arsitektur dan praproses

| Item | Nilai | Sumber |
|---|---|---|
| Backbone | MobileNetV2, pretrained ImageNet | paper |
| Fine-tuning | **seluruh jaringan**, backbone tidak dibekukan | paper |
| Input | 384x384 | paper |
| Resize | tanpa padding, aspect ratio di-warp | paper |
| Kanal | grayscale diduplikasi 3x | paper |
| Normalisasi | mean/std ImageNet | **asumsi kita** — paper tidak menyebutnya |
| Augmentasi | tidak ada | paper: augmentasi justru menurunkan 84,5% ke 83,9% |
| Dimensi fitur | 1280 (GAP per channel) | paper |
| Parameter | 2.236.682 | terhitung dari self-check |

Self-check memverifikasi tiga hal yang mudah salah tanpa terlihat: citra
non-persegi keluar persegi (warp, bukan padding), kanal grayscale benar-benar
diduplikasi identik sebelum normalisasi, dan `requires_grad` seluruh backbone
bernilai True.

## Hyperparameter

SGD momentum 0,9, lr 0,01, batch 40, **200 epoch** (paper §4.2). Early stopping
`patience=15` adalah deviasi kita.

## Anggaran compute — baca sebelum menjalankan

MobileNetV2 pada 384x384 selama 200 epoch, dikali 3 seed, hampir pasti jauh di atas
15 menit per run. Notebook `04_baseline_citra.ipynb` menjalankan estimasi 2 epoch
lebih dulu dan mencetak ekstrapolasinya.

Kalau budget dipotong — epoch dikurangi atau seed dikurangi jadi satu — notebook
mengumpulkannya ke variabel `DEVIASI` dan mencetaknya. **Catat di tabel di bawah**,
jangan dipotong diam-diam.

Deviasi compute yang benar-benar terjadi: diisi setelah eksekusi.

## Hasil

Diisi setelah eksekusi.

| Seed | OA | Macro F1 | Epoch terbaik | Durasi |
|---|---|---|---|---|
| 42 | | | | |
| 43 | | | | |
| 44 | | | | |
| **rata-rata** | | | | |

Target paper: **Tabel 1b 84,5% / 0,82**, dan baris IMAGE Tabel 3 sama.

Pembanding lain di Tabel 3 paper: ensemble CNN Harley et al. 79,9%. MobileNetV2
tunggal mengalahkannya, jadi 84,5% bukan baseline lemah.

## F1 per kelas

Diisi setelah eksekusi. Kolom pembanding adalah baris IMAGE Tabel 3 paper:

| Kelas | Kita | Paper |
|---|---|---|
| Advertisement | | 0,94 |
| Email | | 0,96 |
| Form | | 0,85 |
| Letter | | 0,83 |
| Memo | | 0,90 |
| News | | 0,89 |
| Note | | 0,83 |
| Report | | 0,61 |
| Resume | | 0,80 |
| Scientific | | 0,62 |

Perhatikan **Resume 0,80** melawan TEXT 0,97 di [[03-baseline-cnn1d]]. Itu satu
kelas di mana teks jelas menang, dan bukti paling terang bahwa kedua modalitas
saling melengkapi.

Confusion matrix: `reports/figures/05_confusion_image_seed42.png`.

## Pratinjau oracle

Notebook menghitung oracle begitu kedua baseline tersedia pada seed yang sama: satu
sampel benar bila TEXT **atau** IMAGE benar. Notebook juga mencetak proporsi
"keduanya benar" dan "keduanya salah" — yang terakhir adalah bagian yang **tidak**
bisa diselamatkan skema fusion mana pun yang bekerja dengan memilih.

Target paper: Oracle **92,1%**, yaitu 7,6% absolut di atas baseline terbaik.
FUSION memanen 3,3% dari itu, sekitar 43%.

Hasil oracle kita: diisi setelah eksekusi.

## Temuan

Diisi setelah eksekusi.
