---
judul: FUSION — konkatenasi
tipe: eksperimen
tahap: "06"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/06
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[06-fusion-sum]]"
  - "[[03-baseline-cnn1d]]"
  - "[[04-baseline-mobilenetv2]]"
  - "[[tabel-utama]]"
---

# FUSION — konkatenasi

model:: FUSION-concat
dataset:: Tobacco3482 (citra JPG + QS-OCR-small)
split:: 800 train (termasuk 10% val) / 2682 test, stratified
seed:: 42, 43, 44
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Arsitektur lolos self-check bentuk tensor dan
> aliran gradien, tapi belum pernah dilatih. Field angka kosong sampai dijalankan
> di mesin compute.

## Arsitektur

| Bagian | Keluaran | Sumber |
|---|---|---|
| Cabang citra | MobileNetV2 → GAP → 1280 → FC → **128** | paper |
| Cabang teks | CNN1D → **128** | paper |
| Penggabungan | konkatenasi → 256 | paper |
| Head | Linear 256→256, BN, ReLU, Dropout, Linear 256→10 | **asumsi kita** |
| Parameter | 13.804.810 | terhitung dari self-check |

Classifier kedua cabang diganti `nn.Identity` — paper memotong layer akhir, dan
membiarkan Linear mati di dalamnya berarti setiap checkpoint membawa bobot tak
terpakai. Self-check memverifikasi `state_dict` benar-benar bersih dari keduanya.

## Tiga ambiguitas yang diputuskan di sini

Ketiganya sebelumnya tercatat terbuka di
[[1907.06370-spesifikasi-implementasi]]:

1. **Arsitektur head fusion** (ambiguitas #5). Paper hanya menyebut "multi-layer
   perceptron following [Eitel]". Kita pakai satu layer tersembunyi 256 dengan
   BN/ReLU/Dropout.
2. **Inisialisasi cabang** (ambiguitas #9). "Dilatih end-to-end" kita baca sebagai:
   cabang citra mulai dari bobot ImageNet persis seperti baseline IMAGE, cabang teks
   mulai dari nol. Bukan dari checkpoint baseline yang sudah kita latih sendiri.
3. **Mekanisme penjumlahan adaptif** (ambiguitas #3) — lihat [[06-fusion-sum]].

## Risiko yang harus diperiksa di log

Satu cabang pretrained dilatih bersama satu cabang dari nol pada learning rate yang
sama. Kalau cabang citra konvergen jauh lebih cepat, fusion bisa runtuh menjadi
model citra dengan parameter tambahan — hasilnya akan terlihat "baik" (setara
baseline IMAGE) sambil sebenarnya gagal memakai teks.

Indikator: kalau OA fusion praktis sama dengan baseline IMAGE dan F1 kelas **Resume**
tidak naik mendekati nilai TEXT, curigai ini. [[06-fusion-sum]] memberi indikator
yang lebih langsung lewat bobot cabang terlatihnya.

## Hyperparameter

SGD momentum 0,9, lr 0,01, batch 40, 200 epoch (paper §4.2). Early stopping
`patience=15` adalah deviasi kita.

## Hasil

Diisi setelah eksekusi.

| Seed | OA | Macro F1 | Epoch terbaik | Durasi |
|---|---|---|---|---|
| 42 | | | | |
| 43 | | | | |
| 44 | | | | |
| **rata-rata ± std** | | | | |

Target paper: **OA 87,8% / F1 0,86**, yaitu +3,3% absolut di atas baseline IMAGE.

Deviasi compute yang benar-benar terjadi: diisi setelah eksekusi.

## Pemeriksaan klaim per kelas

Paper menulis gain-nya konsisten dan fusion "hampir tidak pernah" di bawah salah
satu baseline pada kelas mana pun. Tabel 3 paper sendiri punya **dua** pengecualian:
Resume (TEXT 0,97 > FUSION 0,96) dan Advertisement (IMAGE 0,94 > FUSION 0,93).

Notebook menghitung jumlah pengecualian pada hasil kita. Kalau jauh lebih dari dua,
itu `#temuan/negatif` dan dilaporkan apa adanya.

Jumlah kelas di mana FUSION kalah dari baseline terbaiknya: diisi setelah eksekusi.

Confusion matrix: `reports/figures/06_confusion_fusion_concat_seed42.png`.

## Temuan

Diisi setelah eksekusi.
