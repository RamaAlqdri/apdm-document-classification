---
judul: Baseline TEXT — MLP atas embedding SIF
tipe: eksperimen
tahap: "05"
tanggal: 2026-09-28
status: selesai
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
oa:: 0.6780
macro_f1:: 0.6616
durasi:: 48s

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050, CUDA). Angka di bawah dari
> eksekusi nyata `notebooks/03_baseline_teks.ipynb`.

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

| Seed | OA | Macro F1 | Epoch terbaik | Epoch dijalankan | Durasi |
|---|---|---|---|---|---|
| 42 | 0,6779 | 0,6628 | 32 | 47 | 21,2s |
| 43 | 0,6820 | 0,6577 | 21 | 36 | 13,1s |
| 44 | 0,6741 | 0,6644 | 22 | 37 | 13,6s |
| **rata-rata ± std** | **0,6780 ± 0,0032** | **0,6616 ± 0,0029** | 25 | 40 | 48s total |

Target paper (Tabel 1a): **OA 70,8% / F1 0,69**. Selisih kita **−0,030 OA**.

Variansnya sangat kecil (± 0,003) — model ini stabil antar seed, jauh lebih stabil
daripada IMAGE (± 0,021).

## Perbandingan dengan paper

**Urutan paper bertahan: MLP (0,6780) tetap di bawah CNN1D (0,7266), selisih
0,0486.** Paper melaporkan selisih 0,031 (70,8% vs 73,9%), jadi jarak kita bahkan
lebih lebar.

Penjelasan paper untuk kekalahan ini terkonfirmasi secara tidak langsung oleh Tahap
7: merata-ratakan seluruh embedding kata mengencerkan informasi, dan memang ablasi
menunjukkan sinyal teks yang berguna sangat terkonsentrasi — menghapus 75% kata pun
tidak mengubah akurasi fusion. Representasi yang bisa **memilih** (max-pool CNN1D)
memang punya keuntungan struktural di sini.

Kedua angka kita 2-3% di bawah paper, konsisten dengan seluruh baseline lain —
bukan cacat khusus model ini.
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


## Temuan

**#temuan/positif — urutan MLP < CNN1D mereplikasi paper**, dengan margin yang lebih
lebar (0,049 vs 0,031).

**#temuan/positif — model paling stabil di seluruh proyek**, std OA hanya 0,0032.
Wajar: masukannya vektor 300 dimensi yang sudah dihitung sebelumnya, tidak ada
augmentasi, dan modelnya kecil.

Catatan: MLP tidak dipakai di model fusion, sesuai paper. Perannya murni pembanding
untuk membenarkan pemilihan CNN1D.
