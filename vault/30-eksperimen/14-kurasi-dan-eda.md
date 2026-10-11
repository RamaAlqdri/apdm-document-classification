---
judul: Kurasi dataset dan EDA per modalitas
tipe: eksperimen
tahap: "10"
tanggal: 2026-10-11
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/10
  - modal/teks
  - modal/citra
  - komponen/evaluasi
terkait:
  - "[[00-audit-data]]"
  - "[[larson-2025-id-codes]]"
  - "[[13-ocr-multi-mesin]]"
  - "[[peta-eksperimen]]"
---

# Kurasi dataset dan EDA per modalitas

model:: MobileNetV2 pretrained (tanpa fine-tune) + rata-rata FastText
dataset:: Tobacco3482 + QS-OCR, 3.482 dokumen
split:: seed 42 untuk analisis berlabel; seed 42/43/44 di manifest
seed:: 42
oa:: -
macro_f1:: -
durasi::

> **BELUM DIJALANKAN.** Notebook: `notebooks/11_eda.ipynb`, modul: `src/eda.py`.
> Diminta dosen sebagai luaran tambahan: dataset terkurasi, notebook EDA, dan laporan
> singkat temuan EDA.

## Tujuan

Tiga luaran: `data/processed/manifest_kurasi.csv` (split train/val/test plus penanda),
EDA per modalitas, dan bahan untuk laporan singkat. Bagian F (perbandingan antar-mesin
OCR) ditunda sampai [[13-ocr-multi-mesin]] dijalankan.

## Keputusan desain

1. **Kurasi menandai, tidak membuang.** Derau OCR adalah objek penelitian. Penanda:
   teks kosong, teks pendek, OOV tinggi, resolusi rendah, buram, dan duplikat.
2. **Ambang diambil dari ekor distribusi (persentil 5/95)**, bukan angka yang dipilih
   sebelum melihat data. Ini pelajaran dari ambang biner di notebook 07–09.
3. **Analisis berlabel hanya memakai bagian fit seed 42**: kata khas, kemiripan
   antar-kelas, dan classifier kode ID. Data uji hanya untuk akurasi probe.
4. **Duplikat dicek dua tahap.** Tahap pertama: korelasi thumbnail 64×64 yang sudah
   dinormalkan. Tahap kedua: konfirmasi lewat Jaccard teks OCR. Versi pertama memakai
   dHash 64-bit dan, pada fixture sintetis, menandai 77 dari 80 halaman sebagai duplikat.
   Halaman dokumen sebagian besar putih, jadi hash sekasar itu menyamakan semuanya.
   Self-check `src/eda.py` kini menguji kasus itu secara eksplisit.
5. **Kode ID dideteksi secara heuristik** (deret ≥ 6 digit, bukan desimal, tanggal, atau
   telepon). Jalan pintasnya diuji dengan classifier yang hanya diberi 3 digit awal kode;
   pembandingnya 3 karakter awal ID berkas. Lihat [[larson-2025-id-codes]].
6. **Fitur teks diambil dari sekuens notebook 02**, sehingga FastText (15 GB RAM) tidak
   perlu dimuat ulang.

## Verifikasi sebelum diserahkan

`src/eda.py` punya self-check. Notebook juga dieksekusi utuh pada fixture sintetis
(80 halaman, 10 kelas), yang mencakup halaman kosong, duplikat sungguhan, dan halaman
bertemplat sama. Tidak ada galat. Angka dari fixture tidak bermakna dan tidak dicatat di
sini.

## Hasil

> Diisi dari eksekusi nyata di laptop compute.

## Temuan
