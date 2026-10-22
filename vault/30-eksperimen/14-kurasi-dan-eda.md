---
judul: Kurasi dataset dan EDA per modalitas
tipe: eksperimen
tahap: "10"
tanggal: 2026-10-11
status: selesai
tags:
  - tipe/eksperimen
  - tahap/10
  - modal/teks
  - modal/citra
  - komponen/evaluasi
  - temuan/positif
  - temuan/anomali
terkait:
  - "[[00-audit-data]]"
  - "[[larson-2025-id-codes]]"
  - "[[13-ocr-multi-mesin]]"
  - "[[peta-eksperimen]]"
---

# Kurasi dataset dan EDA per modalitas

> Sejak 2026-10-22 EDA ini dijalankan ulang oleh `notebooks/pipeline.ipynb` bagian 4 dan
> 6.1 (logika di `src/eda.py`), dengan keluaran di `reports/fase2/eda_*`. Angka di catatan
> ini tetap dari run notebook 11 (kini `notebooks/arsip/11_eda.ipynb`). Bagian 6.1 menambah
> probe teks per kondisi OCR.

model:: MobileNetV2 pretrained (tanpa fine-tune) + rata-rata FastText
dataset:: Tobacco3482 + QS-OCR, 3.482 dokumen
split:: seed 42 untuk analisis berlabel; seed 42/43/44 di manifest
seed:: 42
oa:: 0.7293 (probe citra) / 0.6685 (probe teks)
macro_f1:: 0.7099 / 0.6491
durasi:: ± 10 menit (ekstraksi CNN 301 dtk di CUDA)

> **DIJALANKAN 2026-10-11** di laptop TUF lewat `notebooks/11_eda.ipynb` (modul `src/eda.py`).
> Laporan singkatnya: `proposal/laporan-eda.pdf`.
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

| Pengukuran | Nilai |
|---|---|
| Pasangan lengkap | 3482/3482, rasio kelas 5,17 |
| Split per seed | 720 train / 80 val / 2682 test, komposisi identik di 42/43/44 |
| Ambang dari data | teks pendek ≤ 23 kata, OOV ≥ 0,369, DPI < 203, ketajaman < 788 |
| Tertandai | teks kosong 37, pendek 163, OOV tinggi 172, resolusi rendah 68, ketajaman rendah 175, duplikat 9 dokumen (12 pasangan) |
| Bocor latih–uji | 1 / 5 / 4 pasangan (seed 42 / 43 / 44) |
| Probe citra (MobileNetV2 beku) | OA 0,7293, macro F1 0,7099 |
| Probe teks (rata-rata FastText) | OA 0,6685, macro F1 0,6491 |
| Kelas mayoritas | 0,1782 |
| Jalan pintas kode ID dari teks OCR | OA 0,2782 |
| Jalan pintas 3 karakter ID berkas | OA 0,4720 |
| Teks: kata per dokumen | median 174, rata-rata 224; OOV median 5,5%, rata-rata 9,8% |

F1 probe teks − citra per kelas: Resume +0,237, Scientific +0,076, Email +0,018; Advertisement
−0,297 dan Note −0,289 (citra jauh lebih unggul).

## Temuan

**#temuan/positif — komplementaritas terukur sebelum pelatihan apa pun.** Teks unggul pada
Resume, Scientific, dan Email; citra unggul pada Advertisement dan Note. Pengelompokan t-SNE
kedua modalitas juga berbeda.

**#temuan/anomali — iklan terputar menghapus teks.** Dokumen dengan teks paling bermasalah
adalah iklan yang dipindai terputar 90°. Konfigurasi `--psm 3` tidak mendeteksi orientasi.
Ini relevan untuk [[13-ocr-multi-mesin]], karena mesin yang mengoreksi orientasi bisa paling
berbeda justru di sini.

**#temuan/anomali — penanda "ketajaman rendah" tercampur kepadatan tinta.** Email dan Note
paling jarang tintanya, sehingga variansi Laplacian-nya rendah walaupun tidak kabur. Ke-94
Email yang "citranya bermasalah" seluruhnya berasal dari penanda ini. Penanda ini tidak
boleh dibaca sebagai "kabur".

**Bias resolusi:** median dan kuartil DPI Email dan Resume sama-sama 203.

**Duplikat:** 6 dari 9 dokumen duplikat adalah Advertisement. Pasangan yang keduanya tanpa
teks terkonfirmasi otomatis, jadi duplikat ini perlu dicek visual sebelum split diubah.

**Kode ID:** prefiks ID berkas memprediksi kelas 47%, tetapi model kita tidak melihat nama
berkas. Kode ID di teks OCR hanya memberi +10 poin di atas mayoritas, padahal deteksinya
heuristik. Lihat [[larson-2025-id-codes]].
