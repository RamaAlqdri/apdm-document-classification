---
judul: OCR multi-mesin — Tesseract, EasyOCR, PaddleOCR pada citra yang sama
tipe: eksperimen
tahap: "10"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/eksperimen
  - tahap/10
  - modal/teks
  - komponen/ocr
  - temuan/positif
  - temuan/anomali
terkait:
  - "[[kashyap-2026-ringkasan]]"
  - "[[francis-2025-perbandingan-ocr]]"
  - "[[ocr-tesseract]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[rencana-prompt]]"
---

# OCR multi-mesin

model:: Tesseract 5 / EasyOCR 1.7.2 / PaddleOCR 3.7.0 (PP-OCRv5 mobile)
dataset:: Tobacco3482, 3.482 JPG Kaggle
split:: - (seluruh dataset, belum ada pelatihan)
seed:: -
oa:: -
macro_f1:: -
durasi:: Tesseract 1,45 / EasyOCR 32,15 / PaddleOCR 6,99 dtk per halaman (median); PaddleOCR penuh 7,38 jam

> **DIJALANKAN** di laptop TUF (RTX 3050 4 GB, Windows) lewat `notebooks/pipeline.ipynb`
> bagian 5, hasil di-push 2026-10-24 (`a55f472`). Angka di bawah disalin dari
> `reports/fase2/ocr_*.csv` dan keluaran sel notebook.

## Tujuan

Langkah pertama untuk menjawab Q1 Fase 2: menyediakan teks Tobacco3482 dari **beberapa
mesin OCR, dibaca dari citra yang sama**. Gap-nya diambil dari
[[kashyap-2026-ringkasan]], yang memakai mesin komersial tanpa nama, dan dari
[[francis-2025-perbandingan-ocr]], yang membandingkan mesin hanya pada tingkat OCR.

Tahap ini **tidak** mengukur akurasi klasifikasi; itu Tahap 11. Di sini yang dihasilkan
adalah teksnya, kotak posisinya untuk Tahap 14, dan proksi kualitasnya.

## Konfigurasi

| Mesin | Pengaturan | Lingkungan |
|---|---|---|
| Tesseract | `--oem 1 --psm 3 -l eng`, sama dengan QS-OCR | utama, biner via winget |
| EasyOCR | `Reader(["en"])`, GPU bila ada | utama, numpy/torch dikunci |
| PaddleOCR | `PP-OCRv5_mobile_det` + `en_PP-OCRv5_mobile_rec`, CPU, tanpa koreksi orientasi | `.venv-paddle/` terpisah |
| QS-OCR | Tesseract 4.0.0-beta.1 pada TIF asli | acuan Fase 1, tidak dijalankan ulang |

Deviasi dan keputusan yang harus dibaca bersama hasilnya:

1. **Tesseract-JPG ≠ QS-OCR.** Versinya beda (5 vs 4-beta) dan citranya beda (JPG
   re-encode vs TIF asli). Selisih keduanya mengukur **efek gabungan**, tidak bisa
   dipisahkan.
2. **PaddleOCR memakai model mobile, bukan default.** paddleocr 3.7 diam-diam pindah ke
   `PP-OCRv6_medium`. Pada uji satu halaman sintetis di Mac, model itu 6,2 dtk/halaman,
   sedangkan pasangan v5 mobile 2,0 dtk/halaman. Teks keluarannya identik di halaman
   uji itu. Untuk 3.482 halaman di CPU, selisihnya beberapa jam. Bisa diganti lewat
   `PADDLE_MODEL = "default"`.
3. **Granularitas kotak berbeda.** Tesseract per **kata**; EasyOCR dan PaddleOCR per
   **baris**. Ini tidak berpengaruh di Tahap 11, karena yang dipakai teksnya. Di
   Tahap 14 harus diperhitungkan.
4. **Urutan baca mesin berbasis kotak direkonstruksi** oleh `reading_order()` (baris
   atas-bawah, kata kiri-kanan, satu kolom). Tesseract memakai urutan bawaannya sendiri.
   Karena itu ukuran kesepakatan sengaja dihitung atas *himpunan* kata, tanpa urutan.
5. **Keras-OCR tidak dijalankan.** Ia butuh TensorFlow dengan API Keras 2, yang hanya ada
   sebelum TF 2.16, padahal TF baru mendukung Python 3.12 sejak 2.16. Dependensinya,
   `imgaug` 0.4.0, juga masih memakai `np.bool` yang sudah dihapus dari numpy. Menjalankannya
   butuh venv Python 3.11 tersendiri. Ini **belum diputuskan**, dan perlu dikonfirmasi
   ke dosen.

## Proksi kualitas (karena CER tidak bisa dihitung)

Tobacco3482 tidak punya transkripsi acuan. Yang diukur:

| Proksi | Arti | Batasan |
|---|---|---|
| kata/halaman, % halaman kosong | mesin yang melewatkan blok teks | tidak membedakan kata benar dan sampah |
| OOV (`wordfreq`, zipf > 0) | kata ≥ 4 huruf di luar daftar kata Inggris | **batas bawah** galat: `tbe` (salah baca *the*) punya zipf 2,56 sehingga terhitung dikenal |
| Jaccard himpunan kata antar-mesin | seberapa sepakat dua mesin | sepakat tidak berarti benar |

Ukuran OOV ini **tidak sebanding** dengan OOV di [[01-ekstraksi-fitur-teks]], yang
memakai kosakata korpus sebagai acuan. Lihat [[noise-ocr-dan-oov]].

## Verifikasi sebelum diserahkan

Dilakukan di mesin agen, tanpa dataset:

- `python -m src.ocr`: self-check urutan baca, pengelompokan baris Tesseract, proksi,
  runner yang bisa dilanjutkan, dan halaman gagal yang dicatat lalu dicoba ulang.
- Ketiga adapter dijalankan sungguhan pada satu memo sintetis. Ketiganya membaca kelima
  baris dengan benar; EasyOCR salah membaca satu titik di akhir kalimat sebagai titik dua.
- Notebook 10 dieksekusi utuh pada fixture 4 halaman sintetis (Tesseract + EasyOCR),
  termasuk satu halaman kosong. Tidak ada galat.
- Pemasangan EasyOCR dengan constraint numpy terbukti mempertahankan numpy 1.26.4;
  opencv turun ke 4.11.0.86, versi terakhir yang menerima numpy 1.x.

## Hasil

Versi yang benar-benar dipakai (`reports/fase2/ocr_meta.json`): Tesseract **5.4.0**
(pytesseract 0.3.13, `--oem 1 --psm 3`), EasyOCR **1.7.2** di GPU (torch 2.5.1+cu121),
PaddleOCR **3.7.0** dengan **paddlepaddle 3.2.2** di CPU, model PP-OCRv5 mobile.

**Deviasi dari rencana:** paddlepaddle 3.3.1 gagal di setiap halaman pada CPU Windows
(oneDNN, `ConvertPirAttribute2RuntimeAttribute not support`). Menonaktifkan oneDNN
berhasil tapi ± 7× lebih lambat, sehingga versi diturunkan ke 3.2.2 (teks keluaran sama
pada uji). Perbaikan ini dibuat pengguna di `src/ocr.py` (`PADDLE_VERSION`). Halaman
Tesseract dan EasyOCR sudah ada dari sesi sebelumnya dan dilanjutkan; `_meta.json` ditulis
ulang pada run ini, jadi metadata mencerminkan lingkungan saat run terakhir.

### Resolusi citra

Lebar 817-3.520 px, median 1.728 px; perkiraan DPI median 203 (rentang 96-414).

### Kecepatan

| Mesin | dtk/halaman (median) | perkiraan 3.482 halaman |
|---|---|---|
| Tesseract (CPU) | 1,45 | 1,4 jam |
| EasyOCR (GPU) | 32,15 | ± 31 jam |
| PaddleOCR (CPU) | 6,99 | 7,38 jam (terukur) |

### Proksi kualitas per mesin

| Kondisi | kata/halaman (median) | halaman kosong | OOV median | OOV rata-rata |
|---|---|---|---|---|
| qs-ocr | 160 | 1,06 % | 0,055 | 0,098 |
| tesseract | 161 | 0,60 % | 0,057 | 0,115 |
| easyocr | 157 | 0,17 % | 0,136 | 0,232 |
| paddleocr | 162 | 0,06 % | 0,054 | 0,124 |

### Kesepakatan antar-mesin (median Jaccard himpunan kata)

| | qs-ocr | tesseract | easyocr | paddleocr |
|---|---|---|---|---|
| qs-ocr | 1 | 0,920 | 0,693 | 0,864 |
| tesseract | | 1 | 0,692 | 0,864 |
| easyocr | | | 1 | 0,734 |

### OOV per kelas (median)

Kelas tersulit untuk semua mesin adalah **Note** (qs 0,250 · tesseract 0,293 · easyocr 0,636 ·
paddleocr 0,345), lalu Advertisement dan News. Report, Letter, dan Memo di bawah 0,05 untuk
semua mesin kecuali EasyOCR.

## Temuan

1. **#temuan/positif — PaddleOCR menghasilkan teks paling lengkap:** halaman kosong hampir
   nol (0,06 %) dan OOV median terendah, meski dibaca dari JPG re-encode. Di
   [[15-grid-teks-ocr]] ini juga menjadi teks dengan akurasi klasifikasi tertinggi.
2. **#temuan/anomali — proksi OOV tidak memprediksi kegunaan teks.** EasyOCR punya OOV
   median 2,5× mesin lain dan kesepakatan terendah, tetapi cabang teksnya lebih akurat
   daripada Tesseract-JPG ([[15-grid-teks-ocr]]). Probe linear tanpa training
   ([[14-kurasi-dan-eda]], bagian 6.1 notebook) lebih baik meramalkan urutannya:
   paddleocr 0,686 > qs-ocr 0,669 > easyocr 0,652 ≈ tesseract 0,650.
3. Tesseract-JPG paling dekat dengan QS-OCR (Jaccard 0,92), wajar karena mesinnya
   sekeluarga; selisih keduanya adalah efek gabungan versi 5 vs 4-beta dan JPG vs TIF.
4. Biaya: EasyOCR di GPU 4 GB adalah yang paling mahal (± 22× Tesseract) walau tidak
   paling akurat.
