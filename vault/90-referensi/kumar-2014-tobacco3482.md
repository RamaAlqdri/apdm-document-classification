---
judul: Kumar et al. (2014) — Tobacco3482
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/citra
terkait:
  - "[[harley-2015-rvl-cdip]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
  - "[[peta-proyek]]"
---

# Kumar et al. (2014) — Tobacco3482

> J. Kumar et al., "Structural similarity for document image classification and
> retrieval," Pattern Recognition Letters, 2014.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya.*

## Kontribusi

Sumber dataset **Tobacco3482**: 3.482 dokumen hitam-putih dengan anotasi 10 kelas,
subset dari arsip litigasi industri tembakau. Paper aslinya mengusulkan ukuran
kemiripan struktural untuk klasifikasi dan temu-kembali citra dokumen — yaitu
pendekatan pra-deep-learning berbasis tata letak.

## Kenapa dikutip di paper kita

Ini dataset utama Audebert et al., dan satu-satunya yang kita pakai. Sepuluh
kelasnya: Advertisement, Email, Form, Letter, Memo, News, Note, Report, Resume,
Scientific.

## Relevansi untuk proyek kita

Dataset inti proyek. Beberapa sifatnya yang menentukan keputusan teknis kita:

- **Tidak ada split resmi.** Paper memakai k-fold dengan 800 train; kita 3 random
  split terstratifikasi, dicatat sebagai deviasi.
- **Kelasnya timpang.** Jumlah sampel per kelas tidak seragam — itu yang diukur di
  EDA Tahap 3, dan alasan F1 class-balanced dilaporkan berdampingan dengan OA.
- **Terlalu bersih.** Semua dokumen berorientasi benar dan dipindai profesional.
  Penulis mengakui ini sebagai limitasi utama; ablasi Tahap 7 kita menguji limitasi
  itu secara empiris.
- **Subset RVL-CDIP**, lihat [[harley-2015-rvl-cdip]].

Format aslinya TIF di server UMIACS yang sering sulit diakses, jadi kita memakai
versi JPG dari Kaggle. Teks QS-OCR di-OCR dari TIF asli — sumber kedua modalitas
kita karena itu tidak identik.
