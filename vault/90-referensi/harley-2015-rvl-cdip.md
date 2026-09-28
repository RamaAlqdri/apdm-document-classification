---
judul: Harley et al. (2015) — RVL-CDIP
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/citra
terkait:
  - "[[1907.06370-ringkasan]]"
  - "[[kumar-2014-tobacco3482]]"
  - "[[peta-metode]]"
---

# Harley et al. (2015) — RVL-CDIP

> A. W. Harley et al., "Evaluation of Deep Convolutional Nets for Document Image
> Classification and Retrieval," ICDAR, Aug. 2015.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya. Perlakukan sebagai orientasi, bukan sumber.*

## Kontribusi

Memperkenalkan **RVL-CDIP**: 400.000 citra dokumen grayscale berlabel, 16 kelas,
25.000 sampel per kelas, diambil dari arsip Truth Tobacco Industry Documents. Split
standarnya 320.000 train / 40.000 val / 40.000 test. Juga mengevaluasi CNN dalam
untuk klasifikasi dan temu-kembali dokumen berbasis citra.

## Kenapa dikutip di paper kita

Tiga peran sekaligus:

1. **Dataset besar** tempat Audebert et al. menunjukkan hasilnya konsisten pada
   skala berbeda (IMAGE 89,1% / TEXT 74,6% / FUSION 90,6%).
2. **Pembanding pada Tobacco3482** — ensemble CNN dari Harley mencapai 79,9%, di
   bawah baseline MobileNetV2 tunggal (84,5%).
3. **Sumber split standar** RVL-CDIP.

## Relevansi untuk proyek kita

Tidak langsung: kita tidak memakai RVL-CDIP karena 400.000 dokumen di luar
anggaran. Yang relevan adalah dua hal. Pertama, angka 79,9% jadi patokan bahwa
84,5% memang bukan baseline lemah. Kedua, **Tobacco3482 adalah subset RVL-CDIP** —
itu sebabnya QS-OCR-Small dan QS-OCR-Large punya irisan, dan mengapa transfer
learning antar keduanya akan bocor. Lihat [[kumar-2014-tobacco3482]].
