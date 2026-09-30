---
judul: Kay (2007) — Tesseract OCR
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/teks
  - komponen/ocr
terkait:
  - "[[ocr-tesseract]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[peta-metode]]"
---

# Kay (2007) — Tesseract OCR

> A. Kay, "Tesseract: An Open-Source Optical Character Recognition Engine,"
> Linux Journal, July 2007.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca artikel aslinya.*

## Kontribusi

Artikel pengenalan mesin OCR open-source Tesseract. Tesseract berasal dari
penelitian HP pada 1980-an, lalu dibuka sebagai open source dan dikembangkan lebih
lanjut. Artikel 2007 ini mendeskripsikan versi yang masih berbasis pencocokan pola
karakter dan analisis komponen terhubung.

Yang dipakai QS-OCR adalah versi 4.0, yang jauh berbeda: mesin pengenalannya
berbasis **LSTM** dan membaca baris teks sebagai sekuens, bukan karakter sebagai
bentuk terisolasi. Jadi sitasi ini menunjuk alat yang sama tapi bukan algoritma yang
sama.

## Kenapa dikutip di paper kita

Sitasi alat. Tesseract adalah satu-satunya sumber modalitas teks di seluruh
pipeline — tanpa OCR tidak ada apa pun untuk difusikan.

## Relevansi untuk proyek kita

Kita tidak menjalankan Tesseract sendiri; teks sudah tersedia lewat QS-OCR-Small.
Yang perlu dipahami adalah konfigurasinya, karena itu menentukan sifat noise yang
harus ditangani cabang teks:

| Parameter | Nilai |
|---|---|
| versi | 4.0.0-beta.1 |
| `--oem 1` | LSTM only |
| `--psm 3` | full page tanpa OSD |
| `-l eng` | English |

Tanpa praproses citra, tanpa pascaproses teks. Perhatikan `--psm 3`: mode itu
**tidak** melakukan deteksi orientasi, bertentangan dengan deskripsi di §3.2 paper.
Lihat [[ocr-tesseract]] untuk pembahasannya, dan [[noise-ocr-dan-oov]] untuk
akibatnya.

Kalau nanti perlu meregenerasi OCR sendiri, script di repo QS-OCR bernama
`tobacco3482.sh` — README-nya menyebut `tobacco3842.sh` (angka tertukar) sehingga
mengikuti README mentah-mentah akan kena *file not found*.

Itu bukan sekadar catatan kaki: menjalankan Tesseract ulang pada citra yang terdegradasi
adalah satu-satunya cara mengukur ketahanan **sistem ujung ke ujung**, dan ketidakmampuan
kami melakukannya adalah limitasi D1 di [[limitasi-dan-lanjutan]]. Lihat juga
[[07-ablasi-degradasi-citra]].
