---
judul: OCR dengan Tesseract
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/teks
  - komponen/ocr
terkait:
  - "[[noise-ocr-dan-oov]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
  - "[[kay-2007-tesseract]]"
  - "[[peta-metode]]"
---

# OCR dengan Tesseract

## Inti gagasan

Dataset Tobacco3482 hanya berisi citra. Teksnya ada, tapi terkubur di piksel.
Tesseract-lah yang menggalinya, dan sifat galian itu menentukan seluruh kualitas
cabang teks. Versi 4.0 yang dipakai berbasis LSTM, bukan lagi pencocokan pola
karakter seperti versi lama — jadi ia membaca baris sebagai sekuens dan bisa
memakai konteks, tapi tetap bisa berhalusinasi kata yang tidak ada.

## Cara kerja

Alurnya: binarisasi dengan thresholding Otsu untuk memisahkan tulisan gelap dari
latar putih, segmentasi halaman untuk menemukan blok teks, lalu pengenalan
per-baris oleh LSTM.

Parameter yang dipakai QS-OCR:

| Parameter | Nilai | Arti |
|---|---|---|
| `--oem 1` | LSTM only | engine neural, bukan legacy |
| `--psm 3` | full page, tanpa OSD | segmentasi otomatis penuh, **tanpa** deteksi orientasi/script |
| `-l eng` | English | model bahasa Inggris |

Tanpa praproses citra dan tanpa pascaproses teks. Salah eja dibiarkan apa adanya —
itu keputusan sadar penulis, karena justru itu yang mau diuji.

## Kaitan dengan paper ini

Teks OCR adalah satu-satunya sumber modalitas kedua. Semua yang rusak di sini
merambat ke [[noise-ocr-dan-oov]] dan menjelaskan mengapa cabang teks berhenti di
73,8% sementara citra mencapai 84,5%.

## Batasan / catatan kritis

**Ada kontradiksi di paper.** Paper §3.2 menulis Tesseract akan mencoba mendeteksi
orientasi teks dan melakukan transformasi afin bila perlu. Itu tidak benar untuk
konfigurasi yang dipakai: `--psm 3` justru mode yang secara eksplisit
**mematikan** orientation and script detection. OSD hanya aktif di `--psm 0`
(hanya OSD) dan `--psm 1` (segmentasi otomatis **dengan** OSD). Jadi dokumen
miring tidak akan diperbaiki.

Ini tidak berdampak pada Tobacco3482 karena semua dokumennya sudah berorientasi
benar — tapi penulis sendiri menyebut orientasi sebagai masalah utama di aplikasi
nyata, dan justru mematikannya di pipeline mereka.

Ablasi rotasi di [[07-ablasi-degradasi-citra]] menunjukkan kenapa itu penting: rotasi 45
derajat menjatuhkan baseline citra dari 0,8315 ke 0,3031. Statistik noise yang dihasilkan
konfigurasi OCR ini diukur di [[01-ekstraksi-fitur-teks]].
