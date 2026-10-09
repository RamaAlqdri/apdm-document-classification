---
judul: Peta Metode
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
  - "[[peta-eksperimen]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
---

# Peta Metode

Peta konsep untuk `20-metode/`. Seluruh catatan di bawah sudah ditulis di Tahap 2.
Spesifikasi teknis lengkapnya ada di [[1907.06370-spesifikasi-implementasi]],
ringkasan dan catatan kritis di [[1907.06370-ringkasan]].

## Cabang teks

Citra → OCR → token → embedding kata → embedding dokumen atau sekuens → model.

- [[ocr-tesseract]] — bagaimana teksnya dihasilkan, dan apa yang rusak
- [[noise-ocr-dan-oov]] — akibatnya: salah eja dan kata di luar kosakata
- [[fasttext-subword]] — kenapa subword menjawab masalah OOV (inti argumen paper)
- [[sif-document-embedding]] — weighted average + PCA removal, input untuk MLP
- [[cnn-1d-teks]] — sekuens 500×300, konvolusi 1D, max-pool-through-time

## Cabang citra

- [[mobilenetv2]] — backbone yang dipilih, alasan efisiensi
- [[inverted-residual-block]] — blok penyusunnya

## Penggabungan & evaluasi

- [[strategi-fusion-concat-vs-sum]] — dua strategi, dan kenapa penjumlahan gagal
- [[oracle-sebagai-batas-atas]] — batas atas teoretis dari dua baseline unimodal
- [[taksonomi-multimodal]] — posisi paper dalam kerangka representation /
  translation / alignment / fusion / co-learning, termasuk keberatan bahwa teks
  di sini **diturunkan** dari citra sehingga bukan modalitas independen

## Urutan baca yang disarankan

[[ocr-tesseract]] → [[noise-ocr-dan-oov]] → [[fasttext-subword]] →
[[cnn-1d-teks]] → [[mobilenetv2]] → [[strategi-fusion-concat-vs-sum]] →
[[oracle-sebagai-batas-atas]] → [[taksonomi-multimodal]]

## Catatan referensi

Delapan catatan pustaka di `90-referensi/`, semuanya ditulis dari cara paper kita
mengutipnya dan bukan dari membaca sumber aslinya:

[[harley-2015-rvl-cdip]] · [[kumar-2014-tobacco3482]] ·
[[sandler-2018-mobilenetv2]] · [[bojanowski-2017-fasttext]] · [[arora-2017-sif]] ·
[[kim-2014-cnn-sentence]] · [[eitel-2015-multimodal-rgbd]] ·
[[kay-2007-tesseract]]

### Fase 2 (≥2025, dibaca dari teks lengkap)

Referensi utama dan pembingkai gap ada di `10-jurnal/`: [[kashyap-2026-ringkasan]] dan
[[mironczuk-2026-review-fusion]]. Pendukung di `90-referensi/`:

- [[francis-2025-perbandingan-ocr]] — Tesseract, EasyOCR, PaddleOCR, Keras-OCR
  dibandingkan, tapi hanya pada tingkat OCR, bukan dampaknya pada klasifikasi
- [[zhang-2025-ocr-hinders-rag]] — galat OCR merambat ke tugas lanjutan
- [[michail-2025-embedding-tahan-ocr]] — embedding yang tahan derau OCR
- [[larson-2025-id-codes]] — jalan pintas kode ID di Tobacco3482; *baru dari abstrak*

## Tiga hal yang harus kita uji sendiri, bukan diwarisi dari paper

1. **Kegagalan fusion penjumlahan** dilaporkan tanpa angka dan tanpa spesifikasi
   mekanisme. Lihat [[strategi-fusion-concat-vs-sum]].
2. **Klaim "fusion hampir tidak pernah kalah per kelas"** dibantah tabel paper
   sendiri di dua kelas (Resume, Advertisement).
3. **Ketahanan terhadap degradasi** disebut sebagai limitasi tapi tidak diuji.
   Itu isi Tahap 7, dengan keterbatasan desain yang dicatat di
   [[taksonomi-multimodal]].
