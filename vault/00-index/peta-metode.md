---
judul: Peta Metode
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: wip
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
  - "[[peta-eksperimen]]"
---

# Peta Metode

Peta konsep untuk `20-metode/`. Catatan yang belum ada ditulis di Tahap 2 —
wikilink merah di bawah ini adalah daftar kerjanya.

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
