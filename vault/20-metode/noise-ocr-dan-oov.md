---
judul: Noise OCR dan masalah out-of-vocabulary
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/teks
  - komponen/ocr
  - komponen/embedding
terkait:
  - "[[ocr-tesseract]]"
  - "[[fasttext-subword]]"
  - "[[1907.06370-ringkasan]]"
  - "[[peta-metode]]"
---

# Noise OCR dan masalah out-of-vocabulary

## Inti gagasan

Embedding kata seperti word2vec dan GloVe dilatih di korpus bersih — Wikipedia,
novel, berita. Asumsinya: tokenisasi mudah, dan kata yang muncul adalah kata
sungguhan. Teks OCR melanggar kedua asumsi itu sekaligus. Akibatnya banyak token
yang tidak punya vektor sama sekali, yaitu **out-of-vocabulary** (OOV).

## Cara kerja

Dua jenis kerusakan, dan penting membedakannya:

1. **Salah eja** — kata nyata yang terbaca sedikit keliru: `specifically` jadi
   `Specificalily`, `filter` jadi `fiilter`. Bentuknya masih mirip aslinya.
2. **Halusinasi** — string yang bukan kata apa pun, muncul dari binarisasi buruk
   atau noise salt-and-pepper. Tidak ada padanan yang bisa dipulihkan.

Skalanya, menurut paper, pada korpus Tobacco3482: rata-rata satu dokumen
menghasilkan 136 kata dengan panjang minimal 4 karakter. Dari jumlah itu hanya
118 ada di GloVe dan 114 ada di kamus Enchant. Secara keseluruhan **sekitar 26%
korpus di luar kamus** dan **23% di luar GloVe**. Distribusinya berekor panjang:
mayoritas dokumen sekitar 10% OOV, tapi ada puluhan dokumen yang isinya hampir
seluruhnya bukan kata bahasa Inggris.

## Kaitan dengan paper ini

Dua penanganan OOV yang biasa dipakai justru berbahaya di sini. **Vektor acak**
menyuntikkan fitur tak diskriminatif. **Pemetaan ke kata terdekat lewat jarak
Levenshtein** tampak masuk akal, tapi kalau 26% korpus harus dipetakan paksa, yang
masuk ke model adalah tebakan dalam jumlah besar.

Karena itulah paper beralih ke embedding berbasis karakter. Lihat
[[fasttext-subword]] — argumen itu adalah poros seluruh cabang teks.

## Batasan / catatan kritis

Salah eja dan halusinasi tidak setara, tapi paper memperlakukannya sebagai satu
masalah. FastText bisa memberi vektor masuk akal untuk `fiilter`; untuk string
sampah ia juga memberi vektor, hanya saja vektor itu tidak berarti apa-apa. Jadi
subword menyelesaikan kerusakan jenis pertama dan sekadar **menyembunyikan** yang
kedua. Ini yang membuat max-pooling di [[cnn-1d-teks]] penting: model perlu bisa
mengabaikan posisi, bukan cuma punya vektor untuk setiap posisi.
