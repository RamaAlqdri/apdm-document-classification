---
judul: SIF sebagai embedding dokumen
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/teks
  - komponen/embedding
terkait:
  - "[[fasttext-subword]]"
  - "[[cnn-1d-teks]]"
  - "[[arora-2017-sif]]"
  - "[[peta-metode]]"
---

# SIF sebagai embedding dokumen

## Inti gagasan

FastText memberi satu vektor per kata. MLP butuh satu vektor per dokumen. SIF
(*Smooth Inverse Frequency*) adalah cara paling murah untuk menjembatani keduanya,
dan ternyata sulit dikalahkan oleh metode yang jauh lebih rumit.

## Cara kerja

Dua langkah.

**Pertama, rata-rata berbobot.** Setiap kata diberi bobot `a / (a + p(w))`, dengan
`p(w)` frekuensi kata itu di korpus dan `a` konstanta kecil (umumnya 1e-3). Kata
yang sering muncul — `the`, `of`, `and` — mendapat bobot kecil; kata yang jarang
mendapat bobot besar. Semangatnya mirip IDF, tapi diturunkan dari model generatif,
bukan dipasang sebagai heuristik.

**Kedua, pembuangan komponen utama.** Hitung PCA atas seluruh embedding dokumen di
korpus, lalu proyeksikan keluar komponen pertama. Komponen itu biasanya menangkap
arah yang dimiliki semua dokumen — semacam bias sintaksis bersama — yang tidak
membedakan apa pun. Membuangnya menajamkan sisanya.

## Kaitan dengan paper ini

SIF dipakai sebagai masukan baseline MLP, dan MLP itu kalah: 70,8% vs 73,9% untuk
CNN1D. Jadi SIF **bukan** yang dipakai di model fusion akhir.

Tapi perbandingannya berguna justru karena kalah. Penjelasan paper: merata-ratakan
seluruh embedding kata mengencerkan informasi yang membedakan, karena hanya
sebagian teks yang relevan. Lebih buruk lagi, vektor dari kata halusinasi OCR ikut
masuk ke rata-rata dan tidak bisa dibuang. Lihat [[cnn-1d-teks]] untuk mekanisme
yang menghindari ini.

Membaca kekalahan SIF sebagai "rata-rata itu buruk" terlalu cepat, sebab 70,8%
dari teks OCR yang rusak tetap bukan angka sepele.

## Batasan / catatan kritis

Paper tidak menyebut nilai `a` maupun berapa komponen PCA yang dibuang. Kita pakai
default SIF (`a = 1e-3`, buang 1 komponen) dan catat itu sebagai asumsi — lihat
tabel Ambiguitas di [[1907.06370-spesifikasi-implementasi]].

Satu jebakan implementasi: PCA harus dipasang **hanya pada data train**, lalu
diterapkan ke test. Menghitung PCA atas seluruh korpus termasuk test adalah
kebocoran informasi, walau halus dan mudah terlewat.
