---
judul: Kim (2014) — CNN untuk klasifikasi kalimat
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/teks
  - komponen/cnn
terkait:
  - "[[cnn-1d-teks]]"
  - "[[sif-document-embedding]]"
  - "[[peta-metode]]"
---

# Kim (2014) — CNN untuk klasifikasi kalimat

> Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP,
> Oct. 2014.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya.*

## Kontribusi

Menunjukkan konvolusi — yang lahir untuk citra — bekerja sangat baik pada teks bila
kalimat direpresentasikan sebagai matriks vektor kata. Arsitekturnya dangkal: satu
lapis konvolusi dengan beberapa ukuran filter, max-pooling-over-time, dropout,
lalu softmax. Sederhana, tapi kompetitif dengan model yang jauh lebih rumit pada
banyak tolok ukur.

Gagasan yang paling diwarisi banyak paper sesudahnya adalah
**max-pooling-over-time**: ambil aktivasi maksimum setiap filter atas seluruh
panjang kalimat. Hasilnya ukuran tetap tanpa peduli panjang masukan, dan model
belajar mendeteksi *apakah* suatu pola ada, bukan di mana posisinya.

## Kenapa dikutip di paper kita

Inspirasi langsung arsitektur CNN1D Audebert et al. Mereka memperdalamnya jadi 4
layer dengan 512 channel dan window 12, tapi max-pooling-through-time-nya berasal
dari sini.

## Relevansi untuk proyek kita

Ini yang menjelaskan **mengapa** CNN1D mengalahkan MLP+SIF pada teks OCR, bukan
sekadar bahwa ia mengalahkannya. Max-pool membiarkan model mengabaikan bagian
dokumen yang tidak berguna; rata-rata memaksa setiap posisi ikut berkontribusi
termasuk posisi yang berisi sampah OCR. Lihat [[cnn-1d-teks]].

Satu catatan: di Kim, konvolusi bekerja pada kalimat pendek dan window filter
kecil (3-5 kata). Audebert et al. memakai window 12 pada dokumen sepanjang 500 kata.
Perluasan skala itu masuk akal, tapi tidak diuji — dan di teks OCR yang urutannya
bisa kacau, tidak jelas apakah jendela selebar itu menangkap frasa atau bertindak
sebagai detektor kemunculan kata longgar.

Arsitektur turunannya berhasil sebagai model mandiri ([[03-baseline-cnn1d]], OA 0,7266)
tapi **kolaps** ketika dipakai sebagai cabang di dalam fusion — varians 38x lebih kecil,
67 dari 128 unit mati. Diukur di [[11-diagnostik-cabang-teks]].
