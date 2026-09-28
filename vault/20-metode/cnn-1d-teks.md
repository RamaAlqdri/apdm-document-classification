---
judul: CNN 1D untuk sekuens teks
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/teks
  - komponen/cnn
terkait:
  - "[[sif-document-embedding]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[kim-2014-cnn-sentence]]"
  - "[[peta-metode]]"
---

# CNN 1D untuk sekuens teks

## Inti gagasan

Kalau dokumen direpresentasikan sebagai sekuens vektor kata dan bukan sebagai satu
vektor rata-rata, model bisa **memilih** bagian mana yang penting. Konvolusi 1D
mengerjakan pemilihan itu: ia menggeser jendela di sepanjang teks mencari pola
lokal, dan max-pooling kemudian menyimpan yang paling kuat sambil membuang sisanya.

## Cara kerja

Masukannya matriks 500 x 300 — 500 posisi kata, masing-masing vektor FastText 300
dimensi, di-zero-pad untuk dokumen yang lebih pendek. Arsitektur di paper:

- 4 layer konvolusi 1D, window 12, 512 channel per layer, aktivasi ReLU
- maxpooling stride 2 diselipkan antar layer
- **max-pooling-through-time** di akhir: ambil nilai maksimum setiap channel atas
  seluruh panjang sekuens
- Dropout, lalu fully connected, lalu softmax

Max-pooling-through-time adalah bagian yang paling menentukan. Ia mengubah sekuens
berapa pun panjangnya menjadi satu vektor berukuran tetap, dan jawabannya hanya
bergantung pada **di mana pola terkuat ditemukan** — bukan pada rata-rata seluruh
dokumen. Satu kalimat pembuka khas email bisa menentukan kelas tanpa terganggu 400
posisi lain yang berisi sampah OCR.

## Kaitan dengan paper ini

CNN1D mengalahkan MLP+SIF, 73,9% vs 70,8%, dan itulah alasan ia dipilih sebagai
cabang teks di model fusion. Mekanismenya persis kebalikan kelemahan
[[sif-document-embedding]]: rata-rata memaksa semua posisi berkontribusi, max-pool
membiarkan model mengabaikan yang tidak berguna. Terhadap masalah di
[[noise-ocr-dan-oov]], ini bukan perbaikan yang kebetulan — ini pertahanan
struktural terhadap noise.

## Batasan / catatan kritis

Window 12 itu lebar untuk teks — n-gram 12 kata. Di teks bersih itu hampir satu
klausa penuh. Di teks OCR yang urutannya bisa kacau karena segmentasi halaman,
tidak jelas apakah jendela selebar itu benar-benar menangkap frasa atau sekadar
bertindak sebagai detektor kemunculan kata longgar. Paper tidak mengujinya.

Padding 500 juga perlu dilihat dengan angka korpus: rata-rata dokumen hanya 136
kata dengan panjang minimal 4 karakter, jadi sebagian besar masukan kemungkinan
besar mayoritas nol. Berapa banyak — itu yang kita ukur sendiri di Tahap 4.

Perlakuan dokumen yang **lebih panjang** dari 500 kata tidak disebut paper sama
sekali. Kita asumsikan truncate.
