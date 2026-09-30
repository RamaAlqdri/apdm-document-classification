---
judul: Inverted residual block
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/citra
  - komponen/cnn
terkait:
  - "[[mobilenetv2]]"
  - "[[sandler-2018-mobilenetv2]]"
  - "[[peta-metode]]"
---

# Inverted residual block

## Inti gagasan

Residual block klasik (ResNet) berbentuk **lebar - sempit - lebar**: kanal
dikompresi di tengah blok lalu dikembangkan lagi, dan koneksi residual
menghubungkan dua ujung yang lebar. Inverted residual membalik urutannya menjadi
**sempit - lebar - sempit**. Konvolusi mahal dikerjakan di bagian lebar di dalam
blok, sementara yang mengalir ke layer berikutnya adalah representasi sempit.

## Cara kerja

Satu blok, tiga langkah:

1. **Expansion** — konvolusi 1x1 dengan aktivasi identitas, menambah jumlah kanal.
2. **Depthwise 3x3** — konvolusi terpisah per kanal, diikuti ReLU. Ini yang murah:
   konvolusi biasa mengalikan semua kanal masukan dengan semua kanal keluaran,
   depthwise memprosesnya satu-satu.
3. **Projection** — konvolusi 1x1 mengembalikan ke jumlah kanal yang sempit.

Koneksi residual dipasang antara dua ujung yang sempit. Karena itulah yang
diteruskan antar blok berukuran kecil, aktivasi yang harus disimpan di memori jauh
lebih sedikit daripada ResNet dengan kapasitas setara.

## Kaitan dengan paper ini

Ini alasan struktural di balik pilihan [[mobilenetv2]]. Efisiensi MobileNetV2 bukan
hasil membuat jaringan lebih dangkal atau lebih sempit begitu saja, melainkan hasil
memindahkan biaya komputasi ke tempat yang paling murah. Itu yang membuat inferensi
60ms di GPU dan 230ms di CPU masuk anggaran latensi paper.

## Batasan / catatan kritis

Aktivasi pada projection layer dibuat **linear**, bukan ReLU — itu bagian dari
judul paper aslinya ("Linear Bottlenecks"). Alasannya: ReLU pada ruang berdimensi
rendah membuang informasi, karena ia memotong semua nilai negatif dan di ruang
sempit tidak ada dimensi cadangan untuk memulihkannya. Paper Audebert et al.
menyebut konvolusi 1x1 terakhir memakai ReLU; deskripsi itu tidak persis sesuai
desain MobileNetV2 aslinya. Kalau kita memakai `torchvision.models.mobilenet_v2`
kita mendapat implementasi resmi, jadi tidak ada yang perlu diperbaiki — hanya
jangan mengutip deskripsi paper sebagai spesifikasi arsitektur. Dicatat juga di
[[sandler-2018-mobilenetv2]].

Efisiensi blok ini terbukti di praktik: baseline citra kami selesai dalam 120 menit untuk
tiga seed di GPU 4 GB ([[04-baseline-mobilenetv2]]).
