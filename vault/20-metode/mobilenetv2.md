---
judul: MobileNetV2 sebagai backbone citra
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
  - "[[inverted-residual-block]]"
  - "[[sandler-2018-mobilenetv2]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
  - "[[peta-metode]]"
---

# MobileNetV2 sebagai backbone citra

## Inti gagasan

Cabang citra tidak memakai arsitektur terkuat yang tersedia, melainkan yang paling
murah dengan performa masih kompetitif. Pilihan ini digerakkan kebutuhan produk,
bukan kebutuhan akurasi: sistemnya harus merespons pengguna dalam hitungan detik,
dan Tesseract sendiri sudah menghabiskan sekitar 910ms dari anggaran itu.

## Cara kerja

MobileNetV2 adalah tumpukan 19 bottleneck residual layer, masing-masing dibangun
dari inverted residual block — lihat [[inverted-residual-block]]. Akurasi top-1
ImageNet-nya setara VGG-16 tapi jauh lebih cepat.

Adaptasi untuk dokumen:

| Item | Nilai | Alasan |
|---|---|---|
| Input | 384 x 384 | jauh di atas 224 standar; tulisan perlu resolusi |
| Resize | tanpa padding, aspect ratio di-warp | Tensmeyer & Martinez melaporkan warp lebih baik daripada padding pada resolusi sama |
| Kanal | grayscale diduplikasi 3x | supaya bobot pretrained RGB bisa dipakai |
| Bobot awal | pretrained ImageNet, di-fine-tune penuh | mempercepat konvergensi dan menaikkan akurasi |

Untuk fusion, lapisan klasifikasinya dipotong: feature map konvolusi terakhir
diringkas dengan global average pooling per channel menjadi vektor 1280 dimensi,
lalu dipetakan lewat fully connected ke 128 dimensi.

## Kaitan dengan paper ini

Hasilnya 84,5% pada Tobacco3482 — sendirian sudah mengalahkan ensemble CNN dari
Harley et al. (79,9%) dan jadi baseline terkuat yang harus dilampaui fusion.

Satu temuan negatif yang berguna: **augmentasi data tidak membantu**, 83,9% dengan
vs 84,5% tanpa. Penjelasan penulis masuk akal — semua dokumennya grayscale,
tulisan gelap di latar putih, baris horizontal, jadi tidak ada variasi warna atau
geometri yang perlu ditiru. Ini sekaligus mengonfirmasi betapa homogen datasetnya,
dan justru itu yang memotivasi ablasi degradasi di Tahap 7.

## Batasan / catatan kritis

Input 384 x 384 dengan aspect ratio di-warp berarti dokumen portrait A4 digencet
horizontal cukup parah. Paper menerimanya berdasarkan temuan orang lain, bukan
pengujian sendiri.

Dua hal yang tidak disebut dan harus kita asumsikan: normalisasi citra (kita pakai
statistik ImageNet, konsisten dengan bobot pretrained) dan apakah ada layer yang
dibekukan (kita fine-tune penuh, sesuai narasi paper). Bobot MobileNetV2
torchvision juga tidak identik dengan versi Keras yang dipakai penulis — deviasi
yang sudah pasti, didaftar sebagai C1 di [[limitasi-dan-lanjutan]].

Hasilnya OA 0,8086 melawan 0,845 di paper ([[04-baseline-mobilenetv2]]). Dan cabang ini
ternyata **mendominasi** model fusion: fiturnya bernorma 3,7x lebih besar daripada fitur
cabang teks, yang membuat head concat mengabaikan teks ([[11-diagnostik-cabang-teks]]).
