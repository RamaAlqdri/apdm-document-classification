---
judul: Sandler et al. (2018) — MobileNetV2
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/citra
  - komponen/cnn
terkait:
  - "[[mobilenetv2]]"
  - "[[inverted-residual-block]]"
  - "[[peta-metode]]"
---

# Sandler et al. (2018) — MobileNetV2

> M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks,"
> CVPR, June 2018.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya.*

## Kontribusi

Arsitektur CNN untuk perangkat dengan sumber daya terbatas. Dua gagasan yang ada di
judulnya:

- **Inverted residuals** — blok sempit-lebar-sempit, koneksi residual antar ujung
  yang sempit, konvolusi depthwise mengerjakan bagian berat di dalam blok.
- **Linear bottlenecks** — projection layer tidak memakai ReLU, karena ReLU di ruang
  berdimensi rendah membuang informasi yang tidak bisa dipulihkan.

Hasilnya akurasi top-1 ImageNet setara VGG-16 dengan biaya komputasi jauh lebih
kecil.

## Kenapa dikutip di paper kita

Backbone cabang citra. Dipilih karena anggaran latensi, bukan karena paling akurat:
Tesseract sudah menghabiskan ~910ms, jadi jaringan citra harus murah. MobileNetV2
berjalan 60ms di GPU dan 230ms di CPU; Xception yang dipertimbangkan penulis butuh
630ms.

## Relevansi untuk proyek kita

Kita pakai `torchvision.models.mobilenet_v2` dengan bobot pretrained ImageNet,
di-fine-tune penuh. Dua hal yang perlu diingat:

1. Bobot torchvision tidak identik dengan versi Keras yang dipakai penulis —
   deviasi yang sudah pasti.
2. Deskripsi blok di paper Audebert et al. menyebut konvolusi 1x1 terakhir memakai
   ReLU, yang tidak sesuai desain linear bottleneck aslinya. Implementasi resmi
   torchvision yang benar; jangan pakai deskripsi paper sebagai spesifikasi. Lihat
   [[inverted-residual-block]].

Vektor 1280 dimensi yang dipakai cabang citra di model fusion adalah keluaran
global average pooling atas feature map konvolusi terakhir MobileNetV2.
