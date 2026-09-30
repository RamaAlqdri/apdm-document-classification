---
judul: Eitel et al. (2015) — fusion multimodal RGB-D
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[taksonomi-multimodal]]"
  - "[[peta-metode]]"
---

# Eitel et al. (2015) — fusion multimodal RGB-D

> A. Eitel et al., "Multimodal deep learning for robust RGB-D object recognition,"
> IROS, Sept. 2015.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya.*

## Kontribusi

Pengenalan objek dengan menggabungkan citra RGB dan peta kedalaman. Polanya: dua
CNN terpisah, satu per modalitas, masing-masing menghasilkan vektor fitur, lalu
jaringan fusion di atasnya menggabungkan keduanya untuk satu keputusan. Ada
penekanan pada ketahanan — bagaimana sistem tetap bekerja ketika salah satu
modalitas berkualitas buruk.

## Kenapa dikutip di paper kita

Ini sumber pola arsitektur fusion yang dipakai Audebert et al.: dua cabang
independen, masing-masing diringkas ke vektor berdimensi sama, digabung, lalu
diumpankan ke MLP. Paper menyebutnya sebagai acuan langsung untuk lapisan fusion.

## Relevansi untuk proyek kita

Dua hal.

**Pertama, ini pembanding yang membuat keberatan soal modalitas jadi tajam.** Pada
RGB-D, dua modalitasnya memang dari sensor berbeda yang menangkap sifat fisik
berbeda. Pada kasus kita, teks diturunkan dari citra yang sama lewat OCR. Polanya
dipinjam dari situasi yang secara struktural berbeda — itu bukan kesalahan, tapi
patut disadari. Lihat [[taksonomi-multimodal]].

**Kedua, tema ketahanannya persis ablasi Tahap 7 kita.** Eitel et al. menyoroti
perilaku sistem ketika satu modalitas memburuk; Audebert et al. tidak mengujinya
pada kasus dokumen meski mengakui di bagian limitasi bahwa datasetnya terlalu
bersih. Itulah celah yang ablasi kita isi: degradasi citra bertahap, missing
modality, dan degradasi teks.

Hasilnya justru berlawanan dengan semangat Eitel: fusion kami **lebih rapuh** daripada
baseline citra ([[07-ablasi-degradasi-citra]]), karena cabang teksnya tidak berfungsi
([[08-ablasi-missing-modality]]). Setelah diperbaiki di [[12-perbaikan-norm-dan-init]],
modelnya baru benar-benar multimodal.
