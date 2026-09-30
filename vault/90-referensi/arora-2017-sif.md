---
judul: Arora et al. — SIF sentence embedding
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/teks
  - komponen/embedding
terkait:
  - "[[sif-document-embedding]]"
  - "[[fasttext-subword]]"
  - "[[peta-metode]]"
---

# Arora et al. — SIF sentence embedding

> S. Arora et al., "A Simple but Tough-to-Beat Baseline for Sentence Embeddings,"
> ICLR (paper kita mengutipnya bertanggal Nov. 2016).

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya.*

## Kontribusi

Baseline embedding kalimat yang sangat sederhana namun sulit dikalahkan metode
bersupervisi yang jauh lebih rumit. Dua langkah:

1. **Rata-rata berbobot** vektor kata, dengan bobot `a / (a + p(w))` — kata sering
   ditekan, kata jarang diangkat. Diturunkan dari model generatif, bukan dipasang
   sebagai heuristik seperti IDF.
2. **Pembuangan komponen utama pertama** lewat PCA. Komponen itu menangkap arah yang
   dimiliki semua kalimat dan tidak membedakan apa pun.

Nama SIF berasal dari skema bobotnya, *smooth inverse frequency*.

## Kenapa dikutip di paper kita

Cara Audebert et al. mengubah sekumpulan vektor kata menjadi satu vektor dokumen
untuk baseline MLP.

## Relevansi untuk proyek kita

Kita implementasikan sebagai representasi kedua di Tahap 4, meski tahu ia akan
kalah: 70,8% vs 73,9% untuk CNN1D. Perbandingannya tetap berguna karena menunjukkan
**mengapa** rata-rata gagal di teks OCR — vektor dari kata halusinasi ikut
merata-rata dan tidak bisa dibuang. Lihat [[sif-document-embedding]].

Dua hal yang harus kita putuskan sendiri karena paper diam: nilai `a` dan jumlah
komponen PCA yang dibuang. Kita pakai default (`a = 1e-3`, buang 1 komponen).

Jebakan implementasi: PCA harus dipasang hanya pada data train lalu diterapkan ke
test. Menghitungnya atas seluruh korpus adalah kebocoran yang halus dan mudah
terlewat — diverifikasi bersih di [[01-ekstraksi-fitur-teks]], sisa proyeksi tinggal
1e-07.

Hasil baseline-nya di [[02-baseline-mlp-sif]]: OA 0,6780, memang di bawah CNN1D seperti
paper, dengan margin yang bahkan lebih lebar (0,049 melawan 0,031).
