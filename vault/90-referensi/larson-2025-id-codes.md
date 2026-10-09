---
judul: Larson et al. (2025) — Spurious cues ID code di RVL-CDIP dan Tobacco3482
tipe: referensi
tahap: "09"
tanggal: 2026-10-09
status: todo
tags:
  - tipe/referensi
  - tahap/09
  - modal/citra
  - komponen/evaluasi
terkait:
  - "[[kumar-2014-tobacco3482]]"
  - "[[kashyap-2026-ringkasan]]"
  - "[[peta-metode]]"
---

# Larson et al. (2025) — Spurious cues ID code

> S. Larson, S. Duwal, B. Vilnrotter, G. Chakkithara, V. Padwal, K. Leach, "Spurious
> Cues in RVL-CDIP and Tobacco3482 Document Classification: The Case of ID Codes,"
> *Proc. ACM Symposium on Document Engineering (DocEng) 2025*, pp. 1–4.
> doi:10.1145/3704268.3748683. Open access CC BY-NC.

*Status `todo`: catatan ini ditulis dari abstrak saja. PDF-nya belum ada di
`references/`; unduhan otomatis ditolak verifikasi bot ACM.*

## Kontribusi (dari abstrak)

Dokumen di RVL-CDIP dan Tobacco3482 bercap **kode ID** arsip. Model dangkal yang hanya
memakai fitur dari kode ID itu sudah mencapai ±40% akurasi di RVL-CDIP dan **±60% di
Tobacco3482**. Classifier mutakhir kehilangan 11 poin akurasi di RVL-CDIP setelah kode ID
dihapus. Penulis merilis versi dataset dengan kode ID dihapus.

## Relevansi untuk proyek kita

Dataset-nya sama dengan yang kita pakai ([[kumar-2014-tobacco3482]]). Ini **wajib dikontrol di tahap "teks sebagai objek"** (Tahap 14). Kode ID adalah objek
teks dengan posisi yang konsisten. Model yang diberi posisi kotak teks bisa belajar jalan
pintas itu, alih-alih belajar tata letak dokumen. Akurasi tinggi tanpa kontrol ini tidak
bisa ditafsirkan.

Hal yang sama berlaku mundur ke Fase 1: OCR membaca kode ID sebagai token. Sebagian
kemampuan cabang teks ([[03-baseline-cnn1d]]) mungkin berasal dari situ. Belum diukur.
