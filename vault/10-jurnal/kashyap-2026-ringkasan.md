---
judul: Ringkasan — Kashyap et al. (2026), Multimodal Fusion and Supervised Contrastive Learning for DIC
tipe: jurnal
tahap: "09"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/jurnal
  - tahap/09
  - modal/fusion
  - komponen/ocr
  - komponen/fusion
terkait:
  - "[[mironczuk-2026-review-fusion]]"
  - "[[1907.06370-ringkasan]]"
  - "[[peta-proyek]]"
  - "[[rencana-prompt]]"
---

# Ringkasan — Kashyap et al. (2026)

> A. Kashyap, M. Mittal, S. Singh, "Multimodal Fusion and Supervised Contrastive Learning
> for Robust Document Image Classification," *International Journal of Intelligent
> Systems*, vol. 2026, art. 4241437, 2026. doi:10.1155/int/4241437. Open access CC-BY.
> PDF: `references/International Journal of Intelligent Systems - 2026 - Kashyap - ….pdf`

**Referensi utama proyek sejak Fase 2**, menggantikan posisi [[1907.06370-ringkasan]]
(Audebert 2019) yang tidak memenuhi syarat dosen: minimal 2025, sudah terbit.
Audebert tetap dipakai sebagai baseline yang sudah kita replikasi.

## Masalah

Klasifikasi citra dokumen (DIC). Pendekatan unimodal rapuh pada citra bising, dan model
multimodal besar (LayoutLMv3, DocFormer) terlalu mahal untuk perangkat terbatas.

## Kontribusi

1. Late fusion dua aliran: citra (EfficientNetB3) dan teks OCR (BERT-base). Ekstraksi
   fitur dipisah dari fusion, dengan klaim mengurangi propagasi galat antar-modalitas.
2. SCSR: hanya teks, dengan embedding `bge-small-en` yang di-fine-tune secara
   supervised contrastive lalu diklasifikasi lewat retrieval FAISS + voting mayoritas
   k=10.
3. Dataset kecil buatan sendiri, RealDoc-16 (160 dokumen; di teks tertulis 120).

## Metode

| Komponen | Pilihan |
|---|---|
| OCR | **"a commercial OCR engine"**, tidak disebut namanya (§3.4.2) |
| Praproses teks | lowercase, buang angka, stopword, tanda baca |
| Teks | BERT-base, fine-tune 10 epoch di RVL-CDIP (~41 jam) |
| Citra | EfficientNetB3 300×300, fine-tune 10 epoch di RVL-CDIP (~22,5 jam) |
| Fusion | **rata-rata logit tanpa bobot** (tingkat keputusan) |
| Hardware | RTX A4000 16 GB, TensorFlow/Keras |

## Dataset & protokol

Semua model dilatih **hanya pada RVL-CDIP** (16 kelas, ~320 ribu dokumen). Tobacco3482
dipakai utuh sebagai data uji tanpa pelatihan di dalamnya. Untuk Tobacco3482 dilaporkan
metrik *weighted*, bukan macro.

## Hasil utama

| | RVL-CDIP | Tobacco3482 |
|---|---|---|
| Teks (BERT) | 89,19 | 90,28 |
| Citra (EfficientNetB3) | 89,13 | 89,05 |
| Late fusion | **93,06** | **92,00** |
| SCSR | 80,76 | 87,89 |

Satu kali run, tanpa simpangan baku.

## Limitasi yang diakui penulis

Arah lanjutan yang mereka sebut: kondisi **degradasi OCR yang parah**, dan **pembobotan
modalitas adaptif / fusion sadar-keyakinan** sebagai pengganti rata-rata tanpa bobot.
Keduanya persis arah Fase 2 kita.

## Catatan kritis saya

1. **Angka 92,00% di Tobacco3482 tidak konsisten dengan protokolnya sendiri.** Penulis
   menulis bahwa Note dan Report tidak ada di RVL-CDIP. Menurut [[00-audit-data]],
   keduanya berisi 201 + 265 = 466 dokumen. Model 16 kelas RVL-CDIP tidak mungkin
   memprediksinya, jadi bila 3.482 dokumen dievaluasi, akurasi maksimum adalah
   3016/3482 = **86,6%**. Tabel per kelasnya memang hanya memuat 8 kelas. Kemungkinan
   besar 92% dihitung pada ±3.016 dokumen, padahal Tabel 2 menyatakan 3.482 dokumen dan
   10 kelas.
2. **Perbandingan dengan Audebert tidak setara.** Tabel 4 menyandingkan 92,00% (dilatih
   320 ribu dokumen RVL-CDIP) dengan 87,8% Audebert (dilatih 800 dokumen Tobacco3482),
   tanpa catatan bahwa protokolnya berbeda.
3. **Potensi tumpang tindih data tidak dibahas.** RVL-CDIP dan Tobacco3482 sama-sama
   diambil dari IIT-CDIP. *Belum saya verifikasi*; untuk memeriksanya dibutuhkan
   RVL-CDIP, yang tidak kita pakai.
4. **Satu run, tanpa uji statistik.** Ini kekurangan yang diukur sebagai masalah
   lapangan oleh [[mironczuk-2026-review-fusion]].
5. **Mesin OCR tidak disebutkan**, jadi pipeline teksnya tidak bisa direproduksi. Seluruh
   argumen "BERT lebih baik dari FastText" di §4.2 bercampur dengan efek OCR yang tidak
   diketahui. Ini **gap utama Fase 2**: OCR diperlakukan sebagai konstanta, tidak pernah
   sebagai variabel.

## Ambiguitas & asumsi

> Detail yang TIDAK disebutkan paper. Jangan ditambal dengan tebakan diam-diam.

- Nama dan versi mesin OCR.
- Apakah dokumen Tobacco3482 yang juga ada di RVL-CDIP dikeluarkan dari evaluasi.
- Bagaimana Note dan Report diperlakukan saat evaluasi.
- Panjang maksimum token BERT untuk dokumen panjang (SCSR memakai chunking, aliran BERT
  tidak dijelaskan).

## Posisi untuk proyek kita

**Tidak direplikasi.** Biayanya (63 jam fine-tuning di GPU 16 GB plus RVL-CDIP ±40 GB)
jauh di luar laptop RTX 3050 4 GB. Yang diambil dari paper ini:

- **embedding BERT** sebagai sumbu kedua di samping FastText;
- **aturan fusion rata-rata logit**, yang bisa diuji pada checkpoint CNN1D dan IMAGE kita
  tanpa training baru;
- **arah lanjutan penulis sendiri** sebagai pembenaran gap.

Rencana lengkapnya di [[rencana-prompt]], bagian Fase 2.
