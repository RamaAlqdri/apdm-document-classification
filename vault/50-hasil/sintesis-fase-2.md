---
judul: Sintesis Fase 2 — mesin OCR sebagai variabel
tipe: hasil
tahap: "13"
tanggal: 2026-10-24
status: selesai
tags:
  - tipe/hasil
  - tahap/13
  - modal/fusion
  - komponen/ocr
  - komponen/fusion
  - temuan/positif
  - temuan/negatif
terkait:
  - "[[13-ocr-multi-mesin]]"
  - "[[15-grid-teks-ocr]]"
  - "[[16-fusi-per-ocr]]"
  - "[[17-missing-modality-lanjutan]]"
  - "[[sintesis-akhir]]"
  - "[[limitasi-dan-lanjutan]]"
  - "[[peta-proyek]]"
---

# Sintesis Fase 2 — mesin OCR sebagai variabel

Satu run penuh `notebooks/pipeline.ipynb` di laptop TUF (RTX 3050 4 GB), hasil di-push
2026-10-24. Tobacco3482 saja, 800 latih (80 val) / 2.682 uji, seed 42/43/44. Setiap angka
di bawah berasal dari catatan sumber yang ditautkan; tidak ada angka yang tidak dijalankan.

## Validitas pipeline

- Cabang TEXT dengan QS-OCR mereproduksi CNN1D Fase 1 **bit-identik** per seed (0,7345 /
  0,7248 / 0,7204, epoch terbaik sama), walau dilatih ulang dari nol
  ([[15-grid-teks-ocr]]). IMAGE 0,8169 ± 0,0257 vs Fase 1 0,8086 ± 0,0210.
- EDA ulang menghasilkan angka yang sama dengan [[14-kurasi-dan-eda]] (probe citra 0,7293,
  teks 0,6685; jalan pintas kode ID 0,2782).

## Q1 — Seberapa besar mesin OCR mengubah cabang teks?

**H1 didukung.** Dengan embedding, model, dan split dikunci, mesin OCR saja menggeser OA
cabang teks **5,4 poin**: tesseract-JPG 0,6970 < easyocr 0,7155 < qs-ocr 0,7266 <
**paddleocr 0,7514**. Tiga pasangan berbeda nyata di 3/3 seed, dua di 2/3 seed (McNemar,
Holm per seed). Sumber: [[15-grid-teks-ocr]].

Dua hal yang tidak terduga:

- **Mesin lebih penting daripada kualitas sumber citra.** PaddleOCR pada JPG re-encode
  mengalahkan QS-OCR yang dibaca dari TIF asli (+2,5 poin), sementara Tesseract pada JPG
  kalah 3,0 poin. Teks QS-OCR yang selama ini dipakai sebagai acuan bukan batas atas.
- **Proksi OOV menyesatkan.** EasyOCR punya OOV median 2,5× mesin lain dan kesepakatan
  antar-mesin terendah, tetapi cabang teksnya lebih akurat daripada Tesseract-JPG. Probe
  linear murah tanpa training meramalkan urutan mesin jauh lebih baik ([[13-ocr-multi-mesin]]).

## Q3 — Apakah teks yang lebih kuat lebih dipakai, dan aturan fusi mana yang lebih tahan?

**Perbaikan OCR terbawa sampai ke fusi.** Dari tesseract ke paddleocr: LATE +2,8, CONCAT
+3,2 poin, nyata di 3/3 seed. Sel terbaik **CONCAT paddleocr 0,8674 ± 0,0014** (macro-F1
0,8524), 3,3 poin di atas fusion end-to-end Fase 1 (0,8342), dan memanen 52 % potensi oracle,
di atas Fase 1 (29 %) dan paper Audebert (43 %). Sumber: [[16-fusi-per-ocr]].

**H3a (teks lebih kuat → lebih dipakai): didukung untuk LATE, lemah untuk CONCAT.**
Spearman OA TEXT terhadap penurunan LATE tanpa teks 0,683 (12 sel), terhadap gain LATE 0,517.
Untuk CONCAT 0,287. Akurasi teks bukan satu-satunya penentu: easyocr lebih lemah dari qs-ocr
tetapi dipanen jauh lebih banyak.

**H3b (LATE lebih tahan cabang timpang): tidak didukung.** Rerata kedua aturan hampir
sama, dan pemenangnya bergantung pada kekuatan cabang citra: LATE unggul saat IMAGE kuat,
CONCAT unggul nyata di 3/4 kondisi saat IMAGE lemah (seed 44). CONCAT 5-9× lebih stabil
antar-seed. Untuk cabang teks yang lemah, keduanya setara.

## Koreksi metodologis atas rancangan sendiri

Uji *missing modality* yang dipra-registrasi (mengganti modalitas dengan keluaran untuk
masukan kosong) **hanya sah untuk LATE tanpa teks**. Untuk CONCAT dan untuk semua versi
"tanpa citra", hasilnya jatuh di bawah model unimodal modalitas lain, yang hanya mungkin
bila masukan penggantinya keluar distribusi. Pemakaian modalitas karena itu dinilai dengan
gain di atas IMAGE dan panen potensi. Lihat [[16-fusi-per-ocr]] temuan 2.

Uji lanjutan [[17-missing-modality-lanjutan]] memastikan penyebabnya:

- **Pengganti di luar distribusi (H4a): didukung sebagian.** Dengan rerata fitur latih
  sebagai pengganti, semua versi tanpa citra mendarat paling jauh 4,2 poin di bawah TEXT
  (sebelumnya 11-35 poin). CONCAT tanpa teks naik dari −0,56 / −0,59 / −0,14 / −0,30
  menjadi −0,108 / −0,130 / −0,026 / −0,027 terhadap IMAGE (qs-ocr / tesseract / easyocr /
  paddleocr). Pada OCR yang lebih baik, CONCAT bisa bekerja setara IMAGE tanpa teks; pada
  qs-ocr dan tesseract tidak.
- **"Teks kosong ⇒ Note" (H4b): tidak didukung.** *Head* tanpa dokumen kosong tetap
  tertarik ke Note, juga pada OCR yang tidak punya dokumen kosong sama sekali.
- **Pengganti rerata tidak netral untuk LATE:** rerata logit latih menjatuhkan LATE tanpa
  teks 7-13 poin di bawah IMAGE. Untuk LATE, pengganti kosong tetap ukuran yang sah.

## Posisi terhadap literatur (dengan catatan protokol)

| Studi | Protokol Tobacco3482 | Fusi |
|---|---|---|
| Audebert (2020) | 800 latih, k-fold | 0,878 |
| Kanchi (2022), tanpa RVL-CDIP | partisi Afzal | 0,903 |
| Fase 1 kita | 800 latih, 3 seed | 0,8342 |
| **Fase 2 kita, CONCAT paddleocr** | 800 latih, 3 seed | **0,8674** |

Angka antarbaris tidak sepenuhnya sebanding (protokol berbeda); yang sebanding langsung
adalah dua baris terakhir.

## Yang tetap terbuka

H2 (BERT memperkecil selisih antar-OCR) belum diuji. Kejatuhan CONCAT tanpa teks terbukti
sebagian besar artefak masukan pengganti, tetapi asal tarikan ke Note belum diketahui
(dugaan belum diuji: Note kelas berteks terpendek, [[17-missing-modality-lanjutan]]). Ringkasan limitasi ada di [[limitasi-dan-lanjutan]] bagian H.
