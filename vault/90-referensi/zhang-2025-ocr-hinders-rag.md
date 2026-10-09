---
judul: Zhang et al. (2025) — OCR Hinders RAG
tipe: referensi
tahap: "09"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/referensi
  - tahap/09
  - modal/teks
  - komponen/ocr
terkait:
  - "[[noise-ocr-dan-oov]]"
  - "[[francis-2025-perbandingan-ocr]]"
  - "[[peta-metode]]"
---

# Zhang et al. (2025) — OCR Hinders RAG

> J. Zhang, Q. Zhang, B. Wang, L. Ouyang, Z. Wen, Y. Li, K.-H. Chow, C. He, W. Zhang,
> "OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented
> Generation," *ICCV 2025*, pp. 17443–17453. doi:10.1109/ICCV51701.2025.01620.
> PDF: `references/OCR_Hinders_RAG_….pdf`

## Kontribusi

Benchmark OHRBench untuk mengukur bagaimana galat OCR merambat ke sistem
retrieval-augmented generation. Galat OCR dipilah menjadi derau semantik (kata salah)
dan derau format (struktur tabel atau rumus rusak).

## Relevansi untuk proyek kita

Bukti dari tugas lain bahwa **pilihan OCR mengubah kinerja tugas lanjutan**, bukan
sekadar skor OCR-nya. Itu premis Fase 2 ([[kashyap-2026-ringkasan]]), kita uji untuk klasifikasi dokumen. Pemilahan
derau semantik vs format juga berguna untuk membaca perbedaan antar-mesin di Tahap 10:
mesin yang membaca per baris (Tesseract) dan per kotak teks (EasyOCR, PaddleOCR)
menghasilkan urutan kata yang berbeda meski kata-katanya benar. Lihat juga
[[noise-ocr-dan-oov]].
