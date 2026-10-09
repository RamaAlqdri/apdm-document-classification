---
judul: Michail et al. (2025) — Embedding yang tahan derau OCR
tipe: referensi
tahap: "09"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/referensi
  - tahap/09
  - modal/teks
  - komponen/embedding
terkait:
  - "[[fasttext-subword]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[peta-metode]]"
---

# Michail et al. (2025) — Embedding yang tahan derau OCR

> A. Michail, J. Opitz, Y. Wang, R. Meister, R. Sennrich, S. Clematide, "Cheap Character
> Noise for OCR-Robust Multilingual Embeddings," *Findings of ACL 2025*.
> https://aclanthology.org/2025.findings-acl.609/
> PDF: `references/2025.findings-acl.609.pdf`

## Kontribusi

Embedding kalimat standar melemah pada teks hasil OCR karena derau OCR jarang ada di data
pelatihannya. Penulis menunjukkan bahwa fine-tuning dengan **derau karakter buatan yang
murah** plus loss kontrastif cukup untuk memulihkan ketahanannya, tanpa data in-domain
berlabel, dan tanpa merusak kinerja pada teks bersih.

## Relevansi untuk proyek kita

Dasar sumbu **embedding** di Fase 2, di samping BERT dari [[kashyap-2026-ringkasan]]. Ada dua cara menangani derau OCR:

- **struktural**, lewat subword n-gram FastText ([[fasttext-subword]]), yang memberi
  vektor pada kata rusak karena berbagi potongan huruf;
- **lewat data**, dengan fine-tuning pada teks bising, seperti di makalah ini.

Injeksi typo di `src/ablation.py` (Tahap 7) adalah mekanisme derau yang sama, sehingga
bisa dipakai ulang sebagai augmentasi bila sumbu embedding diperluas.
