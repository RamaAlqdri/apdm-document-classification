---
judul: Log 2026-10-09 — Tahap 9 Revisi Arah setelah Tanggapan Dosen
tipe: log
tahap: "09"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/log
  - tahap/09
  - status/selesai
terkait:
  - "[[kashyap-2026-ringkasan]]"
  - "[[mironczuk-2026-review-fusion]]"
  - "[[rencana-prompt]]"
  - "[[peta-proyek]]"
---

# Log 2026-10-09 — Tahap 9 Revisi Arah

## Pemicu

Tanggapan dosen atas progress awal. Isinya: (1) jelajahi mesin OCR lain (EasyOCR,
PaddleOCR, Keras-OCR); (2) gap belum ditentukan; (3) embedding bisa dieksplorasi;
(4) teks yang dideteksi sebagai objek untuk tahap berikutnya; (5) referensi utama 2019
dan masih arXiv, padahal minimal harus 2025.

Soal (5), ada satu koreksi fakta: Audebert sebenarnya sudah terbit, di Springer CCIS
1167 (2020). Kesalahan kita ada di proposal, yang mengutip versi arXiv-nya. Tahunnya
tetap tidak memenuhi syarat, jadi referensi utama tetap diganti.

## Yang dikerjakan

1. **Pencarian referensi** lewat Crossref dan Semantic Scholar. Metadata setiap
   referensi diverifikasi lewat DOI, bukan dari ingatan. PDF diunduh pengguna ke
   `references/`. Larson 2025 belum ada, karena unduhan ACM ditolak verifikasi bot.
2. **Teks lengkap [[kashyap-2026-ringkasan]] dibaca.** Ditemukan empat masalah. Yang
   terpenting: OCR-nya mesin komersial tanpa nama, dan angka 92% di Tobacco3482 tidak
   mungkin tercapai pada 10 kelas, karena Note dan Report tidak ada di RVL-CDIP
   (maksimum 86,6%).
3. **[[mironczuk-2026-review-fusion]] dibaca.** Review ini menyatakan kualitas OCR tidak
   bisa dikontrol dalam meta-analisis karena studi primer tidak melaporkannya. Itu
   pembingkai gap yang langsung.
4. Empat catatan pustaka: [[francis-2025-perbandingan-ocr]],
   [[zhang-2025-ocr-hinders-rag]], [[michail-2025-embedding-tahan-ocr]],
   [[larson-2025-id-codes]] (yang terakhir dari abstrak saja, berstatus `todo`).
5. `proposal/proposal.tex` ditulis ulang. Judul baru, gap dalam kotak, Q1–Q3, desain
   faktorial OCR × embedding, uji statistik, dan tahap teks-sebagai-objek. Hasil Fase 1
   masuk sebagai hasil awal yang memotivasi gap. Versi lama tetap ada di commit
   `dbb32e6`.
6. Fase 2 (Tahap 9–14) ditambahkan ke [[rencana-prompt]]. Tag `#tahap/09`–`#tahap/14`
   didaftarkan di [[peta-tag]]. `CLAUDE.md` diperbarui.

## Keputusan

1. **Kashyap dijadikan referensi utama, tapi tidak direplikasi.** Biayanya 63 jam
   fine-tuning di GPU 16 GB plus RVL-CDIP. Yang diambil: embedding BERT, aturan
   rata-rata logit, dan arah lanjutan yang penulis sebut sendiri.
2. **Fase 1 tidak dibuang.** Hasilnya menjadi sel acuan Tesseract × FastText. Uji
   missing modality dan probe linear menjadi alat ukur di setiap sel.
3. **Semua mesin OCR dijalankan pada JPG yang sama**, termasuk Tesseract. Tanpa itu,
   perbandingan Tesseract-TIF (QS-OCR) dengan mesin lain-JPG tercampur efek format citra.
4. **Kualitas OCR dilaporkan sebagai proksi**, karena Tobacco3482 tidak punya
   transkripsi acuan.
5. **Kotak teks disimpan sejak Tahap 10**, supaya Tahap 14 tidak perlu OCR ulang.

## Terbuka

- PaddleOCR disebut dua kali dalam catatan dosen. Perlu dikonfirmasi apakah salah
  satunya maksudnya mesin lain.
- PDF Larson 2025 perlu diunduh manual dari
  https://dl.acm.org/doi/10.1145/3704268.3748683.
- Tahap 10 butuh keputusan instalasi: Tesseract di Windows, `paddlepaddle-gpu` >500 MB.
