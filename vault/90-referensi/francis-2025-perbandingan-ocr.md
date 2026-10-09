---
judul: Francis & Sangeetha (2025) — Perbandingan model OCR
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
  - "[[ocr-tesseract]]"
  - "[[kashyap-2026-ringkasan]]"
  - "[[peta-metode]]"
---

# Francis & Sangeetha (2025) — Perbandingan model OCR

> S. A. Francis, M. Sangeetha, "A comparison study on optical character recognition
> models in mathematical equations and in any language," *Results in Control and
> Optimization*, vol. 18, art. 100532, 2025. doi:10.1016/j.rico.2025.100532.
> PDF: `references/1-s2.0-S2666720725000189-main.pdf`

## Kontribusi

Membandingkan tujuh mesin OCR, termasuk **seluruh mesin yang disarankan dosen**:
Tesseract, EasyOCR, PaddleOCR, Keras-OCR, ditambah MMOCR, i2OCR, dan TradOCR. Citra
ujinya campuran: dokumen cetak, layar, struk, spanduk, dan ekspresi matematika, tanpa
praproses.

## Hasil yang dikutip

Tabel 9 makalah:

| Mesin | Akurasi | F1 |
|---|---|---|
| PaddleOCR | 0,7136 | 0,7123 |
| i2OCR | 0,7037 | 0,7026 |
| EasyOCR | 0,6953 | 0,6942 |
| Tesseract | 0,6868 | 0,6858 |
| Keras-OCR | 0,6784 | 0,6774 |

## Relevansi untuk proyek kita

Dasar memilih mesin pembanding untuk Tahap 10 (lihat [[kashyap-2026-ringkasan]] untuk gap-nya,
dan [[ocr-tesseract]] untuk konfigurasi Tesseract yang dipakai sebagai acuan). Dua hal yang **tidak** dijawab makalah ini dan justru
menjadi gap kita:

1. Yang diukur adalah ketepatan OCR itu sendiri, **bukan dampaknya pada tugas lanjutan**
   seperti klasifikasi dokumen.
2. Selisih antar-mesin hanya ±3,5 poin, pada citra yang sama sekali bukan dokumen
   pindaian. Apakah selisih sekecil itu berpengaruh pada klasifikasi Tobacco3482 adalah
   pertanyaan empiris.

## Catatan kritis

Metode evaluasinya kurang jelas: "akurasi" OCR dihitung lewat model decision tree, dan
cara mendefinisikan positif/negatif untuk keluaran OCR tidak dijelaskan dengan rinci.
Angkanya dikutip sebagai urutan kasar, bukan sebagai ukuran yang presisi.
