---
judul: Grid cabang teks — mesin OCR × FastText + CNN1D
tipe: eksperimen
tahap: "11"
tanggal: 2026-10-22
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/11
  - modal/teks
  - komponen/ocr
  - komponen/embedding
  - komponen/evaluasi
terkait:
  - "[[13-ocr-multi-mesin]]"
  - "[[03-baseline-cnn1d]]"
  - "[[16-fusi-per-ocr]]"
  - "[[kashyap-2026-ringkasan]]"
  - "[[peta-eksperimen]]"
---

# Grid cabang teks — mesin OCR × FastText + CNN1D

model:: CNN1D (4 lapis, kernel 12, 512 kanal) di atas FastText cc.en.300.bin, satu model per kondisi OCR × seed
dataset:: Tobacco3482, 3.482 dokumen; teks dari QS-OCR, Tesseract (JPG), EasyOCR, PaddleOCR
split:: 800 latih terstratifikasi (80 val) / 2.682 uji
seed:: 42, 43, 44
oa::
macro_f1::
durasi::

> **BELUM DIJALANKAN.** Dijalankan oleh `notebooks/pipeline.ipynb` bagian 6-7 dan 9 (H1).
> Angka diisi dari `reports/fase2/grid_ringkasan.csv`, `uji_h1_mcnemar_antar_ocr.csv`,
> dan `f1_per_kelas_text.csv`. Field `oa::` di atas diisi rerata sel QS-OCR; tabel per
> kondisi di bawah.

## Tujuan

Menjawab **Q1 / H1**: seberapa besar pilihan mesin OCR mengubah akurasi cabang teks, bila
embedding, model, split, dan hiperparameter dikunci. Ini celah yang tidak diisi tiga
paper rujukan: pada [[kashyap-2026-ringkasan]] mesin OCR anonim dan model teks berganti
bersamaan, sehingga pengaruh OCR tidak pernah terlihat sendiri.

## Konfigurasi

- Teks per kondisi dari [[13-ocr-multi-mesin]], **mentah**: angka, *stopword*, dan tanda
  baca tidak dibuang, supaya efek mesin tidak tercampur efek praproses.
- spaCy (`en_core_web_sm`) untuk tokenisasi, FastText subword `.bin` untuk vektor kata,
  500 kata pertama, float16. Setelan CNN1D identik dengan [[03-baseline-cnn1d]].
- SGD momentum 0,9, lr 0,01, batch 40, maksimal 100 epoch, *early stopping* patience 15.

## Pemeriksaan kewarasan

Sel QS-OCR memakai setelan yang sama dengan Fase 1, jadi harus mendekati CNN1D Fase 1
(0,7266 ± 0,0059). Notebook mencetak selisihnya; selisih lebih dari 0,02 pada mode penuh
berarti ada yang rusak.

## Hipotesis (ditulis sebelum eksekusi)

H1: setidaknya satu pasangan mesin berbeda secara nyata (McNemar, Holm per seed, α 0,05)
**konsisten di ketiga seed**. Dugaan arah: QS-OCR ≥ Tesseract-JPG karena dibaca dari TIF
asli; EasyOCR dan PaddleOCR bisa lebih baik pada halaman berderau tetapi urutan bacanya
direkonstruksi heuristik (`reading_order`), yang dapat merugikan CNN1D berbasis urutan.

## Hasil

| Kondisi OCR | OA TEXT (rerata ± std) | macro-F1 | pasangan signifikan |
|---|---|---|---|
| qs-ocr | | | |
| tesseract | | | |
| easyocr | | | |
| paddleocr | | | |

## Temuan

(diisi setelah eksekusi)
