---
judul: Grid cabang teks — mesin OCR × FastText + CNN1D
tipe: eksperimen
tahap: "11"
tanggal: 2026-10-22
status: selesai
tags:
  - tipe/eksperimen
  - tahap/11
  - modal/teks
  - komponen/ocr
  - komponen/embedding
  - komponen/evaluasi
  - temuan/positif
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
oa:: 0.7266 (qs-ocr) · 0.6970 (tesseract) · 0.7155 (easyocr) · 0.7514 (paddleocr)
macro_f1:: 0.7017 · 0.6547 · 0.6862 · 0.7200
durasi:: 101-169 dtk per model (12 model); fitur teks 6,4 menit

> **DIJALANKAN** di laptop TUF lewat `notebooks/pipeline.ipynb` bagian 6-7 dan 9, hasil
> di-push 2026-10-24 (`a55f472`). Angka disalin dari `reports/fase2/grid_ringkasan.csv`,
> `unimodal_per_seed.csv`, `uji_h1_mcnemar_antar_ocr.csv`, `f1_per_kelas_text.csv`.
> Deviasi runtime: micro-batch 8 dengan akumulasi gradien ke 40 (BatchNorm per 8), AMP.

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

## Pemeriksaan kewarasan: lolos, bit-identik

TEXT qs-ocr per seed 0,7345 / 0,7248 / 0,7204 dengan epoch terbaik 28 / 25 / 31, **sama
persis** dengan [[03-baseline-cnn1d]] di Fase 1, padahal modelnya dilatih ulang dari nol
(`dari_cache = False`, durasi berbeda: 152 vs 92 dtk). Pipeline baru deterministik dan
setara dengan Fase 1, jadi selisih antar-kondisi di bawah bukan artefak kode baru.

## Hasil

| Kondisi OCR | OA TEXT (rerata ± std) | macro-F1 | OA per seed (42/43/44) |
|---|---|---|---|
| qs-ocr | 0,7266 ± 0,0059 | 0,7017 ± 0,0053 | 0,7345 / 0,7248 / 0,7204 |
| tesseract | 0,6970 ± 0,0047 | 0,6547 ± 0,0083 | 0,6913 / 0,7028 / 0,6969 |
| easyocr | 0,7155 ± 0,0061 | 0,6862 ± 0,0095 | 0,7155 / 0,7081 / 0,7230 |
| paddleocr | **0,7514 ± 0,0043** | **0,7200 ± 0,0057** | 0,7498 / 0,7573 / 0,7472 |

### McNemar antar-kondisi (TEXT, Holm per seed, α 0,05)

| Pasangan (a − b) | selisih OA rerata | seed signifikan |
|---|---|---|
| tesseract − paddleocr | −0,0544 | **3/3** |
| easyocr − paddleocr | −0,0359 | **3/3** |
| qs-ocr − tesseract | +0,0296 | **3/3** |
| qs-ocr − paddleocr | −0,0249 | 2/3 |
| tesseract − easyocr | −0,0185 | 2/3 |
| qs-ocr − easyocr | +0,0111 | 0/3 |

### F1 per kelas TEXT (rerata seed)

| Kelas | qs-ocr | tesseract | easyocr | paddleocr |
|---|---|---|---|---|
| Advertisement | 0,537 | 0,445 | 0,537 | **0,622** |
| Email | 0,953 | 0,950 | 0,952 | 0,952 |
| Form | 0,718 | 0,687 | 0,694 | **0,747** |
| Letter | 0,671 | 0,669 | 0,669 | **0,745** |
| Memo | 0,758 | 0,737 | 0,761 | **0,803** |
| News | **0,709** | 0,596 | 0,659 | 0,682 |
| Note | **0,621** | 0,565 | 0,604 | 0,529 |
| Report | 0,495 | 0,460 | 0,507 | **0,546** |
| Resume | 0,969 | 0,946 | 0,954 | **0,973** |
| Scientific | 0,587 | 0,491 | 0,525 | **0,601** |

## Temuan

1. **#temuan/positif — H1 didukung.** Dengan embedding, model, split, dan hiperparameter
   dikunci, mesin OCR saja menggeser akurasi cabang teks sebesar **5,4 poin** (tesseract
   0,697 → paddleocr 0,751). Tiga pasangan berbeda nyata di ketiga seed, dua lagi di dua
   seed; hanya qs-ocr vs easyocr yang tidak pernah berbeda nyata. Selisih terbesar ± 11×
   simpangan baku antar-seed cabang teks (± 0,005).
2. **Mesin lebih menentukan daripada sumber citra.** PaddleOCR yang membaca JPG re-encode
   mengalahkan QS-OCR yang dibaca dari TIF asli (+2,5 poin, 2/3 seed), sedangkan Tesseract
   pada JPG kalah dari QS-OCR (−3,0 poin, 3/3 seed). Asumsi bahwa teks QS-OCR adalah batas
   atas karena berasal dari TIF tidak berlaku.
3. Kenaikan PaddleOCR tersebar di kelas yang kaya teks (Letter +7,4, Advertisement +8,5,
   Report +5,1, Memo +4,5 poin F1 terhadap qs-ocr), tetapi **Note turun 9,2 poin**: kelas
   dengan banyak tulisan tangan, tempat OOV PaddleOCR juga tinggi ([[13-ocr-multi-mesin]]).
   Tidak ada mesin yang terbaik di semua kelas.
4. Urutan ini **tidak** diramalkan oleh proksi OOV (EasyOCR terburuk di OOV, tapi di atas
   Tesseract), tetapi diramalkan cukup baik oleh probe linear tanpa training. Implikasinya
   untuk praktik: memilih mesin OCR untuk klasifikasi sebaiknya dengan probe kecil, bukan
   dengan statistik OOV.
5. Dugaan arah di hipotesis (QS-OCR ≥ Tesseract-JPG) benar. Kerugian dari urutan baca
   heuristik EasyOCR/PaddleOCR tidak tampak di hasil (PaddleOCR justru terbaik), tetapi
   efeknya tidak diuji terpisah, jadi tidak bisa dikatakan nol.
