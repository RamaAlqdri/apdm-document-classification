---
judul: Log 2026-10-29 — Kata khas dan t-SNE teks per mesin OCR
tipe: log
tahap: "10"
tanggal: 2026-10-29
status: selesai
tags:
  - tipe/log
  - tahap/10
  - komponen/ocr
  - status/selesai
terkait:
  - "[[14-kurasi-dan-eda]]"
  - "[[13-ocr-multi-mesin]]"
  - "[[peta-proyek]]"
---

# Log 2026-10-29 — Kata khas dan t-SNE teks per mesin OCR

Laporan LaTeX kini membandingkan EDA teks per mesin OCR dari CSV yang sudah ada, kecuali kata
khas TF-IDF dan t-SNE yang selama ini hanya dihitung untuk QS-OCR ([[14-kurasi-dan-eda]]).
Kode untuk keduanya ditambahkan, lalu dijalankan pengguna di laptop compute (`00f1c9e`).

## Yang dibuat

- `notebooks/pipeline.ipynb`: bagian baru **6.2** setelah sel *probe* 6.1. Memakai `texts`
  (bagian 5) dan `f_txt` (rata-rata FastText per kondisi, sel 6.1) yang sudah ada, jadi tidak
  ada model yang dilatih. Keluaran: `reports/fase2/eda_kata_khas_per_ocr.csv` dan
  `reports/figures/fase2_eda_tsne_per_ocr.png` (grid 2×2, seed 42, perplexity yang sama).
- `src/plots.py`: `scatter_2d` menerima `ncols` untuk tata letak grid; self-check diperluas.

## Verifikasi di mesin penulis

`python -m src.plots` lolos; sel baru dieksekusi pada data sintetis (200 dokumen, 4 kondisi)
dan menulis CSV serta gambar. Angka dan gambar sintetis tidak bermakna dan tidak dicatat.

## Untuk pengguna

`python run_all.py` di laptop compute. Semua cache (OCR, sekuens FastText, fitur citra,
model) dipakai ulang; yang benar-benar dihitung hanya EDA, bagian 6.2, dan bagian 8-10.
Setelah hasil di-push, kata khas dan t-SNE per mesin dimasukkan ke laporan Subbagian 4.2.

## Hasil (`00f1c9e`)

Run ulang memakai semua cache; pemeriksaan kewarasan bagian 8 identik dengan run sebelumnya
(TEXT qs-ocr 0,7266 ± 0,0059, IMAGE 0,8169 ± 0,0257). Dari
`reports/fase2/eda_kata_khas_per_ocr.csv` dan `fase2_eda_tsne_per_ocr.png`:

- Kelas bertekst bersih (Email, Letter, Form, Resume, Report, Memo): kata khas hampir sama
  di keempat mesin.
- Potongan kata rusak muncul di kelas/mesin ber-OOV tinggi: EasyOCR `tbe`, `thc` (News,
  Scientific), `ana` (Report); Note rusak di semua mesin (`eee`, `fle`, `tle`, `sot`).
- Advertisement: kata kemasan (flavor, filter, box, slims) masuk sepuluh teratas ketiga mesin
  JPG, tidak pada QS-OCR.
- t-SNE: Resume dan Email terpisah jelas di semua mesin; EasyOCR punya satu gugus kecil
  Advertisement/Form/News di luar sebaran utama. Kualitatif saja.

Dimasukkan ke laporan `proposal/naskah` Subbagian 4.2.3 (Tabel 10, Gambar 4).
