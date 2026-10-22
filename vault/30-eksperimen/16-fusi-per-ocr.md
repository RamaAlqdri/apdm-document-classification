---
judul: Fusi per kondisi OCR — LATE (rata-rata logit) vs CONCAT (fitur beku)
tipe: eksperimen
tahap: "12"
tanggal: 2026-10-22
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/12
  - modal/fusion
  - komponen/fusion
  - komponen/evaluasi
terkait:
  - "[[15-grid-teks-ocr]]"
  - "[[08-ablasi-missing-modality]]"
  - "[[12-perbaikan-norm-dan-init]]"
  - "[[kashyap-2026-ringkasan]]"
  - "[[peta-eksperimen]]"
---

# Fusi per kondisi OCR — LATE vs CONCAT

model:: LATE = argmax rata-rata logit IMAGE + TEXT; CONCAT = ConcatHead (proyeksi 1280→128 ⊕ 128, MLP 256) di atas fitur beku
dataset:: Tobacco3482, 4 kondisi OCR
split:: 800 latih terstratifikasi (80 val) / 2.682 uji
seed:: 42, 43, 44
oa::
macro_f1::
durasi::

> **BELUM DIJALANKAN.** Dijalankan oleh `notebooks/pipeline.ipynb` bagian 8-9. Angka diisi
> dari `reports/fase2/tabel_utama_oa.csv`, `tabel_utama_f1.csv`,
> `uji_h3_pemakaian_modalitas.csv`, dan `uji_h3_mcnemar_late_vs_concat.csv`.

## Tujuan

Menjawab **Q3 / H3**: apakah cabang teks yang lebih kuat (dari [[15-grid-teks-ocr]])
membuat fusi lebih *memakai* teks, dan aturan fusi mana yang lebih tahan terhadap cabang
yang timpang. Fase 1 menunjukkan OA fusion yang tinggi bukan bukti pemakaian kedua
modalitas ([[08-ablasi-missing-modality]]).

## Konfigurasi

- IMAGE: MobileNetV2 per seed, satu model untuk semua kondisi OCR.
- LATE: aturan [[kashyap-2026-ringkasan]], tanpa training.
- CONCAT: *head* dilatih pada fitur beku IMAGE dan TEXT (bagian latih, *early stopping*
  pada val). **Deviasi dari Audebert** (end-to-end), disengaja: kedua aturan memakai model
  unimodal yang sama, dan cabang teks datang sudah terlatih, sehingga kolaps yang diukur
  di Fase 1 tidak bisa terjadi ([[12-perbaikan-norm-dan-init]]).
- Batasan yang ditulis di laporan: *head* dilatih pada fitur data latih yang juga dipakai
  melatih unimodal, jadi ia menilai cabang dari perilaku latihnya.
- Missing modality: modalitas diganti keluaran model untuk masukan kosong (teks kosong =
  sekuens nol; citra = tensor nol ternormalisasi, yaitu citra rata-rata ImageNet).
- ORACLE dari prediksi IMAGE dan TEXT pada set uji yang sama.

## Hipotesis (ditulis sebelum eksekusi)

H3a: `turun_tanpa_teks` naik seiring OA cabang teks (Spearman positif di 4 × 3 sel), untuk
kedua aturan.

H3b: LATE kehilangan lebih sedikit daripada CONCAT saat cabang timpang, karena rata-rata
logit memaksa kedua modalitas berkontribusi sedangkan *head* bisa belajar mengabaikan
cabang yang lemah. Diuji juga langsung dengan McNemar LATE vs CONCAT per kondisi.

## Hasil

| Kondisi OCR | IMAGE | TEXT | LATE | CONCAT | ORACLE | LATE turun tanpa teks | CONCAT turun tanpa teks |
|---|---|---|---|---|---|---|---|
| qs-ocr | | | | | | | |
| tesseract | | | | | | | |
| easyocr | | | | | | | |
| paddleocr | | | | | | | |

## Temuan

(diisi setelah eksekusi)
