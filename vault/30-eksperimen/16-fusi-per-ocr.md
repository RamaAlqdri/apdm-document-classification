---
judul: Fusi per kondisi OCR — LATE (rata-rata logit) vs CONCAT (fitur beku)
tipe: eksperimen
tahap: "12"
tanggal: 2026-10-22
status: selesai
tags:
  - tipe/eksperimen
  - tahap/12
  - modal/fusion
  - komponen/fusion
  - komponen/evaluasi
  - temuan/positif
  - temuan/negatif
  - temuan/anomali
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
oa:: LATE 0.8456 / 0.8376 / 0.8538 / 0.8653 · CONCAT 0.8316 / 0.8349 / 0.8567 / 0.8674 (qs-ocr / tesseract / easyocr / paddleocr)
macro_f1:: LATE 0.8320 / 0.8195 / 0.8364 / 0.8481 · CONCAT 0.8221 / 0.8238 / 0.8441 / 0.8524
durasi:: IMAGE 39-49 menit per seed; seluruh grid fusi 72 dtk

> **DIJALANKAN** di laptop TUF lewat `notebooks/pipeline.ipynb` bagian 7-9, hasil di-push
> 2026-10-24 (`a55f472`). Angka disalin dari `reports/fase2/grid_per_seed.csv`,
> `grid_ringkasan.csv`, `uji_h3_pemakaian_modalitas.csv`, `uji_h3_mcnemar_late_vs_concat.csv`;
> turunan (gain, panen potensi) dihitung dari `grid_per_seed.csv` saat analisis.
> Deviasi runtime: micro-batch 8 + akumulasi gradien, AMP.

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

IMAGE (sama untuk semua kondisi): **0,8169 ± 0,0257**, per seed 0,8315 / 0,8386 / 0,7808
(Fase 1: 0,8086 ± 0,0210, lolos pemeriksaan). Seed 44 jelas lebih lemah (early stopping
di epoch 26, terbaik epoch 11).

| Kondisi OCR | IMAGE | TEXT | LATE | CONCAT | ORACLE |
|---|---|---|---|---|---|
| qs-ocr | 0,8169 ± 0,0257 | 0,7266 ± 0,0059 | 0,8456 ± 0,0221 | 0,8316 ± 0,0119 | 0,9023 ± 0,0127 |
| tesseract | 0,8169 ± 0,0257 | 0,6970 ± 0,0047 | 0,8376 ± 0,0174 | 0,8349 ± 0,0102 | 0,8950 ± 0,0086 |
| easyocr | 0,8169 ± 0,0257 | 0,7155 ± 0,0061 | 0,8538 ± 0,0158 | 0,8567 ± 0,0034 | 0,8972 ± 0,0091 |
| paddleocr | 0,8169 ± 0,0257 | 0,7514 ± 0,0043 | **0,8653 ± 0,0122** | **0,8674 ± 0,0014** | 0,9079 ± 0,0105 |

Macro-F1 mengikuti pola yang sama (terbaik CONCAT paddleocr 0,8524 ± 0,0016).

### Seberapa banyak potensi teks dipanen (rerata seed)

Potensi = ORACLE − IMAGE. Panen = (fusi − IMAGE) / potensi. Paper Audebert: 43 %;
Fase 1 (fusion end-to-end, qs-ocr): 29 %.

| Kondisi | potensi | LATE − IMAGE | panen LATE | CONCAT − IMAGE | panen CONCAT |
|---|---|---|---|---|---|
| qs-ocr | 0,0854 | +0,0287 | 34 % | +0,0147 | 13 % |
| tesseract | 0,0781 | +0,0206 | 25 % | +0,0180 | 19 % |
| easyocr | 0,0803 | +0,0369 | 46 % | +0,0398 | 46 % |
| paddleocr | 0,0910 | +0,0483 | 52 % | +0,0505 | 52 % |

LATE di atas IMAGE di **12 dari 12** sel (minimum +0,006). CONCAT tidak: di qs-ocr seed 43
CONCAT 0,8188 < IMAGE 0,8386 (minimum −0,020).

### Uji missing modality

| Kondisi | LATE tanpa teks − IMAGE | CONCAT tanpa teks − IMAGE | LATE tanpa citra − TEXT | CONCAT tanpa citra − TEXT |
|---|---|---|---|---|
| qs-ocr | −0,022 | −0,563 | −0,268 | −0,112 |
| tesseract | −0,008 | −0,591 | −0,271 | −0,135 |
| easyocr | −0,001 | −0,137 | −0,337 | −0,202 |
| paddleocr | −0,003 | −0,301 | −0,349 | −0,216 |

### LATE vs CONCAT (McNemar, Holm per seed)

| Seed | IMAGE | qs-ocr | tesseract | easyocr | paddleocr |
|---|---|---|---|---|---|
| 42 | 0,8315 | LATE lebih baik* | – | – | – |
| 43 | 0,8386 | LATE lebih baik* | – | – | LATE lebih baik* |
| 44 | 0,7808 | CONCAT lebih baik* | – | CONCAT lebih baik* | CONCAT lebih baik* |

`*` signifikan setelah Holm; `–` tidak berbeda nyata. Rerata LATE − CONCAT per seed:
+0,007 / +0,018 / −0,017. Simpangan baku antar-seed CONCAT 0,0014-0,0119, LATE 0,0122-0,0221.

## Temuan

1. **#temuan/positif — OCR yang lebih baik menaikkan fusi, untuk kedua aturan.** Dari
   tesseract ke paddleocr, LATE naik +2,8 dan CONCAT +3,2 poin, berbeda nyata di ketiga
   seed (McNemar antar-OCR di `uji_h1_mcnemar_antar_ocr.csv`). Sel terbaik, CONCAT
   paddleocr 0,8674 ± 0,0014, 3,3 poin di atas fusion end-to-end Fase 1 (0,8342) dan
   memanen 52 % potensi, di atas 43 % paper Audebert.
2. **#temuan/anomali — ukuran "turun tanpa teks" yang dipra-registrasi hanya sah untuk
   LATE.** LATE tanpa teks mendarat tepat di IMAGE (selisih ≤ 2,2 poin), sesuai harapan.
   CONCAT tanpa teks jatuh **jauh di bawah IMAGE sendiri** (sampai −59 poin), dan semua
   versi "tanpa citra" jatuh di bawah TEXT sendiri. Kehilangan satu modalitas tidak mungkin
   lebih buruk daripada model unimodal modalitas lain kecuali masukan penggantinya keluar
   distribusi bagi model. Dugaan untuk CONCAT: *head* belajar "teks kosong ⇒ Note"
   (14,4 % halaman Note pada QS-OCR kosong, [[14-kurasi-dan-eda]]), sehingga vektor teks kosong menarik semua
   prediksi ke Note. Dugaan ini cocok dengan urutan kejatuhannya: paling dalam pada
   qs-ocr/tesseract yang halaman kosongnya paling banyak (1,06 % / 0,60 %), paling dangkal
   pada easyocr/paddleocr (0,17 % / 0,06 %). Belum diuji langsung. Karena itu H3 di bawah
   dinilai dengan **gain di atas IMAGE dan panen potensi**, bukan "turun tanpa teks", kecuali
   untuk LATE.
3. **H3a didukung untuk LATE, lemah untuk CONCAT.** Spearman (OA TEXT, turun tanpa teks
   LATE) = 0,683 di 12 sel; (OA TEXT, gain LATE) = 0,517, (OA TEXT, panen LATE) = 0,65;
   pada rerata 4 kondisi 0,8. Untuk CONCAT, (OA TEXT, gain) = 0,287 (rerata kondisi 0,4).
   Catatan: easyocr punya TEXT lebih lemah dari qs-ocr tetapi panennya jauh lebih besar
   (46 % vs 34 % LATE, 46 % vs 13 % CONCAT), jadi akurasi teks saja tidak menjelaskan
   seberapa banyak teks dipakai.
4. **#temuan/negatif — H3b tidak didukung, arahnya justru terbalik untuk cabang citra yang
   lemah.** Rerata LATE dan CONCAT hampir sama (selisih ≤ 1,4 poin), dan pemenangnya
   bergantung pada seed: LATE unggul nyata saat IMAGE kuat (seed 42/43, terutama qs-ocr),
   CONCAT unggul nyata di 3 dari 4 kondisi saat IMAGE lemah (seed 44, 0,781). CONCAT juga
   jauh lebih stabil antar-seed (paddleocr: std 0,0014 vs 0,0122). Rata-rata tak berbobot
   ikut jatuh bersama cabang yang lemah; *head* yang dilatih menyesuaikan bobot. Untuk cabang
   **teks** yang lemah (tesseract), keduanya tidak berbeda nyata di seed mana pun.
5. Kelas tersulit tetap **Scientific** (recall 0,458 pada CONCAT paddleocr seed 42) dan
   Report (presisi 0,655), sama dengan pola di ketiga paper rujukan.
