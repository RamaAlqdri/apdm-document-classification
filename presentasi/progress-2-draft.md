# Draft Presentasi Progress 2

Gabungan laporan rencana (`proposal/proposal.pdf`) dan laporan EDA
(`proposal/laporan-eda.pdf`). Gayanya disamakan dengan Progress 1 (Canva, 16:9). Gambar
siap tempel ada di `presentasi/gambar/`.

Rencana penelitian tidak memuat hasil eksperimen. Angka di slide 5–8 berasal dari EDA,
bukan dari model.

---

## 01 — Judul

**Pengaruh Mesin OCR dan Embedding Teks terhadap Pemanfaatan Modalitas Teks pada
Klasifikasi Dokumen Multimodal**

Progress 2 · Analisis & Pemrosesan Data Multimodal

Disusun oleh:
- Yulian Anggreaif Rudamaga : 256150100011001
- Zalzabila Rani : 256150100011008
- Ramadhani Al-Qadri : 256150100011010

---

## 02 — Referensi Utama Baru

**Kashyap, Mittal & Singh (2026)**, *International Journal of Intelligent Systems*.
Fusion EfficientNetB3 + BERT, akurasi 92% pada Tobacco3482.

Catatan kritis:
- Mesin OCR **tidak disebutkan**, hanya "mesin OCR komersial".
- Tanpa kelas Note dan Report, akurasi maksimum pada 10 kelas hanya **86,6%**.
- Satu kali run, tanpa uji statistik.

Pendukung: Mirończuk (2026), Francis & Sangeetha (2025), Zhang et al. (2025), Michail et
al. (2025), Larson et al. (2025).

---

## 03 — Celah Penelitian

> Studi fusion teks–citra memperlakukan **mesin OCR sebagai konstanta** dan hanya
> melaporkan akurasi gabungan.

- Meta-analisis 139 studi (Mirończuk, 2026): kualitas OCR **tidak terkontrol**, dan hanya
  **11,8%** studi memakai uji statistik.
- Perbandingan mesin OCR yang ada hanya di tingkat OCR, belum dampaknya pada klasifikasi.
- Akurasi gabungan tinggi ≠ kedua modalitas benar-benar dipakai.

---

## 04 — Rancangan Metode

🖼 `gambar/kerangka.png`

- **Mesin OCR:** Tesseract, EasyOCR, PaddleOCR (Keras-OCR bersyarat)
- **Embedding:** FastText + CNN 1D, BERT
- **Fusion:** rata-rata logit (gaya Kashyap) vs fusion fitur + LayerNorm
- **Evaluasi:** OA, macro F1, **uji missing modality** (Δ teks), uji **McNemar**, 3 seed

---

## 05 — Dataset Terkurasi

| Bagian | Jumlah per seed (42 / 43 / 44) |
|---|---|
| Train | 720 |
| Val | 80 |
| Test | 2.682 |

- 3.482 pasangan citra–teks lengkap, 10 kelas, rasio kelas 5,17 : 1.
- Kurasi **menandai, bukan membuang**:
  - teks kosong 37, teks pendek 163, OOV tinggi 172
  - resolusi rendah 68, ketajaman rendah 175, duplikat 9
- Ambang diambil dari ekor distribusi data (5%), bukan ditetapkan di awal.

---

## 06 — EDA: Citra vs Teks

🖼 `gambar/eda_tsne_citra_teks.png`

| Fitur (tanpa fine-tune) | Akurasi probe |
|---|---|
| Citra: MobileNetV2 | **0,729** |
| Teks: rata-rata FastText | **0,669** |
| Tebak kelas mayoritas | 0,178 |

Kedua modalitas sudah informatif sebelum pelatihan, dan pola pengelompokan kelasnya
**berbeda**.

---

## 07 — EDA: Kedua Modalitas Saling Melengkapi

🖼 `gambar/eda_komplementaritas.png`

- **Teks lebih unggul:** Resume (+0,24), Scientific, Email
- **Citra lebih unggul:** Advertisement (−0,30), Note (−0,29)

Ini kelas-kelas tempat fusion diharapkan paling menolong.

---

## 08 — EDA: Modalitas Hilang dan Bising

🖼 `gambar/eda_contoh_teks_bermasalah.png`

- **Note:** 14,4% tanpa teks, 41,3% teks pendek, 30,8% OOV tinggi
- **Iklan terputar 90°** → Tesseract (`--psm 3`) gagal membaca teksnya
- **Kode ID arsip** (Larson et al., 2025) sebagai jalan pintas: lemah di teks OCR (0,278
  vs mayoritas 0,178), tetapi tetap dikontrol

**Terima kasih**
