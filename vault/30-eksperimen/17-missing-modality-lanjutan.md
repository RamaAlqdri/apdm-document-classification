---
judul: Missing modality lanjutan — pengganti dalam distribusi dan head tanpa teks kosong
tipe: eksperimen
tahap: "12"
tanggal: 2026-10-24
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/12
  - modal/fusion
  - komponen/fusion
  - komponen/evaluasi
terkait:
  - "[[16-fusi-per-ocr]]"
  - "[[14-kurasi-dan-eda]]"
  - "[[08-ablasi-missing-modality]]"
  - "[[limitasi-dan-lanjutan]]"
  - "[[peta-eksperimen]]"
---

# Missing modality lanjutan

model:: LATE dan CONCAT dengan modalitas diganti rerata latih; CONCAT_nonkosong = ConcatHead dilatih ulang tanpa dokumen bertekst kosong
dataset:: Tobacco3482, 4 kondisi OCR (cache keluaran unimodal [[16-fusi-per-ocr]])
split:: 800 latih terstratifikasi (80 val) / 2.682 uji, sama dengan [[16-fusi-per-ocr]]
seed:: 42, 43, 44
oa::
macro_f1::
durasi::

> **BELUM DIJALANKAN.** Kode: `src/experiment.py` (`evaluate_followup`,
> `empty_text_mask`), sel "H3 lanjutan" di `notebooks/pipeline.ipynb` bagian 9.
> Hanya membaca `data/processed/fase2/outputs/*.npz` di mesin compute; tanpa GPU,
> perkiraan beberapa menit (grid fusi asli 72 detik, di sini dua *head* per sel).
> Keluaran: `reports/fase2/uji_lanjutan_missing_modality.csv`, `uji_lanjutan_selisih.csv`,
> `uji_lanjutan_porsi_note.csv`. Self-check dan *dry run* pada data sintetis lolos;
> angka *dry run* tidak bermakna dan tidak dicatat.

## Tujuan

Menguji dugaan di [[16-fusi-per-ocr]] temuan 2. Di sana CONCAT tanpa teks jatuh jauh di
bawah IMAGE sendiri, dan semua versi "tanpa citra" jatuh di bawah TEXT sendiri. Kehilangan
satu modalitas tidak mungkin lebih buruk daripada unimodal modalitas yang tersisa, kecuali
masukan penggantinya (keluaran model untuk masukan kosong) keluar distribusi. Dua
pertanyaan:

1. Apakah kejatuhan itu artefak masukan pengganti? Kalau ya, pengganti yang berada di
   dalam distribusi seharusnya membuat fusi tanpa satu modalitas mendarat dekat unimodal
   modalitas yang tersisa.
2. Apakah *head* CONCAT belajar "teks kosong ⇒ Note"? 14,4 % halaman Note pada QS-OCR
   kosong ([[14-kurasi-dan-eda]]), jadi vektor teks kosong di data latih hampir selalu
   berlabel Note.

## Konfigurasi

- Cache dan split sama dengan [[16-fusi-per-ocr]]; tidak ada model unimodal yang dilatih
  ulang. *Head* CONCAT asli dilatih ulang dengan seed yang sama; notebook memeriksa bahwa
  prediksinya identik dengan bagian 8.
- **Pengganti rerata** (`_rerata`): modalitas yang hilang diganti rata-rata fiturnya
  (CONCAT) atau logitnya (LATE) atas dokumen latih *head*. Untuk LATE ini sama dengan
  menambah bias prior kelas dari modalitas yang hilang.
- **Teks kosong** = dokumen yang fitur teksnya sama dengan fitur masukan kosong (toleransi
  1e-3 relatif), yaitu kosong menurut model: halaman 0 kata atau semua token terbuang saat
  tokenisasi. Notebook mencetak cek silang dengan jumlah halaman 0 kata dari bagian 5.1.
- **CONCAT_nonkosong**: *head* kedua, dilatih dengan dokumen teks kosong dibuang dari
  latih dan val *head*. Set uji tidak berubah (dokumen uji kosong tetap dievaluasi).
- Tambahan ukuran: porsi prediksi Note pada set uji, dan kelas yang paling banyak
  diprediksi, untuk tiap varian tanpa teks.

## Hipotesis (ditulis sebelum eksekusi)

Dinilai pada rerata 3 seed per kondisi OCR. Ambang 5 poin dipilih sebelum eksekusi:
setara dua kali simpangan baku antar-seed IMAGE terbesar di [[16-fusi-per-ocr]] (0,0257).

**H4a — kejatuhan adalah artefak masukan pengganti.** Didukung bila di keempat kondisi:
|CONCAT_tanpa_teks_rerata − IMAGE| ≤ 0,05, LATE_tanpa_citra_rerata ≥ TEXT − 0,05, dan
CONCAT_tanpa_citra_rerata ≥ TEXT − 0,05. Bila CONCAT tanpa teks tetap jauh di bawah IMAGE
dengan pengganti rerata, *head* memang tidak bisa bekerja tanpa teks, dan itu temuan
tentang *head*, bukan tentang ukuran.

**H4b — *head* belajar "teks kosong ⇒ Note".** Prasyarat: |CONCAT_nonkosong − CONCAT| ≤
0,01 (membuang ± 1 % data latih tidak merusak *head*). Didukung bila (i) porsi prediksi
Note CONCAT_tanpa_teks ≥ porsi Note sebenarnya + 0,10, dan (ii) pada CONCAT_nonkosong porsi
itu turun ke ≤ porsi sebenarnya + 0,05, dan (iii) jarak CONCAT_nonkosong_tanpa_teks ke IMAGE
menyusut paling sedikit separuh dibanding CONCAT_tanpa_teks. Tafsir alternatif yang
dibedakan:

- (i) gagal: kejatuhan tidak menarik ke Note; dugaan gugur sejak awal.
- (i) lolos, (ii) gagal: tarikan ke Note tidak berasal dari dokumen kosong di data latih.
- (i) dan (ii) lolos, tetapi kelas terbanyak pindah ke kelas lain dan (iii) gagal:
  masukan kosong keluar distribusi secara umum; *head* tidak belajar aturan khusus Note,
  ia hanya berekstrapolasi.

Batasan yang sudah diketahui: H4b hanya bermakna pada kondisi yang punya dokumen kosong
di bagian latih. Menurut [[16-fusi-per-ocr]] halaman kosong 1,06 % (qs-ocr), 0,60 %
(tesseract), 0,17 % (easyocr), 0,06 % (paddleocr) dari seluruh dokumen, jadi easyocr dan
paddleocr kemungkinan punya 0-2 dokumen kosong di 720 dokumen latih; di sana CONCAT_nonkosong
praktis sama dengan CONCAT. Notebook mencetak `n_kosong_latih` per sel.

## Hasil

> status: belum-dijalankan. Diisi dari `reports/fase2/uji_lanjutan_*.csv` setelah
> pengguna menjalankan sel "H3 lanjutan" di mesin compute.

## Temuan

Belum ada. Bila H4a didukung, ukuran "turun tanpa teks" untuk CONCAT di
[[16-fusi-per-ocr]] dan batasan H.5 di [[limitasi-dan-lanjutan]] dapat dihitung ulang
dengan pengganti rerata.
