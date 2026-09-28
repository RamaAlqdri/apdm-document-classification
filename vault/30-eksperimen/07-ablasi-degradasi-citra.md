---
judul: Ablasi 1 — degradasi citra bertahap
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/07
  - modal/citra
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[taksonomi-multimodal]]"
  - "[[05-fusion-concat]]"
  - "[[04-baseline-mobilenetv2]]"
---

# Ablasi 1 — degradasi citra bertahap

model:: IMAGE + FUSION-concat (checkpoint Tahap 5-6)
dataset:: Tobacco3482, test split seed 42 (2682 dokumen)
split:: seed 42 saja
seed:: 42
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Perturbasi sudah lolos self-check, tapi belum
> dijalankan terhadap checkpoint sungguhan. Field angka kosong.

## Pertanyaan

Kalau cabang citra dirusak bertahap, apakah cabang teks mengompensasi?

Ini pengujian empiris atas limitasi yang **penulis sebut tapi tidak uji**: mereka
mengakui Tobacco3482 seluruhnya berorientasi rapi dan discan profesional sehingga
tidak merepresentasikan kondisi nyata, lalu berhenti di situ.

Tidak ada training. Checkpoint Tahap 5-6 dipakai apa adanya; yang berubah hanya
masukannya.

## Caveat yang menentukan cara membaca hasil

Citra didegradasi, tapi **teksnya tetap dari OCR atas citra bersih aslinya**. Di
sistem nyata, scan yang blur menghasilkan OCR yang lebih buruk juga, jadi kedua
cabang akan memburuk bersamaan.

Jadi ablasi ini mengukur ketahanan **arsitektur**, bukan ketahanan sistem ujung ke
ujung. Pertanyaannya tetap sah — "apakah cabang teks menyelamatkan ketika cabang
citra rusak" — tapi jangan dibaca sebagai "sistem ini tahan dokumen buruk".
Dibahas di [[taksonomi-multimodal]], dan justru konsekuensi langsung dari fakta bahwa
teks di sini diturunkan dari citra.

Mengukur versi ujung-ke-ujungnya butuh menjalankan Tesseract ulang pada setiap citra
terdegradasi — 3482 dokumen x belasan tingkat degradasi x ~910ms. Di luar anggaran,
dan dicatat sebagai arah lanjutan.

## Desain

| Jenis | Tingkat | Catatan |
|---|---|---|
| rotate | 0, 2, 5, 10, 20, 45 derajat | diisi **putih**, bukan hitam |
| blur | 0, 1, 2, 4, 8 px radius gaussian | |
| jpeg | quality 100, 50, 20, 10, 5 | turun = makin buruk |
| noise | sigma 0, 0.02, 0.05, 0.1, 0.2 | di ruang piksel [0,1], sebelum normalisasi |

Level pertama tiap jenis adalah identitas, jadi titik pertama tiap kurva **harus**
sama dengan baseline yang tidak disentuh. Itu sekaligus pemeriksaan bahwa pipeline
degradasinya tidak bocor.

**Kenapa rotasi diisi putih:** dokumen adalah tulisan gelap di latar putih. Sudut
hitam hasil rotasi akan menyuntikkan sinyal artifisial yang jauh lebih besar daripada
efek kemiringannya sendiri, dan ablasi berubah jadi "apakah CNN bisa melihat segitiga
hitam". Self-check memverifikasi sudut hasil rotasi 45 derajat tetap terang.

**Kenapa degradasi PIL di paling depan pipeline:** supaya ia bekerja pada scan
aslinya di resolusi asli, seperti scan buruk yang sungguhan datang — bukan pada citra
yang sudah kita resize ke 384x384.

## Hasil

Diisi setelah eksekusi dari `reports/ablasi1_degradasi_citra.csv`.
Figur: `reports/figures/07_ablasi_degradasi_citra.png` (empat panel, IMAGE dan
FUSION dalam satu grafik per jenis).

| Jenis | Level | OA IMAGE | OA FUSION | Selisih |
|---|---|---|---|---|
| | | | | |

## Yang sebenarnya diukur: apakah selisihnya melebar?

Akurasi yang turun bukan temuan — itu sudah pasti. Yang informatif adalah **jarak
FUSION − IMAGE** sebagai fungsi tingkat degradasi:

- **Melebar** → cabang teks memang mengompensasi. Fusion tidak hanya lebih akurat di
  data bersih, tapi lebih tahan.
- **Datar** → fusion sekadar mewarisi ketahanan cabang citra. Nilai tambahnya hanya
  akurasi pada data bersih, bukan robustness. Ini `#temuan/negatif` dan lebih menarik
  daripada hasil positif, karena membatasi klaim kegunaan multimodal.
- **Menyempit** → fusion lebih rapuh daripada baseline citra. Kalau ini terjadi,
  periksa dulu apakah cabang teks benar-benar terpakai (lihat [[08-ablasi-missing-modality]]).

Notebook mencetak arah perubahannya per jenis.

Arah per jenis: diisi setelah eksekusi.

## Hipotesis sebelum menjalankan

Ditulis sekarang supaya tidak dirasionalisasi setelah melihat angkanya.

1. **Rotasi akan paling merusak.** MobileNetV2 tidak punya invarians rotasi, tidak
   ada augmentasi rotasi saat training, dan seluruh dataset berorientasi benar.
   45 derajat kemungkinan menjatuhkan IMAGE mendekati tebakan acak.
2. **Kompresi JPEG paling ringan.** Paper sendiri mengukur artefak JPEG sebagai
   augmentasi dan menemukannya tidak berpengaruh signifikan — jadi modelnya memang
   sudah tidak sensitif di situ.
3. **Selisih FUSION − IMAGE akan melebar**, tapi hanya sampai titik tertentu: begitu
   cabang citra runtuh total, fusion seharusnya mendatar di sekitar level TEXT
   standalone, bukan lebih tinggi.

Hipotesis 3 adalah yang paling mungkin salah, dan paling berguna kalau salah.

## Keterbatasan

1. Satu seed (42). Rencana memotongnya ke satu seed karena jumlah kondisi yang
   dievaluasi sudah besar. Berarti tidak ada ± std di tahap ini.
2. Teks tidak ikut terdegradasi (lihat caveat di atas).
3. Degradasi diterapkan seragam ke seluruh test set, bukan dicampur. Dokumen nyata
   punya kualitas yang beragam dalam satu batch.

## Temuan

Diisi setelah eksekusi.
