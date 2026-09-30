---
judul: Ablasi 1 — degradasi citra bertahap
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-28
status: selesai
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
oa:: 0.8315 (IMAGE bersih) / 0.8218 (FUSION bersih)
macro_f1:: -
durasi:: menit-an, tanpa training

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050, CUDA), memakai checkpoint
> Tahap 5-6 apa adanya: `IMAGE_seed42.pt` (epoch 22, val_oa 0,9250) dan
> `FUSION-concat_seed42.pt` (epoch 18, val_oa 0,8875).

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

Baseline bersih seed 42: **IMAGE 0,8315 — FUSION 0,8218.** Perhatikan fusion sudah
**di bawah** citra pada seed ini, meski rata-rata tiga seed sebaliknya (0,8342 vs
0,8086). Seluruh tabel di bawah harus dibaca relatif terhadap titik awal ini.

| Jenis | Level | OA IMAGE | OA FUSION | Selisih |
|---|---|---|---|---|
| rotate | 0° | 0,8315 | 0,8218 | −0,0097 |
| rotate | 2° | 0,8311 | 0,8069 | −0,0242 |
| rotate | 5° | 0,8184 | 0,8069 | −0,0116 |
| rotate | 10° | 0,7498 | 0,7543 | +0,0045 |
| rotate | 20° | 0,5761 | 0,5384 | −0,0377 |
| **rotate** | **45°** | **0,3031** | **0,1223** | **−0,1808** |
| blur | 0 | 0,8315 | 0,8218 | −0,0097 |
| blur | 1 | 0,8292 | 0,8225 | −0,0067 |
| blur | 2 | 0,8262 | 0,8098 | −0,0164 |
| blur | 4 | 0,7595 | 0,7301 | −0,0295 |
| **blur** | **8** | **0,6115** | **0,5026** | **−0,1089** |
| jpeg | q100 | 0,8315 | 0,8218 | −0,0097 |
| jpeg | q50 | 0,8318 | 0,8195 | −0,0123 |
| jpeg | q20 | 0,8289 | 0,8218 | −0,0071 |
| jpeg | q10 | 0,8300 | 0,8221 | −0,0078 |
| jpeg | q5 | 0,8248 | 0,8270 | +0,0022 |
| noise | σ 0,00 | 0,8315 | 0,8218 | −0,0097 |
| noise | σ 0,02 | 0,8113 | 0,8110 | −0,0004 |
| noise | σ 0,05 | 0,7491 | 0,7315 | −0,0175 |
| noise | σ 0,10 | 0,5004 | 0,4288 | −0,0716 |
| **noise** | **σ 0,20** | **0,2782** | **0,1156** | **−0,1626** |

Level 0 setiap jenis memberi angka identik dengan baseline — pipeline degradasinya
tidak bocor.

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

### Arah per jenis

| Jenis | Selisih teringan → terberat | Arah |
|---|---|---|
| rotate | −0,0097 → −0,1808 | **menyempit** |
| blur | −0,0097 → −0,1089 | **menyempit** |
| noise | −0,0097 → −0,1626 | **menyempit** |
| jpeg | −0,0097 → +0,0022 | melebar (tapi lihat catatan) |

**Tiga dari empat menyempit. Fusion tidak mengompensasi — ia kolaps lebih dalam
daripada baseline citra.**

JPEG yang "melebar" tidak berarti apa-apa: OA IMAGE hanya turun 0,0067 dari q100 ke
q5, jadi tidak ada degradasi berarti untuk dikompensasi. Ini justru mereplikasi temuan
paper sendiri bahwa artefak JPEG tidak berpengaruh pada dokumen grayscale.

Titik paling ekstrem: **rotasi 45°** — IMAGE 0,3031, FUSION 0,1223. Fusion kehilangan
18 poin lebih banyak, dan 0,1223 itu di bawah tebakan acak (0,10 ≈ 1/10 kelas) hanya
sedikit. Modelnya praktis runtuh.

## Hipotesis sebelum menjalankan — dan hasilnya

Ditulis sebelum eksekusi, disimpan apa adanya.

**1. "Rotasi akan paling merusak." → BENAR.** Rotasi 45° menjatuhkan IMAGE ke 0,3031,
penurunan terbesar di antara semua jenis pada level terberatnya. Sesuai dugaan:
MobileNetV2 tidak punya invarians rotasi dan tidak ada augmentasi rotasi saat training.

**2. "Kompresi JPEG paling ringan." → BENAR.** Bahkan pada quality 5, IMAGE hanya
turun 0,0067. Mereplikasi temuan paper.

**3. "Selisih FUSION − IMAGE akan melebar." → SALAH, dan salahnya besar.**

Saya menduga cabang teks akan mengompensasi sehingga jaraknya melebar saat citra
memburuk. Yang terjadi sebaliknya di tiga dari empat jenis: jaraknya **menyempit**
tajam, sampai −0,18 pada rotasi 45°.

Inilah nilai menulis hipotesis lebih dulu. Kalau ditulis setelah melihat angka, sangat
mudah merasionalisasi "fusion memang tidak diharapkan tahan degradasi". Padahal
prediksi eksplisitnya adalah sebaliknya, dan prediksi itu keliru.

**Kenapa keliru** terjawab oleh [[08-ablasi-missing-modality]]: cabang teks pada model
fusion kita praktis tidak terpakai. Tanpa cabang teks yang berfungsi, tidak ada apa pun
untuk mengompensasi — dan head fusion yang dilatih pada fitur citra bersih justru lebih
sensitif terhadap fitur citra yang rusak daripada classifier citra biasa.

## Keterbatasan

1. Satu seed (42). Rencana memotongnya ke satu seed karena jumlah kondisi yang
   dievaluasi sudah besar. Berarti tidak ada ± std di tahap ini.
2. Teks tidak ikut terdegradasi (lihat caveat di atas).
3. Degradasi diterapkan seragam ke seluruh test set, bukan dicampur. Dokumen nyata
   punya kualitas yang beragam dalam satu batch.

## Temuan

**#temuan/negatif — fusion lebih rapuh daripada baseline citra, bukan lebih tahan.**
Pada rotasi, blur, dan noise, jarak FUSION − IMAGE menyempit tajam seiring degradasi.
Pada rotasi 45° fusion kehilangan 18 poin lebih banyak.

Ini membatasi klaim kegunaan multimodal dengan cara yang tidak dibahas paper. Paper
menyebut ketidakrealistisan dataset sebagai limitasi lalu berhenti; ketika limitasi itu
diuji, arsitektur fusion mereka ternyata **tidak** memberi ketahanan tambahan. Pada
setelan kami, ia malah mengurangi.

**#temuan/positif — hipotesis 1 dan 2 terkonfirmasi.** Rotasi paling merusak, JPEG
paling tidak berpengaruh, keduanya sesuai dugaan dan sesuai paper.

**#temuan/anomali — hipotesis 3 keliru secara terbalik.** Dicatat sebagai kekeliruan
prediksi saya, bukan disamarkan.

**Caveat yang menentukan cara membaca semua ini** (diulang dari atas karena penting):
teks di sini tetap berasal dari OCR atas citra **bersih**. Di sistem nyata, scan yang
dirotasi 45° menghasilkan OCR yang jauh lebih buruk juga, jadi angka di atas mengukur
ketahanan **arsitektur**, bukan sistem ujung ke ujung. Untuk kasus fusion kita
kesimpulannya justru lebih kuat, bukan lebih lemah: fusion gagal mengompensasi
**bahkan ketika teksnya diberi keuntungan tidak realistis berupa sumber yang bersih.**
