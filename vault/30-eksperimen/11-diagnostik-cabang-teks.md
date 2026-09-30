---
judul: Diagnostik — kenapa cabang teks tidak terpakai
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/eksperimen
  - tahap/07
  - modal/teks
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[10-uji-lanjutan-early-stopping]]"
  - "[[12-perbaikan-norm-dan-init]]"
  - "[[03-baseline-cnn1d]]"
  - "[[05-fusion-concat]]"
  - "[[tabel-utama]]"
---

# Diagnostik — kenapa cabang teks tidak terpakai

model:: diagnostik (tanpa training)
dataset:: Tobacco3482, test split seed 42 (2682 dokumen)
split:: seed 42
seed:: 42
oa:: -
macro_f1:: -
durasi:: menit-an

> **DIJALANKAN 2026-09-30** lewat `notebooks/08_diagnostik_cabang_teks.ipynb`. Hanya
> membaca checkpoint; tidak ada training dan tidak ada artefak lama yang ditulis.

## Pertanyaan

[[10-uji-lanjutan-early-stopping]] menetapkan bahwa cabang teks hanya memanen 9-12%
potensi, pada kedua strategi penggabungan dan bahkan setelah 200 epoch penuh. Yang belum
diketahui: **di mana** informasinya hilang. Tiga hipotesis dengan perbaikan berbeda:

- **A — cabang teksnya kolaps.** Perbaikannya: inisialisasi dari checkpoint CNN1D.
- **B — cabang teksnya baik, head-nya mengabaikan.** Perbaikannya berbeda sama sekali:
  normalisasi per cabang atau gerbang eksplisit.
- **C — skalanya tidak seimbang.** Bisa terjadi bersama A atau B.

Menebak salah berarti membuang satu run 3,3 jam pada perbaikan yang salah.

## Metode

Empat pengukuran pada fitur 128 dimensi, dengan **CNN1D standalone sebagai acuan** —
model itu mencapai OA 0,7345 sendirian ([[03-baseline-cnn1d]]), jadi fiturnya kita tahu
berfungsi.

Yang menentukan adalah **linear probe**: regresi logistik di atas fitur yang dibekukan,
di-fit pada split train dan dievaluasi pada test. Probe tinggi berarti fiturnya baik dan
head-nya yang salah; probe gagal berarti cabangnya kolaps.

## Hasil

| Sumber fitur | std | Unit mati | Norma | cos antar dok | **Probe** |
|---|---|---|---|---|---|
| **CNN1D standalone (acuan)** | 1,069 | 18/128 | 21,01 | 0,650 | **0,7479** |
| concat: cabang teks | 0,0317 | 63/128 | 2,56 | 0,982 | 0,3986 |
| **noES: cabang teks** | **0,0279** | **67/128** | **2,50** | **0,978** | **0,4236** |
| sum: cabang teks | 0,0329 | 65/128 | 2,39 | 0,972 | 0,4198 |
| concat: cabang citra | 0,594 | 0/128 | 9,24 | 0,479 | 0,8460 |
| noES: cabang citra | 0,630 | 0/128 | 9,29 | 0,411 | 0,8441 |

Rasio norma citra : teks, konsisten di ketiga model — concat 3,61×, noES **3,72×**,
sum 3,62×.

### Probe-nya sendiri valid

Pemeriksaan yang ditanam sebelum eksekusi: probe atas fitur CNN1D = **0,7479** versus
end-to-end 0,7345. Sedikit di atas, wajar karena regresi logistik tanpa dropout. Cabang
citra: probe 0,8441 versus end-to-end 0,8315. Kedua acuan konsisten, jadi angka 0,4236
bisa dipercaya.

## Hipotesis A terkonfirmasi — cabang teks memang kolaps

Varians **38× lebih kecil** dari acuan, **67 dari 128 unit mati** (acuan 18), cos antar
dokumen 0,978 melawan 0,650. Lewat cabang ini semua dokumen terlihat nyaris identik.
Normanya 8,4× lebih kecil.

## Tapi fiturnya tetap membawa informasi

Probe 0,4236 itu **2,4× di atas kelas mayoritas** (0,178) dan 57% dari acuan. Informasinya
jelas masih ada.

Itu mengubah artinya. Fusion memanen 9% potensi oracle, sementara **regresi logistik
biasa** di atas fitur beku yang sama mencapai 0,42. Head fusion mengabaikan sesuatu yang
bahkan classifier linear bisa pakai. Jadi A benar, dan sebagian B juga benar.

## Hipotesis C terkonfirmasi — dan inilah mekanismenya

Digabung, ketiga temuan memberi satu mekanisme yang konkret, bukan lagi dugaan:

> Fitur teks **pelan** — norma kecil, varians kecil, separuh unit mati — sementara fitur
> citra 3,7× lebih besar. Head concat karena itu didominasi separuh citra secara
> numerik. Cabang teks berkontribusi sedikit **bukan karena tidak berinformasi, tapi
> karena terlalu lirih untuk didengar.**

Perbaikannya karena itu **dua**, bukan satu: inisialisasi dari checkpoint CNN1D untuk
kolapsnya, dan normalisasi per cabang untuk skalanya. Diuji di
[[12-perbaikan-norm-dan-init]].

## Satu metrik ternyata tidak berguna di sini

**Rank efektif gagal jadi pembeda.** CNN1D standalone yang *berfungsi* justru punya rank
efektif **1,62** — nyaris rank-1 juga. Yang kolaps 1,36, rasionya 0,84, tidak memicu apa
pun. Arsitektur ini (max-pool-through-time → FC → ReLU) secara alami memusatkan varians.

Notebook sudah memuat peringatan bahwa tidak ada satu metrik pun yang cukup sendiri,
diverifikasi pada fitur sintetis yang kolapsnya diketahui:

| Bentuk kolaps | `std` | `cos` | `rank efektif` |
|---|---|---|---|
| Isotropik | **0,001 (turun 580×)** | **1,000** | 105,6 — *buta* |
| Terarah | 0,457 — *buta* | 0,494 — *buta* | **1,56** |
| Sehat | 0,582 | 0,321 | 104,8 |

Yang benar-benar membedakan pada kasus nyata kita: **std, unit mati, dan cos** — plus
probe sebagai penentu.

## Temuan

**#temuan/positif — mekanismenya teridentifikasi, bukan lagi dugaan.** Kolaps varians
(38×), separuh unit mati, dan ketimpangan skala 3,7×, ketiganya terukur.

**#temuan/anomali — fitur yang kolaps tetap informatif (probe 0,4236).** Bukan kasus
"tidak ada apa-apa di sana"; head-nya gagal memakai sesuatu yang classifier linear bisa
baca.

**#temuan/negatif — rank efektif tidak bisa dipakai sebagai detektor kolaps pada
arsitektur ini**, karena model acuannya sendiri berrank 1,62.

## Keterbatasan

1. Satu seed (42).
2. Probe linear mengukur informasi yang bisa dipisahkan **secara linear**. Fitur bisa
   memuat informasi non-linear yang tidak tertangkap. Jadi 0,4236 adalah batas bawah.
3. Diagnostik ini korelasional. Ia menunjukkan cabang teks kolaps dan skalanya timpang,
   tapi tidak membuktikan keduanya **penyebab** kontribusi teks yang rendah. Itu yang
   diuji secara intervensional di [[12-perbaikan-norm-dan-init]].
