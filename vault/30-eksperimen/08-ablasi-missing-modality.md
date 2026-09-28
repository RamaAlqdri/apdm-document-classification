---
judul: Ablasi 2 — missing modality saat inferensi
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/07
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[taksonomi-multimodal]]"
  - "[[05-fusion-concat]]"
  - "[[oracle-sebagai-batas-atas]]"
---

# Ablasi 2 — missing modality saat inferensi

model:: FUSION-concat (checkpoint Tahap 6)
dataset:: Tobacco3482, test split seed 42 (2682 dokumen)
split:: seed 42 saja
seed:: 42
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Field angka kosong.

## Pertanyaan

Nol-kan salah satu masukan. Seberapa jauh fusion kolaps?

Ini eksperimen termurah di seluruh Tahap 7 — tidak ada training, tidak ada degradasi
bertahap, hanya dua evaluasi tambahan — tapi kemungkinan yang paling informatif,
karena ia langsung menjawab **apakah model benar-benar memakai kedua modalitas.**

## Kenapa ini penting: risiko yang dicatat sejak Tahap 6

[[05-fusion-concat]] mencatat satu risiko yang tidak bisa diuji tanpa menjalankan
apa pun: cabang citra mulai dari bobot ImageNet sementara cabang teks mulai dari nol,
pada learning rate yang sama. Kalau cabang citra konvergen jauh lebih cepat, fusion
bisa runtuh jadi model citra dengan parameter tambahan.

Yang berbahaya, hasilnya akan **terlihat baik** — setara baseline IMAGE — sambil
sebenarnya gagal total memakai teks. Ablasi ini mendeteksinya langsung: kalau
menolkan teks hampir tidak menurunkan apa pun, itulah yang terjadi.

## Bagaimana "kosong" didefinisikan

**Teks.** Nol persis seperti padding, jadi sekuens yang dinolkan adalah dokumen
kosong yang benar-benar in-distribution. Dataset kita memang sudah memuat dokumen
yang OCR-nya tidak menghasilkan apa pun — persentasenya diukur di
[[01-ekstraksi-fitur-teks]]. Jadi kondisi ini punya padanan alami di data nyata,
bukan sekadar konstruksi.

**Citra.** Tidak sebersih itu. Tensornya sudah ternormalisasi, jadi nol berarti
**citra rata-rata ImageNet**, bukan hitam. Itu pilihan yang paling tidak
mengagetkan — masukan yang tidak berkontribusi apa-apa relatif terhadap statistik
normalisasi — tapi tetap pilihan. Alternatif (hitam, putih, atau noise) akan memberi
angka berbeda, dan ini harus disebut di laporan.

## Hasil

Diisi setelah eksekusi.

| Kondisi | OA | Macro F1 | Turun dari utuh |
|---|---|---|---|
| Keduanya utuh | | | — |
| Teks dinolkan (hanya citra) | | | |
| Citra dinolkan (hanya teks) | | | |

Pembanding yang relevan, dari Tahap 5:

| Pembanding | OA |
|---|---|
| IMAGE standalone | |
| TEXT standalone | |

## Cara membacanya

**Kalau "teks dinolkan" hampir tidak menurunkan apa pun** (turun < ~1%): cabang teks
praktis tidak terpakai. Fusion adalah model citra dengan parameter ekstra, dan
kenaikan akurasinya di data bersih kemungkinan berasal dari kapasitas tambahan, bukan
dari modalitas kedua. `#temuan/negatif`, dan temuan paling serius yang mungkin
muncul di seluruh proyek.

**Kalau "citra dinolkan" jatuh ke sekitar level TEXT standalone**: fusion memakai
kedua modalitas dengan wajar, dan degradasi anggun ketika satu hilang.

**Kalau "citra dinolkan" jatuh jauh di BAWAH TEXT standalone**: model belajar
representasi yang bergantung pada kehadiran kedua masukan sekaligus. Itu bukan
kegagalan, tapi berarti model tidak bisa dipakai pada dokumen tanpa citra — relevan
untuk penerapan nyata dan layak disebut.

## Kaitan dengan oracle

Ablasi ini dan [[oracle-sebagai-batas-atas]] menanyakan hal yang berkaitan dari dua
arah. Oracle bertanya "berapa banyak yang **bisa** dipanen dari kedua modalitas".
Ablasi ini bertanya "berapa banyak yang **benar-benar** dipakai".

Kombinasi yang paling menarik: oracle tinggi (potensi besar) tapi menolkan teks tidak
berdampak (potensi tidak dipakai). Itu akan berarti komplementaritasnya nyata tapi
arsitektur concat gagal memanfaatkannya — dan itulah kondisi yang membuat ablasi 4
(fusion bergerbang) layak dikerjakan.

## Keterbatasan

1. Satu seed (42), tidak ada ± std.
2. Definisi "kosong" untuk citra adalah pilihan, bukan sesuatu yang kanonik.
3. Menolkan masukan di inferensi tidak sama dengan melatih model tanpa modalitas itu.
   Model ini tidak pernah melihat masukan kosong saat training (kecuali dokumen yang
   OCR-nya memang kosong), jadi sebagian penurunan bisa jadi efek out-of-distribution,
   bukan hilangnya informasi.

## Temuan

Diisi setelah eksekusi.
