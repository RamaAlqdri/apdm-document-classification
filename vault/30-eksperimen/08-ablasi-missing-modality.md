---
judul: Ablasi 2 — missing modality saat inferensi
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-28
status: selesai
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
oa:: 0.8218 (utuh) / 0.8203 (tanpa teks) / 0.0731 (tanpa citra)
macro_f1:: 0.8047 / 0.8004 / 0.0419
durasi:: menit-an, tanpa training

> **DIJALANKAN 2026-09-30**, memakai `FUSION-concat_seed42.pt` apa adanya.
>
> **Ini eksperimen paling menentukan di seluruh proyek.** Hasilnya membalik
> interpretasi Tahap 6.

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
| Keduanya utuh | 0,8218 | 0,8047 | — |
| **Teks dinolkan** (hanya citra) | **0,8203** | 0,8004 | **−0,0015** |
| **Citra dinolkan** (hanya teks) | **0,0731** | 0,0419 | **−0,7487** |

Pembanding dari Tahap 5, seed 42:

| Pembanding | OA |
|---|---|
| IMAGE standalone | 0,8315 |
| TEXT standalone | 0,7345 |
| Tebakan acak (1/10 kelas) | 0,1000 |
| Selalu menebak kelas mayoritas (Memo) | 0,1782 |

Dua angka yang menentukan:

**Menolkan teks menurunkan akurasi 0,15 persen poin.** Bukan 15 poin, bukan 1,5 —
**0,15**. Menghapus sepenuhnya satu dari dua modalitas nyaris tidak berpengaruh.

**Menolkan citra menjatuhkan akurasi ke 0,0731**, yaitu di bawah tebakan acak dan jauh
di bawah menebak kelas mayoritas. Model tidak "turun ke level TEXT standalone" (0,7345)
seperti yang diharapkan kalau kedua modalitas dipakai; ia runtuh total.

## Cara membacanya

Tiga kemungkinan ditulis sebelum eksekusi. **Yang terjadi adalah gabungan kemungkinan
pertama dan ketiga — keduanya yang paling buruk.**

*Kemungkinan 1, "teks dinolkan hampir tidak menurunkan apa pun (< ~1%)": TERJADI, jauh
di bawah ambang itu (0,15%).* Cabang teks praktis tidak terpakai. Fusion adalah model
citra dengan parameter ekstra, dan kenaikan +2,56% di data bersih kemungkinan berasal
dari kapasitas tambahan atau efek regularisasi head concat, bukan dari modalitas kedua.
Ini `#temuan/negatif` yang paling serius di seluruh proyek, persis seperti yang
diantisipasi.

*Kemungkinan 2, "citra dinolkan jatuh ke sekitar TEXT standalone": TIDAK terjadi.*
Itu akan menjadi hasil yang sehat. Tidak terjadi.

*Kemungkinan 3, "citra dinolkan jatuh jauh di bawah TEXT standalone": TERJADI, ekstrem.*
0,0731 melawan 0,7345. Bahkan di bawah tebakan acak.

### Kenapa 0,0731 dan bukan sekadar "buruk"

Di bawah tebakan acak berarti model bukan menebak, melainkan memberi jawaban yang
**secara sistematis salah** — kemungkinan besar mengeluarkan satu kelas minoritas untuk
hampir semua masukan. Dua sebab yang mungkin, dan keduanya menunjuk hal yang sama:

1. **Head fusion sepenuhnya bergantung pada 128 dimensi dari cabang citra.** Tanpa itu,
   128 dimensi dari teks tidak cukup mengarahkan keputusan, karena head-nya tidak pernah
   belajar memakainya.
2. **Efek out-of-distribution.** Nol pada tensor citra yang sudah ternormalisasi berarti
   "citra rata-rata ImageNet", masukan konstan yang tidak pernah dilihat saat training,
   termasuk oleh statistik BatchNorm di cabang citra.

Sebab 2 sendiri tidak menjelaskan kenapa cabang teks tidak menyelamatkan apa pun —
kalau cabang teks berfungsi, ia semestinya tetap memberi sinyal berguna. Jadi sebab 1
tetap penjelasan utamanya.

## Kaitan dengan oracle

Ablasi ini dan [[oracle-sebagai-batas-atas]] menanyakan hal yang berkaitan dari dua
arah. Oracle bertanya "berapa banyak yang **bisa** dipanen dari kedua modalitas".
Ablasi ini bertanya "berapa banyak yang **benar-benar** dipakai".

Kombinasi yang paling menarik: oracle tinggi (potensi besar) tapi menolkan teks tidak
berdampak (potensi tidak dipakai). Itu akan berarti komplementaritasnya nyata tapi
arsitektur concat gagal memanfaatkannya — dan itulah kondisi yang membuat ablasi 4
(fusion bergerbang) layak dikerjakan.

**Kombinasi itulah yang terjadi.**

| Pengukuran | Nilai | Artinya |
|---|---|---|
| Oracle | 0,8977 | potensi +8,9% di atas baseline terbaik |
| "Hanya TEXT benar" | 8,9% | 239 dokumen hanya bisa diselamatkan teks |
| Dipanen FUSION | +2,6% | 29% dari potensi |
| Efek menolkan teks | −0,15% | **cabang teks tidak dipakai** |

Komplementaritasnya terukur dan nyata. Yang gagal adalah arsitekturnya memanfaatkannya.
Dengan itu, syarat untuk mengerjakan **ablasi 4 (fusion bergerbang / cross-attention)**
sudah terpenuhi — lihat rekomendasi.

## Keterbatasan

1. Satu seed (42), tidak ada ± std. Perlu dicatat bahwa pada seed inilah FUSION
   kebetulan **di bawah** IMAGE (0,8218 vs 0,8315), sedangkan rata-rata tiga seed
   sebaliknya. Jadi kesimpulan "teks tidak terpakai" sebaiknya dikonfirmasi pada seed
   43 atau 44 — biayanya beberapa menit, checkpoint sudah ada.
2. Definisi "kosong" untuk citra adalah pilihan, bukan sesuatu yang kanonik. Nol pada
   tensor ternormalisasi = citra rata-rata ImageNet. Alternatif (hitam, putih, noise)
   akan memberi angka berbeda.
3. Menolkan masukan di inferensi tidak sama dengan melatih model tanpa modalitas itu.
   Sebagian dari penurunan 0,7487 hampir pasti efek out-of-distribution, bukan
   hilangnya informasi.

   **Tapi keterbatasan ini tidak melemahkan temuan utamanya.** Temuan utamanya adalah
   arah *sebaliknya*: menolkan teks **tidak** menurunkan apa-apa. Efek
   out-of-distribution hanya bisa membuat penurunan tampak **lebih besar** daripada
   kenyataan, tidak lebih kecil. Jadi 0,15% adalah batas atas kontribusi cabang teks,
   bukan batas bawah.
4. **Belum diuji pada checkpoint FUSION-sum**, yang bobot cabang teksnya 0,31 dan
   menyiratkan hasil berbeda. Lihat [[06-fusion-sum]].

## Temuan

**#temuan/negatif — cabang teks pada FUSION-concat praktis tidak terpakai.** Menolkan
seluruh masukan teks menurunkan akurasi 0,0015. Ini temuan utama Tahap 7 dan ia
mengubah kesimpulan Tahap 6: kenaikan +2,56% FUSION di atas baseline citra **bukan**
bukti bahwa modalitas kedua membantu.

**#temuan/negatif — fusion tidak bisa dipakai tanpa citra.** 0,0731, di bawah tebakan
acak. Untuk penerapan nyata ini penting: model tidak degradasi dengan anggun, ia
runtuh.

**#temuan/positif — komplementaritasnya sendiri nyata.** Oracle 0,8977 dan 8,9% sampel
hanya benar lewat teks. Masalahnya bukan pada data atau pada modalitas, melainkan pada
cara fusion dilatih.

Ketiganya bersama membentuk satu kesimpulan yang lebih tajam daripada "replikasi
berhasil": **potensinya ada, arsitekturnya tidak memanennya, dan penyebab paling
mungkin adalah early stopping yang mengakhiri training sebelum cabang teks matang.**
Diagnosis lengkap dan uji yang menentukannya ada di [[05-fusion-concat]] dan
[[tabel-utama]].
