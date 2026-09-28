---
judul: FUSION — penjumlahan adaptif
tipe: eksperimen
tahap: "06"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/06
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[05-fusion-concat]]"
  - "[[04-baseline-mobilenetv2]]"
  - "[[tabel-utama]]"
---

# FUSION — penjumlahan adaptif

model:: FUSION-sum
dataset:: Tobacco3482 (citra JPG + QS-OCR-small)
split:: 800 train (termasuk 10% val) / 2682 test, stratified
seed:: 42, 43, 44
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Field angka kosong sampai dijalankan di mesin
> compute.

## Kenapa eksperimen ini ada

Paper melaporkan strategi penjumlahan "jatuh signifikan di bawah baseline citra
murni", lalu memakai konkatenasi untuk sisa paper. Masalahnya, klaim itu adalah yang
paling lemah landasannya di seluruh paper:

1. **Tidak ada angka.** Disebut sebagai "eksperimen pendahuluan" saja — tanpa OA,
   tanpa F1, tanpa jumlah run.
2. **Tidak ada spesifikasi mekanisme.** Apa yang membuat penjumlahan itu "adaptif"
   tidak pernah dijelaskan.

Karena itu kesimpulannya bisa benar karena alasan yang salah. Lihat
[[strategi-fusion-concat-vs-sum]].

## Mekanisme yang kita pilih

Satu skalar terlatih per cabang, dilewatkan softmax sehingga kedua bobot berjumlah 1:

```
w = softmax([a_citra, a_teks])
v = w[0] * v_citra + w[1] * v_teks
```

Diinisialisasi seimbang (0,5 / 0,5). Ini bacaan yang paling cocok dengan frasa
"adaptive averaging" — sebuah rata-rata, yang bobotnya diadaptasi lewat pelatihan.

**Konsekuensi yang wajib ditulis di laporan:** hasil apa pun di sini berlaku untuk
mekanisme ini, bukan untuk fusion penjumlahan sebagai kelas metode. Dua alternatif
yang tidak kita uji dan berperilaku berbeda: rata-rata tanpa bobot terlatih, dan
gate per-dimensi (satu bobot per indeks vektor, bukan satu per cabang).

Self-check memverifikasi bobotnya benar-benar berjumlah 1, benar-benar menerima
gradien, dan `branch_weights()` menolak dipanggil pada strategi concat.

## Bobot cabang sebagai diagnostik

Bobot terlatih dicetak per run, dan ini informasi yang tidak bisa didapat dari
strategi concat: **kalau bobot teks mendekati nol, model belajar mengabaikan teks
sepenuhnya.** Itu bukan sekadar hasil buruk, itu penjelasan mengapa buruk — dan
sekaligus bukti untuk hipotesis paper bahwa kedua ruang fitur tidak bisa disejajarkan
tanpa merusak daya diskriminatifnya.

Bobot terlatih: diisi setelah eksekusi.

| Seed | Bobot citra | Bobot teks |
|---|---|---|
| 42 | | |
| 43 | | |
| 44 | | |

## Hasil

Diisi setelah eksekusi.

| Seed | OA | Macro F1 | Epoch terbaik | Durasi |
|---|---|---|---|---|
| 42 | | | | |
| 43 | | | | |
| 44 | | | | |
| **rata-rata ± std** | | | | |

Paper tidak memberi angka untuk dibandingkan. Patokannya baseline IMAGE (84,5% di
paper): paper mengklaim penjumlahan turun **di bawah** itu.

## Cara membaca hasilnya

**Kalau penjumlahan memang di bawah baseline citra:** konsisten dengan arah klaim
paper. Tandai `#temuan/positif`, tapi tulis "konsisten untuk mekanisme yang kami
pilih", bukan "berhasil mereplikasi temuan paper" — paper tidak menyebut mekanisme
apa pun untuk direplikasi.

**Kalau tidak:** ini `#temuan/negatif` terhadap paper, dan justru temuan yang lebih
menarik. Kemungkinan besar kegagalan yang dilaporkan paper adalah artefak mekanisme
mereka, bukan sifat fusion penjumlahan. Jangan dihaluskan jadi "hasil kami sedikit
berbeda".

## Temuan

Diisi setelah eksekusi.
