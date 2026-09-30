---
judul: FUSION — penjumlahan adaptif
tipe: eksperimen
tahap: "06"
tanggal: 2026-09-28
status: selesai
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
oa:: 0.8275
macro_f1:: 0.7990
durasi:: 1.95 jam

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050 4 GB, CUDA). Angka di bawah
> dari eksekusi nyata `notebooks/05_fusion.ipynb`.

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

| Seed | Bobot citra | Bobot teks |
|---|---|---|
| 42 | 0,694 | **0,306** |
| 43 | 0,685 | **0,315** |
| 44 | 0,691 | **0,309** |

**Sangat konsisten antar seed, dan teks mendapat bobot yang jauh dari nol.** Kalau
cabang teks benar-benar derau, tekanan pelatihan seharusnya menurunkan bobotnya
mendekati 0. Itu tidak terjadi: ketiga seed berhenti di sekitar 0,31 dengan sebaran
hanya 0,009.

Ini membuat gambarannya menarik sekaligus belum selesai. Pada strategi `concat`,
[[08-ablasi-missing-modality]] menunjukkan cabang teks praktis tidak terpakai. Pada
`sum`, bobot terlatihnya menyiratkan sebaliknya.

**Yang belum diuji:** ablasi missing modality **hanya dijalankan pada checkpoint
concat**. Menjalankannya pada checkpoint `sum` biayanya beberapa menit dan
checkpointnya sudah ada. Sampai itu dikerjakan, bobot 0,31 adalah petunjuk, bukan
bukti — pada penjumlahan, menolkan teks menghasilkan `0,69 × v_citra`, yaitu versi
terskala dari fitur citra yang mungkin masih bisa ditangani head-nya, sehingga
ujinya memang kurang tajam daripada pada concat.

## Hasil

| Seed | OA | Macro F1 | Epoch terbaik | Epoch dijalankan | Durasi |
|---|---|---|---|---|---|
| 42 | 0,8043 | 0,7785 | 12 | 27 | 32,9 menit |
| 43 | 0,8389 | 0,8116 | 20 | 35 | 42,1 menit |
| 44 | 0,8393 | 0,8068 | 19 | 34 | 41,8 menit |
| **rata-rata ± std** | **0,8275 ± 0,0164** | **0,7990 ± 0,0146** | 17 | 32 | 1,95 jam |

Paper tidak memberi angka untuk dibandingkan. Patokannya baseline IMAGE — paper
mengklaim penjumlahan turun **di bawah** itu.

### Perbandingan langsung

| Model | OA | vs baseline IMAGE |
|---|---|---|
| baseline IMAGE | 0,8086 | — |
| FUSION sum | **0,8275 ± 0,0164** | **+0,0189** |
| FUSION concat | 0,8342 ± 0,0105 | +0,0256 |

**Penjumlahan tidak jatuh di bawah baseline citra. Ia berada 1,9% di atasnya.**

Dan selisih concat vs sum hanya **0,0067**, jauh di dalam satu standar deviasi
(concat ± 0,0105, sum ± 0,0164). Pada tiga seed, kedua strategi **tidak terbedakan
secara statistik**.

## Cara membaca hasilnya

Dua cabang yang ditulis sebelum eksekusi, disimpan apa adanya supaya jelas mana yang
terjadi:

- *Kalau penjumlahan memang di bawah baseline citra:* konsisten dengan arah klaim
  paper, tapi ditulis sebagai "konsisten untuk mekanisme yang kami pilih", bukan
  "berhasil mereplikasi temuan paper".
- *Kalau tidak:* `#temuan/negatif` terhadap paper, dan jangan dihaluskan jadi "hasil
  kami sedikit berbeda".

**Yang terjadi adalah cabang kedua.** Penjumlahan berada 1,9% **di atas** baseline
citra dan tidak terbedakan secara statistik dari concat. Ditulis apa adanya di bagian
Temuan.

## Temuan

**#temuan/negatif terhadap paper — klaim kegagalan penjumlahan tidak terreplikasi.**
Paper menyatakan penjumlahan "jatuh signifikan di bawah baseline citra murni". Milik
kita 1,9% di atasnya, dan tidak terbedakan secara statistik dari concat.

Konsekuensinya bukan "paper salah", melainkan lebih spesifik dan lebih berguna:
**alasan paper memilih concat tidak berdasar pada bukti yang bisa diperiksa.** Mereka
tidak melaporkan angka, tidak menyebutkan mekanisme, dan kesimpulannya tidak bertahan
pada satu bacaan yang wajar atas frasa "adaptive averaging".

Caveat yang wajib tetap disebut: mekanisme kita adalah **pilihan kita** — satu skalar
terlatih per cabang lewat softmax. Dua alternatif yang tidak diuji berperilaku beda:
rata-rata tanpa bobot terlatih, dan gate per-dimensi. Kalau paper memakai yang
pertama, kegagalan mereka masuk akal dan tidak bertentangan dengan hasil kita.
Dugaan paling wajar tetap: **kegagalan yang mereka laporkan adalah artefak mekanisme
mereka, bukan sifat fusion penjumlahan sebagai kelas metode.**

**#temuan/anomali — bobot teks 0,31 bertentangan dengan hasil ablasi concat.** Pada
concat, teks praktis tidak terpakai; pada sum, bobotnya jauh dari nol dan stabil
antar seed. Dua model berbeda, jadi bukan kontradiksi logis — tapi ini pertanyaan
terbuka yang **bisa dijawab dalam beberapa menit** dengan menjalankan ablasi missing
modality pada checkpoint sum.

> **SUDAH DIJAWAB, dan jawabannya tidak seperti dugaan.**
> [[10-uji-lanjutan-early-stopping]] menjalankan ablasi itu pada checkpoint `sum`:
> menolkan teks menurunkan OA **0,0097** (0,8043 → 0,7946). Enam setengah kali lipat
> efeknya pada concat, tapi tetap hanya 26 dokumen dari 2682 — 12% dari potensi oracle.
>
> **Bobot 0,31 bukan bukti teks dipakai.** Softmax sekadar tidak punya tekanan kuat untuk
> mendorongnya ke nol. Kontradiksinya hilang: pada **kedua** strategi penggabungan,
> cabang teks nyaris tidak berkontribusi, dan [[11-diagnostik-cabang-teks]] mengukur
> penyebabnya sama pada keduanya — kolaps fitur (std 0,033 melawan acuan 1,069, 65 dari
> 128 unit mati) plus ketimpangan skala 3,62×.
>
> Pelajaran metodologisnya: **bobot gerbang terlatih adalah diagnostik yang lemah.**
> Ia memberitahu apa yang tidak dimatikan model, bukan apa yang dipakainya. Yang
> menentukan adalah uji missing modality dan probe linear.

**#temuan/positif — sum lebih murah.** 1,95 jam vs 2,92 jam untuk concat, karena
head-nya menerima 128 dimensi alih-alih 256 dan early stopping menyala lebih awal.
Kalau keduanya memang setara akurasinya, sum adalah pilihan yang lebih efisien —
kesimpulan yang berlawanan dengan rekomendasi paper.
