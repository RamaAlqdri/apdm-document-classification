---
judul: Ekstraksi Fitur Teks — Split, Sekuens 500x300, SIF
tipe: eksperimen
tahap: "04"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/eksperimen
  - tahap/04
  - modal/teks
  - komponen/embedding
terkait:
  - "[[fasttext-subword]]"
  - "[[sif-document-embedding]]"
  - "[[cnn-1d-teks]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[00-audit-data]]"
---

# Ekstraksi Fitur Teks — Split, Sekuens 500x300, SIF

model:: ekstraksi-fitur
dataset:: QS-OCR-small (3482 dokumen)
split:: 800 train (termasuk 10% val) / sisanya test, stratified, seed 42/43/44
seed:: 42, 43, 44
oa:: 
macro_f1:: 
durasi:: 56,2s (tokenisasi 51,5s + sekuens 4,7s)

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050, CUDA). Semua angka berasal
> dari eksekusi nyata `notebooks/02_text_features.ipynb`.

## Tujuan

Menyiapkan dua representasi teks yang akan dipakai Tahap 5 dan 6, dan menguji
sendiri klaim inti paper bahwa FastText tahan terhadap salah eja OCR.

## Prasyarat

1. `01_data_audit.ipynb` sudah dijalankan → `data/processed/manifest.csv` ada.
2. `python -m spacy download en_core_web_sm`.
3. `models/cc.en.300.bin` (~7 GB di disk, ~15 GB saat dimuat). **Wajib `.bin`**,
   bukan `.vec` — lihat [[fasttext-subword]].

## Split

Deviasi dari paper, bukan replikasi. Paper memakai k-fold cross-validation dan
melatih dengan jumlah epoch tetap; kita memakai tiga random split terstratifikasi
dan memotong 10% validation dari 800 train untuk early stopping.

| Bagian | Jumlah |
|---|---|
| train | 720 |
| val | 80 |
| test | 2682 |

Validation diambil **dari dalam** 800, bukan ditambahkan di atasnya, supaya jumlah
dokumen yang dilihat model saat training tidak melebihi angka paper.

Split disimpan sebagai indeks ke `manifest.csv` plus fingerprint file itu
(`split_seed{42,43,44}.json`). Kalau manifest berubah urutan, `load_split` menolak
dengan error alih-alih menukar label diam-diam.

Stratifikasi terverifikasi — proporsi train mengikuti korpus sampai satu desimal:

| Kelas | train | val | test | train % | korpus % |
|---|---|---|---|---|---|
| Advertisement | 48 | 5 | 177 | 6,7 | 6,6 |
| Email | 124 | 14 | 461 | 17,2 | 17,2 |
| Form | 89 | 10 | 332 | 12,4 | 12,4 |
| Letter | 117 | 13 | 437 | 16,2 | 16,3 |
| Memo | 128 | 14 | 478 | 17,8 | 17,8 |
| News | 39 | 4 | 145 | 5,4 | 5,4 |
| Note | 41 | 5 | 155 | 5,7 | 5,8 |
| Report | 55 | 6 | 204 | 7,6 | 7,6 |
| Resume | 25 | 3 | 92 | 3,5 | 3,4 |
| Scientific | 54 | 6 | 201 | 7,5 | 7,5 |

**Peringatan yang baru terlihat setelah Tahap 5-6 dijalankan:** validation hanya
**80 sampel**, jadi granularitasnya 1,25% per sampel. Itu terlalu kasar untuk
memilih checkpoint, dan ternyata jadi masalah nyata — `patience=15` menyala karena
derau, sehingga IMAGE dan FUSION berhenti pada epoch 11-46 dari 200 yang
direncanakan. Lihat diagnosis di [[tabel-utama]].

## Tokenisasi

spaCy `en_core_web_sm`, punctuation dan whitespace dibuang, satu pass untuk
seluruh korpus. Hasilnya dipakai kedua representasi, jadi tidak ada dokumen yang
ditokenisasi dua kali.

| Metrik | Nilai | Pembanding paper |
|---|---|---|
| Durasi tokenisasi | **51,5s** (15 ms/dokumen) | — |
| Median token per dokumen | **175** | — |
| Rata-rata token (semua) | 226 (maks 1621) | — |
| **Rata-rata token >= 4 karakter** | **135** | **136** |
| Dokumen tanpa token | **27** (0,8%) | — |
| Dokumen > 500 token (di-truncate) | **8,0%** | — |

Baris yang dicetak tebal itu **konfirmasi terkuat di seluruh proyek** bahwa korpus
teks kita adalah korpus yang sama dengan paper: 135 vs 136, dihitung dengan definisi
yang sama. Tidak ada cara mendapatkan kecocokan sedekat itu secara kebetulan.

Truncation 8,0% ternyata tidak menjadi masalah: ablasi Tahap 7 menunjukkan
menghapus **75% kata** pun tidak mengubah akurasi fusion, jadi kehilangan kata di
atas posisi 500 hampir pasti tidak berdampak. Lihat [[09-ablasi-degradasi-teks]].

Angka "token >= 4 karakter" sengaja dihitung karena itulah definisi yang dipakai
paper; menghitung semua token akan memberi angka lebih tinggi dan tidak sebanding.

## Analisis OOV

Pengganti Fig. 4a paper. **Tidak memakai kamus eksternal**: kata yang diulang
korpus minimal 50 kali dianggap kosakata rujukan, sisanya dihitung sebagai
non-rujukan. Konsekuensinya angka kita sebanding **bentuk distribusinya**, bukan
nilainya, dengan angka paper (~26% di luar kamus US English, ~23% di luar GloVe).

Figur: `reports/figures/04_distribusi_oov.png`.

| Metrik | Nilai | Paper (metode berbeda) |
|---|---|---|
| Rata-rata proporsi non-rujukan | **56,8%** | ~26% di luar kamus |
| Median | **53,3%** | ~10% (mode distribusi) |
| Dokumen dengan >90% non-rujukan | **226** (6,5%) | "puluhan dokumen" |

**Angka kita dua kali lipat paper, dan itu memang diperkirakan** — bukan temuan.
Kosakata rujukan kita adalah kata yang diulang korpus minimal 50 kali; itu jauh
lebih ketat daripada kamus Enchant yang memuat ratusan ribu entri. Jadi banyak kata
Inggris yang sah ikut terhitung non-rujukan. Yang sebanding hanyalah **bentuk**
distribusinya.

Bentuknya cocok pada satu hal penting: ekor panjangnya ada, 226 dokumen (6,5%)
isinya >90% non-rujukan. Paper menyebut "puluhan dokumen"; kita dapat ratusan,
konsisten dengan ambang yang lebih ketat. Lihat [[noise-ocr-dan-oov]].

## Reproduksi Fig. 4b

Pasangan salah-eja dicari dari korpus kita sendiri, bukan diambil dari paper.
Heuristiknya: kata nyaris-unik (frekuensi <= 2) yang ejaannya mirip (rasio difflib
>= 0,85) dengan kata yang sering diulang korpus. Ini heuristik, bukan ground truth,
jadi pasangannya dicetak untuk diperiksa mata sebelum dipercaya.

| Metrik | Nilai |
|---|---|
| Jumlah pasangan ditemukan | **40** |
| **Median cosine FastText** | **0,190** |

Pasangan dengan cosine tertinggi:

| Salah eja | Dugaan asli | Cosine |
|---|---|---|
| Natonal | National | 0,725 |
| hought | thought | 0,645 |
| Cener | Center | 0,562 |
| targe | target | 0,512 |
| Chemica | Chemical | 0,463 |

Terendah: `continned`/`continued` 0,120, `Aithough`/`Although` 0,128,
`Houstun`/`Houston` 0,155.

Pembanding dari paper (korpus berbeda, jadi bukan angka yang sama):

| Pasangan | GloVe | ELMo | FastText |
|---|---|---|---|
| specifically / Specificalily | 0,71 | 0,68 | 0,96 |
| filter / fiilter | 0,91 | 0,73 | 0,96 |
| alcohol / Aleohol | 0,40 | 0,69 | 0,88 |
| Largely / Largly | 0,25 | 0,81 | 0,98 |

**Median kita 0,190 versus 0,88-0,98 di paper. Jangan dibaca sebagai "FastText
gagal"** — ada dua perancu yang belum dikendalikan, dan keduanya berpihak pada
paper:

1. **Heuristik kita ikut menjaring yang bukan salah eja.** `Inactivation`→`activation`,
   `analyzer`→`analyzed`, `tess`→`tests` itu kata berbeda, bukan salah baca. Juga
   banyak pemotongan (`Chemica`, `REPOR`, `targe`) yang jaraknya lebih jauh daripada
   substitusi satu karakter seperti contoh paper.
2. **Kata dalam kosakata memakai vektor terlatih, kata OOV memakai jumlah n-gram.**
   `Although` ada di kosakata FastText sehingga mendapat vektor hasil pelatihan;
   `Aithough` tidak, sehingga dibangun dari subword saja. Kedua vektor itu tidak
   hidup di subruang yang sama, dan jaraknya bisa jauh lebih besar daripada yang
   disiratkan paper.

**Uji terkontrol yang belum dikerjakan:** hitung cosine untuk empat pasangan milik
paper sendiri (`specifically`/`Specificalily`, `filter`/`fiilter`,
`alcohol`/`Aleohol`, `Largely`/`Largly`) dengan model kita. Kalau keluar ~0,96,
median rendah kita adalah sifat pemilihan pasangan, bukan sifat FastText. Murah,
dan ia memisahkan artefak metode dari temuan. **Sampai itu dikerjakan, angka 0,190
tidak boleh dipakai sebagai bukti melawan klaim paper.**

**Ini reproduksi sebagian.** Kolom GloVe dan ELMo tidak dihitung — masing-masing
butuh model besar tambahan. Jangan diklaim setara Fig. 4b penuh.

Klaim kecil paper yang diuji: bahwa Tesseract tidak menghasilkan karakter OOV bagi
FastText. **Hasil: klaim itu tidak berlaku di korpus kita.** Dari 109 karakter unik,
satu punya vektor nol yaitu underscore `_`. Dampaknya sepele, tapi klaimnya memang
tidak akurat.

## Representasi 1 — sekuens untuk CNN1D

`data/processed/sequences_500x300_fp16.npy`, memmap `float16`, shape
(3482, 500, 300). Sekitar 1,0 GB; `float32` akan 2,1 GB dan presisi tambahannya
tidak berarti dibanding noise OCR.

Dokumen lebih pendek dari 500 di-zero-pad (sesuai paper). Dokumen lebih panjang
di-truncate — **paper tidak menyebut kasus ini sama sekali**, jadi truncation
adalah asumsi kita.

| Metrik | Nilai |
|---|---|
| Ukuran file | **1,04 GB** |
| Shape | (3482, 500, 300) float16 |
| Durasi | **4,7s** (1 ms/dokumen) |

Padding terverifikasi: dokumen dengan 263 token benar-benar nol mulai posisi 263.

## Representasi 2 — SIF untuk MLP

Rata-rata berbobot `a / (a + p(w))` dengan `a = 1e-3`, lalu komponen utama pertama
dibuang. Keduanya nilai default SIF; paper tidak menyebutkannya.

**Komponen utama dipasang pada baris train saja**, karena itu ada satu file per
seed (`sif_seed{42,43,44}.npy`). Menghitungnya atas seluruh korpus adalah
kebocoran ke test yang halus dan mudah terlewat. Lihat [[sif-document-embedding]].

Terverifikasi bahwa komponen utama benar-benar terbuang: sisa proyeksi pada baris
train tinggal 2,5e-07 / 6,0e-07 / 1,23e-07 untuk seed 42/43/44.

Penyederhanaan yang dicatat: `p(w)` dihitung dari train seed 42 dan dipakai untuk
ketiga seed. Dampaknya kecil karena `p(w)` didominasi stopword, tapi tetap
penyederhanaan, bukan hal yang benar secara ketat.

## Temuan

**#temuan/positif — korpus terkonfirmasi identik dengan paper.** 135 vs 136 token
≥4 karakter.

**#temuan/anomali — cosine salah eja jauh di bawah paper (0,190 vs ~0,96), tapi
belum bisa disimpulkan.** Dua perancu belum dikendalikan; uji empat pasangan milik
paper adalah langkah yang memisahkannya. Ini pertanyaan terbuka, bukan temuan.

**#temuan/negatif — klaim "tidak ada karakter OOV" tidak berlaku.** Underscore `_`
punya vektor nol di korpus kita. Sepele dampaknya, tapi klaimnya salah.

Jawaban atas tiga pertanyaan yang diajukan sebelum eksekusi:

1. **Belum terjawab** — lihat anomali di atas.
2. **Kerugian truncation dapat diabaikan.** 8,0% dokumen terpotong, tapi ablasi
   Tahap 7 menunjukkan menghapus 75% kata pun tidak mengubah akurasi fusion.
   Pertanyaannya jadi tidak relevan dengan cara yang tidak diduga — lihat
   [[09-ablasi-degradasi-teks]].
3. **27 dokumen (0,8%) tanpa token sama sekali**, terkonsentrasi di kelas Note
   (48,3% dokumennya < 20 kata). Ini missing modality yang muncul alami di data.
