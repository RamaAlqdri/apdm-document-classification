---
judul: Ekstraksi Fitur Teks — Split, Sekuens 500x300, SIF
tipe: eksperimen
tahap: "04"
tanggal: 2026-09-28
status: belum-dijalankan
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
durasi:: 

> **STATUS: BELUM DIJALANKAN.** `notebooks/02_text_features.ipynb` sudah ditulis
> dan seluruh sel kodenya lolos pemeriksaan sintaks serta pemeriksaan referensi
> terhadap `src/`, tapi **belum pernah dieksekusi**: mesin penulis kode tidak
> memegang dataset maupun model FastText. Field angka dibiarkan kosong sampai
> dijalankan di mesin compute.

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

Tabel stratifikasi per kelas: diisi setelah eksekusi.

## Tokenisasi

spaCy `en_core_web_sm`, punctuation dan whitespace dibuang, satu pass untuk
seluruh korpus. Hasilnya dipakai kedua representasi, jadi tidak ada dokumen yang
ditokenisasi dua kali.

| Metrik | Nilai | Pembanding paper |
|---|---|---|
| Durasi tokenisasi | | — |
| Median token per dokumen | | — |
| Rata-rata token >= 4 karakter | | 136 |
| Dokumen tanpa token | | — |
| Dokumen > 500 token (di-truncate) | | — |

Angka "token >= 4 karakter" sengaja dihitung karena itulah definisi yang dipakai
paper; menghitung semua token akan memberi angka lebih tinggi dan tidak sebanding.

## Analisis OOV

Pengganti Fig. 4a paper. **Tidak memakai kamus eksternal**: kata yang diulang
korpus minimal 50 kali dianggap kosakata rujukan, sisanya dihitung sebagai
non-rujukan. Konsekuensinya angka kita sebanding **bentuk distribusinya**, bukan
nilainya, dengan angka paper (~26% di luar kamus US English, ~23% di luar GloVe).

Figur: `reports/figures/04_distribusi_oov.png`.

| Metrik | Nilai |
|---|---|
| Rata-rata proporsi non-rujukan | |
| Median | |
| Dokumen dengan >90% non-rujukan | |

Yang dicari: apakah bentuknya juga berekor panjang seperti paper, yaitu mayoritas
dokumen terkonsentrasi di proporsi rendah tapi ada puluhan dokumen yang isinya
hampir seluruhnya bukan kata Inggris. Lihat [[noise-ocr-dan-oov]].

## Reproduksi Fig. 4b

Pasangan salah-eja dicari dari korpus kita sendiri, bukan diambil dari paper.
Heuristiknya: kata nyaris-unik (frekuensi <= 2) yang ejaannya mirip (rasio difflib
>= 0,85) dengan kata yang sering diulang korpus. Ini heuristik, bukan ground truth,
jadi pasangannya dicetak untuk diperiksa mata sebelum dipercaya.

Tabel hasil: diisi setelah eksekusi.

| Metrik | Nilai |
|---|---|
| Jumlah pasangan ditemukan | |
| Median cosine FastText | |

Pembanding dari paper (korpus berbeda, jadi bukan angka yang sama):

| Pasangan | GloVe | ELMo | FastText |
|---|---|---|---|
| specifically / Specificalily | 0,71 | 0,68 | 0,96 |
| filter / fiilter | 0,91 | 0,73 | 0,96 |
| alcohol / Aleohol | 0,40 | 0,69 | 0,88 |
| Largely / Largly | 0,25 | 0,81 | 0,98 |

**Ini reproduksi sebagian.** Kolom GloVe dan ELMo tidak dihitung — masing-masing
butuh model besar tambahan yang tidak diunduh. Jangan diklaim setara Fig. 4b penuh.

Notebook juga menguji satu klaim kecil paper: bahwa Tesseract tidak menghasilkan
karakter yang OOV bagi FastText. Hasil: diisi setelah eksekusi.

## Representasi 1 — sekuens untuk CNN1D

`data/processed/sequences_500x300_fp16.npy`, memmap `float16`, shape
(3482, 500, 300). Sekitar 1,0 GB; `float32` akan 2,1 GB dan presisi tambahannya
tidak berarti dibanding noise OCR.

Dokumen lebih pendek dari 500 di-zero-pad (sesuai paper). Dokumen lebih panjang
di-truncate — **paper tidak menyebut kasus ini sama sekali**, jadi truncation
adalah asumsi kita.

| Metrik | Nilai |
|---|---|
| Ukuran file | |
| Durasi | |

## Representasi 2 — SIF untuk MLP

Rata-rata berbobot `a / (a + p(w))` dengan `a = 1e-3`, lalu komponen utama pertama
dibuang. Keduanya nilai default SIF; paper tidak menyebutkannya.

**Komponen utama dipasang pada baris train saja**, karena itu ada satu file per
seed (`sif_seed{42,43,44}.npy`). Menghitungnya atas seluruh korpus adalah
kebocoran ke test yang halus dan mudah terlewat. Lihat [[sif-document-embedding]].

Penyederhanaan yang dicatat: `p(w)` dihitung dari train seed 42 dan dipakai untuk
ketiga seed. Dampaknya kecil karena `p(w)` didominasi stopword, tapi tetap
penyederhanaan, bukan hal yang benar secara ketat.

## Temuan

Diisi setelah eksekusi, dengan tag `#temuan/positif`, `#temuan/negatif`, atau
`#temuan/anomali`.

Pertanyaan yang ingin dijawab angka-angka di atas:

1. Apakah cosine FastText untuk salah eja di korpus **kita** benar-benar tinggi
   seperti klaim paper, atau contoh di Fig. 4b adalah kasus terbaik?
2. Seberapa besar kerugian truncation pada 500 kata?
3. Berapa dokumen yang teksnya kosong, sehingga di model fusion nanti menjadi
   kasus missing modality alami — relevan untuk ablasi 2 di Tahap 7.
