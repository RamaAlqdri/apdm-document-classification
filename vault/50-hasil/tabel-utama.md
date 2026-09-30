---
judul: Tabel Utama — TEXT / IMAGE / FUSION / Oracle
tipe: hasil
tahap: "06"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/hasil
  - tahap/06
  - modal/fusion
  - komponen/evaluasi
terkait:
  - "[[12-perbaikan-norm-dan-init]]"
  - "[[11-diagnostik-cabang-teks]]"
  - "[[sintesis-akhir]]"
  - "[[03-baseline-cnn1d]]"
  - "[[04-baseline-mobilenetv2]]"
  - "[[05-fusion-concat]]"
  - "[[06-fusion-sum]]"
  - "[[oracle-sebagai-batas-atas]]"
  - "[[peta-proyek]]"
---

# Tabel Utama — TEXT / IMAGE / FUSION / Oracle

> **DIJALANKAN 2026-09-30** di laptop Windows (i7 gen 11, RTX 3050 4 GB, CUDA).
> Sumber angka: `reports/tabel_utama.csv` dari `notebooks/05_fusion.ipynb`.

Padanan Tabel 3 paper, dirata-ratakan atas tiga seed.

## Hasil kita

Rata-rata tiga seed (42/43/44). F1 per kelas; baris OA di kolom kedua.

| Model | OA | Macro F1 | Adv. | Email | Form | Letter | Memo | News | Note | Report | Resume | Sci. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEXT (CNN1D) | 0,7266 | 0,7017 | 0,537 | 0,953 | 0,718 | 0,671 | 0,758 | 0,709 | 0,621 | 0,495 | **0,969** | 0,587 |
| IMAGE (MobileNetV2) | 0,8086 | 0,7837 | 0,902 | 0,955 | 0,804 | 0,805 | 0,844 | 0,890 | 0,771 | 0,527 | 0,784 | 0,554 |
| **FUSION (concat)** | **0,8342** | **0,8105** | 0,894 | 0,961 | 0,848 | 0,835 | 0,879 | 0,883 | 0,780 | 0,638 | 0,803 | 0,583 |
| FUSION (sum) | 0,8275 | 0,7990 | — | — | — | — | — | — | — | — | — | — |
| Oracle | 0,8977 | 0,8850 | 0,930 | 0,983 | 0,902 | 0,898 | 0,920 | 0,938 | 0,876 | 0,735 | 0,951 | 0,718 |

Tiga model tambahan dari uji lanjutan, semuanya **seed 42 saja** sehingga tidak
sebanding langsung dengan baris rata-rata tiga seed di atas:

| Model (seed 42) | OA | Macro F1 | Resume | Memo | Catatan |
|---|---|---|---|---|---|
| FUSION concat, seed 42 | 0,8218 | 0,8047 | 0,803 | 0,879 | run pertama |
| FUSION concat tanpa early stopping | 0,8304 | 0,8108 | 0,814 | 0,902 | [[10-uji-lanjutan-early-stopping]] |
| **FUSION diperbaiki** | 0,8110 | **0,8108** | **0,968** | 0,807 | [[12-perbaikan-norm-dan-init]] |

Model terakhir adalah satu-satunya yang **benar-benar memakai kedua modalitas**, dan
satu-satunya yang mereplikasi Resume paper (0,96). OA-nya justru paling rendah.

F1 per kelas untuk FUSION sum tidak dihitung; notebook hanya menyimpan OA dan macro F1
untuk strategi pembanding.

Standar deviasi antar seed: TEXT ± 0,0059 · IMAGE ± 0,0210 · FUSION concat ± 0,0105 ·
FUSION sum ± 0,0164.

## Tabel 3 paper, sebagai pembanding

| Model | OA | F1 | Adv. | Email | Form | Letter | Memo | News | Note | Report | Resume | Sci. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEXT | 73,8% | 0,71 | 0,60 | 0,96 | 0,76 | 0,71 | 0,79 | 0,67 | 0,62 | 0,43 | 0,97 | 0,57 |
| IMAGE | 84,5% | 0,82 | 0,94 | 0,96 | 0,85 | 0,83 | 0,90 | 0,89 | 0,83 | 0,61 | 0,80 | 0,62 |
| FUSION | 87,8% | 0,86 | 0,93 | 0,98 | 0,88 | 0,86 | 0,90 | 0,90 | 0,85 | 0,71 | 0,96 | 0,68 |
| Oracle | 92,1% | 0,91 | 0,94 | 0,99 | 0,94 | 0,92 | 0,93 | 0,93 | 0,89 | 0,81 | 0,97 | 0,79 |

Paper tidak melaporkan angka untuk FUSION penjumlahan sama sekali.

### Selisih kita − paper

| Model | OA | Sel terburuk |
|---|---|---|
| TEXT | **−0,011** | Advertisement −0,063 |
| IMAGE | −0,036 | Report −0,083 |
| FUSION | −0,044 | **Resume −0,157** |
| Oracle | −0,023 | Report −0,075 |

Semuanya 1-4% di bawah paper, merata di hampir semua kelas — konsisten dengan deviasi
lingkungan, bukan kegagalan pada komponen tertentu. Dua pengecualian: kita **lebih
baik** dari paper pada TEXT Report (+0,065) dan TEXT News (+0,039), dan jauh **lebih
buruk** pada FUSION Resume (−0,157), yang dibahas di bawah.

## Tiga pertanyaan yang tabel ini harus menjawab

### 1. Apakah urutannya konsisten?

Yang wajib terjadi: **TEXT < IMAGE < FUSION < Oracle**. Selisih 2-4% dari angka
paper wajar mengingat framework berbeda (PyTorch vs TensorFlow), split berbeda
(3 random split vs k-fold), dan sumber citra berbeda (JPG re-encode vs TIF asli).
Urutan yang terbalik jauh lebih serius daripada angka yang bergeser.

**Hasil: terpenuhi.** 0,7266 < 0,8086 < 0,8342 < 0,8977. Urutannya sama dengan paper,
dan seluruh selisih dari paper di rentang 1-4% yang sudah diperkirakan.

Ditambah satu konfirmasi independen dari [[01-ekstraksi-fitur-teks]]: rata-rata token
≥4 karakter keluar **135** sementara paper menyebut **136**. Korpus teksnya identik.

### 2. Berapa besar potensi fusion yang benar-benar dipanen?

Ini pertanyaan yang oracle jawab, dan yang mengubah "fusion berhasil" menjadi
pernyataan yang bisa diperdebatkan.

| Besaran | Paper | Kita |
|---|---|---|
| Baseline terbaik | 84,5% | 80,86% |
| Oracle | 92,1% | 89,77% |
| Potensi tersedia | +7,6% | **+8,91%** |
| Dipanen FUSION | +3,3% | +2,56% |
| Proporsi dipanen | 43% | **29%** |

Potensi kita **lebih besar** dari paper (+8,9% vs +7,6%) karena kedua baseline kita
lebih lemah sehingga lebih saling melengkapi. Tapi yang dipanen justru lebih sedikit:
29% versus 43%.

Pemecahan oracle per sampel — angka yang paper tidak laporkan tapi paling informatif:

| Kategori | Kita | Arti |
|---|---|---|
| Keduanya benar | 0,6374 | bagian mudah, tidak butuh fusion |
| **Hanya TEXT benar** | **0,0891** | **komplementaritas langsung**: citra gagal, teks menyelamatkan |
| Hanya IMAGE benar | 0,1711 | citra menyelamatkan |
| Keduanya salah | 0,1023 | tidak bisa diselamatkan skema pemilihan apa pun |

Rata-rata tiga seed. Per seed: hanya TEXT 0,0783 / 0,0854 / 0,1037.

"Hanya TEXT benar" = 8,9% berarti sekitar **239 dari 2682 dokumen** yang cabang citranya
gagal tapi cabang teksnya berhasil. Komplementaritasnya terukur, bukan hipotetis.

Perhatikan identitas: "potensi tersedia" (+8,91%) dan "hanya TEXT benar" (0,0891)
**sama persis**. Bukan kebetulan — oracle = (IMAGE benar ATAU TEXT benar), jadi
oracle − IMAGE = sampel yang hanya TEXT benar. Berguna sebagai pemeriksaan kewarasan.

"Hanya TEXT benar" adalah ukuran paling langsung bahwa modalitas teks membawa
sesuatu. Kalau angka itu mendekati nol, seluruh premis proyek runtuh — lihat
perdebatan di [[taksonomi-multimodal]] soal teks yang diturunkan dari citra.

### 3. Apakah klaim per kelas paper bertahan?

Paper menulis gain-nya konsisten dan fusion hampir tidak pernah di bawah salah satu
baseline pada kelas mana pun. **Tabel 3 paper sendiri membantahnya di dua kelas:**
Resume (TEXT 0,97 > FUSION 0,96) dan Advertisement (IMAGE 0,94 > FUSION 0,93).

**Jumlah pengecualian pada hasil kita: 4 dari 10.** Tapi angka mentah itu menyesatkan —
tiga di antaranya di dalam derau antar-seed (std 0,0105):

| Kelas | FUSION | Baseline terbaik | Selisih |
|---|---|---|---|
| Advertisement | 0,894 | IMAGE 0,902 | −0,008 |
| News | 0,883 | IMAGE 0,890 | −0,007 |
| Scientific | 0,583 | TEXT 0,587 | −0,004 |
| **Resume** | **0,803** | **TEXT 0,969** | **−0,166** |

Yang sungguhan hanya satu. Dan itu persis kelas yang catatan risiko di
[[05-fusion-concat]] tunjuk sebagai indikator: **paper mengangkat Resume ke 0,96
melawan TEXT 0,97; kita hanya 0,803, praktis sama dengan cabang citra (0,784).**

Pada kelas dengan keunggulan teks terbesar di seluruh dataset, fusion kita memilih
berperilaku seperti cabang citra. Itu bukan anomali kecil — itu gejala.

## Deviasi yang berlaku untuk seluruh tabel ini

1. PyTorch, bukan TensorFlow 1.12 + Keras.
2. 3 random split terstratifikasi, bukan k-fold cross-validation.
3. Early stopping dengan validation dipotong dari 800 train, bukan epoch tetap.
4. Citra dari JPG Kaggle re-encode, teks di-OCR dari TIF asli — sumber kedua
   modalitas tidak identik.
5. Mekanisme penjumlahan adaptif adalah pilihan kita; paper tidak menyebutkannya.
6. Head fusion dan inisialisasi cabang adalah asumsi kita.
7. **BatchNorm melihat micro-batch 8, bukan batch 40.** VRAM 4 GB tidak bisa menampung
   batch 40 pada 384x384, jadi batch dipecah dan gradiennya diakumulasi. Update
   optimizer tetap dari 40 sampel seperti paper, tapi statistik BatchNorm berasal dari
   8 sampel. Akumulasi gradien tidak bisa memperbaiki ini.
8. **Mixed precision (AMP) aktif**, tidak dipakai paper.
9. **Epoch efektif 26-61, bukan 200.** Tidak dipotong manual — `EPOCHS` tetap 200 di
   semua notebook. Pemotongannya terjadi sendiri lewat early stopping, dan inilah
   deviasi yang paling mungkin merugikan. Lihat diagnosis di bawah.

Daftar lengkap 14 ambiguitas paper ada di
[[1907.06370-spesifikasi-implementasi]].

## Diagnosis: early stopping mengakhiri training terlalu dini

Ini penjelasan yang menyatukan hampir semua selisih terhadap paper.

| Model | Epoch terbaik (3 seed) | Epoch dijalankan | Rencana |
|---|---|---|---|
| IMAGE | 22 / 14 / 11 | 37 / 29 / 26 | 200 |
| FUSION concat | 18 / 31 / 46 | 33 / 46 / 61 | 200 |
| FUSION sum | 12 / 20 / 19 | 27 / 35 / 34 | 200 |

Kita melatih **sekitar 15% anggaran paper**, dan itu tidak direncanakan.

Penyebab langsungnya: **validation hanya 80 sampel.** Granularitasnya 1,25% per
sampel, sehingga `val_oa` sangat berderau dan `patience=15` bisa menyala karena derau
semata. Bukti konkretnya: IMAGE seed 42 mencatat `val_oa` 0,9250 di epoch 22 — nilai
tertinggi yang pernah dicapai, hampir pasti kebetulan — dan checkpoint itulah yang
disimpan. Test-nya keluar 0,8315, sembilan poin di bawah. Seleksi checkpoint-nya
overfit ke 80 sampel itu.

Dampaknya **berbeda** untuk kedua cabang, dan ini yang penting:

**Untuk IMAGE, masalahnya seleksi, bukan lamanya training.** Train loss saat berhenti
sudah 0,025-0,05, jadi 720 sampel train sudah dihafal. Melatih lebih lama tidak otomatis
menolong; memilih checkpoint dengan validation yang layak yang menolong.

**Untuk FUSION, masalahnya lamanya training.** Cabang citra memakai bobot ImageNet
sehingga berguna sejak epoch 1. Cabang teks dilatih **dari nol** dan butuh ~28 epoch
untuk berguna ketika berdiri sendiri ([[03-baseline-cnn1d]]). Di dalam fusion, cabang
citra sudah menurunkan loss lebih dulu sehingga tidak ada tekanan bagi cabang teks
untuk matang — lalu early stopping mengakhiri run di epoch 33-61, sebelum ia sempat.

Korelasinya terlihat: seed FUSION yang dilatih paling lama (44, 61 epoch) dan yang
terbaik (43, 46 epoch) keduanya di atas seed yang berhenti paling dini (42, 33 epoch,
OA terendah). Bukan bukti, tapi konsisten.

**Paper melatih 200 epoch tetap tanpa early stopping.** Kemungkinan itu bukan detail
sepele melainkan justru alasan cabang teks mereka terpakai.

## Temuan

**#temuan/positif — replikasi berhasil pada tingkat angka.** Urutan TEXT < IMAGE <
FUSION < Oracle terpenuhi, semua OA 1-4% di bawah paper, korpus teks terkonfirmasi
identik (135 vs 136 token ≥4 karakter), dan pola komplementaritas Resume terreplikasi
pada tingkat baseline (TEXT 0,969 vs IMAGE 0,784).

**#temuan/negatif — tapi fusion tidak bekerja seperti yang diklaim paper.** Kenaikan
+2,56% di atas baseline citra **bukan** berasal dari modalitas teks. Tiga bukti
independen di [[08-ablasi-missing-modality]] dan [[09-ablasi-degradasi-teks]]:
menolkan seluruh teks menurunkan akurasi 0,0015; menghapus 0-100% kata mengubahnya
0,0015; dan Resume mendarat di sisi citra. Kenaikan itu kemungkinan dari kapasitas
tambahan atau efek regularisasi head concat.

**#temuan/negatif — fusion lebih rapuh, bukan lebih tahan.** Di bawah rotasi, blur,
dan noise, fusion kolaps lebih dalam daripada baseline citra
([[07-ablasi-degradasi-citra]]). Konsekuensi logis dari poin di atas.

**#temuan/negatif — klaim kegagalan fusion penjumlahan tidak terreplikasi.** Paper
menyebut penjumlahan jatuh signifikan di bawah baseline citra; milik kita 1,9% di
atasnya dan tidak terbedakan secara statistik dari concat (selisih 0,0067 melawan std
0,0105-0,0164). Lihat [[06-fusion-sum]] untuk caveat soal mekanisme.

**#temuan/anomali — komplementaritas ada, arsitekturnya tidak memanennya.** Oracle
0,8977, 8,9% sampel hanya benar lewat teks, tapi hanya 29% potensi yang dipanen dan
cabang teksnya tidak terpakai. Ini kondisi yang membuat ablasi 4 (fusion bergerbang
atau cross-attention) layak dikerjakan — syaratnya yang ditulis di Tahap 7 sudah
terpenuhi.

## Yang harus dikerjakan sebelum kesimpulan ini final

Urut manfaat per biaya:

1. **Latih ulang FUSION satu seed tanpa early stopping, 200 epoch penuh** (± 4,2 jam),
   lalu ulangi ablasi missing modality. Ini menguji langsung apakah cabang teks yang
   tidak terpakai adalah artefak early stopping atau sifat arsitektur concat. **Hasilnya
   menentukan kesimpulan laporan akhir.**
2. **Ablasi missing modality pada checkpoint FUSION-sum** (menit-an, checkpoint sudah
   ada). Bobot cabang teks 0,31 menyiratkan hasil berbeda dari concat.
3. **Perbesar validation ke 15-20% dari train** atau pilih checkpoint pada `val_loss`
   yang dihaluskan. Perkiraan perolehan 1-2% untuk semua model.
4. **Uji empat pasangan salah-eja milik paper** dengan model FastText kita, untuk
   memisahkan artefak heuristik dari sifat model — lihat
   [[01-ekstraksi-fitur-teks]].

## Status keempat langkah itu, per 2026-09-30

| # | Langkah | Status |
|---|---|---|
| 1 | FUSION tanpa early stopping | **selesai** — [[10-uji-lanjutan-early-stopping]] |
| 2 | Missing modality pada FUSION-sum | **selesai** — [[10-uji-lanjutan-early-stopping]] |
| 3 | Perbesar validation | **belum** — lihat [[limitasi-dan-lanjutan]] |
| 4 | Uji empat pasangan paper | **belum** — lihat [[limitasi-dan-lanjutan]] |

Dua langkah pertama memicu dua eksperimen lanjutan yang tidak direncanakan
([[11-diagnostik-cabang-teks]] dan [[12-perbaikan-norm-dan-init]]), dan keduanya
mengubah kesimpulan proyek. Ringkasannya di [[sintesis-akhir]].

### Yang berubah setelah keempat uji itu

Diagnosis "early stopping" di atas **benar sebagian, bukan penyebab utamanya**.
Penyebab sebenarnya: cabang teks **kolaps** (varians 38× lebih kecil dari CNN1D
standalone, 67 dari 128 unit mati) dan skalanya **timpang 3,7×** terhadap cabang citra.
Keduanya bisa diperbaiki, dan setelah diperbaiki cabang teks jadi sama informatifnya
dengan CNN1D standalone (probe 0,7468 melawan 0,7479).

Ongkosnya OA turun 0,0194 sementara macro F1 identik. Yang berubah bukan besarnya
akurasi, melainkan **dari mana akurasinya berasal**.
