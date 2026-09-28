---
judul: Tabel Utama — TEXT / IMAGE / FUSION / Oracle
tipe: hasil
tahap: "06"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/hasil
  - tahap/06
  - modal/fusion
  - komponen/evaluasi
terkait:
  - "[[03-baseline-cnn1d]]"
  - "[[04-baseline-mobilenetv2]]"
  - "[[05-fusion-concat]]"
  - "[[06-fusion-sum]]"
  - "[[oracle-sebagai-batas-atas]]"
  - "[[peta-proyek]]"
---

# Tabel Utama — TEXT / IMAGE / FUSION / Oracle

> **STATUS: BELUM DIJALANKAN.** Kerangka tabel ini sudah siap tapi **semua selnya
> kosong**. Diisi dari `reports/tabel_utama.csv` yang dihasilkan
> `notebooks/05_fusion.ipynb`. Jangan mengisi dari dugaan — aturan 3 `CLAUDE.md`.

Padanan Tabel 3 paper, dirata-ratakan atas tiga seed.

## Hasil kita

| Model | OA | Macro F1 | Adv. | Email | Form | Letter | Memo | News | Note | Report | Resume | Sci. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEXT (CNN1D) | | | | | | | | | | | | |
| IMAGE (MobileNetV2) | | | | | | | | | | | | |
| FUSION (concat) | | | | | | | | | | | | |
| FUSION (sum) | | | | | | | | | | | | |
| Oracle | | | | | | | | | | | | |

## Tabel 3 paper, sebagai pembanding

| Model | OA | F1 | Adv. | Email | Form | Letter | Memo | News | Note | Report | Resume | Sci. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TEXT | 73,8% | 0,71 | 0,60 | 0,96 | 0,76 | 0,71 | 0,79 | 0,67 | 0,62 | 0,43 | 0,97 | 0,57 |
| IMAGE | 84,5% | 0,82 | 0,94 | 0,96 | 0,85 | 0,83 | 0,90 | 0,89 | 0,83 | 0,61 | 0,80 | 0,62 |
| FUSION | 87,8% | 0,86 | 0,93 | 0,98 | 0,88 | 0,86 | 0,90 | 0,90 | 0,85 | 0,71 | 0,96 | 0,68 |
| Oracle | 92,1% | 0,91 | 0,94 | 0,99 | 0,94 | 0,92 | 0,93 | 0,93 | 0,89 | 0,81 | 0,97 | 0,79 |

Paper tidak melaporkan angka untuk FUSION penjumlahan sama sekali.

## Tiga pertanyaan yang tabel ini harus menjawab

### 1. Apakah urutannya konsisten?

Yang wajib terjadi: **TEXT < IMAGE < FUSION < Oracle**. Selisih 2-4% dari angka
paper wajar mengingat framework berbeda (PyTorch vs TensorFlow), split berbeda
(3 random split vs k-fold), dan sumber citra berbeda (JPG re-encode vs TIF asli).
Urutan yang terbalik jauh lebih serius daripada angka yang bergeser.

Hasil: diisi setelah eksekusi.

### 2. Berapa besar potensi fusion yang benar-benar dipanen?

Ini pertanyaan yang oracle jawab, dan yang mengubah "fusion berhasil" menjadi
pernyataan yang bisa diperdebatkan.

| Besaran | Paper | Kita |
|---|---|---|
| Baseline terbaik | 84,5% | |
| Oracle | 92,1% | |
| Potensi tersedia | +7,6% | |
| Dipanen FUSION | +3,3% | |
| Proporsi dipanen | 43% | |

Pemecahan oracle per sampel — angka yang paper tidak laporkan tapi paling informatif:

| Kategori | Kita | Arti |
|---|---|---|
| Keduanya benar | | bagian mudah, tidak butuh fusion |
| Hanya TEXT benar | | **komplementaritas langsung**: citra gagal, teks menyelamatkan |
| Hanya IMAGE benar | | citra menyelamatkan |
| Keduanya salah | | tidak bisa diselamatkan skema pemilihan apa pun |

"Hanya TEXT benar" adalah ukuran paling langsung bahwa modalitas teks membawa
sesuatu. Kalau angka itu mendekati nol, seluruh premis proyek runtuh — lihat
perdebatan di [[taksonomi-multimodal]] soal teks yang diturunkan dari citra.

### 3. Apakah klaim per kelas paper bertahan?

Paper menulis gain-nya konsisten dan fusion hampir tidak pernah di bawah salah satu
baseline pada kelas mana pun. **Tabel 3 paper sendiri membantahnya di dua kelas:**
Resume (TEXT 0,97 > FUSION 0,96) dan Advertisement (IMAGE 0,94 > FUSION 0,93).

Jumlah pengecualian pada hasil kita: diisi setelah eksekusi.

Kelas yang paling perlu diperhatikan adalah **Resume**: paper memberi TEXT 0,97
melawan IMAGE 0,80, selisih terbesar di seluruh tabel. Kalau fusion kita tidak
mengangkat Resume mendekati nilai TEXT, kemungkinan cabang teks tidak benar-benar
terpakai — lihat catatan risiko di [[05-fusion-concat]].

## Deviasi yang berlaku untuk seluruh tabel ini

1. PyTorch, bukan TensorFlow 1.12 + Keras.
2. 3 random split terstratifikasi, bukan k-fold cross-validation.
3. Early stopping dengan validation dipotong dari 800 train, bukan epoch tetap.
4. Citra dari JPG Kaggle re-encode, teks di-OCR dari TIF asli — sumber kedua
   modalitas tidak identik.
5. Mekanisme penjumlahan adaptif adalah pilihan kita; paper tidak menyebutkannya.
6. Head fusion dan inisialisasi cabang adalah asumsi kita.
7. **BatchNorm melihat micro-batch, bukan batch 40.** Di GPU dengan VRAM terbatas,
   batch 40 dipecah jadi micro-batch dan gradiennya diakumulasi. Update optimizer
   tetap dihitung dari 40 sampel seperti paper, tapi statistik BatchNorm berasal dari
   ukuran micro-batch. Akumulasi gradien tidak bisa memperbaiki ini.
   Micro-batch yang benar-benar dipakai: diisi setelah eksekusi.
8. **Mixed precision (AMP) aktif di GPU CUDA**, tidak dipakai paper. Berpengaruh pada
   presisi numerik meski secara praktis dapat diabaikan untuk klasifikasi.
9. Pemotongan budget epoch, kalau terjadi: diisi setelah eksekusi.

Daftar lengkap 14 ambiguitas paper ada di
[[1907.06370-spesifikasi-implementasi]].

## Temuan

Diisi setelah eksekusi, dengan `#temuan/positif`, `#temuan/negatif`, atau
`#temuan/anomali`.
