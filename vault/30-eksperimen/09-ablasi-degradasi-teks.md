---
judul: Ablasi 3 — degradasi teks
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-28
status: belum-dijalankan
tags:
  - tipe/eksperimen
  - tahap/07
  - modal/teks
  - modal/fusion
  - komponen/embedding
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[taksonomi-multimodal]]"
  - "[[fasttext-subword]]"
  - "[[noise-ocr-dan-oov]]"
---

# Ablasi 3 — degradasi teks

model:: FUSION-concat (checkpoint Tahap 6)
dataset:: Tobacco3482, test split seed 42 (2682 dokumen)
split:: seed 42 saja
seed:: 42
oa:: 
macro_f1:: 
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Field angka kosong.

## Pertanyaan

Dua bentuk kerusakan teks, dan keduanya menanyakan hal berbeda:

1. **Hapus x% kata** — informasi berkurang. Berapa banyak teks yang sebenarnya
   dibutuhkan fusion?
2. **Injeksi typo** — informasi tetap ada tapi ejaannya rusak. Inilah yang
   sesungguhnya dilakukan OCR.

Perbandingan keduanya pada fraksi yang sama adalah **uji langsung atas argumen inti
paper**: kalau typo jauh lebih ringan dampaknya daripada penghapusan, itu bukti
subword FastText memang bekerja — diuji pada model terlatih, bukan pada cosine
similarity pasangan kata seperti di [[fasttext-subword]].

## Desain

| Perturbasi | Tingkat |
|---|---|
| Hapus kata | 0%, 10%, 25%, 50%, 75%, 100% |
| Injeksi typo | 0%, 10%, 25%, 50% |

**Penghapusan kata merapatkan sekuens, bukan menolkan di tempat.** Saat training, nol
hanya pernah muncul sebagai padding di ekor sekuens. Melubangi bagian tengah akan
out-of-distribution dengan cara yang mencampur "teks lebih sedikit" dengan "masukan
yang belum pernah dilihat model" — dua variabel sekaligus, dan hasilnya tidak bisa
ditafsirkan. Kata yang lolos dirapatkan ke depan dengan urutan dipertahankan, lalu
di-pad di ujung. Self-check memverifikasi keduanya.

**Injeksi typo meniru apa yang benar-benar dilakukan Tesseract**: menggandakan huruf,
menghilangkan huruf, atau menukar tetangga — keluarga kerusakan yang sama dengan
contoh paper sendiri (`specifically`/`Specificalily`, `filter`/`fiilter`,
`Largely`/`Largly`). Token di bawah 4 karakter dilewati.

## Pemeriksaan silang bawaan

Menghapus **100%** kata harus setara dengan menolkan teks di
[[08-ablasi-missing-modality]]. Notebook meng-assert selisihnya di bawah 2% — kalau
gagal, ada bug di salah satu dari keduanya, bukan temuan ilmiah.

Hasil pemeriksaan silang: diisi setelah eksekusi.

## Hasil — penghapusan kata

Diisi setelah eksekusi. Figur: `reports/figures/07_ablasi_degradasi_teks.png`.

| Fraksi dihapus | OA | Macro F1 | Turun dari utuh |
|---|---|---|---|
| 0% | | | — |
| 10% | | | |
| 25% | | | |
| 50% | | | |
| 75% | | | |
| 100% | | | |

Garis acuan pada grafik adalah **IMAGE standalone**. Kalau kurva berhenti di atas
garis itu bahkan saat teks habis, fusion masih mendapat sesuatu dari arsitekturnya
sendiri. Kalau jatuh di bawahnya, fusion lebih buruk daripada memakai citra saja
ketika teks hilang — kerapuhan yang harus dilaporkan, bukan disembunyikan.

## Hasil — injeksi typo

Diisi setelah eksekusi.

| Fraksi typo | OA | Macro F1 |
|---|---|---|
| 0% | | |
| 10% | | |
| 25% | | |
| 50% | | |

**Bagian ini mungkin tidak dikerjakan.** Merusak karakter di dalam token berarti
token itu harus di-embed ulang, jadi `cc.en.300.bin` (~7 GB di disk, ~15 GB saat
dimuat) harus tersedia lagi. Fitur yang sudah dihitung tidak cukup.

Notebook melewati bagian ini kalau modelnya tidak ada dan mencetak peringatan. Kalau
dilewati, **tulis di sini bahwa ia tidak dikerjakan** — jangan hilangkan diam-diam
dari laporan.

Status injeksi typo: diisi setelah eksekusi.

## Hipotesis sebelum menjalankan

1. **Typo pada fraksi x akan jauh lebih ringan daripada penghapusan pada fraksi x.**
   Kalau tidak, argumen subword paper tidak bertahan pada model terlatih, hanya pada
   cosine similarity — dan itu `#temuan/negatif` yang penting.
2. **Kurva penghapusan akan mendatar lebih awal dari dugaan.** Dokumen rata-rata cuma
   ~136 kata berguna, dan CNN1D memakai max-pool-through-time yang hanya butuh pola
   terkuat muncul di suatu tempat. Menghapus 25% kata mungkin nyaris tidak berefek.
3. **100% penghapusan akan mendarat di sekitar IMAGE standalone**, bukan di bawahnya.

Hipotesis 2 adalah yang paling spesifik dan karena itu paling berguna kalau salah.

## Keterbatasan

1. Satu seed (42).
2. Typo buatan tidak sama dengan noise OCR asli. Tesseract juga berhalusinasi string
   yang bukan kata sama sekali dan mengacaukan urutan baris — lihat
   [[noise-ocr-dan-oov]]. Kita hanya meniru kerusakan ejaan, bagian yang paling
   mudah disimulasikan dan justru bagian yang paling bisa ditangani FastText. Jadi
   ablasi ini kemungkinan **melebih-lebihkan** ketahanan.
3. Degradasi seragam ke seluruh test set, bukan dicampur.

## Temuan

Diisi setelah eksekusi.
