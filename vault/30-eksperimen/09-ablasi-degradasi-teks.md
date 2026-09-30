---
judul: Ablasi 3 — degradasi teks
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-28
status: selesai
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
oa:: 0.8218 (0% hapus) → 0.8203 (100% hapus)
macro_f1:: 0.8047 → 0.8004
durasi:: menit-an, tanpa training

> **DIJALANKAN 2026-09-30**, memakai `FUSION-concat_seed42.pt` apa adanya. Injeksi
> typo **ikut dikerjakan** — `cc.en.300.bin` masih tersedia.

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

**Hasil: selisihnya 0,0000.** Menghapus 100% kata memberi OA 0,8203, identik dengan
menolkan teks di [[08-ablasi-missing-modality]]. Kedua jalur kode sepakat sampai empat
angka desimal, jadi tidak ada bug di salah satunya.

## Hasil — penghapusan kata

Diisi setelah eksekusi. Figur: `reports/figures/07_ablasi_degradasi_teks.png`.

| Fraksi dihapus | OA | Macro F1 | Turun dari utuh |
|---|---|---|---|
| 0% | 0,8218 | 0,8047 | — |
| 10% | 0,8225 | 0,8050 | **+0,0007** |
| 25% | 0,8225 | 0,8047 | **+0,0007** |
| 50% | 0,8229 | 0,8057 | **+0,0011** |
| 75% | 0,8233 | 0,8065 | **+0,0015** |
| 100% | 0,8203 | 0,8004 | −0,0015 |

Baca kolom terakhir sekali lagi: **menghapus 10% sampai 75% kata membuat akurasi
sedikit NAIK.** Bukan turun. Puncaknya justru di 75% penghapusan.

Rentang seluruh kurva dari 0% sampai 100% hanya **0,0030** — tiga persepuluh persen.
Dengan 2682 sampel test, itu selisih 8 dokumen. Seluruh kurva rata dalam derau.

Garis acuan pada grafik adalah **IMAGE standalone**. Kalau kurva berhenti di atas
garis itu bahkan saat teks habis, fusion masih mendapat sesuatu dari arsitekturnya
sendiri. Kalau jatuh di bawahnya, fusion lebih buruk daripada memakai citra saja
ketika teks hilang — kerapuhan yang harus dilaporkan, bukan disembunyikan.

## Hasil — injeksi typo

Diisi setelah eksekusi.

| Fraksi typo | OA | Macro F1 |
|---|---|---|
| 0% | 0,8218 | 0,8047 |
| 10% | 0,8218 | 0,8049 |
| 25% | 0,8218 | 0,8049 |
| 50% | 0,8210 | 0,8038 |

Merusak setengah token korpus mengubah OA sebesar **0,0008**. Sama ratanya dengan
kurva penghapusan.

**Status injeksi typo: DIKERJAKAN.** `cc.en.300.bin` masih tersedia di mesin compute,
sehingga token yang dirusak bisa di-embed ulang.

Catatan teknis: loop inilah yang memicu `PermissionError: [WinError 32]` di Windows —
memory-map `.npy` sementara menahan file sehingga `unlink()` gagal. Diperbaiki di
`src/` lewat `close_datasets()` dan context manager `scratch_npy()`, bukan ditambal di
notebook. Lihat log 2026-09-30.

## Hipotesis sebelum menjalankan — dan hasilnya

**1. "Typo jauh lebih ringan daripada penghapusan." → TIDAK BISA DIJAWAB.** Keduanya
tidak berefek sama sekali (0,0008 vs 0,0030), jadi tidak ada selisih untuk
dibandingkan. Pertanyaannya jadi tidak bisa dijawab pada model ini — bukan karena
subword FastText bekerja atau gagal, melainkan karena **cabang teksnya tidak dipakai
sama sekali**. Perbandingan ini harus diulang pada model yang benar-benar memakai teks.

**2. "Kurva penghapusan mendatar lebih awal dari dugaan." → BENAR, dan jauh lebih
ekstrem dari dugaan.** Saya menduga 25% penghapusan "mungkin nyaris tidak berefek".
Kenyataannya **100% penghapusan** nyaris tidak berefek. Dugaannya benar arahnya tapi
meleset besar soal derajatnya — dan alasan sebenarnya bukan max-pool-through-time
seperti yang saya kira, melainkan sesuatu yang jauh lebih mendasar.

**3. "100% penghapusan mendarat di sekitar IMAGE standalone." → BENAR.** 0,8203 versus
IMAGE standalone 0,8315. Tapi ini benar untuk alasan yang salah: saya mengira fusion
akan turun *ke* level citra setelah kehilangan teks; kenyataannya ia **sudah** berada di
situ sejak awal, karena teksnya tidak pernah berkontribusi.

## Keterbatasan

1. Satu seed (42).
2. **Hipotesis 1 tidak terjawab** karena cabang teks tidak dipakai, bukan karena
   pengukurannya gagal. Ulangi pada model yang memakai teks.
3. Typo buatan tidak sama dengan noise OCR asli. Tesseract juga berhalusinasi string
   yang bukan kata sama sekali dan mengacaukan urutan baris — lihat
   [[noise-ocr-dan-oov]]. Kita hanya meniru kerusakan ejaan, bagian yang paling
   mudah disimulasikan dan justru bagian yang paling bisa ditangani FastText. Jadi
   ablasi ini kemungkinan **melebih-lebihkan** ketahanan.
4. Degradasi seragam ke seluruh test set, bukan dicampur.

## Temuan

**#temuan/negatif — kurva degradasi teks benar-benar rata, dan itu bukan bukti
ketahanan.** Menghapus 100% kata mengubah akurasi 0,0015; menghapus 75% justru
menaikkannya 0,0015. Merusak setengah token dengan typo mengubah 0,0008.

Godaannya besar untuk membaca ini sebagai "fusion sangat tahan terhadap noise teks".
Itu salah, dan [[08-ablasi-missing-modality]] membuktikannya: model tahan karena tidak
memakai teks sama sekali. **Ketahanan dan ketidakpedulian memberi kurva yang sama
bentuknya; yang membedakan hanyalah uji missing modality.** Tanpa ablasi 2, ablasi 3
akan disalahtafsirkan sebagai hasil positif.

**#temuan/positif — pemeriksaan silang lolos sempurna.** Menghapus 100% kata dan
menolkan teks memberi angka identik (selisih 0,0000). Dua jalur kode yang berbeda,
satu hasil. Assert ini ditanam sebelum eksekusi justru untuk menangkap bug, dan
hasilnya menaikkan kepercayaan pada kedua ablasi.

**#temuan/anomali — akurasi naik sedikit saat kata dihapus.** Puncak di 75%
penghapusan (+0,0015). Besarnya di dalam derau (8 dokumen dari 2682), jadi jangan
ditafsirkan berlebihan. Tapi arahnya konsisten: kalau cabang teks menyumbang derau
alih-alih sinyal, menguranginya memang sedikit membantu.

**Apa yang harus dikerjakan sebelum hipotesis 1 bisa dijawab:** latih ulang fusion
tanpa early stopping sehingga cabang teks benar-benar terpakai, lalu ulangi ablasi ini.
Perbandingan typo-versus-penghapusan baru punya arti pada model seperti itu. Sampai
saat itu, ablasi 3 tidak mengatakan apa pun tentang subword FastText — hanya tentang
model fusion kita.
