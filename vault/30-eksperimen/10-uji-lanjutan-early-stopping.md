---
judul: Uji Lanjutan — missing modality pada sum, dan concat tanpa early stopping
tipe: eksperimen
tahap: "07"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/eksperimen
  - tahap/07
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[05-fusion-concat]]"
  - "[[06-fusion-sum]]"
  - "[[08-ablasi-missing-modality]]"
  - "[[11-diagnostik-cabang-teks]]"
  - "[[tabel-utama]]"
---

# Uji Lanjutan — missing modality pada sum, dan concat tanpa early stopping

model:: FUSION-concat-noES
dataset:: Tobacco3482 (citra JPG + QS-OCR-small)
split:: 800 train (termasuk 10% val) / 2682 test, seed 42
seed:: 42
oa:: 0.8304
macro_f1:: 0.8108
durasi:: 3.32 jam

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050 4 GB, CUDA), lewat
> `notebooks/07_uji_lanjutan.ipynb`. Seluruh artefaknya memakai nama tersendiri
> (`FUSION-concat-noES_*`, `uji_lanjutan.csv`) sehingga hasil run pertama tidak
> tersentuh.

## Dua pertanyaan yang ditinggalkan Tahap 6-7

**Uji A.** Pada `concat`, menolkan teks hanya menurunkan OA 0,0015 sehingga cabang teks
dinilai tidak terpakai ([[08-ablasi-missing-modality]]). Tapi pada `sum`, bobot cabang
teks terlatih justru 0,31 di ketiga seed ([[06-fusion-sum]]). Itu satu-satunya
kontradiksi yang belum terselesaikan.

**Uji B.** [[tabel-utama]] menduga cabang teks tidak terpakai **karena early stopping**
mengakhiri run di epoch 33, sebelum cabang teks yang dilatih dari nol sempat matang.

## Pemeriksaan silang lolos bersih

Sebelum apa pun ditafsirkan, `A.evaluate_zeroed` yang baru harus mereproduksi angka dari
loop inline notebook 06. Hasilnya **0,8203 tepat sama**, dan notebook meng-assert itu
sebelum melanjutkan. Checkpoint `sum` juga keluar 0,8043, identik dengan notebook 05.

Perlu karena notebook 06 sengaja **tidak** diregenerasi — regenerasi akan menghapus
outputnya. Duplikasi kodenya dibiarkan, tapi tidak bisa menyimpang tanpa ketahuan.

## Uji A — missing modality pada FUSION-sum

| Kondisi | OA | Macro F1 | Turun |
|---|---|---|---|
| utuh | 0,8043 | 0,7785 | — |
| teks dinolkan | 0,7946 | 0,7679 | **−0,0097** |
| citra dinolkan | 0,0761 | 0,0141 | −0,7282 |

**Bobot 0,31 ternyata bukan bukti teks dipakai.** Efeknya 0,0097 — 6,5× lipat efek pada
concat (0,0015), tapi tetap hanya 26 dokumen dari 2682. Softmax sekadar tidak punya
tekanan kuat untuk mendorong bobotnya ke nol.

Kontradiksi di [[06-fusion-sum]] dengan itu terselesaikan: pada **kedua** strategi
penggabungan, cabang teks nyaris tidak berkontribusi. Masalahnya bukan di cara
menggabungkan.

## Uji B — concat tanpa early stopping, 200 epoch penuh

`patience=0`. Estimasi notebook 3,19 jam, aktual 3,32 jam.

| | Run pertama | **Tanpa early stopping** |
|---|---|---|
| Epoch dijalankan | 33 | **200** |
| Epoch terbaik | 18 | **102** |
| val_oa terbaik | 0,8875 | 0,9375 |
| **OA test** | 0,8218 | **0,8304** (+0,0086) |
| Macro F1 | 0,8047 | **0,8108** |
| Teks dinolkan | 0,8203 (−0,0015) | 0,8233 (**−0,0071**) |
| Citra dinolkan | 0,0731 | **0,1946** |

**Early stopping memang merusak.** Epoch terbaik melompat 18 → 102, jadi run pertama
benar-benar berhenti di sekitar 15% anggaran paper. Memperbaikinya memberi tiga
perbaikan searah: OA +0,0086, ketergantungan teks 4,7×, dan ketahanan tanpa citra 2,7×
(dari bawah tebakan acak menjadi di atas kelas mayoritas).

**Tapi tidak cukup.** Dibaca sebagai proporsi potensi — oracle seed 42 menyebut 210
dokumen hanya bisa diselamatkan teks:

| Model | Dokumen hilang saat teks dinolkan | % dari 210 |
|---|---|---|
| concat run pertama | 4 | 1,9% |
| **concat-noES** | 19 | **9,1%** |
| sum | 26 | 12,4% |

Naik lima kali lipat, tetap hampir tidak ada. Konfirmasi keduanya di Resume: 0,803 →
0,814, sementara paper 0,96 dan TEXT standalone 0,969.

## Kekeliruan pada ambang yang saya pakai

Sel kesimpulan notebook memakai ambang 0,01 untuk memutuskan apakah sebuah modalitas
"dipakai", lalu mencetak **"bukan early stopping, bukan strategi penggabungan"** karena
0,0097 dan 0,0071 keduanya jatuh di bawahnya.

Vonis itu menyesatkan. 0,0097 versus 0,01 adalah lemparan koin, dan **kedua efeknya
nyata serta searah prediksi**. Ambang biner sewenang-wenang itu keputusan desain yang
buruk; yang benar adalah membaca besaran dan rasionya. Di
[[11-diagnostik-cabang-teks]] semua ambang diganti menjadi relatif terhadap acuan yang
terukur.

## Temuan

**#temuan/positif — diagnosis early stopping terkonfirmasi sebagian.** Epoch terbaik
18 → 102 membuktikan run pertama berhenti jauh terlalu dini, dan memperbaikinya menaikkan
ketiga indikator sekaligus.

**#temuan/negatif — memperbaiki early stopping tidak menyelesaikan masalah intinya.**
Cabang teks tetap hanya memanen 9% potensi yang tersedia setelah 200 epoch penuh.

**#temuan/positif — kontradiksi bobot 0,31 terselesaikan.** Pada `sum` pun cabang teks
nyaris tidak dipakai (0,0097). Bobot softmax yang tidak mendekati nol bukan bukti
kontribusi.

Ketiganya bersama mempersempit pertanyaannya: bukan lama training, bukan strategi
penggabungan. Yang tersisa adalah **apa yang terjadi pada cabang teks itu sendiri** —
dijawab di [[11-diagnostik-cabang-teks]].

## Keterbatasan

1. Satu seed (42), tidak ada ± std.
2. Uji B hanya pada `concat`. Melatih `sum` tanpa early stopping butuh 3,3 jam lagi dan
   tidak dikerjakan.
3. Ambang 0,01 pada sel kesimpulan tidak punya dasar; vonisnya diabaikan dan besarannya
   yang dibaca.
