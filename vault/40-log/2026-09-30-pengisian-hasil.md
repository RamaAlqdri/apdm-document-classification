---
judul: Log 2026-09-30 — Pengisian Hasil Eksekusi ke Catatan
tipe: log
tahap: "07"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/log
  - tahap/07
  - status/selesai
  - komponen/evaluasi
terkait:
  - "[[tabel-utama]]"
  - "[[08-ablasi-missing-modality]]"
  - "[[05-fusion-concat]]"
  - "[[peta-proyek]]"
---

# Log 2026-09-30 — Pengisian Hasil Eksekusi ke Catatan

## Yang dikerjakan

Keenam notebook sudah dijalankan penuh di laptop Windows (i7 gen 11, 24 GB RAM,
RTX 3050 4 GB, CUDA). Sebelas catatan diisi dari keluaran nyata dan statusnya diubah
dari `belum-dijalankan` menjadi `selesai`.

Sumber angka: sel ringkasan tiap notebook, `reports/tabel_utama.csv`,
`reports/ablasi1_degradasi_citra.csv`, dan 12 file `reports/log_*.csv`.

Tidak ada satu angka pun yang tidak berasal dari eksekusi. `data/processed/` dan
`models/` kosong di mesin penulisan karena gitignored — checkpoint dan fitur ada di
mesin compute.

## Catatan yang diisi

`vault/30-eksperimen/`: [[00-audit-data]], [[01-ekstraksi-fitur-teks]],
[[02-baseline-mlp-sif]], [[03-baseline-cnn1d]], [[04-baseline-mobilenetv2]],
[[05-fusion-concat]], [[06-fusion-sum]], [[07-ablasi-degradasi-citra]],
[[08-ablasi-missing-modality]], [[09-ablasi-degradasi-teks]].

`vault/50-hasil/`: [[tabel-utama]].

## Hasil utama

| Model | Kita | Paper |
|---|---|---|
| TEXT (CNN1D) | 0,7266 ± 0,0059 | 0,738 |
| IMAGE | 0,8086 ± 0,0210 | 0,845 |
| FUSION concat | 0,8342 ± 0,0105 | 0,878 |
| FUSION sum | 0,8275 ± 0,0164 | tidak dilaporkan |
| Oracle | 0,8977 | 0,921 |

Urutan TEXT < IMAGE < FUSION < Oracle terpenuhi, semuanya 1-4% di bawah paper.

## Temuan yang mengubah kesimpulan proyek

**Cabang teks pada FUSION-concat praktis tidak terpakai.** Tiga pengukuran independen:

1. Menolkan seluruh masukan teks menurunkan OA **0,0015**.
2. Menghapus 0% sampai 100% kata mengubah OA **0,0015**; menghapus 75% justru sedikit
   menaikkannya.
3. Kelas Resume — keunggulan teks terbesar di dataset — mendarat di 0,803, dekat ke
   IMAGE (0,784) dan jauh dari TEXT (0,969). Paper menjaganya di 0,96.

Jadi kenaikan +2,56% FUSION di atas baseline citra bukan berasal dari modalitas kedua.
Komplementaritasnya sendiri terbukti ada — oracle 0,8977, dan 239 dari 2682 dokumen
hanya benar lewat teks — tapi hanya 29% potensinya dipanen (paper 43%).

**Konsekuensinya: fusion lebih rapuh, bukan lebih tahan.** Di bawah rotasi, blur, dan
noise, jarak FUSION − IMAGE menyempit tajam; pada rotasi 45° fusion kehilangan 18 poin
lebih banyak. Ini membalik hipotesis yang saya tulis sebelum eksekusi.

**Klaim kegagalan fusion penjumlahan tidak terreplikasi.** Paper menyebut penjumlahan
jatuh signifikan di bawah baseline citra. Milik kita 1,9% **di atasnya**, dan selisih
concat vs sum (0,0067) jauh di dalam satu standar deviasi — keduanya tidak terbedakan
secara statistik. Bobot cabang terlatih konsisten di tiga seed: citra 0,69 / teks 0,31.

## Diagnosis: early stopping, bukan arsitektur

Epoch terbaik semuanya sangat dini: IMAGE 22/14/11 dan FUSION 18/31/46, dari 200 yang
direncanakan. Kita melatih sekitar 15% anggaran paper, dan itu **tidak direncanakan** —
`EPOCHS` tetap 200 di semua notebook, pemotongannya terjadi sendiri lewat early
stopping.

Penyebab langsungnya validation hanya **80 sampel**, granularitas 1,25% per sampel.
Buktinya konkret: IMAGE seed 42 mencatat `val_oa` 0,9250 di epoch 22 lalu checkpoint
itu yang disimpan; test-nya 0,8315, sembilan poin di bawah.

Dampaknya berbeda untuk kedua cabang, dan pemisahan ini yang penting:

- **IMAGE:** masalahnya **seleksi checkpoint**, bukan lamanya training. Train loss saat
  berhenti sudah 0,025-0,05, jadi 720 sampel train sudah dihafal.
- **FUSION:** masalahnya **lamanya training**. Cabang citra berguna sejak epoch 1 karena
  bobot ImageNet; cabang teks dari nol butuh ~28 epoch untuk berguna sendiri. Di dalam
  fusion, cabang citra sudah menurunkan loss lebih dulu sehingga tidak ada tekanan bagi
  cabang teks untuk matang, lalu early stopping mengakhiri run sebelum ia sempat.

200 epoch tetap tanpa early stopping di paper kemungkinan bukan detail sepele melainkan
justru alasan cabang teks mereka terpakai.

## Keputusan editorial saat mengisi

1. **Hipotesis pra-eksekusi disimpan apa adanya, termasuk yang keliru.** Hipotesis 3 di
   [[07-ablasi-degradasi-citra]] memprediksi jarak FUSION − IMAGE akan melebar; yang
   terjadi sebaliknya. Ditulis sebagai kekeliruan prediksi, bukan dihapus. Kalau
   hipotesis ditulis setelah melihat angka, sangat mudah merasionalisasi hasil apa pun.
2. **"4 dari 10 kelas FUSION kalah" tidak ditulis sebagai angka mentah.** Tiga di
   antaranya −0,004 sampai −0,008, di dalam derau antar-seed. Yang sungguhan hanya
   Resume (−0,166). Melaporkan "4 vs 2 milik paper" akan melebih-lebihkan.
3. **Kurva degradasi teks yang rata TIDAK ditulis sebagai ketahanan.** Ketahanan dan
   ketidakpedulian memberi kurva yang bentuknya sama; yang membedakan hanya uji missing
   modality. Tanpa ablasi 2, ablasi 3 akan disalahtafsirkan sebagai hasil positif.
4. **Cosine salah eja 0,190 vs ~0,96 paper tidak diklaim sebagai bukti melawan
   FastText.** Dua perancu belum dikendalikan: heuristik kita ikut menjaring yang bukan
   salah eja, dan kata dalam kosakata memakai vektor terlatih sementara kata OOV memakai
   jumlah n-gram. Ditandai `#temuan/anomali` dan pertanyaan terbuka, bukan temuan.
5. **Keterbatasan ablasi 2 dinyatakan arahnya.** Efek out-of-distribution hanya bisa
   membuat penurunan tampak lebih besar, tidak lebih kecil. Jadi 0,15% adalah batas
   **atas** kontribusi cabang teks, bukan batas bawah — dan keterbatasan itu tidak
   melemahkan temuan utamanya.

## Yang belum dikerjakan

Empat langkah, urut manfaat per biaya, tercatat lengkap di [[tabel-utama]]:

1. Latih ulang FUSION satu seed tanpa early stopping, 200 epoch (± 4,2 jam), lalu ulangi
   ablasi missing modality. **Ini yang menentukan kesimpulan laporan akhir.**
2. Ablasi missing modality pada checkpoint FUSION-sum (menit-an, checkpoint sudah ada).
3. Perbesar validation ke 15-20% dari train.
4. Uji empat pasangan salah-eja milik paper dengan model FastText kita.

Tahap 8 (sintesis akhir dan limitasi) sebaiknya menunggu langkah 1 dan 2, karena
keduanya bisa mengubah jawaban atas pertanyaan "apakah fusion memberi nilai tambah dan
dari mana asalnya".
