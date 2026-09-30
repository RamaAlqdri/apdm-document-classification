---
judul: Baseline IMAGE — MobileNetV2 384x384
tipe: eksperimen
tahap: "05"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/eksperimen
  - tahap/05
  - modal/citra
  - komponen/cnn
terkait:
  - "[[mobilenetv2]]"
  - "[[inverted-residual-block]]"
  - "[[03-baseline-cnn1d]]"
  - "[[oracle-sebagai-batas-atas]]"
---

# Baseline IMAGE — MobileNetV2 384x384

model:: MobileNetV2
dataset:: Tobacco3482 (Tobacco3482-jpg dari Kaggle)
split:: 800 train (termasuk 10% val) / 2682 test, stratified
seed:: 42, 43, 44
oa:: 0.8086
macro_f1:: 0.7837
durasi:: 120.1 menit

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050 4 GB, CUDA). Angka di bawah
> dari eksekusi nyata `notebooks/04_baseline_citra.ipynb`.

## Arsitektur dan praproses

| Item | Nilai | Sumber |
|---|---|---|
| Backbone | MobileNetV2, pretrained ImageNet | paper |
| Fine-tuning | **seluruh jaringan**, backbone tidak dibekukan | paper |
| Input | 384x384 | paper |
| Resize | tanpa padding, aspect ratio di-warp | paper |
| Kanal | grayscale diduplikasi 3x | paper |
| Normalisasi | mean/std ImageNet | **asumsi kita** — paper tidak menyebutnya |
| Augmentasi | tidak ada | paper: augmentasi justru menurunkan 84,5% ke 83,9% |
| Dimensi fitur | 1280 (GAP per channel) | paper |
| Parameter | 2.236.682 | terhitung dari self-check |

Self-check memverifikasi tiga hal yang mudah salah tanpa terlihat: citra
non-persegi keluar persegi (warp, bukan padding), kanal grayscale benar-benar
diduplikasi identik sebelum normalisasi, dan `requires_grad` seluruh backbone
bernilai True.

## Hyperparameter

SGD momentum 0,9, lr 0,01, batch 40, **200 epoch** (paper §4.2). Early stopping
`patience=15` adalah deviasi kita.

## Anggaran compute — baca sebelum menjalankan

MobileNetV2 pada 384x384 selama 200 epoch, dikali 3 seed, hampir pasti jauh di atas
15 menit per run. Notebook `04_baseline_citra.ipynb` menjalankan estimasi 2 epoch
lebih dulu dan mencetak ekstrapolasinya.

Kalau budget dipotong — epoch dikurangi atau seed dikurangi jadi satu — notebook
mengumpulkannya ke variabel `DEVIASI` dan mencetaknya. **Catat di tabel di bawah**,
jangan dipotong diam-diam.

### Yang benar-benar terjadi

Estimasi notebook: **75,5 s/epoch → 4,2 jam per seed** untuk 200 epoch, 12,6 jam
untuk 3 seed.

Yang terjadi: **early stopping memotongnya drastis.** Total hanya 120,1 menit untuk
ketiga seed — sekitar **16% dari anggaran yang direncanakan**.

| Deviasi | Keterangan |
|---|---|
| Micro-batch 8 + akumulasi gradien | update optimizer tetap dari 40 sampel, tapi **BatchNorm menormalisasi per 8 sampel** |
| Mixed precision (AMP) aktif | tidak dipakai paper |
| Epoch efektif 26-37, bukan 200 | early stopping `patience=15`, **tidak direncanakan sebagai pemotongan budget** |

Epoch tidak dipotong manual — `EPOCHS` tetap 200. Pemotongannya terjadi sendiri
lewat early stopping, dan itu justru yang mengkhawatirkan. Lihat diagnosis di
[[tabel-utama]].

## Hasil

| Seed | OA | Macro F1 | Epoch terbaik | Epoch dijalankan | Durasi |
|---|---|---|---|---|---|
| 42 | 0,8315 | 0,8119 | **22** | 37 | 46,1 menit |
| 43 | 0,8136 | 0,7995 | **14** | 29 | 38,9 menit |
| 44 | 0,7808 | 0,7397 | **11** | 26 | 35,2 menit |
| **rata-rata ± std** | **0,8086 ± 0,0210** | **0,7837 ± 0,0315** | 16 | 31 | 120,1 menit |

Target paper: **Tabel 1b 84,5% / 0,82**. Selisih kita **−0,036 OA**.

Dua hal yang mencolok dan saling terkait:

**Varians antar seed jauh lebih besar dari model teks** (± 0,021 vs ± 0,006).
Seed 44 hanya 0,7808 sementara seed 42 mencapai 0,8315 — rentang 5 poin.

**Epoch terbaik semuanya sangat dini: 22, 14, 11 dari 200.** Dan ada korelasi jelas
dengan hasilnya: seed yang berhenti paling awal (44, terbaik di epoch 11) adalah yang
terburuk. Ini bukan kebetulan — validation hanya 80 sampel, jadi `val_oa` berderau
±1,25% per sampel dan `patience=15` bisa menyala karena derau semata.

Contoh konkret: seed 42 mencatat `val_oa` 0,9250 di epoch 22 — nilai tertinggi yang
pernah dicapai, dan hampir pasti kebetulan. Checkpoint itulah yang disimpan dan
dievaluasi, hasilnya test OA 0,8315. Jarak 9 poin antara val dan test menunjukkan
seleksi checkpoint-nya memang overfit ke 80 sampel itu.

Train loss saat berhenti sudah 0,025-0,05, jadi model **sudah menghafal 720 sampel
train**. Artinya melatih lebih lama tidak otomatis menolong untuk cabang citra —
yang bermasalah adalah **pemilihan checkpoint**, bukan lamanya training. Itu berbeda
dari kasus fusion, lihat [[05-fusion-concat]].

Pembanding lain di Tabel 3 paper: ensemble CNN Harley et al. 79,9%. MobileNetV2
tunggal mengalahkannya, jadi 84,5% bukan baseline lemah.

## F1 per kelas

Kolom pembanding adalah baris IMAGE Tabel 3 paper:

| Kelas | Kita | Paper | Selisih |
|---|---|---|---|
| Advertisement | 0,902 | 0,94 | −0,038 |
| Email | 0,955 | 0,96 | −0,005 |
| Form | 0,804 | 0,85 | −0,046 |
| Letter | 0,805 | 0,83 | −0,025 |
| Memo | 0,844 | 0,90 | −0,056 |
| News | 0,890 | 0,89 | 0,000 |
| Note | 0,771 | 0,83 | −0,059 |
| Report | 0,527 | 0,61 | −0,083 |
| **Resume** | **0,784** | 0,80 | −0,016 |
| Scientific | 0,554 | 0,62 | −0,066 |

Seluruhnya di bawah atau setara paper, tidak ada yang di atas — pola yang konsisten
dengan penurunan menyeluruh 3,6%, bukan kegagalan pada kelas tertentu.

**Resume 0,784 melawan TEXT 0,969** di [[03-baseline-cnn1d]]. Selisih 18,5 poin itu
bukti terang bahwa kedua modalitas saling melengkapi, dan pola ini persis sama
dengan paper (0,80 vs 0,97). Yang kemudian tidak terjadi adalah fusion memanfaatkan
keunggulan itu: FUSION hanya mencapai 0,803 pada Resume, nyaris sama dengan cabang
citra. Lihat [[05-fusion-concat]].

Varians per kelas paling ekstrem di **Report**: 0,584 / 0,660 / **0,338** untuk seed
42/43/44. Seed 44 praktis gagal di kelas ini, dan itulah penyumbang utama OA-nya yang
rendah.

Confusion matrix: `reports/figures/05_confusion_image_seed42.png`.

## Pratinjau oracle

Notebook menghitung oracle begitu kedua baseline tersedia pada seed yang sama: satu
sampel benar bila TEXT **atau** IMAGE benar. Notebook juga mencetak proporsi
"keduanya benar" dan "keduanya salah" — yang terakhir adalah bagian yang **tidak**
bisa diselamatkan skema fusion mana pun yang bekerja dengan memilih.

Target paper: Oracle **92,1%**, yaitu 7,6% absolut di atas baseline terbaik.
FUSION memanen 3,3% dari itu, sekitar 43%.

### Hasil oracle kita (seed 42, dari notebook 04)

| Besaran | Nilai |
|---|---|
| TEXT | 0,7345 |
| IMAGE | 0,8315 |
| **Oracle** | **0,9098** (macro F1 0,8968) |
| Potensi di atas baseline terbaik | **+0,0783** |
| Keduanya benar | 0,6562 |
| **Hanya TEXT benar** | **0,0783** |
| Hanya IMAGE benar | 0,1752 |
| Keduanya salah | 0,0902 |

"Hanya TEXT benar" = 7,8% sampel: **209 dokumen dari 2682 yang citranya gagal tapi
teksnya berhasil.** Komplementaritasnya nyata dan terukur, bukan hipotetis.

Perhatikan identitas yang muncul: "potensi di atas baseline terbaik" dan "hanya TEXT
benar" nilainya **sama persis (0,0783)**. Itu bukan kebetulan — secara definisi
oracle = (IMAGE benar ATAU TEXT benar), jadi oracle − IMAGE = sampel yang hanya TEXT
benar. Berguna sebagai pemeriksaan kewarasan: kalau kedua angka itu berbeda, ada bug.

Rata-rata tiga seed dan pembacaan lengkapnya ada di [[tabel-utama]].

## Temuan

**#temuan/positif — pola komplementaritas paper terreplikasi.** Resume 0,784 (citra)
vs 0,969 (teks), dan oracle 0,9098 jauh di atas kedua baseline. 7,8% sampel hanya
bisa diselamatkan teks.

**#temuan/negatif — 3,6% di bawah paper, merata di semua kelas.** Tidak ada satu
kelas pun yang di atas paper. Tiga tersangka, semuanya deviasi yang sudah tercatat:
BatchNorm pada micro-batch 8 alih-alih 40, train 720 alih-alih 800 sampel (validation
dipotong dari dalamnya), dan citra JPG re-encode alih-alih TIF asli.

**#temuan/anomali — seleksi checkpoint tidak bisa dipercaya.** Validation 80 sampel,
epoch terbaik 11-22 dari 200, dan jarak val-test sampai 9 poin. Seed yang berhenti
paling dini adalah yang terburuk. Ini bukan soal kurang lama dilatih — train loss
sudah 0,025 — melainkan soal memilih checkpoint berdasarkan 80 sampel yang berderau.

**Rekomendasi:** perbesar validation ke 15-20% dari train, atau pilih checkpoint pada
`val_loss` yang dihaluskan alih-alih `val_oa` mentah. Perkiraan perolehannya 1-2%.
