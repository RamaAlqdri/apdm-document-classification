---
judul: FUSION — konkatenasi
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
  - "[[06-fusion-sum]]"
  - "[[03-baseline-cnn1d]]"
  - "[[04-baseline-mobilenetv2]]"
  - "[[tabel-utama]]"
---

# FUSION — konkatenasi

model:: FUSION-concat
dataset:: Tobacco3482 (citra JPG + QS-OCR-small)
split:: 800 train (termasuk 10% val) / 2682 test, stratified
seed:: 42, 43, 44
oa:: 0.8342
macro_f1:: 0.8105
durasi:: 2.92 jam

> **DIJALANKAN 2026-09-30** di laptop Windows (RTX 3050 4 GB, CUDA). Angka di bawah
> dari eksekusi nyata `notebooks/05_fusion.ipynb`.

## Arsitektur

| Bagian | Keluaran | Sumber |
|---|---|---|
| Cabang citra | MobileNetV2 → GAP → 1280 → FC → **128** | paper |
| Cabang teks | CNN1D → **128** | paper |
| Penggabungan | konkatenasi → 256 | paper |
| Head | Linear 256→256, BN, ReLU, Dropout, Linear 256→10 | **asumsi kita** |
| Parameter | 13.804.810 | terhitung dari self-check |

Classifier kedua cabang diganti `nn.Identity` — paper memotong layer akhir, dan
membiarkan Linear mati di dalamnya berarti setiap checkpoint membawa bobot tak
terpakai. Self-check memverifikasi `state_dict` benar-benar bersih dari keduanya.

## Tiga ambiguitas yang diputuskan di sini

Ketiganya sebelumnya tercatat terbuka di
[[1907.06370-spesifikasi-implementasi]]:

1. **Arsitektur head fusion** (ambiguitas #5). Paper hanya menyebut "multi-layer
   perceptron following [Eitel]". Kita pakai satu layer tersembunyi 256 dengan
   BN/ReLU/Dropout.
2. **Inisialisasi cabang** (ambiguitas #9). "Dilatih end-to-end" kita baca sebagai:
   cabang citra mulai dari bobot ImageNet persis seperti baseline IMAGE, cabang teks
   mulai dari nol. Bukan dari checkpoint baseline yang sudah kita latih sendiri.
3. **Mekanisme penjumlahan adaptif** (ambiguitas #3) — lihat [[06-fusion-sum]].

## Risiko yang harus diperiksa di log

Satu cabang pretrained dilatih bersama satu cabang dari nol pada learning rate yang
sama. Kalau cabang citra konvergen jauh lebih cepat, fusion bisa runtuh menjadi
model citra dengan parameter tambahan — hasilnya akan terlihat "baik" (setara
baseline IMAGE) sambil sebenarnya gagal memakai teks.

Indikator: kalau OA fusion praktis sama dengan baseline IMAGE dan F1 kelas **Resume**
tidak naik mendekati nilai TEXT, curigai ini. [[06-fusion-sum]] memberi indikator
yang lebih langsung lewat bobot cabang terlatihnya.

> **RISIKO INI TERBUKTI TERJADI.** Resume hanya 0,803 melawan TEXT 0,969, dan
> [[08-ablasi-missing-modality]] menunjukkan menolkan seluruh masukan teks hanya
> menurunkan akurasi **0,0015**. Cabang teks praktis tidak terpakai. Pembahasannya di
> bawah dan di [[tabel-utama]].

## Hyperparameter

SGD momentum 0,9, lr 0,01, batch 40, 200 epoch (paper §4.2). Early stopping
`patience=15` adalah deviasi kita.

## Hasil

| Seed | OA | Macro F1 | Epoch terbaik | Epoch dijalankan | Durasi |
|---|---|---|---|---|---|
| 42 | 0,8218 | 0,8047 | 18 | 33 | 42,4 menit |
| 43 | 0,8475 | 0,8183 | 31 | 46 | 57,6 menit |
| 44 | 0,8333 | 0,8086 | 46 | 61 | 74,9 menit |
| **rata-rata ± std** | **0,8342 ± 0,0105** | **0,8105 ± 0,0057** | 32 | 47 | 2,92 jam |

Target paper: **OA 87,8% / F1 0,86**. Selisih kita **−0,044 OA**.

Kenaikan di atas baseline IMAGE (0,8086): **+0,0256** absolut. Paper melaporkan
+3,3%; kita +2,6%. Dengan std 0,0105, kenaikan itu sekitar 2,4σ — nyata, tapi tidak
besar.

**Yang penting: kenaikan itu bukan berasal dari modalitas teks.** Lihat bagian
temuan.

### Deviasi runtime yang benar-benar terjadi

Estimasi notebook: 75,7 s/epoch → 4,2 jam per run, 25,2 jam untuk 2 strategi x 3 seed.
Aktual concat: 2,92 jam total, karena early stopping.

| Deviasi | Keterangan |
|---|---|
| Micro-batch 8 + akumulasi gradien | update dari 40 sampel, tapi BatchNorm per 8 |
| Mixed precision (AMP) aktif | tidak dipakai paper |
| Epoch efektif 33-61, bukan 200 | early stopping, tidak direncanakan |

Perhatikan korelasinya: seed yang dilatih **paling lama** (44, 61 epoch) dan yang
**terbaik** (43, 46 epoch) keduanya jauh di atas seed 42 yang berhenti paling dini
(33 epoch, OA terendah 0,8218). Konsisten dengan hipotesis bahwa cabang teks butuh
waktu untuk matang dan early stopping mengakhirinya terlalu cepat.

## Pemeriksaan klaim per kelas

Paper menulis gain-nya konsisten dan fusion "hampir tidak pernah" di bawah salah
satu baseline pada kelas mana pun. Tabel 3 paper sendiri punya **dua** pengecualian:
Resume (TEXT 0,97 > FUSION 0,96) dan Advertisement (IMAGE 0,94 > FUSION 0,93).

Notebook menghitung jumlah pengecualian pada hasil kita. Kalau jauh lebih dari dua,
itu `#temuan/negatif` dan dilaporkan apa adanya.

**Jumlah kelas di mana FUSION kalah dari baseline terbaiknya: 4 dari 10.** Paper
punya 2. Tapi angka mentah itu menyesatkan — tiga di antaranya sepele:

| Kelas | FUSION | TEXT | IMAGE | Selisih |
|---|---|---|---|---|
| Advertisement | 0,894 | 0,537 | 0,902 | −0,008 |
| News | 0,883 | 0,709 | 0,890 | −0,007 |
| Scientific | 0,583 | 0,587 | 0,554 | −0,004 |
| **Resume** | **0,803** | **0,969** | 0,784 | **−0,166** |

Tiga yang pertama di dalam derau antar-seed (std concat 0,0105). Yang sungguhan hanya
satu, dan besarnya luar biasa: **Resume −0,166**.

Bandingkan dengan paper, yang menjaga Resume di 0,96 melawan TEXT 0,97 — nyaris tidak
kehilangan apa pun. Selisih kita terhadap paper di sel itu **−0,157**, penyimpangan
terbesar di seluruh tabel.

Yang membuatnya diagnostik: FUSION Resume 0,803 sangat dekat dengan IMAGE 0,784, dan
sangat jauh dari TEXT 0,969. Pada kelas di mana teks paling unggul, fusion memilih
berperilaku seperti cabang citra.

Confusion matrix: `reports/figures/06_confusion_fusion_concat_seed42.png`.

## Temuan

**#temuan/positif — fusion mengalahkan kedua baseline.** 0,8342 vs IMAGE 0,8086 dan
TEXT 0,7266. Urutan TEXT < IMAGE < FUSION < Oracle terpenuhi seperti paper.

**#temuan/negatif — kenaikan itu bukan dari modalitas teks.** Tiga bukti independen:

1. Menolkan seluruh masukan teks menurunkan OA hanya **0,0015**
   ([[08-ablasi-missing-modality]]).
2. Menghapus 0% sampai 100% kata mengubah OA **0,0015**; menghapus 75% kata justru
   sedikit menaikkannya ([[09-ablasi-degradasi-teks]]).
3. Resume — kelas dengan keunggulan teks terbesar — mendarat di sisi citra.

Jadi +2,56% itu kemungkinan berasal dari kapasitas tambahan atau efek regularisasi
head concat, bukan dari informasi teks. **Klaim "fusion berhasil karena
komplementaritas" tidak didukung data kita**, meski komplementaritasnya sendiri
terbukti ada (oracle 0,8977, "hanya TEXT benar" 8,9%).

**#temuan/negatif — fusion lebih rapuh, bukan lebih tahan.** Di bawah degradasi citra,
fusion kolaps lebih dalam daripada baseline citra pada 3 dari 4 jenis degradasi. Itu
konsekuensi logis dari poin di atas — tidak ada cabang teks yang bekerja untuk
mengompensasi. Lihat [[07-ablasi-degradasi-citra]].

**Diagnosis yang paling mungkin, dan cara mengujinya.** Cabang citra memakai bobot
ImageNet sehingga berguna sejak epoch 1; cabang teks dilatih dari nol dan butuh ~28
epoch untuk berguna ketika berdiri sendiri ([[03-baseline-cnn1d]]). Di dalam fusion,
cabang citra sudah menurunkan loss lebih dulu sehingga tidak ada tekanan bagi cabang
teks untuk matang — dan early stopping mengakhiri run di epoch 33-61, sebelum ia
sempat. Paper melatih 200 epoch tetap tanpa early stopping, dan itu mungkin bukan
detail sepele melainkan justru yang membuat cabang teksnya terpakai.

**Uji yang menentukan:** latih ulang satu seed tanpa early stopping, 200 epoch penuh
(± 4,2 jam), lalu ulangi ablasi missing modality. Kalau menolkan teks kemudian
benar-benar menurunkan akurasi, penyebabnya early stopping. Kalau tetap tidak,
penyebabnya arsitektur concat itu sendiri dan gerbang eksplisit jadi layak dicoba.

## Terselesaikan — apa yang akhirnya ditemukan

Ketiga uji lanjutannya sudah dijalankan, dan diagnosis di atas **benar sebagian**.

**Early stopping memang merusak, tapi bukan penyebab utamanya.** Tanpa early stopping,
epoch terbaik melompat 18 → 102 dan kontribusi teks naik 4,7× — tapi tetap hanya 9% dari
potensi. Lihat [[10-uji-lanjutan-early-stopping]].

**Penyebab sebenarnya terukur di [[11-diagnostik-cabang-teks]]:** cabang teks **kolaps**
(varians 38× lebih kecil dari CNN1D standalone, 67 dari 128 unit mati) **dan** skalanya
timpang 3,7× terhadap cabang citra. Fitur teks terlalu lirih untuk didengar head concat —
meski probe linear membuktikan fiturnya masih membawa informasi.

**Keduanya bisa diperbaiki.** [[12-perbaikan-norm-dan-init]]: LayerNorm per cabang plus
inisialisasi cabang teks dari `CNN1D_seed42.pt` membuat probe naik 0,4236 → **0,7468**
(acuan 0,7479), unit mati 67 → **4**, dan efek menolkan teks 0,0071 → **0,6887**. Resume
0,803 → **0,968**, menutup penyimpangan per-kelas terbesar kita terhadap paper.

Ongkosnya: OA turun 0,0194 sementara **macro F1 identik** (0,8108). Perbaikannya
memindahkan akurasi dari kelas besar yang dikuasai citra ke kelas kecil yang butuh teks.

Kesimpulan yang dibawa ke [[sintesis-akhir]]: **OA fusion yang tinggi bukan bukti bahwa
fusion memakai kedua modalitas.** Model yang secara diam-diam unimodal bisa mencetak OA
lebih tinggi daripada model yang benar-benar multimodal pada dataset ini.
