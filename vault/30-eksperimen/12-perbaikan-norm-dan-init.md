---
judul: Perbaikan — LayerNorm per cabang + inisialisasi cabang teks
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
  - "[[11-diagnostik-cabang-teks]]"
  - "[[10-uji-lanjutan-early-stopping]]"
  - "[[05-fusion-concat]]"
  - "[[03-baseline-cnn1d]]"
  - "[[sintesis-akhir]]"
---

# Perbaikan — LayerNorm per cabang + inisialisasi cabang teks

model:: FUSION-fix
dataset:: Tobacco3482 (citra JPG + QS-OCR-small)
split:: 800 train (termasuk 10% val) / 2682 test, seed 42
seed:: 42
oa:: 0.8110
macro_f1:: 0.8108
durasi:: 4.04 jam

> **DIJALANKAN 2026-09-30** lewat `notebooks/09_uji_perbaikan.ipynb`. Checkpoint
> `CNN1D_seed42.pt` hanya dibaca; artefaknya memakai nama `FUSION-fix_*` sehingga
> seluruh hasil sebelumnya utuh.

## Dua perbaikan, menyerang dua penyebab yang terukur

[[11-diagnostik-cabang-teks]] mengukur kolaps cabang teks (varians 38× lebih kecil, 67
dari 128 unit mati) **dan** ketimpangan skala 3,7× terhadap cabang citra. Masing-masing
diperbaiki:

1. **`init_text_from(CNN1D_seed42.pt)`** — cabang teks mulai dari bobot CNN1D yang sudah
   terlatih alih-alih dari nol, menghapus asimetri terhadap cabang citra yang memakai
   bobot ImageNet.
2. **`normalize_branches=True`** — LayerNorm pada masing-masing vektor 128 sebelum
   digabung. LayerNorm, bukan BatchNorm, karena ia menormalkan **per sampel** sehingga
   tidak terpengaruh micro-batch 8 yang dipaksakan GPU 4 GB.

Keduanya aditif di `src/models/fusion.py`; default `normalize_branches=False` sehingga
checkpoint lama tetap bisa dimuat. Tambahan parameternya hanya 512 (dua LayerNorm 128).

## Pemeriksaan pra-training

Notebook berhenti sebelum training kalau inisialisasinya gagal diam-diam. Hasilnya:

- selisih maksimum fitur cabang teks versus CNN1D standalone: **0,00e+00** — identik
- norma setelah LayerNorm: teks **11,31**, citra **11,31**, rasio **1,00×** (sebelumnya
  3,72×)

Tanpa pemeriksaan ini, inisialisasi yang gagal akan menghabiskan 4 jam tanpa ada yang tahu.

## Hasil — mekanismenya benar-benar diperbaiki

| Pengukuran | Sebelum (noES) | **Diperbaiki** | Acuan CNN1D |
|---|---|---|---|
| **Probe linear fitur teks** | 0,4236 | **0,7468** | **0,7479** |
| std fitur teks | 0,0279 | 0,3642 | 1,0690 |
| Unit mati | 67/128 | **4/128** | 18/128 |
| cos antar dokumen | 0,978 | **0,4465** | 0,650 |
| Efek menolkan teks | −0,0071 | **−0,6887** | — |
| OA tanpa citra | 0,1946 | **0,6469** | — |

**Probe 0,7468 melawan acuan 0,7479** — selisih 0,0011. Cabang teks di dalam fusion
sekarang **sama informatifnya dengan CNN1D standalone**. Unit matinya bahkan lebih
sedikit daripada acuan (4 vs 18), dan fiturnya lebih beragam (cos 0,447 vs 0,650).

Modelnya kini benar-benar multimodal. Menolkan teks menjatuhkannya ke 0,1223; menolkan
citra ke 0,6469, dekat TEXT standalone 0,7345. Degradasinya anggun di kedua arah — tidak
lagi runtuh seperti sebelumnya (0,1946).

Epoch terbaik 109 dari 200, durasi 4,04 jam (estimasi 4,59 jam).

## Tapi OA turun, sementara macro F1 tidak bergerak

| | noES | Diperbaiki |
|---|---|---|
| OA | 0,8304 | **0,8110** (−0,0194) |
| **Macro F1** | **0,8108** | **0,8108** |

Sama persis sampai empat desimal. Penjelasannya ada di per kelas:

| Kelas | noES | Diperbaiki | Delta | n |
|---|---|---|---|---|
| **Resume** | 0,814 | **0,968** | **+0,154** | 92 |
| Scientific | 0,566 | 0,625 | +0,059 | 201 |
| Email | 0,956 | 0,967 | +0,011 | 461 |
| Advertisement | 0,898 | 0,899 | +0,001 | 177 |
| Note | 0,791 | 0,786 | −0,005 | 155 |
| Report | 0,634 | 0,628 | −0,006 | 204 |
| Form | 0,844 | 0,812 | −0,032 | 332 |
| Letter | 0,800 | 0,758 | −0,042 | 437 |
| News | 0,903 | 0,858 | −0,045 | 145 |
| **Memo** | 0,902 | **0,807** | **−0,095** | 478 |

**Jumlah seluruh delta = 0,000.** Kelas yang memburuk berukuran besar (Memo 478,
Letter 437, Form 332 — total 1392 sampel), yang membaik lebih kecil (total 931). Jadi OA
turun sementara macro F1 — yang menimbang semua kelas sama — tidak bergerak sama sekali.

**Resume 0,814 → 0,968.** Paper 0,96, TEXT standalone 0,969. Itu penyimpangan per-kelas
terbesar kita terhadap paper (dulu −0,157), dan perbaikan ini menutupnya sepenuhnya.

## Apa yang sebenarnya terjadi

Perbaikannya berhasil pada yang ditargetkan, dan menyingkap pertukaran yang sebelumnya
tersembunyi: **fusion asli mendapat OA tinggi dengan menjadi model citra berparameter
lebih.** Ketika dipaksa memakai kedua modalitas, ia jadi lebih baik di kelas yang butuh
teks, lebih buruk di kelas besar yang dikuasai citra, dan setara pada metrik seimbang.

Cabang citranya ikut berubah: cos antar dokumennya naik 0,411 → **0,833**, artinya fitur
citra jadi lebih seragam, dan std-nya turun 0,630 → 0,404. Kapasitas bersamanya bergeser
ke teks. Ini bukan sekadar "menyeimbangkan bobot" — kedua cabang berubah.

## Dua kekeliruan pada sel kesimpulan otomatis saya

**1. `std_rasio > 0,5` sebagai kriteria "tidak kolaps" itu keliru.** std 0,341 memicu
vonis "fiturnya masih terkompresi", padahal probe membuktikan fiturnya sama informatifnya
dengan acuan (0,7468 vs 0,7479). Skala absolut tidak relevan — ada LayerNorm tepat di
belakangnya. [[11-diagnostik-cabang-teks]] sendiri menetapkan probe sebagai pengukuran
yang menentukan, lalu sel saya memakai std untuk memvonis.

**2. "Normalisasi yang bekerja, bukan inisialisasi" tidak punya dasar.** Kedua perbaikan
dijalankan bersamaan. Sel itu mengklaim atribusi yang notebooknya sendiri nyatakan tidak
bisa dilakukan.

## Satu metrik saya jebol di sini

`persen_potensi_dipanen: 879,6%` bukan temuan, itu formula yang rusak di luar rentangnya.
Ia didefinisikan sebagai |efek teks| dibagi potensi oracle (0,0783), yang hanya berarti
saat model memanen **sebagian** dari sampel text-only-correct. Sekarang model
**bergantung** pada teks sehingga menolkannya menghancurkan segalanya — fenomena berbeda.
Angka itu diabaikan.

## Temuan

**#temuan/positif — kolaps cabang teks teratasi sepenuhnya.** Probe 0,7468 melawan acuan
0,7479, unit mati 4 melawan 18. Diagnosis [[11-diagnostik-cabang-teks]] terbukti secara
intervensional, bukan hanya korelasional.

**#temuan/positif — model kini benar-benar multimodal.** Menolkan modalitas mana pun
berdampak besar, dan degradasinya anggun di kedua arah.

**#temuan/positif — penyimpangan per-kelas terbesar terhadap paper tertutup.** Resume
0,814 → 0,968 melawan paper 0,96.

**#temuan/anomali — OA turun 0,0194 sementara macro F1 identik.** Jumlah delta per kelas
tepat nol. Perbaikannya memindahkan akurasi dari kelas besar yang dikuasai citra ke kelas
kecil yang butuh teks, bukan menambah atau mengurangi akurasi secara keseluruhan.

**Implikasi yang paling penting untuk laporan:** OA fusion yang tinggi **bukan bukti
bahwa fusion memakai kedua modalitas.** Model yang secara diam-diam unimodal bisa
mencetak OA lebih tinggi daripada model yang benar-benar multimodal pada dataset ini.
Kalau paper hanya melaporkan OA dan F1 tanpa uji missing modality, tidak ada cara
membedakan keduanya.

## Keterbatasan

1. **Satu seed (42), satu run.** Selisih OA 0,0194 berada di dekat std antar-seed concat
   (0,0105), jadi arah penurunannya tidak bisa dipastikan tanpa seed tambahan.
2. **Kedua perbaikan diterapkan bersamaan**, jadi atribusi per perbaikan tidak diketahui.
   Memisahkannya butuh dua run lagi (± 6,6 jam) dan tidak dikerjakan.
3. Tidak diuji pada strategi `sum`, dan tidak diuji dengan ablasi degradasi citra Tahap 7
   — menarik untuk tahu apakah model yang benar-benar multimodal ini akhirnya **lebih
   tahan** degradasi, yang gagal ditunjukkan model sebelumnya di
   [[07-ablasi-degradasi-citra]]. Itu arah lanjutan F1 di [[limitasi-dan-lanjutan]], dan
   yang termurah dari semuanya.
