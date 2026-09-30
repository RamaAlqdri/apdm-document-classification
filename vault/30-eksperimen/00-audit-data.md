---
judul: Audit Data — Tobacco3482 JPG + QS-OCR-Small
tipe: eksperimen
tahap: "03"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/eksperimen
  - tahap/03
  - modal/citra
  - modal/teks
  - komponen/ocr
terkait:
  - "[[kumar-2014-tobacco3482]]"
  - "[[ocr-tesseract]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[peta-eksperimen]]"
---

# Audit Data — Tobacco3482 JPG + QS-OCR-Small

model:: audit
dataset:: Tobacco3482-jpg (Kaggle) + QS-OCR-small (GitHub v1.0)
split:: -
seed:: 42
oa:: 
macro_f1:: 
durasi:: -

> **DIJALANKAN 2026-09-30** di laptop Windows (i7 gen 11, RTX 3050 4 GB, CUDA).
> Semua angka di bawah berasal dari eksekusi nyata `notebooks/01_data_audit.ipynb`, bukan dugaan.

## Tujuan

Memastikan kedua modalitas benar-benar bisa dipasangkan sebelum satu baris model
ditulis. Tiga hal yang diperiksa: apakah jumlah filenya sesuai (3482 per modalitas,
10 kelas), apakah nama kelasnya konsisten antar arsip, dan berapa pasangan yang
benar-benar terbentuk.

## Sumber

| Modalitas | Sumber | Ukuran |
|---|---|---|
| Citra | Kaggle `patrickaudriaz/tobacco3482jpg` | 3,29 GB |
| Teks | GitHub `QuickSign/ocrized-text-dataset`, Releases tag `v1.0`, aset `QS-OCR-small.tar.gz` | 2,5 MB |

Citra tidak dikopi ke `data/raw/` — memakai cache kagglehub, dengan symlink di
`data/raw/tobacco3482-jpg` agar tata letaknya sesuai dokumentasi.

## Hasil verifikasi

| Pemeriksaan | Harapan | Hasil |
|---|---|---|
| Total file citra | 3482 | **3482** ✓ |
| Total file teks | 3482 | **3482** ✓ |
| Jumlah folder kelas (citra) | 10 | **10** ✓ |
| Jumlah folder kelas (teks) | 10 | **10** ✓ |
| Folder bersarang duplikat | tidak ada | **ADA, terdeteksi** — lihat di bawah |
| Stem muncul di >1 kelas | tidak ada | **0** ✓ |

### Arsip citra memang bersarang dua kali

Audit menemukan **2 folder** yang masing-masing punya 10 kelas:

```
.../versions/1/Tobacco3482-jpg/                    <- dipilih
.../versions/1/Tobacco3482-jpg/Tobacco3482-jpg/
```

`pick_dataset_root` mencetak peringatan dan memilih path terpendek. Inilah kondisi
yang membuat `find_class_dirs` sengaja mengembalikan **semua** kandidat alih-alih
menebak satu: kalau ia langsung menebak, duplikasi ini tidak akan pernah terlihat.

Efek sampingnya, folder bersarang itu ikut terbaca sebagai kandidat nama kelas dan
dilaporkan `tidak_dikenali_citra: ['Tobacco3482-jpg']`. Kosmetik — folder itu tidak
berisi file `.jpg` di levelnya sendiri (0 citra, 0 teks di tabel hitungan), dan
pairing hanya memakai 10 kelas yang benar-benar terpetakan.

## Rekonsiliasi nama kelas

Arsip citra memakai kode pendek, arsip teks memakai nama panjang menurut README.
Pemetaan dilakukan lewat `CLASS_ALIASES` di [src/data.py](../../src/data.py) secara
eksplisit — apa pun yang tidak tercakup dilaporkan sebagai tidak dikenali dan
diputuskan manual, bukan dinormalisasi diam-diam.

Hasil: **10 dari 10 kelas cocok.** Ternyata kedua arsip memakai kode pendek yang
sama, jadi tidak ada perbedaan ejaan yang perlu dijembatani:

| Kanonik | Folder citra | Folder teks |
|---|---|---|
| Advertisement | ADVE | ADVE |
| Email … Scientific | sama | sama |

Catatan: arsip **teks** juga memakai `ADVE`, bukan `Advertisement` seperti yang
disiratkan README repo. Alias `ADVE → Advertisement` di `CLASS_ALIASES` tetap
diperlukan, hanya saja dipakai untuk kedua sisi, bukan satu sisi.

## Pairing

| Kelas | Pasangan | Citra yatim | Teks yatim |
|---|---|---|---|
| Advertisement | 230 | 0 | 0 |
| Email | 599 | 0 | 0 |
| Form | 431 | 0 | 0 |
| Letter | 567 | 0 | 0 |
| Memo | 620 | 0 | 0 |
| News | 188 | 0 | 0 |
| Note | 201 | 0 | 0 |
| Report | 265 | 0 | 0 |
| Resume | 120 | 0 | 0 |
| Scientific | 261 | 0 | 0 |
| **TOTAL** | **3482** | **0** | **0** |

**Pairing sempurna: tidak ada satu pun file yatim.** Tidak ada contoh nama yang
gagal cocok karena tidak ada yang gagal. Stem nama file citra dan teks identik
untuk seluruh 3482 dokumen.

## EDA

Figur tersimpan di `reports/figures/`:

- `03_distribusi_kelas.png` — ketimpangan sampel per kelas
- `03_panjang_teks.png` — histogram `n_kata` dengan garis batas padding 500, plus
  boxplot per kelas
- `03_contoh_pasangan.png` — 4 pasangan citra dan teks OCR-nya berdampingan

| Metrik | Nilai |
|---|---|
| Rasio kelas terbesar : terkecil | **5,17** (Memo 620 : Resume 120) |
| Median `n_kata` | **174** |
| Rata-rata `n_kata` | 223,9 (std 200,9; maks 1579) |
| Persen teks kosong | **0,7%** (26 dokumen) |
| Persen teks < 20 kata | **4,3%** (150 dokumen) |
| Persen dokumen > 500 kata | **7,8%** |

Distribusi kelas lengkap: Memo 620, Email 599, Letter 567, Form 431, Report 265,
Scientific 261, Advertisement 230, Note 201, News 188, Resume 120.

Teks pendek sangat tidak merata antar kelas:

| Kelas | Dokumen < 20 kata | Persen |
|---|---|---|
| Note | 97 | **48,3%** |
| Advertisement | 42 | **18,3%** |
| News | 2 | 1,1% |
| Form | 3 | 0,7% |
| Email | 4 | 0,7% |
| Letter | 1 | 0,2% |
| Memo | 1 | 0,2% |
| Report, Resume, Scientific | 0 | 0% |

Note dan Advertisement adalah kelas yang cabang teksnya nyaris tidak punya bahan.
Itu langsung terlihat di hasil Tahap 5: F1 TEXT untuk Advertisement hanya 0,537 dan
Note 0,621 — dua yang terendah setelah Report. Lihat [[03-baseline-cnn1d]].

Tiga angka yang paling menentukan keputusan berikutnya:

- **Persen teks kosong/pendek** adalah batas atas kualitas cabang teks. Dokumen
  tanpa teks hanya bisa diklasifikasi dari citranya, dan di model fusion ia menjadi
  kasus missing modality alami — relevan untuk ablasi 2 di Tahap 7.
- **Persen dokumen > 500 kata** menentukan seberapa besar kerugian truncation.
  Paper menyebut rata-rata 136 kata (panjang minimal 4 karakter), jadi harapannya
  kecil. Lihat [[cnn-1d-teks]].
- **Rasio ketimpangan kelas** menjelaskan mengapa F1 class-balanced dilaporkan
  berdampingan dengan OA, bukan sebagai pelengkap.

## Konteks yang wajib dibaca bersama hasil

1. **Cara teks dihasilkan.** Tesseract 4.0.0-beta.1, `--oem 1 --psm 3 -l eng`,
   tanpa praproses citra dan tanpa pascaproses teks. Salah eja dibiarkan apa adanya
   secara sengaja — itu justru objek yang diuji paper. Lihat [[ocr-tesseract]].
2. **`--psm 3` berarti tanpa deteksi orientasi.** Paper §3.2 menyatakan Tesseract
   mencoba mendeteksi orientasi teks; itu tidak benar untuk konfigurasi ini.
3. **Tidak ada split bawaan.** README QS-OCR-Small menyarankan k-fold. Kita memakai
   3 random split terstratifikasi di Tahap 4 dan mencatatnya sebagai deviasi.
4. **Sumber kedua modalitas tidak identik.** Teks di-OCR dari TIF asli, citra kita
   JPG hasil re-encode. Ini limitasi yang ikut ke laporan akhir, bukan bug.
5. **Typo di README repo.** README menyebut `./tobacco3842.sh`, file sebenarnya
   `tobacco3482.sh`. Relevan hanya bila meregenerasi OCR sendiri.

## Koreksi terhadap rencana proyek

Rencana di `vault/00-index/rencana-prompt.md` menyebut release `v1.0` punya **4
aset arsip**. Diverifikasi lewat GitHub API: hanya ada **2** —
`QS-OCR-small.tar.gz` (2,5 MB) dan `QS-OCR-Large.tar.gz` (251,9 MB). Nama filenya
juga `QS-OCR-small` dengan huruf kecil. Rencana sudah diperbaiki.

## Temuan

**#temuan/positif — integritas data sempurna.** 3482/3482 pada kedua modalitas,
0 yatim, 0 stem ganda, 10/10 kelas terpetakan. Tidak ada satu pun kompromi data
yang perlu dicatat sebagai limitasi tambahan.

**#temuan/positif — korpus teks kita identik dengan korpus paper.** Konfirmasi
independennya ada di [[01-ekstraksi-fitur-teks]]: rata-rata token dengan panjang
minimal 4 karakter keluar **135**, sementara paper menyebut **136**. Dua pengukuran
yang dihitung dengan cara sama pada korpus yang diklaim sama, dan hasilnya cocok
dalam 1 token.

**#temuan/anomali — arsip Kaggle bersarang duplikat.** Bukan masalah, tapi kalau
audit ini tidak dirancang melaporkan semua kandidat, ada kemungkinan nyata memakai
folder yang salah tanpa pernah tahu.

**Yang perlu diingat untuk Tahap 7:** 26 dokumen (0,7%) teksnya kosong sama sekali.
Itu kasus missing modality yang terjadi secara alami di data, bukan konstruksi
ablasi — lihat [[08-ablasi-missing-modality]].
