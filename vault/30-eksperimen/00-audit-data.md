---
judul: Audit Data — Tobacco3482 JPG + QS-OCR-Small
tipe: eksperimen
tahap: "03"
tanggal: 2026-09-28
status: belum-dijalankan
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
durasi:: 

> **STATUS: BELUM DIJALANKAN.** Notebook `01_data_audit.ipynb` sudah ditulis dan
> sel kodenya lolos pemeriksaan sintaks, tapi **belum pernah dieksekusi**: mesin
> tempat kode ini ditulis tidak memegang dataset. Download dan eksekusi dilakukan
> di mesin compute.
>
> Setiap field angka di bawah sengaja dibiarkan kosong. Isi hanya dari keluaran sel
> terakhir notebook, lalu ubah `status:` menjadi `selesai`. Jangan mengisi dari
> dugaan — lihat aturan 3 di `CLAUDE.md`.

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

Diisi setelah eksekusi.

| Pemeriksaan | Harapan | Hasil |
|---|---|---|
| Total file citra | 3482 | |
| Total file teks | 3482 | |
| Jumlah folder kelas (citra) | 10 | |
| Jumlah folder kelas (teks) | 10 | |
| Folder bersarang duplikat | tidak ada | |
| Stem muncul di >1 kelas | tidak ada | |

## Rekonsiliasi nama kelas

Arsip citra memakai kode pendek, arsip teks memakai nama panjang menurut README.
Pemetaan dilakukan lewat `CLASS_ALIASES` di [src/data.py](../../src/data.py) secara
eksplisit — apa pun yang tidak tercakup dilaporkan sebagai tidak dikenali dan
diputuskan manual, bukan dinormalisasi diam-diam.

Tabel pemetaan yang benar-benar terbentuk: diisi setelah eksekusi.

## Pairing

Diisi setelah eksekusi.

| Kelas | Pasangan | Citra yatim | Teks yatim |
|---|---|---|---|
| Advertisement | | | |
| Email | | | |
| Form | | | |
| Letter | | | |
| Memo | | | |
| News | | | |
| Note | | | |
| Report | | | |
| Resume | | | |
| Scientific | | | |
| **TOTAL** | | | |

Contoh nama yang gagal cocok: diisi setelah eksekusi.

## EDA

Diisi setelah eksekusi. Figur tersimpan di `reports/figures/`:

- `03_distribusi_kelas.png` — ketimpangan sampel per kelas
- `03_panjang_teks.png` — histogram `n_kata` dengan garis batas padding 500, plus
  boxplot per kelas
- `03_contoh_pasangan.png` — 4 pasangan citra dan teks OCR-nya berdampingan

| Metrik | Nilai |
|---|---|
| Rasio kelas terbesar : terkecil | |
| Median `n_kata` | |
| Persen teks kosong | |
| Persen teks < 20 kata | |
| Persen dokumen > 500 kata | |

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

Diisi setelah eksekusi, dengan tag `#temuan/positif`, `#temuan/negatif`, atau
`#temuan/anomali` sesuai hasilnya.
