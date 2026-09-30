---
judul: Peta Tag
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
  - "[[papan-progress]]"
---

# Peta Tag

Daftar **resmi** tag. Jangan bikin varian baru tanpa mendaftarkannya di sini.
Semua nested, huruf kecil, Bahasa Indonesia.

## `#tipe/*` — jenis catatan (wajib, satu per catatan)

| Tag | Dipakai untuk |
|---|---|
| `#tipe/jurnal` | Ringkasan dan spesifikasi paper di `10-jurnal/` |
| `#tipe/konsep` | Catatan atomik satu konsep di `20-metode/` |
| `#tipe/eksperimen` | Satu run nyata di `30-eksperimen/`, wajib ada field Dataview |
| `#tipe/log` | Log harian di `40-log/` |
| `#tipe/hasil` | Agregasi dan sintesis di `50-hasil/` |
| `#tipe/referensi` | MOC, catatan pustaka di `90-referensi/`, dokumen rujukan |

## `#tahap/*` — tahap kerja (wajib)

Didaftar satu per satu, bukan sebagai rentang, supaya bisa diperiksa otomatis:

`#tahap/00` `#tahap/01` `#tahap/02` `#tahap/03` `#tahap/04` `#tahap/05`
`#tahap/06` `#tahap/07` `#tahap/08`

Sesuai daftar tahap di [[peta-proyek]]. Catatan uji lanjutan (notebook 07-09) memakai
`#tahap/07` karena merupakan kelanjutan ablasi, bukan tahap baru.

## `#modal/*` — modalitas yang dibahas

| Tag | Dipakai untuk |
|---|---|
| `#modal/teks` | Hanya menyangkut cabang teks (OCR, embedding, CNN1D) |
| `#modal/citra` | Hanya menyangkut cabang citra (MobileNetV2, praproses citra) |
| `#modal/fusion` | Menyangkut penggabungan keduanya, termasuk oracle |

## `#komponen/*` — komponen pipeline

| Tag | Dipakai untuk |
|---|---|
| `#komponen/ocr` | Tesseract, kualitas OCR, noise, OOV |
| `#komponen/embedding` | FastText, SIF, representasi kata/dokumen |
| `#komponen/cnn` | Arsitektur konvolusi (1D untuk teks, 2D untuk citra) |
| `#komponen/fusion` | Strategi penggabungan, concat vs penjumlahan |
| `#komponen/evaluasi` | Metrik, split, protokol, confusion matrix, oracle |

## `#status/*` — status kerja

| Tag | Dipakai untuk |
|---|---|
| `#status/todo` | Belum dikerjakan |
| `#status/wip` | Sedang dikerjakan, isinya belum lengkap |
| `#status/selesai` | Tuntas dan sudah diverifikasi |

Nilai frontmatter `status:` memakai kosakata yang sama, plus
`ditunda` dan `belum-dijalankan` (khusus catatan eksperimen yang belum
benar-benar dieksekusi — lihat aturan 3 di `CLAUDE.md`).

## `#temuan/*` — penanda temuan (khusus catatan eksperimen dan hasil)

| Tag | Dipakai untuk |
|---|---|
| `#temuan/positif` | Hasil sesuai hipotesis atau mereplikasi paper |
| `#temuan/negatif` | Hasil bertentangan dengan hipotesis atau paper |
| `#temuan/anomali` | Hasil yang belum bisa dijelaskan, perlu ditelusuri |

`#temuan/negatif` **bukan** alasan menyembunyikan hasil. Paper melaporkan fusion
penjumlahan gagal; kalau kita mendapat hasil sebaliknya, itu temuan, bukan bug.

Dan itulah yang terjadi — lihat [[06-fusion-sum]]. Agregasi seluruh temuan bertanda ada
di [[peta-eksperimen]], statusnya di [[papan-progress]].

Dan itulah yang terjadi — lihat [[06-fusion-sum]]. Agregasi seluruh temuan bertanda ada
di [[peta-eksperimen]], statusnya di [[papan-progress]].
