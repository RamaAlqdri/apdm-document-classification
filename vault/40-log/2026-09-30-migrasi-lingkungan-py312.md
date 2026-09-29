---
judul: Log 2026-09-30 — Migrasi ke Python 3.12 dan Perbaikan Memory-Map Windows
tipe: log
tahap: "07"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/log
  - tahap/07
  - komponen/evaluasi
  - status/selesai
terkait:
  - "[[09-ablasi-degradasi-teks]]"
  - "[[peta-proyek]]"
  - "[[papan-progress]]"
  - "[[noise-ocr-dan-oov]]"
---

# Log 2026-09-30 — Migrasi ke Python 3.12 dan Perbaikan Memory-Map Windows

## Pemicu

Loop injeksi typo di [[09-ablasi-degradasi-teks]] (`notebooks/06_ablasi.ipynb`)
berhenti dengan `PermissionError: [WinError 32]` saat `tmp.unlink()`. Sebelumnya
`download_text()` juga gagal karena `tarfile.extractall(filter="data")`, argumen
yang belum ada di Python 3.10.

## Bug 1 — memory-map menahan file di Windows

`SequenceDataset` dan `VectorDataset` membuka `.npy` dengan `mmap_mode="r"`. Selama
dataset dan `DataLoader`-nya masih hidup, map itu terbuka, dan Windows menolak
menghapus file yang sedang dipetakan. Di Linux kode yang sama lolos, karena POSIX
mengizinkan unlink atas file yang masih ter-map. **Bug ini khusus Windows dan tidak
ada hubungannya dengan versi Python.**

Perbaikan di `src/`, bukan tambalan di notebook:

- `_MemmapBacked.close()` — melepas map secara eksplisit
- `close_datasets(*ds)` — menelusuri `DataLoader.dataset` dan `PairDataset`
- `scratch_npy(path)` — context manager: hapus sisa lama di awal, hapus lagi di
  akhir walau body gagal, dan kalau masih ter-map, error-nya menyebut sebabnya
  alih-alih `WinError 32` mentah

Efek samping yang berguna: level ablasi yang gagal di tengah tidak lagi
meninggalkan file ~800 MB di `data/processed/`. Loop yang diperbaiki inilah yang menguji klaim subword FastText atas noise OCR di [[noise-ocr-dan-oov]].

## Bug 2 — dua assertion yang ternyata belum pernah dijalankan

`_self_check()` di `src/text_features.py` juga tidak melepas map-nya, sehingga
`TemporaryDirectory` gagal dibersihkan. Karena crash itu terjadi di tengah, semua
assertion **setelahnya** tidak pernah dieksekusi. Setelah map-nya dilepas, muncul:

`assert cosine(v, v * 3) == 1.0` gagal. Nilai aktualnya `0.9999999999996666`,
selisih 3,3e-13. Penyebabnya `cosine()` sengaja menambah `+1e-12` di penyebut
supaya vektor nol tidak memicu pembagian nol — dan korpus ini memang punya dokumen
kosong. Epsilon itu membuat hasil tepat `1.0` mustahil. Jadi yang salah adalah
perbandingan float eksaknya, bukan epsilon-nya; `== 1.0` dan `== -1.0` diganti
pembandingan bertoleransi. Fungsinya tidak diubah.

## Bug 3 — assertion `drop_last` yang rusak tergantung mesin

`assert sum(...) == 10` di `src/datasets.py` mengasumsikan batch 5, padahal
`make_loaders` memakai `CONFIG["micro_batch_size"]` yang di-derive dari VRAM GPU
(di mesin ini 8). Rusak sejak commit akumulasi gradien. Angkanya sekarang dihitung
dari `loader.batch_size`, jadi bebas mesin.

## Migrasi lingkungan

Python 3.12.10 dipasang per-user berdampingan; `python` default tetap 3.10.9
sebagai jaring pengaman. Proyek memakai `.venv` sendiri, kernel Jupyter
`apdam-py312`.

Plafon versi proyek ini adalah **Python 3.12**, bukan yang terbaru:
`fasttext-wheel` 0.9.2 hanya punya wheel Windows sampai cp312, tidak ada cp313.
Di 3.13 `fasttext` harus dibangun dari source dan butuh MSVC.

`requirements.txt` diperbarui: `--extra-index-url` ke index cu121 plus pin
`torch==2.5.1+cu121`. Tanpa itu, `pip install -r requirements.txt` memasang wheel
PyPI yang untuk Windows adalah build **CPU-only** — gagal senyap, bukan error.

### Versi: yang berhasil dipin dan yang tidak

| Paket | Lama (3.10) | Baru (3.12) | |
|---|---|---|---|
| torch | 2.5.1+cu121 | 2.5.1+cu121 | sama |
| torchvision | 0.20.1+cu121 | 0.20.1+cu121 | sama |
| numpy | 1.26.4 | 1.26.4 | sama |
| pandas | 2.3.2 | 2.3.2 | sama |
| spacy | 3.8.16 | 3.8.16 | sama |
| fasttext-wheel | 0.9.2 | 0.9.2 | sama |
| scikit-learn | 1.2.2 | 1.9.1 | **naik paksa** |
| matplotlib | 3.7.1 | 3.11.2 | **naik paksa** |

Dua yang terakhir tidak punya wheel cp312, jadi tidak bisa dipin. `matplotlib`
hanya memengaruhi gambar. `scikit-learn` memengaruhi split dan metrik, jadi diuji.

## Verifikasi parity — dua uji nyata

**Split.** Fingerprint SHA dari `make_split` untuk seed 42/43/44 × train/val/test,
sembilan-sembilannya **identik** antara sklearn 1.2.2 dan 1.9.1 (n=3482,
train 720 / val 80 / test 2682). Jadi kenaikan paksa sklearn tidak menggeser split.

**Metrik.** Checkpoint `MLP_seed42.pt` (epoch 32, val_oa 0,6875) dievaluasi di CPU
di kedua lingkungan — CPU sengaja, supaya perbedaan tidak tertutup nondeterminisme
cuDNN:

| | py3.10 / sklearn 1.2.2 | py3.12 / sklearn 1.9.1 |
|---|---|---|
| oa | 0,6778523490 | 0,6778523490 |
| macro_f1 | 0,6627705606 | 0,6627705606 |
| sha prediksi | `a82b2b2374f87e38` | `a82b2b2374f87e38` |

Identik sampai 10 desimal, dan hash prediksinya sama. **Hasil Tahap 1–6 tetap
sebanding dengan angka yang akan dihasilkan di lingkungan baru.**

Keenam self-check `src/` lolos di kedua interpreter: `config`, `data`, `datasets`,
`splits`, `text_features`, `ablation`.

## Yang belum dikerjakan

- Tahap 7 (injeksi typo) **belum dijalankan ulang**. Angkanya masih kosong; lihat
  [[09-ablasi-degradasi-teks]].
- Sisa `data/processed/tmp_typo_0.0.npy` (804 MB) belum bisa dihapus karena kernel
  Jupyter lama masih memegang map-nya. Setelah restart kernel, `scratch_npy`
  menghapusnya sendiri di awal run.
- `src/utils.py` belum punya self-check. Bukan kegagalan, cuma belum ada.
