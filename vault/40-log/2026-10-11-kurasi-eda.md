---
judul: Log 2026-10-11 — Notebook kurasi dan EDA
tipe: log
tahap: "10"
tanggal: 2026-10-11
status: selesai
tags:
  - tipe/log
  - tahap/10
  - status/selesai
terkait:
  - "[[14-kurasi-dan-eda]]"
  - "[[peta-proyek]]"
---

# Log 2026-10-11 — Notebook kurasi dan EDA

Dosen meminta tiga luaran tambahan: dataset terkurasi dengan split, notebook EDA, dan
laporan singkat temuan EDA. Bagian F (antar-mesin OCR) ditunda atas permintaan pengguna.

Yang dibuat:

- `src/eda.py`: statistik citra, thumbnail dan deteksi duplikat, kebocoran split, kode ID,
  penanda ekor distribusi, vektor dokumen dari sekuens, dan kemiripan pusat kelas. Self-check
  lolos; total self-check kini 12 modul.
- `notebooks/11_eda.ipynb`: bagian A–E, G, H, I.
- Notebook 11 didaftarkan di `run_all.py` dan RUNBOOK.

Kekeliruan yang tertangkap sebelum diserahkan: deteksi duplikat dengan dHash 64-bit
menandai hampir semua halaman fixture sebagai duplikat. Rinciannya dan perbaikannya ada
di [[14-kurasi-dan-eda]].

Laporan singkat EDA baru ditulis setelah notebook dijalankan di laptop TUF.
