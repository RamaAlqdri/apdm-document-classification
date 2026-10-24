---
judul: Log 2026-10-24 — Kode uji missing modality lanjutan
tipe: log
tahap: "12"
tanggal: 2026-10-24
status: selesai
tags:
  - tipe/log
  - tahap/12
  - status/selesai
terkait:
  - "[[17-missing-modality-lanjutan]]"
  - "[[16-fusi-per-ocr]]"
  - "[[2026-10-24-hasil-fase-2]]"
  - "[[peta-eksperimen]]"
---

# Log 2026-10-24 — Kode uji missing modality lanjutan

Menindaklanjuti [[16-fusi-per-ocr]] temuan 2 (CONCAT tanpa teks jatuh jauh di bawah IMAGE,
diduga karena masukan pengganti keluar distribusi dan *head* belajar "teks kosong ⇒ Note").
Hanya kode dan rancangan; **belum dijalankan pada data asli**.

## Yang dibuat

- `src/experiment.py`:
  - `empty_text_mask`: dokumen yang fitur teksnya sama dengan fitur masukan kosong.
  - `evaluate_followup`: (a) modalitas hilang diganti rerata fitur/logit bagian latih,
    untuk LATE dan CONCAT; (b) ConcatHead kedua dilatih tanpa dokumen teks kosong di latih
    dan val *head*, lalu uji tanpa teks diulang. Tiap baris juga mencatat kelas yang paling
    banyak diprediksi.
  - `run_grid` menerima argumen `cell`, jadi grid yang sama menjalankan `evaluate_followup`.
  - Self-check diperluas: kondisi sintetis ketiga dengan sepertiga kelas 3 bertekst kosong;
    memeriksa mask, jumlah dokumen kosong per bagian, *head* identik dengan grid, dan bahwa
    pengganti rerata mendarat dekat IMAGE.
- `notebooks/pipeline.ipynb`: empat sel "H3 lanjutan" setelah sel matriks konfusi H3
  (markdown, eksekusi + cek silang dokumen kosong, tabel selisih + porsi Note, penilaian
  kriteria H4a/H4b). Output sel lama tidak disentuh.
- [[17-missing-modality-lanjutan]]: catatan eksperimen, `status: belum-dijalankan`,
  hipotesis H4a/H4b dan ambangnya ditulis sebelum eksekusi.
- Tautan dari [[16-fusi-per-ocr]] temuan 2 dan [[limitasi-dan-lanjutan]] H.5.

## Verifikasi di mesin penulis

- `python -m src.experiment`: self-check lolos.
- *Dry run* sel bagian 8 sampai 10 (termasuk empat sel baru) pada cache unimodal sintetis
  (500 dokumen, 10 kelas, model linear kecil, dokumen kosong disisipkan di kelas Note) di
  folder sementara: semua sel jalan, 13 berkas laporan tertulis. Angka *dry run* tidak
  bermakna dan tidak dicatat.

## Untuk pengguna (mesin compute)

Jalankan ulang notebook (`python run_all.py`); semua cache dipakai ulang, jadi yang benar-benar
dihitung hanya bagian 8-10. Lalu isi [[17-missing-modality-lanjutan]] dari
`reports/fase2/uji_lanjutan_*.csv`.
