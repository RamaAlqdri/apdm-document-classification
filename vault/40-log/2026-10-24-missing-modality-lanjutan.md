---
judul: Log 2026-10-24 — Uji missing modality lanjutan (kode dan hasil)
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

# Log 2026-10-24 — Uji missing modality lanjutan

Menindaklanjuti [[16-fusi-per-ocr]] temuan 2 (CONCAT tanpa teks jatuh jauh di bawah IMAGE,
diduga karena masukan pengganti keluar distribusi dan *head* belajar "teks kosong ⇒ Note").
Kode dan rancangan ditulis dulu, lalu dijalankan pengguna di mesin compute pada hari yang sama.

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

## Eksekusi (mesin compute, `f4cbe39`)

Pengguna menjalankan ulang notebook; semua model unimodal diambil dari cache
(`dari_cache` = True di `unimodal_per_seed.csv`), grid utama identik, *head* CONCAT yang
dilatih ulang identik dengan bagian 8. Sel uji lanjutan 133 detik. Keluaran:
`reports/fase2/uji_lanjutan_missing_modality.csv`, `uji_lanjutan_selisih.csv`,
`uji_lanjutan_porsi_note.csv`.

## Pokok hasil

Rincian di [[17-missing-modality-lanjutan]]; semua angka dari CSV di atas.

- H4a didukung untuk tanpa citra (paling jauh 4,2 poin di bawah TEXT dengan pengganti rerata)
  dan untuk CONCAT tanpa teks pada easyocr/paddleocr (−2,6 / −2,7 poin terhadap IMAGE); pada
  qs-ocr/tesseract tersisa −10,8 / −13,0 poin.
- H4b tidak didukung: tarikan ke Note bertahan tanpa dokumen kosong di data latih, dan muncul
  juga pada OCR yang tidak punya dokumen kosong.
- Anomali: rerata logit latih bukan pengganti netral untuk LATE tanpa teks (−7 sampai −13 poin).
- Dokumen kosong menurut model (27 / 11 / 0 / 0) lebih sedikit daripada halaman 0 kata
  (37 / 21 / 6 / 2); penyebabnya belum diverifikasi.

## Catatan yang diperbarui

[[17-missing-modality-lanjutan]] (status → selesai), [[16-fusi-per-ocr]] temuan 2,
[[limitasi-dan-lanjutan]] H.5, [[sintesis-fase-2]] bagian koreksi metodologis dan yang tetap
terbuka.
