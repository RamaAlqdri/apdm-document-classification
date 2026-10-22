# Arsip notebook per tahap (tidak dijalankan lagi)

Notebook di folder ini adalah riwayat kerja Fase 1 (Tahap 3-8, replikasi Audebert) dan
dua tahap awal Fase 2 (10: OCR multi-mesin, 11: kurasi + EDA). Output eksekusinya
disimpan apa adanya karena angka di `vault/30-eksperimen/` merujuk ke sini.

Notebook ini **tidak untuk dijalankan ulang** dari folder ini: path `ROOT` di sel
pertamanya mengandaikan lokasi lama `notebooks/`. Seluruh logikanya yang masih dipakai
sudah dipindah ke `src/` dan dipanggil oleh satu notebook aktif:

    notebooks/pipeline.ipynb      (atau: python run_all.py)

| Notebook | Isi | Sekarang di |
|---|---|---|
| 00 | setup + menjalankan 01-06 | `pipeline.ipynb` bagian 1, `src/setup_env.py` |
| 01 | audit + manifest | `src/data.py: prepare_dataset()` |
| 02 | split + fitur teks QS-OCR | `src/splits.py`, `src/text_features.py: build_condition_sequences()` |
| 03-05 | baseline TEXT/IMAGE, fusion end-to-end, oracle | `src/experiment.py` (fusi Fase 2: LATE + CONCAT di atas model beku) |
| 06-09 | ablasi, uji lanjutan, diagnostik, perbaikan | riwayat Fase 1; `src/ablation.py` tetap ada |
| 10 | OCR multi-mesin | `src/ocr.py` + `pipeline.ipynb` bagian 5 |
| 11 | kurasi + EDA | `src/eda.py` + `pipeline.ipynb` bagian 4 dan 6.1 |
