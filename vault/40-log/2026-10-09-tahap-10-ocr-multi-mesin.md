---
judul: Log 2026-10-09 — Tahap 10 OCR Multi-mesin (kode siap)
tipe: log
tahap: "10"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/log
  - tahap/10
  - status/selesai
terkait:
  - "[[13-ocr-multi-mesin]]"
  - "[[rencana-prompt]]"
  - "[[peta-proyek]]"
---

# Log 2026-10-09 — Tahap 10 OCR Multi-mesin

Status log ini `selesai` untuk bagian **kode**. Eksperimennya sendiri,
[[13-ocr-multi-mesin]], masih `belum-dijalankan` sampai notebook 10 dijalankan di laptop
compute.

## Yang dibuat

- `src/ocr.py`:
  - adapter Tesseract, EasyOCR, dan PaddleOCR dengan satu format keluaran (teks + kotak);
  - `reading_order()` untuk mesin berbasis kotak;
  - runner yang bisa dilanjutkan dan menulis secara atomik;
  - proksi kualitas (OOV, Jaccard);
  - CLI per mesin dan `_meta.json` berisi versi pustaka.
- `notebooks/10_ocr_multi_mesin.ipynb`: memasang dependensinya sendiri, mengaudit
  resolusi, menjalankan uji coba dengan perkiraan waktu, menjalankan proses penuh, lalu
  menghitung proksi.
- `run_all.py`: notebook 10 masuk daftar, dikecualikan dari penjaga timpa, dan self-check
  `src/ocr` ditambahkan (11 modul). RUNBOOK ditambah satu bagian.

## Keputusan yang didorong temuan saat verifikasi

1. **PaddleOCR di venv terpisah.** Metadata PyPI menunjukkan paddleocr 3.7 menarik
   `opencv-contrib-python==4.10.0.84`. Paket itu bentrok dengan `opencv-python-headless`
   milik EasyOCR, karena keduanya menyediakan `cv2`. Dependensi paddlex juga membawa
   numpy 2.3.5 ke venv uji.
2. **EasyOCR dipasang dengan constraint numpy dan torch.** opencv terbaru menuntut
   numpy ≥ 2, jadi `pip install easyocr` polos akan meng-upgrade numpy dan menggeser
   lingkungan yang hasil Fase 1 bergantung padanya. Dengan constraint, pip memilih opencv
   4.11.0.86 dan numpy tetap 1.26.4. Ini terverifikasi.
3. **PaddleOCR memakai model v5 mobile.** Default baru (PP-OCRv6_medium) 3× lebih lambat
   di CPU pada uji satu halaman. Bisa diganti lewat satu variabel.
4. **Keras-OCR dikeluarkan.** Konflik TF/Keras/Python 3.12 dan `imgaug` diperiksa dari
   kode sumber paketnya, bukan diduga. Adapternya dihapus, karena kode yang tidak bisa
   diuji tidak boleh ikut. Alasannya tertulis di [[13-ocr-multi-mesin]].
5. **Nama gambar diberi awalan `ocr_`, bukan `10_`.** Notebook 09 sudah memakai awalan
   `10_` untuk gambarnya.

## Kekeliruan saya di tahap ini

- **Venv uji PaddleOCR ternyata 951 MB**, di atas ambang 500 MB aturan 4, dan saya tidak
  memperkirakannya lebih dulu. Venv itu hanya di scratchpad dan sudah dihapus beserta
  model unduhannya (`~/.paddlex`, 133 MB).
- **Venv pengembangan di Mac ternyata memakai numpy 2.5.3, bukan pin 1.26.4.**
  Constraint saat memasang EasyOCR menurunkannya dan merusak `contourpy`. Venv Mac kini
  disamakan dengan pin Windows (numpy 1.26.4, contourpy 1.3.3), `pip check` bersih, dan
  ke-11 self-check lolos. EasyOCR dan model-nya (94 MB) tetap terpasang di venv Mac.

## Berikutnya

Jalankan notebook 10 di laptop TUF, lalu isi [[13-ocr-multi-mesin]]. Tahap 11 (grid
OCR × embedding, prompt-nya di [[rencana-prompt]]) membaca `data/interim/ocr/` hasil
notebook ini.
