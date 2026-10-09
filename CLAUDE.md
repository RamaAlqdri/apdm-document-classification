# Brief Proyek: Implementasi Jurnal Multimodal (Teks + Citra)

## Konteks
Proyek tugas matakuliah "Analisis dan Pemrosesan Data Multimodal".

Fase 1 (Tahap 0-8, selesai): replikasi Audebert et al., "Multimodal deep
networks for text and image-based document classification", ECML PKDD 2019
Workshops, CCIS 1167, Springer 2020 (PDF: references/1907.06370v1.pdf).

Fase 2 (Tahap 9-14, sejak tanggapan dosen 2026-10-09): mesin OCR dan embedding
dijadikan variabel. Referensi utama: Kashyap et al. (2026), Int. J. Intelligent
Systems, doi:10.1155/int/4241437 (PDF di references/). Syarat dosen: referensi
utama minimal 2025 dan sudah terbit, BUKAN preprint arXiv. Rencana per tahap di
vault/00-index/rencana-prompt.md.

Dataset:
- Citra: Tobacco3482 versi JPG dari Kaggle (patrickaudriaz/tobacco3482jpg)
- Teks: QS-OCR-Small dari GitHub QuickSign/ocrized-text-dataset, tab Releases
  tag v1.0. File dataset TIDAK ada di root repo.
RVL-CDIP dan QS-OCR-Large TIDAK dipakai (terlalu besar untuk lingkup tugas).

PENTING: repo GitHub tersebut adalah repo dataset, bukan repo implementasi.
Tidak ada kode model di sana, hanya script OCR. Seluruh arsitektur harus
ditulis ulang dari spesifikasi di paper.

## Struktur folder
proyek/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/{raw,interim,processed}/     # digitignore
├── references/                       # PDF paper (sudah ada)
├── notebooks/                        # 01_..., 02_..., dst
├── src/                              # modul reusable, di-import notebook
├── models/                           # checkpoint, digitignore
├── reports/figures/
└── vault/                            # Obsidian vault
    ├── 00-index/
    ├── 10-jurnal/
    ├── 20-metode/
    ├── 30-eksperimen/
    ├── 40-log/
    ├── 50-hasil/
    ├── 90-referensi/
    ├── templates/
    └── attachments/

## Aturan kerja
1. BERTAHAP. Kerjakan hanya tahap yang saya minta. Di akhir tiap tahap:
   berhenti, ringkas apa yang dibuat, tunggu konfirmasi. Jangan lompat maju.
2. Logika berat tinggal di src/ sebagai fungsi yang bisa di-unit-test.
   Notebook hanya untuk orkestrasi, visualisasi, dan narasi.
3. JANGAN PERNAH mengarang angka hasil. Semua metrik di catatan harus berasal
   dari eksekusi nyata. Kalau belum dijalankan, tulis `status: belum-dijalankan`.
4. Tanya dulu sebelum: install dependency >500MB, download data >1GB, atau
   menjalankan training yang diperkirakan >15 menit.
5. Reproducibility: seed global (numpy, random, torch) di satu tempat, semua
   hyperparameter di satu dict CONFIG di sel paling atas notebook.
6. Bahasa: catatan dan narasi notebook dalam Bahasa Indonesia; nama variabel,
   fungsi, dan komentar kode dalam Bahasa Inggris.
7. Setiap akhir tahap: update catatan log harian di vault/40-log/ dan update
   MOC terkait di vault/00-index/.
8. Git: repo sudah diinisialisasi dan sudah ada commit awal. Commit di akhir
   setiap tahap dengan pesan konvensional (feat:, docs:, exp:). Jangan git init
   ulang.

## Konvensi Obsidian
Semua catatan .md wajib punya YAML frontmatter:

```yaml
---
judul:
tipe:        # jurnal | konsep | eksperimen | log | hasil | referensi
tahap:       # 00 sampai 14
tanggal:
status:      # todo | wip | selesai | ditunda
tags: []
terkait: []  # daftar [[wikilink]]
---
```

Taksonomi tag (gunakan nested tag, konsisten, jangan bikin varian baru
tanpa mendaftarkannya di vault/00-index/peta-tag.md):
- #tipe/jurnal #tipe/konsep #tipe/eksperimen #tipe/log #tipe/hasil #tipe/referensi
- #tahap/00 ... #tahap/14
- #modal/teks #modal/citra #modal/fusion
- #komponen/ocr #komponen/embedding #komponen/cnn #komponen/fusion #komponen/evaluasi
- #status/todo #status/wip #status/selesai
- #temuan/positif #temuan/negatif #temuan/anomali

Keterkaitan: pakai [[wikilink]] di badan teks, bukan hanya di frontmatter.
Setiap catatan minimal punya 2 link keluar dan terhubung ke MOC-nya.
Catatan eksperimen wajib punya inline field Dataview:
  model:: | dataset:: | split:: | seed:: | oa:: | macro_f1:: | durasi::
supaya bisa diagregasi jadi tabel otomatis.

## Catatan MOC yang harus ada di 00-index/
- peta-proyek.md (MOC induk)
- peta-tag.md (daftar resmi tag + artinya)
- peta-metode.md
- peta-eksperimen.md (berisi query Dataview ke 30-eksperimen/)
- papan-progress.md (query Dataview status per tahap)
