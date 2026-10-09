---
judul: Rencana Prompt Bertahap - Implementasi Jurnal Multimodal
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
---

# Rencana Prompt Bertahap

Kumpulan prompt untuk menjalankan implementasi paper Audebert et al. (arXiv:1907.06370) secara lokal dengan agen coding. Kirim satu blok per sesi, jangan digabung.

**Fase 1 (Tahap 0–8)** adalah replikasi Audebert dan sudah selesai. **Fase 2 (Tahap 9–14)**
ditambahkan setelah tanggapan dosen pada 2026-10-09: referensi utama harus terbit
minimal 2025, dan gap harus dirumuskan sendiri. Lihat bagian [Fase 2](#fase-2--ocr-dan-embedding-sebagai-variabel).

---

## Basis rujukan tiap perintah

Penting dipahami sebelum mulai, supaya tidak ada ekspektasi yang salah.

### Yang terverifikasi dari repo dataset

Repo `github.com/QuickSign/ocrized-text-dataset` adalah **repo dataset, bukan repo implementasi model**. Isinya: `to_text.py`, `tobacco3482.sh`, `rvl-cdip.sh`, `setup.sh`, `Dockerfile`, `Taskfile.yml`, `datasheet.md`/`datasheet.pdf`, `readme.md`, `requirements.txt`, `pyproject.toml`. Lisensi Apache-2.0. Semua script itu hanya untuk meregenerasi teks OCR dari citra, tidak ada kode training.

Fakta yang dipakai dalam prompt di bawah:

| Fakta | Nilai |
|---|---|
| Lokasi file dataset teks | Tab **Releases**, tag `v1.0`, **2** aset arsip (`QS-OCR-small.tar.gz` 2,5 MB dan `QS-OCR-Large.tar.gz` 251,9 MB). Bukan di root repo. Terverifikasi lewat GitHub API 2026-09-28; rencana awal menyebut 4 aset dan itu salah. |
| QS-OCR-Small | 3.482 file teks, 10 kelas, dari Tobacco3482 |
| QS-OCR-Large | 400.000 file teks, dari RVL-CDIP (tidak dipakai di proyek ini) |
| Kelas QS-OCR-Small | Advertisement (ADVE), Email, Form, Letter, Memo, News, Note, Report, Resume, Scientific |
| Struktur folder | Mengikuti dataset asli, satu folder per kelas |
| Split bawaan | Tidak ada untuk versi Small. README menyarankan k-fold cross-validation. |
| Versi Tesseract | 4.0.0-beta.1 |
| Parameter OCR | `--oem 1` (LSTM), `--psm 3` (full page, tanpa deteksi orientasi/script), `-l eng` |
| Otomasi | library `pytesseract` |
| Praproses citra | Tidak ada. TIFF grayscale asli dipakai langsung. |
| Pascaproses teks | Tidak ada. Output Tesseract mentah, termasuk salah ejanya. |
| Peringatan overlap | Ada irisan sebagian antara QS-OCR-Small dan QS-OCR-Large karena Tobacco3482 adalah subset RVL-CDIP. Relevan hanya kalau melakukan transfer learning antar keduanya. |

**Bug di README repo:** bagian "How to reproduce" menginstruksikan `./tobacco3842.sh`, tetapi file yang benar-benar ada di repo bernama `tobacco3482.sh`. Angka 8 dan 4 tertukar. Kalau mengikuti README mentah-mentah akan kena *file not found*.

**Kontradiksi paper vs parameter:** paper §3.2 menulis Tesseract "will try to detect the text orientation", padahal `--psm 3` justru mode full-page **tanpa** OSD — deteksi orientasi/script hanya aktif di `--psm 0`/`--psm 1`. Parameter di tabel di atas yang benar; kalimat paper tidak akurat. Tandai ini eksplisit di `ocr-tesseract.md` supaya catatan Tahap 2 tidak diam-diam bertentangan dengan tabel ini.

### Yang terverifikasi dari Kaggle

`kaggle.com/datasets/patrickaudriaz/tobacco3482jpg`, judul "Tobacco3482", versi JPG dari dataset asli. Dataset asli berformat TIF dan dihosting di server UMIACS yang sering sulit diakses, sehingga versi Kaggle ini lazim dipakai sebagai pengganti.

**Konsekuensi yang harus ditulis sebagai limitasi:** teks QS-OCR di-OCR dari TIF asli, sedangkan citra kita dari JPG hasil re-encode. Sumber kedua modalitas tidak identik.

### Yang berbasis paper saja

Seluruh spesifikasi model: arsitektur, dimensi layer, optimizer, learning rate, jumlah epoch, batch size, panjang padding, strategi fusion, protokol split. Tidak ada kode resmi dari penulis untuk mengecek silang. Karena itu Tahap 2 mewajibkan pemisahan section "Ambiguitas & asumsi".

Hyperparameter yang disebut eksplisit di paper §4.2. Dipakai langsung di Tahap 5 dan 6, jangan ditebak ulang:

| Model | Optimizer | LR | Momentum | Batch | Epoch |
|---|---|---|---|---|---|
| TEXT (CNN1D) | SGD + momentum | 0,01 | 0,9 | 40 | 100 |
| IMAGE (MobileNetV2) | SGD + momentum | 0,01 | 0,9 | 40 | 200 |
| FUSION | SGD + momentum | 0,01 | 0,9 | 40 | 200 |

Catatan baca PDF: batch TEXT terbaca "406" karena marker footnote 6 menempel di angkanya. Nilainya 40.

Yang **tidak** disebut paper dan wajib masuk "Ambiguitas & asumsi": normalisasi citra (mean/std ImageNet?), truncation dokumen >500 kata (paper hanya menyebut padding untuk yang <500 kata), mekanisme "adaptive averaging" pada fusion penjumlahan, rasio dropout, dan ada/tidaknya early stopping.

Deviasi yang sudah pasti sejak awal: paper memakai TensorFlow 1.12 + Keras, kita PyTorch. Bobot pretrained MobileNetV2 torchvision tidak identik dengan versi Keras.

### Yang berbasis rekomendasi umum

Struktur folder proyek, konvensi vault Obsidian, taksonomi tag, dan seluruh rancangan ablasi di Tahap 7. Ini bukan bagian dari paper maupun repo, melainkan tambahan agar proyek punya nilai analitis di luar replikasi.

---

## Langkah 0 — Brief utama

Buat folder proyek kosong, masuk ke dalamnya, jalankan agen, kirim prompt ini sendirian. Periksa hasilnya sebelum lanjut, karena semua tahap berikutnya bergantung pada file ini.

````text
Buatkan file CLAUDE.md di root folder ini dengan isi persis seperti di bawah,
lalu berhenti dan tunggu instruksi saya. Jangan buat file lain dulu.

---

# Brief Proyek: Implementasi Jurnal Multimodal (Teks + Citra)

## Konteks
Proyek tugas matakuliah "Analisis dan Pemrosesan Data Multimodal".
Mereplikasi paper: Audebert et al., "Multimodal deep networks for text and
image-based document classification" (arXiv:1907.06370). PDF ada di lokal.

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
8. Git: init di tahap 1, commit di akhir setiap tahap dengan pesan konvensional
   (feat:, docs:, exp:).

## Konvensi Obsidian
Semua catatan .md wajib punya YAML frontmatter:

```yaml
---
judul:
tipe:        # jurnal | konsep | eksperimen | log | hasil | referensi
tahap:       # 00 sampai 08
tanggal:
status:      # todo | wip | selesai | ditunda
tags: []
terkait: []  # daftar `[[wikilink]]`
---
```

Taksonomi tag (gunakan nested tag, konsisten, jangan bikin varian baru
tanpa mendaftarkannya di vault/00-index/peta-tag.md):
- #tipe/jurnal #tipe/konsep #tipe/eksperimen #tipe/log #tipe/hasil #tipe/referensi
- #tahap/00 ... #tahap/08
- #modal/teks #modal/citra #modal/fusion
- #komponen/ocr #komponen/embedding #komponen/cnn #komponen/fusion #komponen/evaluasi
- #status/todo #status/wip #status/selesai
- #temuan/positif #temuan/negatif #temuan/anomali

Keterkaitan: pakai `[[wikilink]]` di badan teks, bukan hanya di frontmatter.
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
````

---

## Tahap 1 — Scaffolding + vault

```text
TAHAP 1: Scaffolding.

Buat seluruh struktur folder sesuai CLAUDE.md, plus:
- .gitignore (data/, models/, *.ipynb_checkpoints, .obsidian/workspace*, venv)
- requirements.txt (PyTorch, torchvision, numpy, pandas, scikit-learn,
  matplotlib, seaborn, spacy, fasttext-wheel, kagglehub, pillow, tqdm,
  jupyter). Jangan install dulu, cukup tulis filenya. Pakai `fasttext-wheel`,
  bukan `fasttext` — yang terakhir sering gagal build dari source di Python
  versi baru. Catat di README bahwa setelah install wajib jalan
  `python -m spacy download en_core_web_sm`, karena model spaCy tidak ikut pip.
- README.md singkat: tujuan, cara setup, peta folder.
- src/ dengan __init__.py, config.py (CONFIG dict + set_seed()), utils.py.

Di vault/templates/ buat 4 template: catatan-konsep.md, catatan-eksperimen.md,
catatan-log.md, catatan-jurnal.md, masing-masing dengan frontmatter lengkap
dan heading kosong yang sudah terstruktur.

Di vault/00-index/ buat kelima MOC. peta-tag.md harus mendaftar seluruh tag
beserta definisi kapan dipakai. peta-eksperimen.md dan papan-progress.md
berisi query Dataview yang valid (pakai blok ```dataview```).

Buat vault/00-index/peta-proyek.md sebagai pintu masuk vault: ringkas tujuan,
daftar 9 tahap (00-08) sebagai checklist, dan link ke MOC lain.

Git sudah diinisialisasi dan sudah punya commit awal di folder ini, jadi cukup
commit hasil scaffolding — jangan git init ulang. Lalu berhenti dan laporkan.
```

---

## Tahap 2 — Sintesis jurnal ke vault

```text
TAHAP 2: Sintesis jurnal.

PDF papernya ada di references/1907.06370v1.pdf. Baca penuh, lalu:

A. vault/10-jurnal/1907.06370-ringkasan.md — ringkasan terstruktur: masalah,
   kontribusi, metode, dataset, hasil utama, limitasi yang diakui penulis.
   Tulis ulang dengan kalimat sendiri, jangan menyalin kalimat paper.

B. vault/10-jurnal/1907.06370-spesifikasi-implementasi.md — SEMUA detail teknis
   untuk replikasi, diekstrak apa adanya dari paper: arsitektur tiap cabang,
   dimensi tiap layer, optimizer, learning rate, momentum, batch size, jumlah
   epoch per model, ukuran input citra, panjang padding sekuens teks, strategi
   fusion, protokol split dan evaluasi.

   Ingat: tidak ada kode resmi dari penulis, jadi paper adalah satu-satunya
   sumber. Kalau ada detail yang TIDAK disebutkan paper, buat section terpisah
   "Ambiguitas & asumsi" dan daftarkan eksplisit. Jangan menambal dengan
   tebakan diam-diam.

C. Catatan atomik di vault/20-metode/, satu konsep satu file, masing-masing
   ringkas (150-300 kata) dan saling terhubung:
   ocr-tesseract.md, noise-ocr-dan-oov.md, fasttext-subword.md,
   sif-document-embedding.md, cnn-1d-teks.md, mobilenetv2.md,
   inverted-residual-block.md, strategi-fusion-concat-vs-sum.md,
   oracle-sebagai-batas-atas.md, taksonomi-multimodal.md.

   Khusus taksonomi-multimodal.md: posisikan paper ini dalam kerangka
   representation / translation / alignment / fusion / co-learning, dan bahas
   secara jujur isu "teks diturunkan dari citra lewat OCR, jadi bukan modalitas
   independen" — argumen pro dan kontra, bukan hanya pembelaan.

D. vault/90-referensi/ — catatan pendek untuk 6-8 referensi terpenting yang
   dikutip paper (Harley RVL-CDIP, Kumar Tobacco3482, Sandler MobileNetV2,
   Bojanowski FastText, Arora SIF, Kim CNN sentence classification).

E. Update peta-metode.md dan peta-proyek.md, tulis log tahap 2, commit.

Jangan sentuh data atau kode di tahap ini. Berhenti setelah selesai.
```

---

## Tahap 3 — Data: akuisisi, pairing, verifikasi

```text
TAHAP 3: Data.

Sumber:
- Citra: Kaggle patrickaudriaz/tobacco3482jpg → ekstrak ke data/raw/
  Download Kaggle butuh kredensial: `~/.kaggle/kaggle.json` atau env
  KAGGLE_USERNAME/KAGGLE_KEY. Kalau belum ada, BERHENTI dan minta saya
  menyiapkannya. Jangan cari jalan pintas lewat scraping.
- Teks: QS-OCR-Small, dari tab RELEASES repo QuickSign/ocrized-text-dataset
  tag v1.0 (bukan dari root repo, bukan hasil git clone) → ekstrak ke data/raw/

Buat notebooks/01_data_audit.ipynb + src/data.py yang:

1. Memverifikasi arsip citra: total harus 3482 file dalam 10 folder kelas.
   Deteksi dan laporkan kalau ada folder bersarang duplikat atau file ganda.
2. Memverifikasi arsip teks: 3482 file .txt dalam 10 folder kelas. Nama kelas
   menurut README repo: Advertisement (ADVE), Email, Form, Letter, Memo, News,
   Note, Report, Resume, Scientific. Laporkan kalau nama folder di arsip citra
   dan arsip teks ternyata berbeda ejaan/kapitalisasi, dan buat mapping
   eksplisit — jangan diam-diam menormalisasi.
3. PAIRING: cocokkan stem nama file citra dengan stem nama file teks per kelas.
   Laporkan eksplisit: berapa pasangan berhasil, berapa citra yatim, berapa
   teks yatim, dan tampilkan contoh nama yang gagal cocok.
4. Menghasilkan data/processed/manifest.csv: kolom id, kelas, path_citra,
   path_teks, n_kata, n_karakter.
5. EDA: distribusi sampel per kelas (tunjukkan ketimpangannya), distribusi
   panjang teks, persentase dokumen dengan teks kosong atau sangat pendek
   (<20 kata), dan tampilkan 4 contoh pasangan citra+teks berdampingan untuk
   menilai kualitas OCR secara visual.
6. Menyimpan figur ke reports/figures/.

Catatan konteks untuk ditulis di laporan audit:
- QS-OCR dihasilkan dengan Tesseract 4.0.0-beta.1, parameter --oem 1 --psm 3
  -l eng, tanpa praproses citra dan tanpa pascaproses teks. Salah eja dibiarkan
  apa adanya secara sengaja.
- QS-OCR-Small tidak punya split bawaan; README menyarankan k-fold.
- Teks di-OCR dari TIF asli, sedangkan citra kita dari JPG re-encode. Tulis ini
  sebagai limitasi.
- Kalau nanti memakai script repo untuk regenerasi OCR: README menyebut
  ./tobacco3842.sh tetapi file sebenarnya bernama tobacco3482.sh (typo di
  README).

Tulis vault/30-eksperimen/00-audit-data.md berisi temuan NYATA dari eksekusi.
Update log + commit. Berhenti.
```

---

## Tahap 4 — Split & pipeline teks

```text
TAHAP 4: Split dan representasi teks.

A. src/splits.py: 800 dokumen train, sisanya test, terstratifikasi per kelas,
   3 seed berbeda (42/43/44), disimpan sebagai
   data/processed/split_seed{N}.json agar identik di semua eksperimen.
   Sisihkan juga validation kecil dari train untuk early stopping.

   PENTING — ini DEVIASI, bukan protokol paper. Paper memakai k-fold
   cross-validation dengan 800 dokumen train, kita memakai 3 random split
   terstratifikasi. Paper juga melatih dengan jumlah epoch tetap tanpa early
   stopping (hyperparameter di-tune di subset terpisah), sedangkan kita
   memotong 800 train untuk validation. Tulis kedua deviasi ini eksplisit, dan
   jangan sebut split ini "protokol paper" di catatan mana pun.

B. src/text_features.py:
   - Tokenisasi + pembuangan punctuation pakai spaCy en_core_web_sm.
   - Embedding FastText. WAJIB memakai model .bin (cc.en.300.bin), BUKAN .vec.
     Hanya .bin yang bisa menginferensi vektor untuk kata out-of-vocabulary
     lewat subword, dan itulah inti argumen paper soal noise OCR. Ukurannya
     ~7GB di disk dan ~15GB saat dimuat.

     JANGAN mengandalkan reduce_model sebagai solusi RAM: fungsi itu harus
     memuat model 300 dimensi penuh lebih dulu, jadi puncak pemakaian RAM-nya
     sama saja. Strategi yang benar: fitur teks cuma dihitung SEKALI ke disk,
     jadi jalankan satu pass ekstraksi (biarkan lambat kalau kena swap), tulis
     hasilnya, lalu lepas model dari memori. Model FastText tidak boleh ikut
     dimuat saat training.
   - Dua representasi: (1) sekuens 500x300 zero-padded/truncated untuk CNN1D;
     (2) SIF weighted average + PCA removal untuk baseline MLP.
   - Precompute ke disk sebagai .npy memmap (pertimbangkan float16), jangan
     hitung ulang tiap epoch.

C. notebooks/02_text_features.ipynb: jalankan, ukur waktu, dan reproduksi
   analisis Fig. 4b paper — hitung cosine similarity FastText untuk beberapa
   pasangan kata salah eja yang BENAR-BENAR muncul di korpus kita. Cari
   sendiri contohnya, jangan pakai contoh dari paper.

D. Perbarui catatan vault/20-metode/ bila ada temuan; tulis
   vault/30-eksperimen/01-ekstraksi-fitur-teks.md. Log + commit. Berhenti.
```

---

## Tahap 5 — Baseline unimodal

```text
TAHAP 5: Dua baseline unimodal.

A. src/models/text_models.py: MLP (lebar 2048, ReLU+Dropout+BN, output 128,
   He init) dan CNN1D (4 layer, kernel 12, 512 channel, maxpool stride 2,
   max-pool-through-time, dropout, FC→softmax), sesuai
   vault/10-jurnal/1907.06370-spesifikasi-implementasi.md.

B. src/models/image_model.py: MobileNetV2 pretrained ImageNet, input 384x384,
   grayscale diduplikasi 3 kanal, resize tanpa padding (aspect ratio sengaja
   di-warp, sesuai paper). SELURUH jaringan di-fine-tune, backbone JANGAN
   dibekukan — paper melakukan fine-tuning penuh. Backbone beku bikin akurasi
   jatuh jauh dan selisihnya akan tersalahartikan sebagai gap replikasi.

C. src/train.py: loop training generik, logging per-epoch ke CSV, checkpoint
   best-on-val, seed terkontrol. Hyperparameter dari paper §4.2, jangan
   ditebak: SGD momentum 0,9, lr 0,01, batch 40 untuk semua model; TEXT 100
   epoch, IMAGE 200 epoch, FUSION 200 epoch.

D. notebooks/03_baseline_teks.ipynb — latih MLP dan CNN1D, bandingkan.
   Ini replikasi Tabel 1a paper.
   notebooks/04_baseline_citra.ipynb — latih MobileNetV2. Replikasi Tabel 1b.

Soal durasi: MobileNetV2 384x384 selama 200 epoch hampir pasti jauh di atas 15
menit per run di mesin ini, dan Tahap 6-7 mengalikannya dengan jumlah seed dan
kondisi ablasi. Aturan ">15 menit tanya dulu" akan kepicu di hampir semua run
dan berhenti berfungsi sebagai pengaman.

Prosedurnya: (1) smoke test 2 epoch untuk memastikan pipeline jalan sekaligus
mengukur detik/epoch nyata, (2) laporkan ekstrapolasinya ke saya, (3) baru
sepakati budget epoch sebenarnya. Kalau kita potong dari 200 epoch, catat
sebagai deviasi eksplisit beserta alasannya. Jangan dipotong diam-diam.

Setelah run nyata selesai: buat catatan eksperimen terpisah untuk TIAP run di
vault/30-eksperimen/ dengan inline field Dataview terisi angka sungguhan
(oa::, macro_f1::, durasi::), plus confusion matrix tersimpan di
reports/figures/ dan di-embed ke catatan. Bandingkan dengan angka paper dan
jelaskan selisihnya secara jujur.

Log + commit. Berhenti.
```

---

## Tahap 6 — Model fusion

```text
TAHAP 6: Model multimodal.

src/models/fusion.py: cabang citra (GAP → vektor 1280 → FC → 128), cabang teks
(CNN1D → 128), digabung, lalu MLP → softmax. Dilatih end-to-end.

Implementasikan DUA strategi fusion yang bisa di-switch: concat dan penjumlahan
adaptif ("adaptive averaging"). Paper melaporkan penjumlahan gagal jauh di bawah
baseline citra — kita uji sendiri, jangan diasumsikan benar tanpa bukti dari run
kita.

Paper TIDAK menjelaskan apa yang membuat penjumlahan itu "adaptif" (skalar
terlatih? gate per-dimensi? rata-rata biasa?). Pilih satu, tulis pilihannya
sebagai asumsi eksplisit, dan sebutkan bahwa hasil negatif di strategi ini bisa
jadi artefak pilihan kita, bukan replikasi temuan paper.

Hyperparameter: SGD momentum 0,9, lr 0,01, batch 40, 200 epoch (paper §4.2).

notebooks/05_fusion.ipynb: latih keduanya, 3 seed, laporkan rata-rata ± std.

Tambahkan perhitungan ORACLE. Definisi yang benar menurut paper: ambil prediksi
per-sampel dari DUA MODEL BASELINE UNIMODAL yang sudah dilatih di Tahap 5 (TEXT
standalone dan IMAGE standalone) pada test split yang sama; satu sampel dihitung
benar bila salah satu dari keduanya benar. BUKAN dari cabang internal model
fusion. Kalau ini salah tafsir, angka oracle-nya tidak sebanding dengan 92,1%
paper dan seluruh argumen komplementaritas rusak.

Ini batas atas teoretis fusion dan WAJIB dilaporkan, karena inilah bukti
kuantitatif bahwa kedua modalitas saling melengkapi.

Hasil akhir harus bisa mengisi tabel setara Tabel 3 paper: OA dan F1 per kelas
untuk TEXT / IMAGE / FUSION / Oracle.

Catatan eksperimen per konfigurasi + vault/50-hasil/tabel-utama.md yang
menyatukan semuanya. Tandai temuan dengan #temuan/positif, #temuan/negatif,
atau #temuan/anomali. Log + commit. Berhenti.
```

---

## Tahap 7 — Ablasi & analisis multimodal

Bagian ini bukan bagian dari paper. Inilah yang membedakan proyek dari sekadar replikasi.

```text
TAHAP 7: Ablasi.

Kerjakan berurutan, laporkan tiap sub-eksperimen sebelum lanjut.

Catatan compute: ablasi 1-3 adalah perturbasi pada INFERENSI, jadi pakai
checkpoint TEXT/IMAGE/FUSION yang sudah dilatih di Tahap 5-6 dan jangan melatih
ulang apa pun. Hanya ablasi 4 yang butuh training baru. Pakai satu seed (42) di
seluruh tahap ini dan catat sebagai keterbatasan.

1. Degradasi modalitas citra: rotasi, blur, noise, kompresi JPEG agresif secara
   bertahap. Ukur apakah cabang teks mengompensasi. Plot kurva akurasi vs
   tingkat degradasi untuk IMAGE dan FUSION dalam satu grafik.
2. Missing modality saat inferensi: nol-kan input teks, lalu nol-kan input
   citra. Seberapa jauh fusion kolaps?
3. Degradasi teks: hapus x% kata secara acak, atau injeksi typo buatan.
4. (Opsional, bila waktu cukup) fusion bergerbang atau cross-attention
   sederhana sebagai pembanding concat.

Catatan: paper mengakui di bagian limitasi bahwa dataset Tobacco sudah
ter-orientasi rapi dan discan profesional, sehingga tidak merepresentasikan
kondisi nyata. Ablasi 1 dan 3 adalah pengujian empiris atas limitasi yang
penulis sebut tapi tidak uji.

Tiap sub-eksperimen = satu catatan di vault/30-eksperimen/ yang terhubung ke
[[strategi-fusion-concat-vs-sum]] dan [[taksonomi-multimodal]].
Log + commit. Berhenti.
```

---

## Tahap 8 — Sintesis akhir

```text
TAHAP 8: Sintesis.

A. vault/50-hasil/sintesis-akhir.md: gabungkan seluruh temuan menjadi narasi
   yang menjawab tiga pertanyaan: (1) apakah replikasi berhasil dan seberapa
   dekat dengan paper, (2) apakah fusion benar-benar memberi nilai tambah dan
   dari mana asalnya, (3) kapan multimodal menolong dan kapan tidak,
   berdasarkan hasil ablasi kita sendiri.
   Wajib link ke catatan eksperimen sumbernya. Jangan mengulang angka tanpa
   jejak.

B. vault/50-hasil/limitasi-dan-lanjutan.md: limitasi jujur (sumber citra JPG vs
   teks dari TIF, tidak memakai RVL-CDIP, jumlah seed terbatas, tidak ada kode
   resmi penulis sehingga sebagian hyperparameter adalah asumsi) dan arah
   lanjutan (LayoutLM/DiT sebagai pendekatan multimodal modern).

C. Rapikan vault: pastikan tidak ada orphan note, semua tag sesuai peta-tag.md,
   semua checklist di peta-proyek.md tercentang, papan-progress.md akurat.
   Laporkan kalau ada catatan berstatus wip yang tertinggal.

D. README.md final + commit terakhir.
```

---

# Fase 2 — OCR dan embedding sebagai variabel

## Tanggapan dosen (2026-10-09) dan cara membacanya

| Poin dosen | Ditafsirkan sebagai |
|---|---|
| Eksplorasi OCR: EasyOCR, PaddleOCR, Keras-OCR | Mesin OCR jadi **variabel**, bukan konstanta |
| Gap belum ditentukan | Proposal butuh gap eksplisit, bukan sekadar replikasi |
| Kemungkinan eksplorasi embedding | Sumbu kedua: FastText vs BERT |
| Teks yang dideteksi sebagai objek, untuk tahap selanjutnya | Kotak posisi teks sebagai fitur tata letak → Tahap 14 |
| Referensi 2019 dan arXiv; minimal 2025 | Referensi utama diganti ke [[kashyap-2026-ringkasan]] |

PaddleOCR disebut dua kali dalam catatan dosen. Mungkin salah satunya maksudnya mesin
lain (docTR? TrOCR?). **Perlu dikonfirmasi ke dosen.**

## Gap

> Fusion teks–citra terbaru ([[kashyap-2026-ringkasan]]) memakai mesin OCR komersial yang
> tidak disebutkan, dan fusionnya rata-rata logit tanpa bobot. Meta-analisis
> [[mironczuk-2026-review-fusion]] tidak bisa mengontrol kualitas OCR karena studi primer
> tidak melaporkannya. Belum diukur bagaimana **pilihan mesin OCR × jenis embedding**
> memengaruhi kekuatan cabang teks, dan **seberapa jauh fusion benar-benar memakainya**.

Bagian kedua gap ini lahir dari Fase 1: dengan Tesseract + FastText, cabang teks fusion
praktis tidak terpakai ([[11-diagnostik-cabang-teks]]). Pertanyaannya sekarang: apakah
OCR atau embedding yang lebih baik mengubah itu?

## Pertanyaan penelitian Fase 2

- **Q1.** Seberapa besar pilihan mesin OCR mengubah akurasi cabang teks pada Tobacco3482?
- **Q2.** Apakah embedding kontekstual (BERT) mengecilkan selisih antar-mesin OCR
  dibanding FastText, yaitu apakah ada interaksi OCR × embedding?
- **Q3.** Apakah cabang teks yang lebih kuat membuat fusion lebih *memakai* teks, diukur
  dengan uji missing modality? Dan apakah aturan fusion Kashyap (rata-rata logit) lebih
  tahan terhadap cabang yang timpang daripada fusion fitur?

## Prinsip yang dipertahankan dari Fase 1

- Protokol Tobacco3482 yang sama: 800 dokumen latih, split seed 42/43/44, indeks split
  dari `src/splits.py`. **Tidak memakai RVL-CDIP** (alasan biaya; lihat catatan kritis di
  [[kashyap-2026-ringkasan]]).
- Uji missing modality wajib untuk setiap model fusion.
- **Baru:** uji statistik. McNemar berpasangan per seed antar-kondisi pada test set yang
  sama, ditambah rata-rata ± simpangan baku 3 seed. Ini menanggapi angka 11,8% di
  [[mironczuk-2026-review-fusion]].

---

## Tahap 9 — Revisi landasan

```text
TAHAP 9: Revisi arah setelah tanggapan dosen.

A. Cari referensi utama ≥2025 yang sudah terbit. Verifikasi metadata lewat
   Crossref/Semantic Scholar, jangan dari ingatan. Baca teks lengkap referensi
   utama, bukan abstraknya saja.
B. Catatan jurnal di vault/10-jurnal/ dan catatan pustaka di vault/90-referensi/.
C. Revisi proposal/proposal.tex: judul, gap, pertanyaan penelitian, referensi.
   Hasil Fase 1 masuk sebagai hasil awal yang memotivasi gap.
D. Tahap baru di rencana-prompt.md, tag tahap baru di peta-tag.md, CLAUDE.md.
Log + commit. Berhenti.
```

## Tahap 10 — Ekstraksi OCR multi-mesin

```text
TAHAP 10: OCR sendiri dengan beberapa mesin.

Tujuan: teks Tobacco3482 dari tiap mesin, dari CITRA YANG SAMA (JPG Kaggle),
plus kotak posisi teks untuk Tahap 14.

A. src/ocr.py, satu fungsi per mesin, keluaran seragam:
   [{"text": str, "box": [[x,y]x4], "conf": float}, ...]
   - tesseract: pytesseract, --oem 1 --psm 3 -l eng (konfigurasi QS-OCR)
   - easyocr: Reader(["en"], gpu=True)
   - paddleocr: lang="en"
   - keras-ocr: OPSIONAL (lihat risiko)
   Fungsi reading_order(boxes): urutkan per baris (atas→bawah, kiri→kanan)
   lalu gabung jadi teks. Mesin berbasis kotak tidak menjamin urutan baca.
   Self-check pakai kotak sintetis, tanpa model OCR.
B. Simpan per mesin: data/interim/ocr/<mesin>/<doc_id>.json (kotak + teks).
   Bisa dilanjutkan: lewati dokumen yang sudah ada, catat detik per halaman.
C. Kualitas OCR tanpa transkripsi acuan (Tobacco3482 tidak punya CER):
   - jumlah token, halaman kosong
   - laju OOV terhadap kosakata FastText (pakai oov_rate yang sudah ada)
   - kesepakatan antar-mesin (Jaccard himpunan kata, kemiripan karakter)
   - Tesseract-JPG vs QS-OCR (Tesseract-TIF): mengukur efek JPG vs TIF
   Laporkan sebagai PROKSI, bukan akurasi OCR.
D. Notebook 10_ocr_multi_mesin.ipynb. Audit resolusi citra JPG di sel pertama,
   karena resolusi menentukan kualitas OCR.

Risiko yang harus diputuskan sebelum menulis kode:
- Tesseract di Windows bukan paket pip. Notebook mencoba memasangnya sendiri
  (winget); kalau gagal, dokumentasikan langkah manualnya.
- paddlepaddle-gpu besar (wheel PyPI 2.6.2 saja 758 MB): tanya dulu (aturan 4).
  Alternatif: versi CPU.
- Keras-OCR berbasis TensorFlow; TF GPU tidak didukung di Windows native sejak
  versi 2.11, dan rilis terakhir pustakanya 0.9.3 (November 2023). Kemungkinan hanya CPU
  dan lambat. Boleh dilewati dengan alasan tertulis.
Estimasi: ±1–3 jam per mesin untuk 3.482 halaman di RTX 3050. Jalankan di
laptop compute, bukan di mesin agen.

Catatan eksperimen 13-ocr-multi-mesin.md. Log + commit. Berhenti.
```

## Tahap 11 — Grid teks: OCR × embedding

```text
TAHAP 11: Cabang teks untuk setiap kombinasi.

Kondisi OCR: qs-ocr (Tesseract-TIF, acuan Fase 1), tesseract-jpg, easyocr,
paddleocr (+ keras-ocr bila ada).
Kondisi embedding:
- FastText + CNN1D: pipeline Fase 1 apa adanya (src/text_features.py,
  src/models/text_models.py). Fitur per mesin ke memmap terpisah.
- BERT-base fine-tune (src/models/bert_text.py), maksimal 512 token, AMP,
  micro-batch + akumulasi gradien agar muat di 4 GB. Fallback DistilBERT
  bila tidak muat, dan catat sebagai deviasi dari Kashyap.
Teks masuk TANPA praproses gaya Kashyap (buang angka/stopword) di kedua
embedding, supaya efek embedding tidak tercampur efek praproses.

3 seed per sel. Laporkan OA, macro F1, F1 per kelas, McNemar antar-sel per
seed. Sel qs-ocr × FastText harus mereproduksi CNN1D Fase 1 (0,7266 ± 0,0059);
kalau tidak, ada yang rusak, berhenti dan laporkan.

Catatan eksperimen per embedding + tabel grid di vault/50-hasil/. Tanya dulu
sebelum menjalankan bila estimasi total >15 menit (pasti). Log + commit. Berhenti.
```

## Tahap 12 — Fusion per kondisi OCR

```text
TAHAP 12: Apakah teks yang lebih kuat membuat fusion memakai teks?

A. Aturan fusion Kashyap: rata-rata logit TEXT dan IMAGE. Tanpa training, jadi
   jalankan untuk SEMUA sel Tahap 11 × 3 seed. Checkpoint IMAGE dari Fase 1.
B. Fusion fitur dengan arsitektur yang sudah diperbaiki (LayerNorm per cabang +
   inisialisasi dari checkpoint teks, src/models/fusion.py) hanya untuk kondisi
   OCR terbaik dan terburuk, embedding FastText. Seed 42 dulu; tiga seed bila
   waktunya cukup.
C. Untuk setiap model fusion: OA, macro F1, oracle, dan uji missing modality
   (nolkan teks / nolkan citra, src/ablation.py).
Pertanyaan yang harus dijawab: apakah penurunan saat teks dinolkan membesar
seiring akurasi cabang teks? Apakah rata-rata logit lebih tahan terhadap
cabang yang timpang daripada fusion fitur?

Catatan eksperimen + tabel. Log + commit. Berhenti.
```

## Tahap 13 — Sintesis Fase 2

```text
TAHAP 13: Jawab Q1–Q3 di vault/50-hasil/sintesis-fase-2.md dengan link ke
catatan sumber tiap angka. Perbarui limitasi-dan-lanjutan.md. Siapkan bahan
laporan/presentasi: tabel grid, kurva missing modality vs akurasi teks.
Log + commit. Berhenti.
```

## Tahap 14 — Teks sebagai objek (tahap lanjutan dari dosen)

```text
TAHAP 14: Rancangan dulu, jangan langsung implementasi.

Bahan: kotak teks yang sudah disimpan di Tahap 10, jadi tidak perlu OCR ulang.
Kandidat representasi:
- posisi kotak sebagai fitur 2D di samping embedding kata (gaya LayoutLM);
- peta kotak teks sebagai citra, yaitu teks digambar sebagai kotak berwarna
  (Noce et al., dikutip Kashyap);
- graf antar-kotak teks.
WAJIB: kontrol jalan pintas kode ID ([[larson-2025-id-codes]]). Kode ID adalah
objek teks berposisi tetap; model berbasis posisi bisa belajar jalan pintas itu.
Bandingkan dengan dan tanpa kotak kode ID.

Tulis rancangan + referensi ≥2025 untuk tahap ini. Berhenti, tunggu
persetujuan sebelum implementasi.
```

---

## Checklist eksekusi

- [x] Langkah 0 — CLAUDE.md dibuat dan diperiksa manual
- [x] Tahap 1 — Scaffolding + vault + git init
- [x] Tahap 2 — Sintesis jurnal (isi path PDF dulu)
- [x] Tahap 3 — Audit & pairing data
- [x] Tahap 4 — Split + fitur teks
- [x] Tahap 5 — Baseline TEXT dan IMAGE
- [x] Tahap 6 — FUSION + Oracle
- [x] Tahap 7 — Ablasi (ablasi 4 dilewati)
- [x] Tahap 8 — Sintesis akhir (plus 3 eksperimen lanjutan di luar rencana)
- [x] Tahap 9 — Revisi landasan: referensi ≥2025, gap, proposal (2026-10-09)
- [ ] Tahap 10 — Ekstraksi OCR multi-mesin
- [ ] Tahap 11 — Grid teks OCR × embedding
- [ ] Tahap 12 — Fusion per kondisi OCR + rata-rata logit
- [ ] Tahap 13 — Sintesis Fase 2
- [ ] Tahap 14 — Rancangan teks sebagai objek

## Catatan praktis

1. Kirim Langkah 0 sendirian dan baca `CLAUDE.md` yang dihasilkan sebelum lanjut. Semua tahap berikutnya bergantung padanya.
2. Path PDF sudah terisi di Tahap 2: `references/1907.06370v1.pdf`.
3. Model FastText `.bin` sekitar 7GB di disk dan ~15GB saat dimuat. `reduce_model` TIDAK menolong soal RAM karena harus memuat model penuh lebih dulu. Yang menolong: ekstraksi fitur dijalankan sekali ke disk lalu model dilepas. Lihat Tahap 4B.
4. Jangan berharap angka persis 87,8%. Selisih 2-4% wajar. Yang penting pola relatifnya konsisten: TEXT < IMAGE < FUSION < Oracle.
5. File ini sendiri bisa ditaruh di `vault/00-index/rencana-prompt.md` supaya ikut terindeks di graf Obsidian.
6. Siapkan kredensial Kaggle (`~/.kaggle/kaggle.json`) sebelum Tahap 3, kalau tidak agen akan mentok di langkah download.
7. Setelah install dependency, jalankan `python -m spacy download en_core_web_sm`. Model spaCy tidak ikut lewat pip.
