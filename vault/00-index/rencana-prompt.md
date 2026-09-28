---
judul: Rencana Prompt Bertahap - Implementasi Jurnal Multimodal
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: wip
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
---

# Rencana Prompt Bertahap

Kumpulan prompt untuk menjalankan implementasi paper Audebert et al. (arXiv:1907.06370) secara lokal dengan agen coding. Kirim satu blok per sesi, jangan digabung.

---

## Basis rujukan tiap perintah

Penting dipahami sebelum mulai, supaya tidak ada ekspektasi yang salah.

### Yang terverifikasi dari repo dataset

Repo `github.com/QuickSign/ocrized-text-dataset` adalah **repo dataset, bukan repo implementasi model**. Isinya: `to_text.py`, `tobacco3482.sh`, `rvl-cdip.sh`, `setup.sh`, `Dockerfile`, `Taskfile.yml`, `datasheet.md`/`datasheet.pdf`, `readme.md`, `requirements.txt`, `pyproject.toml`. Lisensi Apache-2.0. Semua script itu hanya untuk meregenerasi teks OCR dari citra, tidak ada kode training.

Fakta yang dipakai dalam prompt di bawah:

| Fakta | Nilai |
|---|---|
| Lokasi file dataset teks | Tab **Releases**, tag `v1.0`, 4 aset arsip. Bukan di root repo. |
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

### Yang terverifikasi dari Kaggle

`kaggle.com/datasets/patrickaudriaz/tobacco3482jpg`, judul "Tobacco3482", versi JPG dari dataset asli. Dataset asli berformat TIF dan dihosting di server UMIACS yang sering sulit diakses, sehingga versi Kaggle ini lazim dipakai sebagai pengganti.

**Konsekuensi yang harus ditulis sebagai limitasi:** teks QS-OCR di-OCR dari TIF asli, sedangkan citra kita dari JPG hasil re-encode. Sumber kedua modalitas tidak identik.

### Yang berbasis paper saja

Seluruh spesifikasi model: arsitektur, dimensi layer, optimizer, learning rate, jumlah epoch, batch size, panjang padding, strategi fusion, protokol split. Tidak ada kode resmi dari penulis untuk mengecek silang. Karena itu Tahap 2 mewajibkan pemisahan section "Ambiguitas & asumsi".

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
terkait: []  # daftar [[wikilink]]
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
````

---

## Tahap 1 — Scaffolding + vault

```text
TAHAP 1: Scaffolding.

Buat seluruh struktur folder sesuai CLAUDE.md, plus:
- .gitignore (data/, models/, *.ipynb_checkpoints, .obsidian/workspace*, venv)
- requirements.txt (PyTorch, torchvision, numpy, pandas, scikit-learn,
  matplotlib, seaborn, spacy, fasttext, pillow, tqdm, jupyter). Jangan install
  dulu, cukup tulis filenya.
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

Terakhir: git init + commit awal. Lalu berhenti dan laporkan.
```

---

## Tahap 2 — Sintesis jurnal ke vault

```text
TAHAP 2: Sintesis jurnal.

PDF papernya ada di [ISI PATH LOKAL ANDA]. Baca penuh, lalu:

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

A. src/splits.py: implementasi protokol paper — 800 dokumen train, sisanya
   test, terstratifikasi per kelas, 3 seed berbeda (42/43/44), disimpan sebagai
   data/processed/split_seed{N}.json agar identik di semua eksperimen.
   Sisihkan juga validation kecil dari train untuk early stopping.
   Catatan: dataset tidak menyediakan split resmi, jadi split ini buatan kita
   sendiri mengikuti deskripsi paper. Dokumentasikan.

B. src/text_features.py:
   - Tokenisasi + pembuangan punctuation pakai spaCy en_core_web_sm.
   - Embedding FastText. WAJIB memakai model .bin (cc.en.300.bin), BUKAN .vec.
     Hanya .bin yang bisa menginferensi vektor untuk kata out-of-vocabulary
     lewat subword, dan itulah inti argumen paper soal noise OCR. File ~7GB.
     Sediakan opsi reduce_model ke dimensi lebih kecil kalau RAM tidak cukup,
     dan catat konsekuensinya sebagai deviasi dari paper.
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
   di-warp, sesuai paper).

C. src/train.py: loop training generik, SGD+momentum sesuai paper, logging
   per-epoch ke CSV, checkpoint best-on-val, seed terkontrol.

D. notebooks/03_baseline_teks.ipynb — latih MLP dan CNN1D, bandingkan.
   Ini replikasi Tabel 1a paper.
   notebooks/04_baseline_citra.ipynb — latih MobileNetV2. Replikasi Tabel 1b.

Sebelum training penuh, estimasi durasinya dan laporkan ke saya. Kalau >15
menit per run, tawarkan mode cepat (epoch dikurangi) untuk smoke test dulu.

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
adaptif. Paper melaporkan penjumlahan gagal — kita uji sendiri, jangan
diasumsikan benar tanpa bukti dari run kita.

notebooks/05_fusion.ipynb: latih keduanya, 3 seed, laporkan rata-rata ± std.

Tambahkan perhitungan ORACLE: untuk tiap sampel test, benar bila cabang teks
ATAU cabang citra benar. Ini batas atas teoretis fusion dan WAJIB dilaporkan,
karena inilah bukti kuantitatif bahwa kedua modalitas saling melengkapi.

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

Kerjakan berurutan, laporkan tiap sub-eksperimen sebelum lanjut:

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

## Checklist eksekusi

- [ ] Langkah 0 — CLAUDE.md dibuat dan diperiksa manual
- [ ] Tahap 1 — Scaffolding + vault + git init
- [ ] Tahap 2 — Sintesis jurnal (isi path PDF dulu)
- [ ] Tahap 3 — Audit & pairing data
- [ ] Tahap 4 — Split + fitur teks (siapkan ruang ~7GB untuk cc.en.300.bin)
- [ ] Tahap 5 — Baseline TEXT dan IMAGE
- [ ] Tahap 6 — FUSION + Oracle
- [ ] Tahap 7 — Ablasi
- [ ] Tahap 8 — Sintesis akhir

## Catatan praktis

1. Kirim Langkah 0 sendirian dan baca `CLAUDE.md` yang dihasilkan sebelum lanjut. Semua tahap berikutnya bergantung padanya.
2. Ganti `[ISI PATH LOKAL ANDA]` di Tahap 2 dengan path PDF paper di mesin Anda.
3. Model FastText `.bin` berukuran sekitar 7GB. Kalau RAM di bawah 16GB, minta agen langsung memakai `reduce_model` ke 100 dimensi sejak awal dan mencatatnya sebagai deviasi eksplisit.
4. Jangan berharap angka persis 87,8%. Selisih 2-4% wajar. Yang penting pola relatifnya konsisten: TEXT < IMAGE < FUSION < Oracle.
5. File ini sendiri bisa ditaruh di `vault/00-index/rencana-prompt.md` supaya ikut terindeks di graf Obsidian.
