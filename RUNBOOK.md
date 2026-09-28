# Runbook — menjalankan di mesin compute

Kode ditulis di mesin lain dan **belum pernah dieksekusi**. Dokumen ini adalah
prosedur menjalankannya dari nol sampai seluruh catatan eksperimen terisi angka.

Semua catatan di `vault/30-eksperimen/` dan `vault/50-hasil/` saat ini berstatus
`belum-dijalankan` dengan field angka kosong. Itu disengaja — aturan 3 di
`CLAUDE.md` melarang mengisi angka yang belum dihitung.

---

## 0. Yang perlu disiapkan

| Kebutuhan | Ukuran | Catatan |
|---|---|---|
| Disk kosong | **± 15 GB** | citra 3,3 GB + FastText 7 GB + sekuens 1 GB + checkpoint ± 1 GB |
| RAM | **16 GB minimum** | FastText butuh ± 15 GB saat dimuat, hanya di notebook 02 |
| GPU | opsional tapi sangat membantu | tanpa GPU, notebook 04-05 bisa berjam-jam |
| Python | **3.12** | 3.13/3.14 belum punya wheel `torch`/`spacy`/`fasttext` |
| Akun Kaggle | — | butuh API token untuk mengunduh dataset citra |

### Profil yang sudah diperhitungkan: ASUS TUF, i7 gen 11, 24 GB RAM, RTX 3050 4 GB

| Aspek | Status |
|---|---|
| RAM 24 GB | **Aman.** FastText butuh ± 15 GB, masih lapang |
| Disk | perlu ± 15 GB kosong |
| VRAM 4 GB | **Ini kendalanya** — lihat di bawah |
| CUDA | RTX 3050 = Ampere, punya tensor core, jadi AMP aktif otomatis |

**Batch 40 pada MobileNetV2 384×384 tidak muat di 4 GB VRAM.** Parameternya kecil
(9 MB), yang memakan adalah peta aktivasi yang harus disimpan untuk backward —
kasarnya 60-80 MB per sampel, jadi batch 40 ≈ 2,5-3 GB sebelum menghitung workspace
dan fragmentasi. Di kartu 4 GB yang efektifnya ± 3,6 GB, itu akan OOM.

Solusinya sudah terpasang: **akumulasi gradien**. Notebook 04 dan 05 punya sel

```python
CONFIG["micro_batch_size"] = 8
```

tepat sebelum sel `EPOCHS`. DataLoader menyajikan micro-batch 8, `train_model`
mengakumulasi 5 di antaranya, dan optimizer tetap melangkah dari 40 sampel persis
seperti paper. Kalau masih OOM, turunkan ke 4.

**Yang tidak setara, dan wajib dicatat sebagai deviasi:** BatchNorm menormalisasi
per micro-batch, jadi statistiknya berasal dari 8 sampel, bukan 40. Akumulasi
gradien tidak bisa memperbaiki itu. Self-check `src/train.py` memverifikasi bagian
yang memang setara — gradien hasil akumulasi identik dengan gradien batch penuh pada
model tanpa BatchNorm.

**AMP (mixed precision) menyala otomatis di CUDA**, memangkas memori aktivasi
sekitar separuh dan mempercepat 2-3× di Ampere. Matikan dengan
`train_model(..., amp=False)` kalau mencurigai masalah numerik.

Perkiraan durasi di 3050 (kasar, **ukur sendiri dengan sel estimasi**): IMAGE
200 epoch ± 1-1,5 jam per seed; FUSION lebih berat, ± 2 jam per seed per strategi.
Total untuk 3 seed penuh ± 12-18 jam. Lihat urutan pemotongan di bagian 5.

### Kalau RAM hanya 16 GB

Notebook 02 akan kena swap saat memuat `cc.en.300.bin`. Biarkan — ia hanya berjalan
sekali, lalu fitur ditulis ke disk dan model dilepas. **Jangan** memakai
`fasttext.util.reduce_model` sebagai jalan keluar: fungsi itu harus memuat model
300 dimensi penuh lebih dulu, jadi puncak RAM-nya sama saja.

Tutup aplikasi lain sebelum menjalankan notebook 02, dan pastikan swap/pagefile
cukup besar.

---

## 1. Ambil kode

```bash
git clone https://github.com/RamaAlqdri/apdm-document-classification.git
cd apdm-document-classification
```

Jangan menyalin `data/processed/manifest.csv` dari mesin lain: isinya **path
absolut** milik mesin yang membuatnya. File itu dibuat ulang oleh notebook 01.

---

## 2. Environment

### Linux / WSL2

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### PyTorch dengan CUDA

`requirements.txt` memasang `torch` versi default, yang di Linux/Windows **belum
tentu** membawa CUDA. Periksa:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kalau `False` padahal ada GPU NVIDIA, pasang ulang dari indeks CUDA — sesuaikan
`cu124` dengan versi driver Anda (`nvidia-smi` menunjukkan CUDA version):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

`src/train.py:pick_device` memilih `cuda` lebih dulu, jadi tidak ada yang perlu
diubah di kode.

---

## 3. Kredensial Kaggle

Buat API token di kaggle.com → Settings → API → Create New Token. Simpan
`kaggle.json`:

- Linux/WSL: `~/.kaggle/kaggle.json`, lalu `chmod 600 ~/.kaggle/kaggle.json`
- Windows: `C:\Users\<nama>\.kaggle\kaggle.json`

---

## 4. Verifikasi sebelum menyentuh data

Sepuluh modul punya self-check yang berjalan **tanpa dataset**. Jalankan semuanya
dulu — kalau ada yang gagal di sini, tidak ada gunanya mengunduh 3 GB:

```bash
for m in config data splits text_features datasets ablation models.text_models models.image_model models.fusion train; do python -m src.$m; done
```

Windows PowerShell:

```powershell
foreach ($m in "config","data","splits","text_features","datasets","ablation","models.text_models","models.image_model","models.fusion","train") { python -m src.$m }
```

Harus muncul sepuluh baris `... self-check ok`.

---

## 5. Urutan menjalankan notebook

```bash
jupyter lab
```

Jalankan **berurutan**. Tiap notebook bergantung pada keluaran sebelumnya.

| # | Notebook | Perkiraan | Menghasilkan |
|---|---|---|---|
| 01 | `01_data_audit.ipynb` | 15-30 menit (mayoritas download 3,3 GB) | `manifest.csv`, figur EDA |
| 02 | `02_text_features.ipynb` | 30-90 menit (+ download FastText 7 GB) | sekuens 500×300, vektor SIF, split |
| 03 | `03_baseline_teks.ipynb` | menit-an di GPU | checkpoint MLP + CNN1D, prediksi TEXT |
| 04 | `04_baseline_citra.ipynb` | **jam-an** | checkpoint IMAGE, prediksi IMAGE |
| 05 | `05_fusion.ipynb` | **paling berat** | checkpoint FUSION, tabel utama |
| 06 | `06_ablasi.ipynb` | menit-an, tanpa training | CSV + figur ablasi |

Angka di atas kasar. **Notebook 04, 05, dan 06 masing-masing punya sel estimasi
durasi** yang menjalankan 2 epoch lalu mengekstrapolasi ke target. Jalankan sel itu
dan baca angkanya sebelum melepas training penuh.

### Kalau terlalu lama

Tiap notebook punya sel `EPOCHS` dan `SEEDS` tepat setelah estimasi. Turunkan di
situ. Notebook mengumpulkan pemotongan ke variabel `DEVIASI` dan mencetaknya di
akhir — **salin ke catatan eksperimen**. Jangan dipotong diam-diam; seluruh nilai
proyek ini bergantung pada deviasi yang tercatat.

Urutan pemotongan yang paling murah dampaknya:

1. Kurangi `SEEDS` jadi `[42]` untuk strategi fusion `sum` (paper pun hanya
   menyebutnya eksperimen pendahuluan)
2. Turunkan `EPOCHS` untuk IMAGE dan FUSION dari 200 ke 60-80 — ada early stopping
   `patience=15`, jadi kemungkinan besar berhenti sendiri sebelum 200
3. Terakhir baru kurangi `SEEDS` untuk model utama, karena itu menghapus ± std

---

## 6. Setelah selesai: isi catatan

Tiap notebook mencetak blok ringkasan siap-salin di sel terakhir. Untuk tiap
catatan: isi tabelnya, isi field `oa::`/`macro_f1::`/`durasi::` di bagian atas, lalu
ubah `status: belum-dijalankan` menjadi `status: selesai`.

| Notebook | Catatan yang diisi |
|---|---|
| 01 | `vault/30-eksperimen/00-audit-data.md` |
| 02 | `vault/30-eksperimen/01-ekstraksi-fitur-teks.md` |
| 03 | `02-baseline-mlp-sif.md`, `03-baseline-cnn1d.md` |
| 04 | `04-baseline-mobilenetv2.md` |
| 05 | `05-fusion-concat.md`, `06-fusion-sum.md`, `vault/50-hasil/tabel-utama.md` |
| 06 | `07-ablasi-degradasi-citra.md`, `08-ablasi-missing-modality.md`, `09-ablasi-degradasi-teks.md` |

Beberapa catatan sudah memuat **hipotesis yang ditulis sebelum eksekusi**. Biarkan
apa adanya meski meleset — hipotesis yang salah lebih berharga daripada hipotesis
yang diedit setelah melihat hasilnya.

Setelah semua terisi, Tahap 8 (sintesis akhir) baru bisa dikerjakan.

---

## 7. Masalah yang sudah diantisipasi

**`DataLoader` hang atau crash di Windows.** `CONFIG["num_workers"]` otomatis 0 di
Windows karena multiprocessing di dalam kernel Jupyter sering menggantung di sana.
Kalau Anda pakai WSL2, nilainya 4 dan semuanya normal. WSL2 memang lebih disarankan.

**`symlink` gagal di Windows.** Windows menolak symlink tanpa Developer Mode atau
hak admin. `download_images` menangkapnya, mencetak peringatan, dan menulis lokasi
cache ke `data/raw/LOKASI_CITRA.txt`. Tidak ada yang rusak — symlink itu kosmetik.

**Notebook 01 gagal di `assert rec["n_kelas_cocok"] == 10`.** Berarti nama folder
kelas di arsip tidak tercakup `CLASS_ALIASES`. Notebook mencetak nama yang tidak
dikenali. Tambahkan aliasnya di `src/data.py`, **jangan** diakali di sel notebook —
normalisasi diam-diam persis hal yang audit ini dimaksudkan untuk menangkap.

**Notebook 02 kehabisan RAM.** Lihat bagian 0. Tutup aplikasi lain; jangan pakai
`reduce_model`.

**`torch.OutOfMemoryError` di notebook 04/05.** Turunkan
`CONFIG["micro_batch_size"]` ke 4, lalu restart kernel — VRAM tidak dilepas
sepenuhnya setelah OOM. Kalau masih gagal, pastikan tidak ada proses lain memakai
GPU (`nvidia-smi`), dan tutup notebook sebelumnya: kernel Jupyter yang masih hidup
tetap memegang VRAM-nya.

**Notebook 06 gagal memuat checkpoint.** Nama filenya mengikuti seed dan strategi:
`IMAGE_seed42.pt`, `CNN1D_seed42.pt`, `FUSION-concat_seed42.pt`. Kalau Anda mengubah
`SEEDS` di notebook sebelumnya, ubah juga `SEED` di sel pertama notebook 06.

**Injeksi typo di notebook 06 dilewati.** Itu normal kalau `cc.en.300.bin` sudah
dihapus untuk menghemat disk. Notebook mencetak peringatan. Catat di
`09-ablasi-degradasi-teks.md` bahwa bagian itu tidak dikerjakan.
