---
judul: Missing modality lanjutan — pengganti dalam distribusi dan head tanpa teks kosong
tipe: eksperimen
tahap: "12"
tanggal: 2026-10-24
status: selesai
tags:
  - tipe/eksperimen
  - tahap/12
  - modal/fusion
  - komponen/fusion
  - komponen/evaluasi
  - temuan/positif
  - temuan/negatif
  - temuan/anomali
terkait:
  - "[[16-fusi-per-ocr]]"
  - "[[14-kurasi-dan-eda]]"
  - "[[08-ablasi-missing-modality]]"
  - "[[limitasi-dan-lanjutan]]"
  - "[[00-audit-data]]"
  - "[[sintesis-fase-2]]"
  - "[[peta-eksperimen]]"
---

# Missing modality lanjutan

model:: LATE dan CONCAT dengan modalitas diganti rerata latih; CONCAT_nonkosong = ConcatHead dilatih ulang tanpa dokumen berteks kosong
dataset:: Tobacco3482, 4 kondisi OCR (cache keluaran unimodal [[16-fusi-per-ocr]])
split:: 800 latih terstratifikasi (80 val) / 2.682 uji, sama dengan [[16-fusi-per-ocr]]
seed:: 42, 43, 44
oa:: CONCAT tanpa teks, pengganti kosong → rerata: 0.2539 → 0.7089 / 0.2260 → 0.6868 / 0.6798 → 0.7912 / 0.5162 → 0.7900 (qs-ocr / tesseract / easyocr / paddleocr; IMAGE 0.8169)
macro_f1:: CONCAT tanpa teks rerata 0.6726 / 0.6545 / 0.7590 / 0.7295
durasi:: 133 dtk untuk seluruh 12 sel (CPU)

> **DIJALANKAN** di mesin compute lewat sel "H3 lanjutan" `notebooks/pipeline.ipynb`, hasil
> di-push 2026-10-24 (`f4cbe39`). Semua cache unimodal dipakai ulang (`dari_cache` = True),
> *head* CONCAT yang dilatih ulang identik dengan bagian 8. Angka disalin dari
> `reports/fase2/uji_lanjutan_missing_modality.csv`, `uji_lanjutan_selisih.csv`,
> `uji_lanjutan_porsi_note.csv` dan output sel; rerata dan simpangan baku antar-seed
> (populasi) dihitung dari CSV per seed saat analisis.

## Tujuan

Menguji dugaan di [[16-fusi-per-ocr]] temuan 2. Di sana CONCAT tanpa teks jatuh jauh di
bawah IMAGE sendiri, dan semua versi "tanpa citra" jatuh di bawah TEXT sendiri. Kehilangan
satu modalitas tidak mungkin lebih buruk daripada unimodal modalitas yang tersisa, kecuali
masukan penggantinya (keluaran model untuk masukan kosong) keluar distribusi. Dua
pertanyaan:

1. Apakah kejatuhan itu artefak masukan pengganti? Kalau ya, pengganti yang berada di
   dalam distribusi seharusnya membuat fusi tanpa satu modalitas mendarat dekat unimodal
   modalitas yang tersisa.
2. Apakah *head* CONCAT belajar "teks kosong ⇒ Note"? 14,4 % halaman Note pada QS-OCR
   kosong ([[14-kurasi-dan-eda]]), jadi vektor teks kosong di data latih hampir selalu
   berlabel Note.

## Konfigurasi

- Cache dan split sama dengan [[16-fusi-per-ocr]]; tidak ada model unimodal yang dilatih
  ulang. *Head* CONCAT asli dilatih ulang dengan seed yang sama; notebook memeriksa bahwa
  prediksinya identik dengan bagian 8.
- **Pengganti rerata** (`_rerata`): modalitas yang hilang diganti rata-rata fiturnya
  (CONCAT) atau logitnya (LATE) atas dokumen latih *head*. Untuk LATE ini sama dengan
  menambah bias prior kelas dari modalitas yang hilang.
- **Teks kosong** = dokumen yang fitur teksnya sama dengan fitur masukan kosong (toleransi
  1e-3 relatif), yaitu kosong menurut model: halaman 0 kata atau semua token terbuang saat
  tokenisasi. Notebook mencetak cek silang dengan jumlah halaman 0 kata dari bagian 5.1.
- **CONCAT_nonkosong**: *head* kedua, dilatih dengan dokumen teks kosong dibuang dari
  latih dan val *head*. Set uji tidak berubah (dokumen uji kosong tetap dievaluasi).
- Tambahan ukuran: porsi prediksi Note pada set uji, dan kelas yang paling banyak
  diprediksi, untuk tiap varian tanpa teks.

## Hipotesis (ditulis sebelum eksekusi)

Dinilai pada rerata 3 seed per kondisi OCR. Ambang 5 poin dipilih sebelum eksekusi:
setara dua kali simpangan baku antar-seed IMAGE terbesar di [[16-fusi-per-ocr]] (0,0257).

**H4a — kejatuhan adalah artefak masukan pengganti.** Didukung bila di keempat kondisi:
|CONCAT_tanpa_teks_rerata − IMAGE| ≤ 0,05, LATE_tanpa_citra_rerata ≥ TEXT − 0,05, dan
CONCAT_tanpa_citra_rerata ≥ TEXT − 0,05. Bila CONCAT tanpa teks tetap jauh di bawah IMAGE
dengan pengganti rerata, *head* memang tidak bisa bekerja tanpa teks, dan itu temuan
tentang *head*, bukan tentang ukuran.

**H4b — *head* belajar "teks kosong ⇒ Note".** Prasyarat: |CONCAT_nonkosong − CONCAT| ≤
0,01 (membuang ± 1 % data latih tidak merusak *head*). Didukung bila (i) porsi prediksi
Note CONCAT_tanpa_teks ≥ porsi Note sebenarnya + 0,10, dan (ii) pada CONCAT_nonkosong porsi
itu turun ke ≤ porsi sebenarnya + 0,05, dan (iii) jarak CONCAT_nonkosong_tanpa_teks ke IMAGE
menyusut paling sedikit separuh dibanding CONCAT_tanpa_teks. Tafsir alternatif yang
dibedakan:

- (i) gagal: kejatuhan tidak menarik ke Note; dugaan gugur sejak awal.
- (i) lolos, (ii) gagal: tarikan ke Note tidak berasal dari dokumen kosong di data latih.
- (i) dan (ii) lolos, tetapi kelas terbanyak pindah ke kelas lain dan (iii) gagal:
  masukan kosong keluar distribusi secara umum; *head* tidak belajar aturan khusus Note,
  ia hanya berekstrapolasi.

Batasan yang sudah diketahui: H4b hanya bermakna pada kondisi yang punya dokumen kosong
di bagian latih. Menurut [[16-fusi-per-ocr]] halaman kosong 1,06 % (qs-ocr), 0,60 %
(tesseract), 0,17 % (easyocr), 0,06 % (paddleocr) dari seluruh dokumen, jadi easyocr dan
paddleocr kemungkinan punya 0-2 dokumen kosong di 720 dokumen latih; di sana CONCAT_nonkosong
praktis sama dengan CONCAT. Notebook mencetak `n_kosong_latih` per sel.

## Hasil

### Dokumen kosong menurut model

| Kondisi | kosong menurut model | halaman 0 kata (bagian 5.1) | Note | Advertisement | di latih *head* (seed 42/43/44) |
|---|---|---|---|---|---|
| qs-ocr | 27 | 37 | 20 | 7 | 4 / 8 / 6 |
| tesseract | 11 | 21 | 6 | 5 | 2 / 4 / 2 |
| easyocr | 0 | 6 | 0 | 0 | 0 |
| paddleocr | 0 | 2 | 0 | 0 | 0 |

Mask sama di ketiga seed. Jumlahnya **lebih kecil** daripada halaman 0 kata, kebalikan dari
yang diperkirakan di Konfigurasi. Dugaan (belum diverifikasi): penghitung kata OCR hanya
menghitung kata alfabetis, sedangkan tokenizer tetap menyimpan angka, jadi halaman berisi
angka saja tidak kosong bagi model. Di val *head* hanya 0-1 dokumen kosong per sel.

### OA rerata seed (± simpangan baku antar-seed)

| Kondisi | IMAGE | TEXT | CONCAT tanpa teks, kosong | CONCAT tanpa teks, rerata | CONCAT_nonkosong tanpa teks, kosong | LATE tanpa teks, kosong | LATE tanpa teks, rerata |
|---|---|---|---|---|---|---|---|
| qs-ocr | 0,8169 | 0,7266 | 0,2539 ± 0,1160 | 0,7089 ± 0,0324 | 0,3213 ± 0,1273 | 0,7949 | 0,6847 ± 0,0694 |
| tesseract | 0,8169 | 0,6970 | 0,2260 ± 0,0530 | 0,6868 ± 0,0487 | 0,1839 ± 0,0114 | 0,8086 | 0,6882 ± 0,0231 |
| easyocr | 0,8169 | 0,7155 | 0,6798 ± 0,0707 | 0,7912 ± 0,0286 | 0,6798 ± 0,0707 | 0,8160 | 0,7412 ± 0,0488 |
| paddleocr | 0,8169 | 0,7514 | 0,5162 ± 0,0664 | 0,7900 ± 0,0088 | 0,5162 ± 0,0664 | 0,8143 | 0,7425 ± 0,0149 |

LATE tanpa teks (kosong) = IMAGE + selisih di `uji_lanjutan_selisih.csv` (−0,0220 / −0,0083 /
−0,0009 / −0,0026). Pada easyocr dan paddleocr tidak ada dokumen yang dibuang, jadi
CONCAT_nonkosong = CONCAT persis.

### Selisih terhadap unimodal modalitas yang tersisa (rerata seed)

| Selisih | qs-ocr | tesseract | easyocr | paddleocr |
|---|---|---|---|---|
| CONCAT tanpa teks, kosong − IMAGE | −0,5630 | −0,5910 | −0,1371 | −0,3008 |
| CONCAT tanpa teks, rerata − IMAGE | −0,1080 | −0,1301 | −0,0257 | −0,0270 |
| CONCAT_nonkosong tanpa teks, kosong − IMAGE | −0,4957 | −0,6330 | −0,1371 | −0,3008 |
| CONCAT_nonkosong tanpa teks, rerata − IMAGE | −0,0997 | −0,1480 | −0,0257 | −0,0270 |
| LATE tanpa teks, rerata − IMAGE | −0,1322 | −0,1288 | −0,0757 | −0,0744 |
| LATE tanpa citra, kosong − TEXT | −0,2678 | −0,2711 | −0,3368 | −0,3485 |
| LATE tanpa citra, rerata − TEXT | −0,0285 | −0,0358 | −0,0403 | −0,0424 |
| CONCAT tanpa citra, kosong − TEXT | −0,1115 | −0,1346 | −0,2018 | −0,2160 |
| CONCAT tanpa citra, rerata − TEXT | −0,0045 | +0,0061 | −0,0172 | −0,0142 |
| CONCAT_nonkosong − CONCAT (masukan lengkap) | +0,0062 | +0,0031 | 0 | 0 |

Per seed, CONCAT tanpa teks rerata − IMAGE berkisar −0,200 (tesseract seed 42) sampai
−0,003 (paddleocr seed 44); LATE tanpa teks rerata − IMAGE di setiap sel negatif, −0,055
sampai −0,194.

### Porsi prediksi Note pada set uji (porsi sebenarnya 0,058)

| Kondisi | CONCAT | tanpa teks, kosong | tanpa teks, rerata | nonkosong, tanpa teks kosong | nonkosong, tanpa teks rerata |
|---|---|---|---|---|---|
| qs-ocr | 0,075 | 0,727 (0,463 / 0,942 / 0,776) | 0,072 | 0,483 (0,262 / 0,892 / 0,294) | 0,058 |
| tesseract | 0,067 | 0,756 (0,703 / 0,738 / 0,826) | 0,084 | 0,817 (0,743 / 0,864 / 0,843) | 0,100 |
| easyocr | 0,061 | 0,192 (0,255 / 0,131 / 0,191) | 0,043 | sama dengan CONCAT | 0,043 |
| paddleocr | 0,057 | 0,320 (0,044 / 0,532 / 0,383) | 0,059 | sama dengan CONCAT | 0,059 |

Dalam kurung: seed 42 / 43 / 44. Dengan teks kosong, kelas terbanyak adalah Note di 9 dari
12 sel *head* asli (sisanya Email dua kali dan Memo sekali) dan di 7 dari 12 sel *head*
nonkosong.

### Penilaian kriteria pra-registrasi

| Kriteria | qs-ocr | tesseract | easyocr | paddleocr |
|---|---|---|---|---|
| H4a: \|CONCAT tanpa teks rerata − IMAGE\| ≤ 0,05 | tidak | tidak | ya | ya |
| H4a: LATE tanpa citra rerata ≥ TEXT − 0,05 | ya | ya | ya | ya |
| H4a: CONCAT tanpa citra rerata ≥ TEXT − 0,05 | ya | ya | ya | ya |
| H4b prasyarat: \|CONCAT_nonkosong − CONCAT\| ≤ 0,01 | ya | ya | ya | ya |
| H4b (i): Note membengkak pada *head* asli | ya | ya | ya | ya |
| H4b (ii): Note normal pada *head* nonkosong | tidak | tidak | tidak | tidak |
| H4b (iii): jarak ke IMAGE menyusut separuh | tidak | tidak | tidak | tidak |

## Temuan

1. **#temuan/positif — H4a didukung untuk "tanpa citra", sebagian untuk CONCAT tanpa
   teks.** Dengan pengganti rerata, kedua aturan tanpa citra mendarat paling jauh 4,2 poin
   di bawah TEXT di semua kondisi (dengan pengganti kosong: 11-35 poin di bawahnya). Kejatuhan "tanpa
   citra" di [[16-fusi-per-ocr]] karena itu artefak masukan pengganti. Untuk CONCAT tanpa
   teks, pengganti rerata memulihkan sebagian besar kejatuhan (qs-ocr −0,563 → −0,108,
   tesseract −0,591 → −0,130) dan mendarat dekat IMAGE pada easyocr dan paddleocr
   (−0,026 / −0,027). Pada qs-ocr dan tesseract masih tersisa 11-13 poin di bawah IMAGE,
   dengan sebaran besar antar-seed (−0,03 sampai −0,20): di sana *head* tidak bisa bekerja
   setara IMAGE tanpa teks, walau penggantinya titik pusat data latih.
2. **#temuan/negatif — H4b tidak didukung: tarikan ke Note tidak berasal dari dokumen
   kosong di data latih.** Kriteria (i) lolos, (ii) gagal di keempat kondisi, jadi menurut
   aturan tafsir yang ditulis sebelum eksekusi, dugaan "teks kosong ⇒ Note" gugur. Bukti
   paling langsung: easyocr dan paddleocr tidak punya satu pun dokumen kosong, tetapi teks
   kosong tetap mendorong porsi Note ke 0,19 dan 0,32. Membuang dokumen kosong menurunkan
   tarikan pada qs-ocr di 2 dari 3 seed (0,463 → 0,262; 0,776 → 0,294) tetapi tidak ke
   normal, dan menaikkannya pada tesseract (0,756 → 0,817). Dengan pengganti rerata, porsi
   Note pada *head* asli kembali normal (0,04-0,09 per sel), jadi tarikan itu milik vektor teks kosong,
   bukan sifat *head* terhadap Note secara umum. Dugaan baru, **belum diuji**: 48,3 % halaman
   Note berisi < 20 kata ([[00-audit-data]]), jadi Note adalah kelas yang teksnya paling pendek, sehingga fiturnya yang paling dekat ke fitur teks kosong, dan
   *head* berekstrapolasi dari kedekatan itu. Besarnya tarikan berubah tajam antar-seed
   (paddleocr: 0,044 / 0,532 / 0,383), sesuai dengan ekstrapolasi di luar distribusi.
3. **#temuan/anomali — pengganti rerata logit merusak LATE tanpa teks.** LATE tanpa teks
   dengan pengganti kosong mendarat di IMAGE (≤ 2,2 poin), tetapi dengan rerata logit latih
   TEXT ia jatuh 7-13 poin di bawah IMAGE (rerata kondisi; per sel 5,5-19,4 poin, negatif di
   setiap sel). Rerata logit bukan pengganti
   netral: ia menambah bias kelas tetap ke logit citra, dan logit TEXT pada dokumen latih
   berasal dari model yang dilatih pada dokumen itu sendiri (batasan H.4 di
   [[limitasi-dan-lanjutan]]). Jadi untuk LATE, pengganti kosong yang lebih sah; untuk
   CONCAT dan semua versi tanpa citra, pengganti rerata yang lebih sah. Kasus ini tidak
   tercakup kriteria H4a dan dicatat sebagai temuan, bukan disembunyikan.
4. Prasyarat aman: membuang 2-8 dokumen kosong dari 720 dokumen latih menggeser CONCAT
   dengan masukan lengkap +0,3 sampai +0,6 poin, di dalam simpangan baku antar-seed CONCAT.
5. **Ukuran pemakaian modalitas yang dipakai mulai sekarang:** "turun tanpa teks" LATE dengan
   pengganti kosong (seperti di [[16-fusi-per-ocr]]); CONCAT tanpa teks dan semua "tanpa
   citra" dengan pengganti rerata. Dengan ukuran ini CONCAT tanpa teks tetap paling dalam
   pada qs-ocr/tesseract (−0,108 / −0,130) dan dangkal pada easyocr/paddleocr (−0,026 /
   −0,027).
