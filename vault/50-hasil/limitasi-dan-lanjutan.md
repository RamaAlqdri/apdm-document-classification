---
judul: Limitasi dan Arah Lanjutan
tipe: hasil
tahap: "08"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/hasil
  - tahap/08
  - komponen/evaluasi
terkait:
  - "[[sintesis-akhir]]"
  - "[[tabel-utama]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
  - "[[taksonomi-multimodal]]"
  - "[[12-perbaikan-norm-dan-init]]"
---

# Limitasi dan Arah Lanjutan

Daftar ini disusun untuk bisa dipakai menilai seberapa jauh kesimpulan di
[[sintesis-akhir]] boleh dipercaya. Yang lemah ditulis sebagai lemah.

---

## A. Limitasi data

**A1. Sumber kedua modalitas tidak identik.** Teks QS-OCR di-OCR dari TIF asli, sedangkan
citra kami JPG hasil re-encode dari Kaggle. Dataset TIF asli dihosting di server UMIACS
yang sering sulit diakses. Artinya teks kami sedikit "lebih baik" daripada yang akan
dihasilkan OCR atas citra yang benar-benar kami pakai. Arah biasnya menguntungkan cabang
teks, jadi temuan bahwa cabang teks tidak terpakai **tidak** bisa dijelaskan oleh ini.

**A2. Hanya Tobacco3482, tanpa RVL-CDIP.** Paper memverifikasi pada 400.000 dokumen; kami
pada 3.482. Semua kesimpulan kami berlaku pada rezim data kecil — 720 sampel train setelah
validation dipotong. Beberapa temuan kami kemungkinan **spesifik untuk rezim itu**,
terutama kolapsnya cabang teks: dengan 720 sampel, cabang yang dilatih dari nol punya
sedikit kesempatan melawan cabang yang sudah pretrained.

**A3. Kelas timpang 5,17:1** (Memo 620, Resume 120). Itu sebabnya OA dan macro F1 bisa
bergerak ke arah berbeda, seperti yang terjadi di [[12-perbaikan-norm-dan-init]].

**A4. Sebagian kelas nyaris tidak punya teks.** 48,3% dokumen Note dan 18,3% Advertisement
punya kurang dari 20 kata; 26 dokumen (0,7%) teksnya kosong sepenuhnya. Pada kelas itu
tidak ada arsitektur yang bisa menolong.

---

## B. Limitasi protokol

**B1. Tiga random split terstratifikasi, bukan k-fold cross-validation.** Paper memakai
k-fold; kami tidak. Perbandingan antar-model kami tetap adil karena splitnya identik, tapi
estimasi variansnya lebih lemah.

**B2. Early stopping yang tidak direncanakan, dan validation 80 sampel.** Ini limitasi
terparah di seluruh proyek. `EPOCHS` diset 200 di semua notebook, tapi early stopping
`patience=15` menghentikan training di epoch 26-61 — sekitar **15% anggaran paper**.
Penyebabnya validation hanya 80 sampel, granularitas 1,25% per sampel, sehingga `val_oa`
berderau dan patience menyala karena derau.

Bukti konkretnya: IMAGE seed 42 mencatat `val_oa` 0,9250 di epoch 22 lalu checkpoint itu
disimpan; test-nya 0,8315 — jarak sembilan poin. Seleksi checkpointnya overfit ke 80 sampel
tersebut.

**B3. Uji lanjutan hanya satu seed.** [[10-uji-lanjutan-early-stopping]],
[[11-diagnostik-cabang-teks]], dan [[12-perbaikan-norm-dan-init]] semuanya seed 42.
Penurunan OA 0,0194 pada model yang diperbaiki berada di dekat std antar-seed concat
(0,0105), jadi **arah penurunannya tidak bisa dipastikan** tanpa seed tambahan. Itu
kelemahan paling langsung pada temuan terbaru kami.

**B4. Kedua perbaikan diterapkan bersamaan.** LayerNorm dan inisialisasi cabang teks
dijalankan dalam satu run, jadi **atribusi per perbaikan tidak diketahui**. Memisahkannya
butuh dua run lagi (± 6,6 jam).

---

## C. Limitasi lingkungan dan implementasi

**C1. PyTorch, bukan TensorFlow 1.12 + Keras.** Bobot pretrained MobileNetV2 torchvision
tidak identik dengan versi Keras.

**C2. BatchNorm melihat micro-batch 8, bukan batch 40.** VRAM 4 GB tidak bisa menampung
batch 40 pada 384×384, jadi batch dipecah dan gradiennya diakumulasi. Update optimizer
tetap dari 40 sampel seperti paper, tapi statistik BatchNorm dari 8 sampel — dan akumulasi
gradien tidak bisa memperbaiki itu. Tersangka utama selisih 3,6% pada baseline citra.

**C3. Mixed precision (AMP) aktif**, tidak dipakai paper.

**C4. 14 ambiguitas paper yang harus kami putuskan sendiri.** Daftar lengkapnya di
[[1907.06370-spesifikasi-implementasi]]. Yang paling berdampak: jumlah layer MLP (paper
hanya menetapkan lebarnya), posisi maxpool CNN1D, mekanisme "adaptive averaging", dan
arsitektur head fusion. Tidak ada kode resmi dari penulis untuk mengecek silang.

---

## D. Limitasi pada ablasi

**D1. Degradasi citra tidak diikuti degradasi OCR.** Teks tetap dari OCR atas citra bersih
aslinya, jadi ablasi mengukur ketahanan **arsitektur**, bukan sistem ujung ke ujung.
Mengukur versi sungguhannya butuh menjalankan Tesseract ulang pada setiap citra
terdegradasi — 3482 dokumen × belasan tingkat × ~910ms, di luar anggaran.

**D2. Ablasi 4 (fusion bergerbang / cross-attention) tidak dikerjakan.** Satu-satunya
ablasi yang butuh training dari awal.

**D3. Definisi "modalitas kosong" untuk citra adalah pilihan.** Nol pada tensor yang sudah
ternormalisasi berarti citra rata-rata ImageNet, bukan hitam. Alternatif akan memberi angka
berbeda. Untuk teks definisinya bersih: nol persis seperti padding, dan dataset memang
memuat dokumen ber-OCR kosong.

**D4. Reproduksi Fig. 4b hanya kolom FastText.** Kolom GloVe dan ELMo tidak dihitung, jadi
klaim **komparatif** paper — bahwa subword lebih baik daripada alternatifnya — tetap tidak
terverifikasi oleh kami. Dan median cosine kami 0,190 melawan ~0,96 milik paper belum bisa
ditafsirkan: heuristik pencari pasangan kami ikut menjaring yang bukan salah eja, dan kata
dalam kosakata memakai vektor terlatih sementara kata OOV memakai jumlah n-gram.

**D5. Probe linear mengukur informasi yang terpisah secara linear.** Angka 0,4236 untuk
fitur yang kolaps adalah **batas bawah**; bisa ada informasi non-linear yang tidak
tertangkap.

---

## E. Kekeliruan metodologis kami sendiri

Dicatat karena berguna, bukan untuk kerendahan hati.

**E1. Tiga ambang biner sewenang-wenang memberi vonis menyesatkan.** Sel kesimpulan
otomatis di notebook 07 memakai ambang 0,01 untuk memutuskan apakah sebuah modalitas
"dipakai", lalu mencetak "bukan early stopping, bukan strategi penggabungan" karena 0,0097
dan 0,0071 keduanya jatuh di bawahnya. 0,0097 versus 0,01 adalah lemparan koin. Notebook 09
mengulangi kesalahan sejenis dengan `std_rasio > 0,5`, memvonis "fiturnya masih
terkompresi" padahal probe membuktikan fiturnya sama informatifnya dengan acuan.

**Pelajarannya:** ambang harus relatif terhadap acuan yang terukur, dan pengukuran yang
paling menentukan (di sini: probe linear) tidak boleh dikalahkan oleh proksi yang lebih
lemah.

**E2. Satu sel mengklaim atribusi yang notebooknya sendiri nyatakan tidak mungkin.**
Notebook 09 mencetak "normalisasi yang bekerja, bukan inisialisasi" padahal keduanya
dijalankan bersamaan.

**E3. Satu metrik jebol di luar rentangnya.** `persen_potensi_dipanen` mencapai 879,6%
setelah perbaikan, karena formulanya mengandaikan model memanen **sebagian** sampel
text-only-correct. Setelah model bergantung pada teks, menolkannya menghancurkan
segalanya — fenomena berbeda yang formulanya tidak mewakili.

**E4. Rank efektif tidak berguna sebagai detektor kolaps pada arsitektur ini**, karena
model acuannya sendiri berrank 1,62. Diverifikasi pada fitur sintetis bahwa tidak ada satu
metrik pun yang cukup sendiri.

**E5. Satu hipotesis pra-eksekusi keliru secara terbalik.** Kami memprediksi jarak
FUSION − IMAGE akan melebar saat citra dirusak; ia menyempit tajam. Hipotesisnya ditulis
sebelum eksekusi dan disimpan apa adanya — kalau ditulis sesudahnya, sangat mudah
merasionalisasi hasil apa pun.

---

## F. Arah lanjutan

Urut manfaat per biaya.

**F1. Ablasi degradasi citra pada model yang sudah diperbaiki** — menit-an, checkpoint
sudah ada. Ini uji paling menarik yang tersisa. Model sebelumnya gagal menunjukkan
ketahanan tambahan karena cabang teksnya mati; sekarang cabang teksnya hidup. Kalau jarak
FUSION − IMAGE akhirnya **melebar**, itu menutup lingkaran dengan
[[07-ablasi-degradasi-citra]] dan membalikkan hipotesis yang keliru di E5.

**F2. Ulangi perbaikan pada seed 43 dan 44** — ± 8 jam. Menyelesaikan B3: apakah OA
benar-benar turun, atau itu derau satu seed.

**F3. Pisahkan atribusi kedua perbaikan** — ± 6,6 jam. Menyelesaikan B4.

**F4. Perbesar validation ke 15-20% dari train**, atau pilih checkpoint pada `val_loss`
yang dihaluskan. Menyelesaikan B2, perkiraan perolehan 1-2% untuk semua model.

**F5. Uji empat pasangan salah-eja milik paper** dengan model FastText kami — menit-an.
Menyelesaikan D4 sebagian: memisahkan artefak heuristik dari sifat model.

**F6. Fusion bergerbang atau cross-attention** (ablasi 4). Setelah F1, pertanyaannya jadi
lebih tajam: apakah gerbang eksplisit mengalahkan LayerNorm + inisialisasi, atau keduanya
mencapai hal yang sama lewat jalan berbeda.

**F7. Ulangi pada RVL-CDIP.** Menyelesaikan A2, dan menguji apakah kolapsnya cabang teks
memang artefak rezim data kecil. Di luar anggaran tugas, tapi ini pertanyaan ilmiah yang
paling penting dari seluruh proyek.

---

## G. Pendekatan modern sebagai pembanding

Paper ini dari 2019, dan pendekatannya — dua encoder terpisah lalu digabung di akhir —
sudah digantikan.

**LayoutLM dan turunannya** menyatukan teks, tata letak, dan citra dalam satu transformer
dengan koordinat kata sebagai masukan eksplisit. Itu menyelesaikan masalah **alignment**
tingkat elemen yang paper ini sepenuhnya lewati: paper tidak pernah mencocokkan kata
tertentu dengan wilayah citra tertentu. Lihat [[taksonomi-multimodal]].

**DiT** (Document Image Transformer) menunjukkan pretraining self-supervised pada citra
dokumen saja bisa mengalahkan pendekatan multimodal yang naif — relevan bagi kami, karena
temuan kami adalah cabang citra yang mendominasi.

Yang paling menarik untuk ditindaklanjuti: **temuan kami menyiratkan bahwa keunggulan
pendekatan modern mungkin sebagian bukan soal arsitektur, melainkan soal menghindari
ketimpangan modalitas.** Transformer yang menyatukan kedua modalitas dari awal tidak punya
dua cabang terpisah yang bisa kalah skala satu terhadap yang lain. Itu hipotesis yang bisa
diuji, dan kami tidak menemukannya dinyatakan di tempat lain.
