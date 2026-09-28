---
judul: Oracle sebagai batas atas fusion
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/fusion
  - komponen/evaluasi
terkait:
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
  - "[[taksonomi-multimodal]]"
  - "[[peta-metode]]"
---

# Oracle sebagai batas atas fusion

## Inti gagasan

Kalau fusion naik 3,3% di atas baseline terbaik, apakah itu bagus? Pertanyaan itu
tidak bisa dijawab tanpa tahu berapa banyak yang **tersedia** untuk dipanen. Oracle
menjawabnya: ia model hipotetis yang melihat prediksi kedua baseline unimodal lalu
selalu memilih yang benar.

## Cara kerja

Untuk setiap sampel test, satu sampel dihitung benar bila prediksi TEXT **atau**
prediksi IMAGE benar. Akurasinya adalah proporsi sampel seperti itu.

Definisi ini punya dua sifat yang harus dipahami. Pertama, ia dihitung dari
**prediksi per-sampel dua model standalone** — bukan dari cabang internal model
fusion. Kedua, ia bukan batas atas yang bisa dicapai model nyata, karena tidak ada
model yang tahu jawaban benar saat memilih. Ia batas atas atas *kelas* metode yang
bekerja dengan memilih antara kedua baseline.

## Kaitan dengan paper ini

Pada Tobacco3482: TEXT 73,8% - IMAGE 84,5% - FUSION 87,8% - **Oracle 92,1%**.

Cara membacanya: potensi maksimal di atas baseline terbaik adalah 7,6% absolut
(84,5 ke 92,1). Fusion memanen 3,3% dari itu — sekitar **43% dari potensi yang
tersedia**. Jadi kalimat yang jujur bukan "fusion berhasil", melainkan "fusion
memanen kurang dari separuh komplementaritas yang ada, dan sisanya masih terbuka".

Oracle juga membuktikan hal yang lebih mendasar: 92,1% jauh di atas kedua baseline,
artinya kedua modalitas memang salah pada **sampel yang berbeda**. Kalau keduanya
salah pada sampel yang sama, oracle akan mendekati baseline terbaik dan seluruh
proyek fusion tidak ada gunanya. Ini bukti kuantitatif komplementaritas, dan itulah
sebabnya oracle wajib dilaporkan.

Bukti per kelas paling terang di **Resume**: TEXT 0,97 sementara IMAGE hanya 0,80.
Ada kelas yang memang ranahnya teks.

## Batasan / catatan kritis

Oracle **bukan** target yang harus didekati. Ia mengasumsikan pengetahuan yang tidak
tersedia saat inferensi, jadi selisih yang tersisa tidak seluruhnya bisa dipanen
oleh metode apa pun.

Oracle juga hanya mengukur potensi *seleksi*, bukan potensi *kombinasi*. Model yang
menggabungkan bukti dari kedua modalitas secara prinsipnya bisa benar pada sampel
di mana **keduanya** salah — dan sampel seperti itu tidak terhitung di oracle. Jadi
92,1% bukan plafon mutlak, hanya plafon untuk satu cara berpikir tentang fusion.

Satu jebakan implementasi: oracle harus dihitung dari prediksi kedua baseline pada
**split test yang sama persis**. Kalau TEXT dan IMAGE dilatih dengan split berbeda,
angka oracle-nya tidak berarti apa-apa.
