---
judul: Strategi fusion — konkatenasi vs penjumlahan
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/fusion
  - komponen/fusion
terkait:
  - "[[oracle-sebagai-batas-atas]]"
  - "[[taksonomi-multimodal]]"
  - "[[eitel-2015-multimodal-rgbd]]"
  - "[[1907.06370-spesifikasi-implementasi]]"
---

# Strategi fusion — konkatenasi vs penjumlahan

## Inti gagasan

Kedua cabang menghasilkan vektor 128 dimensi. Untuk menggabungkannya ada dua
pilihan yang tampak setara tapi mengandaikan hal yang sangat berbeda.

**Penjumlahan** (paper menyebutnya *adaptive averaging*) mengandaikan kedua ruang
fitur bisa **disejajarkan**: dimensi ke-*i* vektor teks harus punya makna yang
sebanding dengan dimensi ke-*i* vektor citra, kalau tidak menjumlahkannya adalah
menjumlahkan dua besaran yang tidak sejenis.

**Konkatenasi** tidak mengandaikan apa pun. Ia menyerahkan tugas menemukan
hubungan antar kedua domain kepada MLP di atasnya, dengan biaya vektor masukan dua
kali lebih panjang dan parameter lebih banyak.

## Cara kerja

Konkatenasi: `[v_teks ; v_citra]` menjadi vektor 256, masuk MLP, keluar softmax.
Penjumlahan: `v_teks + v_citra` menjadi vektor 128, masuk MLP yang lebih kecil.
Keduanya diferensiabel, jadi seluruh jaringan tetap bisa dilatih end-to-end.

## Kaitan dengan paper ini

Penulis memilih konkatenasi. Penjumlahan dilaporkan **jatuh signifikan di bawah
baseline citra murni**, dengan dugaan penyebab kedua ruang fitur tidak bisa
disejajarkan tanpa merusak daya diskriminatif masing-masing.

Argumennya masuk akal secara intuitif: ruang fitur teks disusun dari kemunculan
kata, ruang fitur citra dari tata letak dan tekstur. Memaksa keduanya berbagi
sistem koordinat berarti memaksa jaringan mengorbankan struktur di salah satunya.

## Batasan / catatan kritis

**Klaim ini yang paling lemah landasannya di seluruh paper**, dan itulah alasan
Tahap 6 mengujinya sendiri:

1. **Tidak ada angka.** Disebut sebagai "eksperimen pendahuluan" saja. Tidak ada
   OA, tidak ada F1, tidak ada jumlah run.
2. **Tidak ada spesifikasi.** Apa yang membuat penjumlahan itu "adaptif" tidak
   pernah dijelaskan. Skalar terlatih per cabang? Gate per-dimensi? Rata-rata biasa?
   Ketiganya perilakunya berbeda jauh, dan yang paling sederhana memang paling
   mungkin gagal.
3. **Kesimpulannya mungkin benar karena alasan yang salah.** Kalau yang
   diimplementasikan adalah rata-rata biasa tanpa bobot terlatih, kegagalannya
   memberitahu kita soal implementasi itu, bukan soal fusion penjumlahan sebagai
   kelas metode.

Konsekuensi untuk kita: pilih satu mekanisme, **catat pilihannya sebagai asumsi**,
dan kalau hasil kita juga negatif jangan laporkan sebagai "berhasil mereplikasi
temuan paper" — laporkan sebagai satu titik data untuk mekanisme yang kita pilih.
Kalau hasil kita justru positif, itu temuan yang lebih menarik lagi, tandai
`#temuan/negatif` terhadap paper.
