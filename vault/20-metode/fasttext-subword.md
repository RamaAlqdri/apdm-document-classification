---
judul: FastText dan informasi subword
tipe: konsep
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/konsep
  - tahap/02
  - modal/teks
  - komponen/embedding
terkait:
  - "[[noise-ocr-dan-oov]]"
  - "[[sif-document-embedding]]"
  - "[[bojanowski-2017-fasttext]]"
  - "[[peta-metode]]"
---

# FastText dan informasi subword

## Inti gagasan

word2vec dan GloVe memperlakukan kata sebagai simbol utuh: satu kata, satu baris
di tabel vektor. Kata yang tidak ada di tabel tidak punya vektor, titik. FastText
memecah kata jadi n-gram karakter dan menyimpan vektor untuk **potongannya**.
Vektor sebuah kata adalah jumlah vektor n-gram penyusunnya, jadi kata yang belum
pernah terlihat pun bisa dibangun dari potongan yang sudah dikenal.

## Cara kerja

`fiilter` tidak ada di kosakata mana pun. Tapi n-gram penyusunnya — `fii`, `ilt`,
`lte`, `ter` — sangat tumpang tindih dengan n-gram `filter`. Vektor hasil
penjumlahannya otomatis berdekatan dengan kata aslinya. Kemiripannya lahir dari
ejaan, bukan dari pernah bertemu kata itu di korpus latih.

Paper mengukur ini dan angkanya tajam (Fig. 4b):

| Pasangan | GloVe | ELMo | FastText |
|---|---|---|---|
| specifically / Specificalily | 0,71 | 0,68 | 0,96 |
| filter / fiilter | 0,91 | 0,73 | 0,96 |
| alcohol / Aleohol | 0,40 | 0,69 | 0,88 |
| Largely / Largly | 0,25 | 0,81 | 0,98 |

Kasus `Largely`/`Largly` paling telak: GloVe 0,25, FastText 0,98.

FastText dipilih di atas ELMo bukan karena lebih akurat, tapi karena lebih cepat —
ELMo harus melewatkan konteks melalui LSTM bidireksional saat inferensi, dan
paper punya anggaran latensi.

## Kaitan dengan paper ini

Inilah sendi teknis yang membuat cabang teks berfungsi sama sekali. Tanpa subword,
26% korpus jadi noise. Lihat [[noise-ocr-dan-oov]].

Konsekuensi praktis untuk implementasi kita: **wajib memakai model `.bin`, bukan
`.vec`**. File `.vec` hanya berisi tabel vektor kata jadi — n-gram-nya tidak ikut,
sehingga kemampuan inferensi OOV hilang seluruhnya. Memakai `.vec` berarti
membuang argumen utama paper tanpa sadar.

## Batasan / catatan kritis

Subword bergantung pada karakter yang sudah dikenal FastText. Penulis menyebut
mereka memeriksa ini dan tidak menemukan karakter OOV pada dokumen yang dipakai —
klaim yang bisa kita verifikasi sendiri di korpus kita di Tahap 4.

Yang lebih penting: FastText tetap memberi vektor untuk string sampah hasil
halusinasi OCR. Kemiripan ejaan tidak membedakan `fiilter` dari `Oztrlseezloz`.
Jadi subword mengurangi masalah, tidak menghapusnya.

**Reproduksi kita hanya sebagian.** Tabel di atas punya tiga kolom; kita menghitung
FastText saja. Menambahkan GloVe dan ELMo berarti mengunduh dua model besar lagi
hanya untuk satu tabel pembanding, dan itu di luar anggaran proyek. Konsekuensinya
kita bisa menunjukkan bahwa cosine FastText untuk salah eja di korpus kita tinggi,
tapi **tidak** bisa menunjukkan bahwa ia lebih tinggi daripada alternatifnya. Klaim
komparatif paper tetap tidak terverifikasi oleh kita. Dicatat di
[[01-ekstraksi-fitur-teks]].

Satu klaim kecil paper yang bisa kita uji murah: bahwa Tesseract tidak menghasilkan
karakter yang OOV bagi FastText. Notebook Tahap 4 memeriksanya di korpus kita.
