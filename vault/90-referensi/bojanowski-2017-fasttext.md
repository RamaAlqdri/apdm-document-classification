---
judul: Bojanowski et al. (2017) — FastText
tipe: referensi
tahap: "02"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/02
  - modal/teks
  - komponen/embedding
terkait:
  - "[[fasttext-subword]]"
  - "[[noise-ocr-dan-oov]]"
  - "[[peta-metode]]"
---

# Bojanowski et al. (2017) — FastText

> P. Bojanowski et al., "Enriching Word Vectors with Subword Information,"
> Transactions of the ACL, 2017.

*Catatan ini ditulis dari cara Audebert et al. mengutipnya dan pengetahuan umum,
bukan dari membaca paper aslinya.*

## Kontribusi

Memperluas skipgram word2vec dengan **informasi subword**. Setiap kata direpresentasi
sebagai kantong n-gram karakter, dan vektor kata adalah jumlah vektor n-gram
penyusunnya. Konsekuensinya besar: model bisa membangun vektor untuk kata yang
**belum pernah dilihat**, selama n-gram penyusunnya pernah dilihat. Ini juga
membantu bahasa dengan morfologi kaya, tempat satu akar kata bercabang jadi banyak
bentuk.

Paper terkait dari kelompok yang sama, Joulin et al. "Bag of Tricks for Efficient
Text Classification" (EACL 2017), juga dikutip Audebert et al.

## Kenapa dikutip di paper kita

Inti argumen cabang teks. Teks OCR penuh salah eja, dan pendekatan OOV konvensional
(vektor acak atau pemetaan Levenshtein) menyuntikkan fitur tak diskriminatif dalam
jumlah besar ketika 26% korpus di luar kamus. Subword menyelesaikan ini secara
struktural: `fiilter` otomatis dekat dengan `filter` karena ejaannya mirip.

Data pendukungnya di Fig. 4b paper: pada pasangan `Largely`/`Largly`, GloVe hanya
0,25 sementara FastText 0,98.

## Relevansi untuk proyek kita

Satu konsekuensi implementasi yang tidak boleh keliru: **wajib memakai model `.bin`
(`cc.en.300.bin`), bukan `.vec`**. File `.vec` hanya berisi tabel vektor kata yang
sudah dihitung — vektor n-gram-nya tidak disertakan, jadi inferensi OOV tidak mungkin
dilakukan. Memakai `.vec` berarti membuang seluruh alasan FastText dipilih.

Ukurannya ~7GB di disk dan ~15GB saat dimuat. `reduce_model` tidak menolong soal RAM
karena harus memuat model penuh lebih dulu. Lihat [[fasttext-subword]].
