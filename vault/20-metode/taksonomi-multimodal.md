---
judul: Taksonomi multimodal dan posisi paper ini
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
  - "[[strategi-fusion-concat-vs-sum]]"
  - "[[oracle-sebagai-batas-atas]]"
  - "[[ocr-tesseract]]"
  - "[[1907.06370-ringkasan]]"
---

# Taksonomi multimodal dan posisi paper ini

## Inti gagasan

Kerangka yang lazim dipakai membagi tantangan pembelajaran multimodal jadi lima:
**representation** (bagaimana merepresentasikan beberapa modalitas sekaligus),
**translation** (memetakan satu modalitas ke modalitas lain), **alignment**
(mencocokkan elemen antar modalitas), **fusion** (menggabungkan informasi untuk
satu prediksi), dan **co-learning** (satu modalitas membantu pembelajaran yang lain
saat data timpang).

## Posisi paper ini

Terutama **fusion**, dan itu diakui sendiri: dua ekstraktor terpisah, dua vektor
128 dimensi, digabung untuk satu keputusan klasifikasi.

Sedikit menyentuh **representation** lewat joint representation yang dipelajari
end-to-end. Diskusi konkatenasi vs penjumlahan di
[[strategi-fusion-concat-vs-sum]] sebetulnya adalah diskusi **alignment**:
penjumlahan mengasumsikan kedua ruang sudah sejajar, konkatenasi menolak asumsi itu.
Menarik bahwa paper memilih untuk **tidak** menyejajarkan, dan itu keputusan yang
tepat untuk kasusnya.

Yang **tidak** dikerjakan: tidak ada translation eksplisit, tidak ada alignment
tingkat elemen (tidak ada usaha mencocokkan kata tertentu dengan wilayah citra
tertentu — beda dengan pendekatan modern seperti LayoutLM yang memakai koordinat
kata), dan tidak ada co-learning.

## Isu utama: teks di sini bukan modalitas independen

Ini keberatan yang paling serius dan harus dibahas dua arah, bukan sekadar dibela.

**Argumen kontra — ini tidak benar-benar multimodal.** Teksnya diturunkan dari citra
yang sama lewat OCR. Secara informasi-teoretis tidak ada informasi baru: seluruh isi
teks sudah terkandung di piksel, dan OCR adalah fungsi deterministik atas piksel
itu. Multimodal yang sesungguhnya melibatkan sensor berbeda yang menangkap aspek
berbeda dari dunia — kamera dan mikrofon, RGB dan depth. Di sini hanya ada satu
sensor. Yang dilakukan paper lebih tepat disebut **ekstraksi fitur dua jalur**, dan
menyebutnya multimodal melebihkan kebaruannya.

**Argumen pro — pembedaannya tidak sepenting yang terdengar.** Tiga alasan.

*Pertama*, OCR bukan transformasi yang bisa dipelajari CNN begitu saja. Ia membawa
pengetahuan luar yang besar — model bahasa LSTM Tesseract yang dilatih pada
korpus terpisah. Jadi teks bukan turunan piksel semata, melainkan piksel **plus**
prior linguistik yang tidak ada di jaringan citra. Sama halnya vektor FastText
membawa statistik Common Crawl yang tidak mungkin dipelajari dari 800 dokumen latih.

*Kedua*, ada buktinya. Kalau teks benar-benar tidak menambah apa pun, oracle akan
berada di sekitar baseline terbaik. Oracle 92,1% vs IMAGE 84,5% menunjukkan kedua
jalur salah pada sampel yang berbeda — lihat [[oracle-sebagai-batas-atas]]. Kelas
Resume paling telak: teks 0,97 vs citra 0,80.

*Ketiga*, dalam praktik industri pembedaannya lenyap. Dokumen datang sebagai citra;
apakah teksnya disebut modalitas kedua atau fitur turunan tidak mengubah apa pun
tentang sistem yang harus dibangun.

**Posisi saya.** Keberatannya sah dan mengubah cara membaca hasilnya, tapi tidak
membatalkannya. Yang lebih tepat: ini kasus fusion di mana **satu modalitas
diperoleh dengan biaya tambahan dari modalitas lain**. Itu menjelaskan mengapa
bagian besar pembahasan paper justru soal latensi — OCR bukan masukan gratis, ia
menghabiskan 910ms dari anggaran 1000ms. Framing itu lebih jujur sekaligus lebih
berguna daripada memperdebatkan apakah dua jalur layak disebut dua modalitas.

## Konsekuensi untuk ablasi kita

Kalau teks turunan dari citra, maka **mendegradasi citra seharusnya mendegradasi
teks juga** — di sistem nyata, citra yang dirotasi atau diblur menghasilkan OCR yang
lebih buruk, bukan hanya fitur visual yang lebih buruk. Ablasi Tahap 7 kita
mendegradasi citra sementara teks tetap dari OCR sumber bersih, jadi hasilnya
mengukur ketahanan **arsitektur**, bukan ketahanan sistem ujung ke ujung. Itu
keterbatasan desain ablasi kita dan harus ditulis eksplisit, bukan dilewatkan.
