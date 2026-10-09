# Panduan Template LaTeX Naskah Akademik FILKOM UB

Dokumen ini menjelaskan cara memakai template, apa saja yang sudah diatur, dan apa yang
harus Anda putuskan sendiri. Baca Bagian 1 sampai 3 sebelum mulai menulis; sisanya menjadi
rujukan saat dibutuhkan.

---

## 1. Apa ini

Kerangka LaTeX yang formatnya direplikasi dari template Word resmi FILKOM UB
(`Template Skripsi.docx` dan `Template Tesis V2`). Seluruh aturan format berada di satu
berkas, `gaya/ubtesis.sty`, sehingga naskah Anda hanya berisi tulisan.

Yang sudah terbukti jalan pada kompilasi uji: sampul dengan jarak vertikal sesuai template,
perpindahan margin halaman awal ke halaman isi, penomoran Romawi kecil lalu Arab, empat
halaman daftar, penomoran bab, dan halaman lampiran yang terisi otomatis.

**Batas yang jujur.** Template ini direplikasi dari dokumen Word, dan beberapa keputusan
tidak ada di dokumen itu sehingga diputuskan sendiri. Baris bertanda `KEPUTUSAN` di
`gaya/ubtesis.sty` menandai hal-hal tersebut. Bila prodi Anda meminta lain, ubah di situ.
Yang tidak berasal dari template resmi: catatan kaki 10 pt, istilah "et al." alih-alih
"dkk.", dan detail tata letak daftar referensi.

---

## 2. Prasyarat

| Kebutuhan | Keterangan |
|---|---|
| Distribusi TeX | MacTeX, TeX Live, atau MiKTeX lengkap |
| Mesin | pdfLaTeX. **Bukan** XeLaTeX atau LuaLaTeX |
| Bibliografi | `biber`, bukan `bibtex` |
| Huruf | Carlito, sudah ada di TeX Live standar |
| `-shell-escape` | Tidak diperlukan |

Carlito adalah klon Calibri yang identik secara metrik. Template Word memakai Calibri, yang
tidak boleh didistribusikan ulang; Carlito menghasilkan panjang baris dan pemutusan halaman
yang sama.

Di **Overleaf**: pilih pdfLaTeX dan Biber di menu Menu → Compiler. Keduanya tersedia.

---

## 3. Memulai proyek baru

```bash
cp -r template-latex-ub ~/proyek-baru/naskah
cd ~/proyek-baru/naskah
latexmk main.tex
```

Bila berhasil, `main.pdf` berisi 8 halaman contoh. Setelah itu kerjakan tiga hal berikut,
berurutan.

**Pertama, ganti data sampul** di `main.tex`. Hanya blok ini yang wajib disunting sebelum
naskah tampil dengan identitas yang benar:

```latex
\newcommand\judul{...}
\newcommand\jenisnaskah{PROPOSAL TESIS}   % atau TESIS, SKRIPSI, PROPOSAL SKRIPSI
\newcommand\penulis{...}
\newcommand\nim{...}
\newcommand\programstudi{PROGRAM STUDI ...}
\newcommand\departemen{DEPARTEMEN ...}
\newcommand\fakultas{FAKULTAS ILMU KOMPUTER}
\newcommand\universitas{UNIVERSITAS BRAWIJAYA}
\newcommand\kota{MALANG}
\newcommand\tahun{20XX}
\newcommand\pembimbing{...}
```

Judul ditulis dengan Huruf Kapital di Awal Setiap Kata, bukan seluruhnya kapital; berkas
gaya tidak mengubahnya menjadi kapital. Judul panjang akan menambah baris dan menggeser blok
di bawahnya, persis seperti di Word.

**Kedua, kosongkan isi contoh.** Hapus isi `bab/bab1.tex`, `lampiran/lampiran-a.tex`, dan
entri `contoh2020` di `referensi.bib`. Simpan dulu satu salinan `bab1.tex` bila ingin
menengok contoh sintaksnya.

**Ketiga, periksa daftar `\input`** di `main.tex`. Bab 2 dan 3 masih dikomentari; buka
komentarnya saat berkasnya sudah ada.

---

## 4. Yang harus diganti, ringkas

| Berkas | Tindakan |
|---|---|
| `main.tex` | Ganti blok data sampul. Daftarkan setiap bab baru |
| `bab/bab1.tex` | Hapus isi contoh, tulis naskah Anda |
| `referensi.bib` | Hapus `contoh2020`, isi referensi sebenarnya |
| `lampiran/lampiran-a.tex` | Isi, atau hapus blok `\appendix` di `main.tex` bila tanpa lampiran |
| `gambar/logo-ub.jpg` | Biarkan, kecuali institusinya berbeda |
| `gaya/ubtesis.sty` | **Jangan disentuh** kecuali memang hendak mengubah aturan format |

---

## 5. Struktur folder

```
naskah/
├── main.tex              hanya merangkai; tidak memuat tulisan
├── gaya/ubtesis.sty      seluruh aturan format
├── awal/sampul.tex       halaman sampul
├── bab/                  satu berkas per bab
├── lampiran/             satu berkas per lampiran
├── tabel/                tabel panjang yang dipisah dari naskah
├── gambar/               gambar dan berkas TikZ
├── referensi.bib         basis data referensi
└── latexmkrc             pengaturan kompilasi
```

Pemisahan ini bukan selera. Berkas `.tex` yang memuat seluruh naskah menjadi lambat
dikompilasi dan sulit ditelusuri saat ada galat; satu berkas per bab membuat pesan galat
menunjuk ke baris yang benar.

---

## 6. Format yang sudah dikunci

Daftar ini berguna bila pembimbing atau bagian akademik menanyakan kesesuaian format.

**Halaman.** A4. Margin kiri 4 cm, kanan 3 cm, bawah 3 cm. Margin atas 3,5 cm pada halaman
awal dan 3 cm pada isi. Tanpa header. Nomor halaman di tengah bawah, 10 pt, termasuk pada
halaman pembuka bab. Halaman awal bernomor Romawi kecil, isi bernomor Arab.

**Huruf.** Carlito 12 pt, satu spasi (*leading* 14,65 pt, yaitu 1,22 × ukuran huruf
sebagaimana Word menghitung satu spasi untuk Calibri). Matematika memakai newtxmath bergaya
Times Italic, meniru Equation Editor.

**Judul.**

| Tingkat | Ukuran | Gaya |
|---|---|---|
| Bab | 16 pt | Tebal, KAPITAL, rata tengah, ganti halaman |
| 1.1 | 14 pt | Tebal, rata kiri |
| 1.1.1 | 13 pt | Tebal, gantung 1,26 cm |
| 1.1.1.1 | 12 pt | Tebal miring, gantung 1,30 cm |

Penomoran sampai empat tingkat; daftar isi memuat tiga tingkat. Judul tidak akan tertinggal
sendirian di kaki halaman, karena setiap judul menuntut ruang empat baris.

**Paragraf.** Rata kiri-kanan, baris pertama menjorok 0,75 cm, jarak antarparagraf 6 pt.

**Tabel dan gambar.** Keterangan tabel di **atas**, keterangan gambar di **bawah**, keduanya
tebal, rata tengah, tanpa titik akhir, bernomor mengikuti bab. Isi tabel 12 pt dengan garis
penuh. Baris sumber ditulis dengan `\sumber{...}`.

**Persamaan.** Rata kiri dengan indentasi 0,38 cm, nomor di batas kanan, penomoran mengikuti
bab.

**Daftar berbutir.** Gantung 0,60 cm, jarak antarbutir 6 pt.

**Angka.** `siunitx` disetel ke kaidah Indonesia: koma sebagai pemisah desimal, titik sebagai
pemisah ribuan.

**Pemenggalan kata dimatikan**, mengikuti template Word yang tidak memenggal kata. Sebagai
gantinya, toleransi perataan dilonggarkan (`\tolerance=3000`, `\emergencystretch=3em`) agar
tidak ada baris yang menabrak margin kanan. Akibatnya jarak antarkata bisa terlihat renggang
pada paragraf tertentu, persis seperti di Word. Bila Anda lebih suka pemenggalan LaTeX, hapus
`\hyphenpenalty=10000` dan `\emergencystretch=3em` di `gaya/ubtesis.sty` sekitar baris 78.
Tanda hubung yang memang Anda ketik, misalnya pada "non-linear", tetap boleh menjadi titik
patah baris.

---

## 7. Sitasi dan daftar referensi

Gaya Harvard varian Anglia Ruskin, dilokalkan ke bahasa Indonesia, dijalankan biblatex dengan
mesin biber.

```latex
\parencite{kunci}              % (Brown, 2005)
\textcite{kunci}               % Brown (2005)
\parencite[p.~45]{kunci}       % (Brown, 2005, p.45)
```

Nama penulis dibalik seluruhnya, inisial tanpa spasi, tanpa koma sebelum "dan":
`Cox, C., Brown, J.T. dan Tumpington, W.T.` Lebih dari tiga penulis disingkat "et al."
Bagian akhir berjudul **Daftar Referensi**, bukan "Daftar Pustaka".

Tanggal ditulis gaya Indonesia, `15 September 2026`. Bulan sudah diterjemahkan.

**Pemenggalan per entri.** Setiap entri dianggap berbahasa Inggris. Untuk judul berbahasa
Indonesia, tambahkan `langid = {indonesian}` pada entri tersebut agar pemenggalannya benar.

**Jangan menyusun entri BibTeX dari ingatan.** Salin dari DOI, halaman penerbit, atau
pengelola referensi. Entri yang dikarang menghasilkan nomor volume dan halaman yang salah,
dan itu baru ketahuan saat sidang.

---

## 8. Menambah bab dan lampiran

Bab baru: buat `bab/bab2.tex` yang diawali `\chapter{Judul}`, lalu daftarkan di `main.tex`
di bawah `\input{bab/bab1}`. Urutan `\input` menentukan urutan dan penomoran bab.

Lampiran: buat `lampiran/lampiran-b.tex` yang juga diawali `\chapter{Judul}`, lalu
daftarkan **sesudah** `\appendix`. Setiap `\chapter` sesudah `\appendix` otomatis menjadi
"LAMPIRAN A", "LAMPIRAN B", dan seterusnya, dan otomatis tercatat di DAFTAR LAMPIRAN. Anda
tidak perlu menulis apa pun di halaman daftar itu.

Bila naskah tidak berlampiran, hapus `\daftarlampiran` dan seluruh blok `\appendix`.

---

## 9. Gambar

Simpan di `gambar/` lalu panggil **tanpa awalan folder**, karena `\graphicspath` sudah
disetel:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{nama-berkas}
  \caption{Keterangan gambar ditulis di bawah}
  \label{fig:contoh}
  \sumber{Penulis, 2026.}
\end{figure}
```

Gunakan penempatan `[H]` bila gambar harus tetap persis di tempatnya, misalnya diagram alur
yang harus muncul sebelum subbab tertentu. Paket `float` sudah dimuat.

Diagram TikZ sebaiknya ditulis sebagai berkas `.tex` tersendiri di `gambar/` lalu di-`input`.
Pustaka `arrows.meta`, `positioning`, `fit`, `calc`, dan `shapes.geometric` sudah dimuat.

---

## 10. Tabel

Tabel pendek memakai `tabular` biasa dengan garis penuh. Tabel yang melampaui satu halaman
memakai `xltabular`, dan baris penanda sambungannya memakai `\lanjutan{n}` dengan `n` =
jumlah kolom, sehingga halaman berikutnya memuat "Tabel 2.2 (lanjutan)".

Tabel yang terlalu lebar boleh memperkecil hurufnya sendiri dengan `\fontsize{10}{12}\selectfont`
tepat sebelum lingkungannya. Tabel yang sangat lebar dapat diputar dengan `pdflscape`, yang
sudah dimuat.

---

## 11. Penanda pekerjaan yang belum selesai

Tiga perintah ini menghasilkan teks merah di PDF, sehingga mudah ditemukan saat memeriksa:

```latex
\todo{lengkapi angka ini}
\rujukandiperlukan{klaim ini}
\konfirmasi{tanyakan ke pembimbing}
```

Cari seluruhnya sebelum menyerahkan naskah:

```bash
grep -rn "todo{\|rujukandiperlukan{\|konfirmasi{" bab lampiran
```

---

## 12. Kompilasi

```bash
latexmk main.tex      # kompilasi penuh, biber dijalankan otomatis
latexmk -c            # bersihkan berkas antara, PDF tetap
latexmk -C            # bersihkan termasuk PDF
```

Satu kali `latexmk` sudah cukup; ia mengulang sendiri sampai rujukan silang dan sitasi
terselesaikan. Peringatan "undefined" pada bagian awal log berasal dari lintasan pertama dan
normal — yang perlu diperiksa adalah lintasan terakhir.

Sebelum menyerahkan, pastikan tiga angka ini nol:

```bash
latexmk main.tex 2>&1 | grep -c "Overfull"
latexmk main.tex 2>&1 | grep -c "undefined"
grep -rc "todo{" bab lampiran
```

---

## 13. Masalah yang sering muncul

**`Font Carlito not found`.** Distribusi TeX Anda tidak lengkap. Di TeX Live:
`tlmgr install carlito`. Di MiKTeX, paketnya terpasang otomatis saat pertama dipakai.

**Daftar referensi kosong, sitasi tampil sebagai tanda tanya.** Mesin bibliografinya salah.
Harus `biber`, bukan `bibtex`. Hapus berkas antara dengan `latexmk -C`, lalu kompilasi ulang.

**DAFTAR LAMPIRAN kosong.** Berkas lampiran di-`input` sebelum `\appendix`. Pindahkan ke
bawahnya. Halaman itu terisi dari `\chapter` yang muncul sesudah `\appendix`.

**`Overfull \hbox`.** Ada yang melewati margin kanan, biasanya tabel terlalu lebar atau kata
panjang yang tidak bisa dipenggal. Untuk tabel, kecilkan hurufnya atau sempitkan kolom. Untuk
kata panjang, sisipkan `\-` sebagai titik penggalan yang diizinkan.

**Judul tabel panjang terpisah dari tabelnya.** Pakai `[H]`, atau pecah tabelnya dengan
`xltabular`.

**Baris pertama kelompok di `longtable` tertinggal sendirian di kaki halaman.** `\nopagebreak`
sering tidak berpengaruh di dalam `longtable`. Yang bekerja adalah `\pagebreak` eksplisit
sebelum kelompok tersebut — tetapi berilah komentar, karena penanda itu harus dicabut bila
isi tabelnya berubah.

**Persamaan tampak rata tengah.** Jangan memuat ulang `amsmath` tanpa opsi `fleqn`; opsi itu
sudah diteruskan di berkas gaya.

**Kompilasi macet tanpa pesan jelas.** Hapus berkas antara: `latexmk -C`, lalu ulangi.

---

## 14. Daftar periksa sebelum menyerahkan

- [ ] Seluruh data sampul terisi, tidak ada `...` atau `20XX` tersisa
- [ ] Entri `contoh2020` sudah dihapus dari `referensi.bib`
- [ ] Tidak ada `\todo`, `\rujukandiperlukan`, atau `\konfirmasi` yang tertinggal
- [ ] `Overfull \hbox` nol
- [ ] Tidak ada rujukan silang atau sitasi menggantung di lintasan terakhir
- [ ] Setiap tabel dan gambar punya keterangan dan dirujuk dari dalam teks
- [ ] Setiap entri bibliografi berasal dari sumber sebenarnya, bukan dari ingatan
- [ ] Nomor halaman berlanjut dengan benar dari Romawi ke Arab

---

## 15. Mengubah aturan format

Seluruhnya di `gaya/ubtesis.sty`, yang tersusun menurut bagian: huruf, halaman, judul,
daftar isi, tabel, persamaan, daftar, dan sitasi. Setiap aturan disertai komentar `[spek X]`
yang menunjuk bagian spesifikasi asalnya, serta angka hasil pengukuran terhadap PDF template.

Bila mengubah sesuatu, tulis alasannya pada komentar di baris itu. Enam bulan kemudian Anda
tidak akan ingat mengapa jaraknya 11 pt dan bukan 12 pt, dan tanpa catatan itu Anda akan
mengubahnya kembali.
