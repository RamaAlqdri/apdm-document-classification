---
judul: Sintesis Akhir
tipe: hasil
tahap: "08"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/hasil
  - tahap/08
  - modal/fusion
  - komponen/evaluasi
terkait:
  - "[[tabel-utama]]"
  - "[[limitasi-dan-lanjutan]]"
  - "[[1907.06370-ringkasan]]"
  - "[[12-perbaikan-norm-dan-init]]"
  - "[[oracle-sebagai-batas-atas]]"
  - "[[taksonomi-multimodal]]"
  - "[[peta-proyek]]"
---

# Sintesis Akhir

Replikasi Audebert et al. (arXiv:1907.06370) pada Tobacco3482, plus ablasi dan diagnostik
yang tidak ada di paper. Seluruh angka di bawah berasal dari eksekusi nyata dan bisa
dilacak ke catatan eksperimennya masing-masing.

---

## 1. Apakah replikasinya berhasil, dan seberapa dekat dengan paper?

**Berhasil pada tingkat angka.** Urutan yang harus terlihat, terlihat.

| Model | Kita | Paper | Selisih |
|---|---|---|---|
| TEXT (CNN1D) | 0,7266 ± 0,0059 | 0,738 | −0,011 |
| IMAGE (MobileNetV2) | 0,8086 ± 0,0210 | 0,845 | −0,036 |
| FUSION (concat) | 0,8342 ± 0,0105 | 0,878 | −0,044 |
| Oracle | 0,8977 | 0,921 | −0,023 |

TEXT < IMAGE < FUSION < Oracle terpenuhi, semuanya 1-4% di bawah paper. Sumber angkanya:
[[03-baseline-cnn1d]], [[04-baseline-mobilenetv2]], [[05-fusion-concat]], [[tabel-utama]].

**Korpus teksnya terkonfirmasi identik dengan korpus paper.** Rata-rata token dengan
panjang minimal 4 karakter keluar **135**; paper menyebut **136**. Dua pengukuran dengan
definisi yang sama pada korpus yang diklaim sama, cocok dalam satu token
([[01-ekstraksi-fitur-teks]]).

Data pairingnya juga sempurna: 3482/3482 pada kedua modalitas, nol yatim, nol stem ganda,
10 dari 10 kelas terpetakan ([[00-audit-data]]).

**Selisih 1-4% itu punya tiga tersangka**, semuanya deviasi yang tercatat: BatchNorm pada
micro-batch 8 alih-alih 40 (GPU 4 GB tidak bisa menampung batch 40 pada 384×384), train
720 alih-alih 800 sampel karena validation dipotong dari dalamnya, dan citra JPG re-encode
alih-alih TIF asli. Ditambah satu yang tidak direncanakan: early stopping menghentikan
training di sekitar 15% anggaran epoch paper.

**Satu klaim paper tidak terreplikasi.** Paper menyatakan strategi fusion penjumlahan
"jatuh signifikan di bawah baseline citra murni". Milik kita **1,9% di atasnya** (0,8275
versus 0,8086), dan selisihnya terhadap concat (0,0067) jauh di dalam satu standar
deviasi — kedua strategi **tidak terbedakan secara statistik**. Paper tidak melaporkan
angka maupun mekanisme untuk klaim itu, jadi alasan mereka memilih concat tidak berdasar
pada bukti yang bisa diperiksa ([[06-fusion-sum]]).

---

## 2. Apakah fusion benar-benar memberi nilai tambah, dan dari mana asalnya?

**Ini pertanyaan yang jawabannya paling jauh dari dugaan awal, dan jawabannya ada dua
lapis.**

### Lapis pertama: angkanya bilang ya

FUSION 0,8342 melawan baseline terbaik IMAGE 0,8086 — naik **+2,56%** absolut, sekitar
2,4σ. Paper melaporkan +3,3%. Kalau proyek berhenti di sini, kesimpulannya "replikasi
berhasil, fusion memberi nilai tambah".

### Lapis kedua: kenaikan itu bukan dari modalitas teks

Tiga pengukuran independen mengatakan cabang teks nyaris tidak berkontribusi:

1. **Menolkan seluruh masukan teks menurunkan OA 0,0015** — empat dokumen dari 2682
   ([[08-ablasi-missing-modality]]).
2. **Menghapus 0% sampai 100% kata mengubah OA 0,0015**; menghapus 75% kata justru
   sedikit menaikkannya ([[09-ablasi-degradasi-teks]]).
3. **Resume** — kelas dengan keunggulan teks terbesar di dataset (TEXT 0,969 melawan
   IMAGE 0,784) — mendarat di 0,803, di sisi citra. Paper menjaganya di 0,96.

Jadi +2,56% itu kemungkinan berasal dari kapasitas tambahan atau efek regularisasi head
concat. **Klaim "fusion berhasil karena komplementaritas" tidak didukung data kita** pada
model seperti itu.

### Padahal komplementaritasnya nyata dan terukur

Oracle 0,8977, dan **8,9% sampel hanya benar lewat teks** — sekitar 239 dokumen yang
cabang citranya gagal tapi cabang teksnya berhasil. Potensinya ada; yang gagal adalah
arsitekturnya memanennya. Fusion memanen **29%** dari potensi itu, sementara paper 43%
([[oracle-sebagai-batas-atas]]).

### Kenapa gagal, dan di mana

Diagnosis berjalan tiga tahap, masing-masing mempersempit jawabannya.

**Bukan lama training.** Tanpa early stopping, epoch terbaik melompat 18 → 102 dan
kontribusi teks naik 4,7× — tapi tetap hanya 9% potensi
([[10-uji-lanjutan-early-stopping]]).

**Bukan strategi penggabungan.** Pada `sum` pun efeknya hanya 0,0097. Bobot cabang teks
0,31 yang tampak menjanjikan ternyata bukan bukti kontribusi — softmax sekadar tidak punya
tekanan untuk mendorongnya ke nol.

**Penyebabnya cabang teksnya sendiri.** [[11-diagnostik-cabang-teks]] mengukurnya: varians
fitur **38× lebih kecil** dari CNN1D standalone, **67 dari 128 unit mati**, cos antar
dokumen 0,978 melawan 0,650 — lewat cabang itu semua dokumen terlihat nyaris identik. Dan
skalanya **timpang 3,7×** terhadap cabang citra.

Mekanismenya:

> Fitur teks **pelan** — norma kecil, varians kecil, separuh unit mati — sementara fitur
> citra 3,7× lebih besar. Head concat karena itu didominasi separuh citra secara numerik.
> Cabang teks berkontribusi sedikit bukan karena tidak berinformasi, tapi karena terlalu
> lirih untuk didengar.

Bahwa ia masih berinformasi terbukti dari probe linear: regresi logistik di atas fitur
beku yang kolaps itu mencapai **0,4236**, yaitu 2,4× di atas kelas mayoritas. Head fusion
mengabaikan sesuatu yang bahkan classifier linear bisa baca.

### Dan penyebabnya bisa diperbaiki

[[12-perbaikan-norm-dan-init]]: LayerNorm per cabang plus inisialisasi cabang teks dari
`CNN1D_seed42.pt`.

| Pengukuran | Sebelum | **Sesudah** | Acuan CNN1D |
|---|---|---|---|
| Probe linear fitur teks | 0,4236 | **0,7468** | **0,7479** |
| Unit mati | 67/128 | **4/128** | 18/128 |
| Efek menolkan teks | −0,0071 | **−0,6887** | — |
| OA tanpa citra | 0,1946 | **0,6469** | — |
| **Resume F1** | 0,814 | **0,968** | paper 0,96 |

Cabang teksnya kini **sama informatifnya dengan CNN1D standalone** (selisih probe 0,0011),
modelnya benar-benar multimodal, dan penyimpangan per-kelas terbesar kita terhadap paper
tertutup sepenuhnya.

**Ongkosnya: OA turun 0,0194, sementara macro F1 identik** (0,8108 keduanya, jumlah delta
per kelas tepat 0,000). Perbaikannya memindahkan akurasi dari kelas besar yang dikuasai
citra (Memo −0,095, n=478) ke kelas kecil yang butuh teks (Resume +0,154, n=92).

### Jawaban atas pertanyaan 2

**Fusion memberi nilai tambah, tapi pada replikasi apa adanya nilai tambah itu bukan
berasal dari modalitas kedua.** Komplementaritas antar modalitas nyata dan terukur (8,9%
sampel), tapi arsitektur concat pada setelan kami gagal memanennya karena cabang teksnya
kolaps dan kalah skala. Setelah keduanya diperbaiki, modelnya benar-benar multimodal dan
mereplikasi pola per-kelas paper — dengan OA sedikit lebih rendah dan metrik seimbang yang
tidak berubah.

Temuan metodologis yang dibawa keluar dari sini, dan yang paling berguna bagi orang lain:

> **OA fusion yang tinggi bukan bukti bahwa fusion memakai kedua modalitas.** Model yang
> secara diam-diam unimodal bisa mencetak OA **lebih tinggi** daripada model yang
> benar-benar multimodal pada dataset ini. Tanpa uji missing modality, keduanya tidak bisa
> dibedakan dari tabel hasil.

Paper hanya melaporkan OA dan F1. Kami tidak bisa tahu cabang teks mereka benar-benar
terpakai — dan Resume 0,96 mereka menunjukkan kemungkinan besar iya, tapi itu inferensi
dari satu sel tabel, bukan pengukuran.

---

## 3. Kapan multimodal menolong dan kapan tidak?

Berdasarkan ablasi kami sendiri, bukan klaim paper.

### Menolong: pada kelas yang tampilannya generik tapi isinya khas

**Resume** adalah kasus terbaiknya. TEXT 0,969 melawan IMAGE 0,784 — selisih 18,5 poin.
Riwayat hidup punya tata letak yang mirip dokumen lain, tapi kosakatanya sangat khas.
Ketika fusion benar-benar memakai teks, kelas ini melonjak ke 0,968.

**Scientific** mengikuti pola yang sama, lebih lemah: TEXT 0,587 melawan IMAGE 0,554, dan
naik ke 0,625 setelah perbaikan.

### Tidak menolong: pada kelas yang teksnya tidak ada

[[00-audit-data]] mengukur ini langsung. **48,3% dokumen kelas Note** punya kurang dari 20
kata, dan **18,3% Advertisement**. Akibatnya terlihat di F1 TEXT: Advertisement 0,537 dan
Note 0,621, dua terendah setelah Report.

Pada kelas seperti itu modalitas kedua secara harfiah tidak ada isinya, dan tidak ada
arsitektur yang bisa memperbaikinya. 26 dokumen (0,7%) teksnya kosong sepenuhnya — kasus
missing modality yang muncul alami di data, bukan konstruksi ablasi.

### Merugikan: pada kelas besar yang dikuasai citra

Ini yang tidak diduga. Memaksa model memakai teks **menurunkan** Memo (−0,095), Letter
(−0,042), News (−0,045), dan Form (−0,032). Keempatnya kelas dengan tata letak sangat khas
yang cabang citra sudah menangani dengan baik, dan kapasitas yang bergeser ke teks diambil
dari sana.

Jadi bukan "multimodal selalu lebih baik", melainkan **pertukaran antar kelas** yang
tersembunyi kalau hanya OA yang dilaporkan.

### Tidak menambah ketahanan — malah mengurangi

[[07-ablasi-degradasi-citra]] menguji limitasi yang penulis sebut tapi tidak uji: mereka
mengakui Tobacco3482 seluruhnya berorientasi rapi dan discan profesional.

Hipotesis kami sebelum eksekusi: jarak FUSION − IMAGE akan **melebar** saat citra dirusak,
karena cabang teks mengompensasi. **Hipotesis itu keliru, dan keliru secara terbalik.**

| Degradasi | IMAGE | FUSION | Jarak |
|---|---|---|---|
| rotasi 45° | 0,303 | **0,122** | −0,181 |
| noise σ=0,2 | 0,278 | **0,116** | −0,163 |
| blur r=8 | 0,612 | 0,503 | −0,109 |
| JPEG q=5 | 0,825 | 0,827 | +0,002 |

Fusion kolaps **lebih dalam** daripada baseline citra pada tiga dari empat jenis. Itu
konsekuensi logis dari temuan di pertanyaan 2: tanpa cabang teks yang berfungsi, tidak ada
apa pun untuk mengompensasi, dan head yang dilatih pada fitur citra bersih justru lebih
sensitif terhadap fitur citra yang rusak.

JPEG yang datar bukan pengecualian yang membantah — OA IMAGE hanya turun 0,0067 dari q100
ke q5, jadi tidak ada degradasi berarti untuk dikompensasi. Itu justru mereplikasi temuan
paper sendiri bahwa artefak JPEG tidak berpengaruh pada dokumen grayscale.

Apakah model yang **sudah diperbaiki** akhirnya lebih tahan degradasi adalah pertanyaan
yang belum diuji, dan itu uji paling menarik yang tersisa —
lihat [[limitasi-dan-lanjutan]].

### Satu catatan yang membatasi seluruh bagian ini

Ablasi degradasi kami mendegradasi citra sementara **teksnya tetap dari OCR atas citra
bersih aslinya**. Di sistem nyata, scan yang dirotasi 45° menghasilkan OCR yang jauh lebih
buruk juga. Jadi angka di atas mengukur ketahanan **arsitektur**, bukan sistem ujung ke
ujung.

Untuk kesimpulan kami arah biasnya justru menguatkan: fusion gagal mengompensasi **bahkan
ketika teksnya diberi keuntungan tidak realistis berupa sumber yang bersih.** Konsekuensi
langsung dari keberatan di [[taksonomi-multimodal]] bahwa teks di sini diturunkan dari
citra, sehingga mendegradasi satu modalitas tanpa yang lain adalah kondisi yang tidak bisa
terjadi di dunia nyata.

---

## Ringkasan dalam empat kalimat

Replikasi berhasil pada tingkat angka: urutan TEXT < IMAGE < FUSION < Oracle terpenuhi,
semuanya 1-4% di bawah paper, dan korpus teksnya terkonfirmasi identik.

Satu klaim paper tidak terreplikasi: fusion penjumlahan tidak jatuh di bawah baseline
citra, dan tidak terbedakan secara statistik dari konkatenasi.

Kenaikan fusion +2,56% pada replikasi apa adanya **bukan** berasal dari modalitas teks —
cabang teksnya kolaps dan kalah skala 3,7×, terukur lewat empat cara berbeda — meski
komplementaritas antar modalitas nyata dan mencapai 8,9% sampel.

Kedua penyebabnya bisa diperbaiki, dan setelah diperbaiki modelnya benar-benar multimodal
dengan macro F1 tak berubah dan Resume mereplikasi paper — yang menghasilkan pelajaran
metodologis bahwa **OA fusion yang tinggi bukan bukti fusion memakai kedua modalitas.**
