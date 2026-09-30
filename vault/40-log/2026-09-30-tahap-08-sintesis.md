---
judul: Log 2026-09-30 — Tahap 8 Sintesis Akhir
tipe: log
tahap: "08"
tanggal: 2026-09-30
status: selesai
tags:
  - tipe/log
  - tahap/08
  - status/selesai
terkait:
  - "[[sintesis-akhir]]"
  - "[[limitasi-dan-lanjutan]]"
  - "[[12-perbaikan-norm-dan-init]]"
  - "[[peta-proyek]]"
---

# Log 2026-09-30 — Tahap 8 Sintesis Akhir

## Yang dikerjakan

Tiga catatan eksperimen untuk notebook 07-09, pembaruan empat catatan lama yang
pertanyaan terbukanya sudah terjawab, dua dokumen Tahap 8, dan pembersihan vault.

## File yang dibuat

- [[10-uji-lanjutan-early-stopping]] — notebook 07
- [[11-diagnostik-cabang-teks]] — notebook 08
- [[12-perbaikan-norm-dan-init]] — notebook 09
- [[sintesis-akhir]] — menjawab tiga pertanyaan Tahap 8
- [[limitasi-dan-lanjutan]] — 22 limitasi dalam 5 kategori, plus 7 arah lanjutan

## File yang diperbarui

[[05-fusion-concat]] dan [[06-fusion-sum]]: pertanyaan terbukanya ditutup dengan
jawabannya, bukan dihapus — supaya urutan penalarannya tetap terbaca.
[[tabel-utama]]: tiga model tambahan dan status keempat rekomendasi.
[[peta-proyek]], [[peta-eksperimen]], [[papan-progress]], [[peta-tag]],
[[rencana-prompt]]: checklist dicentang, status `wip` jadi `selesai`.
README: bagian hasil.

## Pembersihan vault

Audit otomatis menemukan lima jenis masalah, semuanya diperbaiki:

1. **43 tag "tak terdaftar"** — ternyata false positive checker saya. `peta-tag.md`
   menulis tag tahap sebagai rentang (`#tahap/00` … `#tahap/08`) sehingga hanya 00 dan 08
   yang terbaca literal. Diperbaiki dengan mendaftarkan kesembilannya satu per satu, jadi
   konvensinya bisa diperiksa otomatis.
2. **8 log yatim** — tidak ada catatan yang menaut log. `papan-progress` punya query
   Dataview tapi query bukan wikilink, jadi di graf Obsidian mereka mengapung. Ditambah
   indeks log eksplisit.
3. **16 catatan dengan kurang dari 2 link keluar di badan teks**, melanggar konvensi
   `CLAUDE.md`. Ditambah taut yang **membawa informasi**, bukan basa-basi: tiap catatan
   pustaka sekarang menunjuk ke hasil eksperimen yang memakainya, dan tiap catatan konsep
   menunjuk ke temuan yang mengujinya.
4. **`[[wikilink]]` literal di `rencana-prompt`** terhitung sebagai link rusak. Dijadikan
   inline code.
5. **Empat MOC masih berstatus `wip`.** Jadi `selesai`.

Audit ulang: **55 catatan, bersih** — frontmatter lengkap, semua tag terdaftar, tidak ada
wikilink rusak, tidak ada orphan, tidak ada yang masih `wip` atau `belum-dijalankan`.

## Keputusan editorial

1. **Pertanyaan terbuka di catatan lama ditutup dengan jawabannya di tempat, bukan
   diganti.** [[05-fusion-concat]] masih memuat diagnosis awal "kemungkinan early
   stopping", diikuti bagian yang menyatakan diagnosis itu benar sebagian. Urutan
   penalarannya jadi terbaca, termasuk yang keliru.

2. **Lima kekeliruan metodologis saya sendiri didaftar sebagai kategori E di
   [[limitasi-dan-lanjutan]]**, bukan disebar jadi catatan kaki. Tiga ambang biner
   sewenang-wenang yang memberi vonis menyesatkan, satu klaim atribusi tanpa dasar, satu
   metrik yang jebol di luar rentangnya, satu metrik yang ternyata tidak berguna, dan satu
   hipotesis pra-eksekusi yang keliru terbalik.

3. **Hipotesis pra-eksekusi yang keliru tetap disimpan** di
   [[07-ablasi-degradasi-citra]] dan [[09-ablasi-degradasi-teks]], lengkap dengan
   penandaan mana yang benar dan mana yang salah.

4. **"4 dari 10 kelas FUSION kalah" tidak pernah ditulis sebagai angka mentah.** Tiga di
   antaranya di dalam derau antar-seed; yang sungguhan hanya Resume (−0,166).

5. **Kurva degradasi teks yang rata tidak ditulis sebagai ketahanan.** Ketahanan dan
   ketidakpedulian menghasilkan kurva berbentuk sama; yang membedakan hanya uji missing
   modality.

6. **Penemuan "OA turun tapi macro F1 identik" dijadikan temuan utama, bukan catatan
   kaki.** Jumlah delta per kelas tepat 0,000 — perbaikannya memindahkan akurasi dari
   kelas besar yang dikuasai citra ke kelas kecil yang butuh teks. Itu jauh lebih
   informatif daripada "akurasi ditukar dengan keseimbangan".

## Pelajaran yang dibawa keluar dari proyek ini

**OA fusion yang tinggi bukan bukti bahwa fusion memakai kedua modalitas.** Model yang
secara diam-diam unimodal bisa mencetak OA **lebih tinggi** daripada model yang
benar-benar multimodal pada dataset ini. Tanpa uji missing modality, keduanya tidak bisa
dibedakan dari tabel hasil.

Konsekuensinya untuk membaca paper aslinya: mereka hanya melaporkan OA dan F1. Kami tidak
bisa tahu cabang teks mereka benar-benar terpakai. Resume 0,96 mereka menunjukkan
kemungkinan besar iya — tapi itu inferensi dari satu sel tabel, bukan pengukuran.

## Yang belum dikerjakan

Tujuh arah lanjutan di [[limitasi-dan-lanjutan]]. Yang termurah dan paling menarik:
**ablasi degradasi citra pada model yang sudah diperbaiki** (F1, menit-an, checkpoint
sudah ada). Model sebelumnya gagal menunjukkan ketahanan tambahan karena cabang teksnya
mati; sekarang cabang teksnya hidup. Kalau jarak FUSION − IMAGE akhirnya melebar, itu
menutup lingkaran dengan [[07-ablasi-degradasi-citra]] dan membalikkan hipotesis yang
keliru.
