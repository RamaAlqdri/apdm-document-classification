---
judul: Ringkasan — Mirończuk (2026), Systematic Review Fusion untuk Klasifikasi Dokumen
tipe: jurnal
tahap: "09"
tanggal: 2026-10-09
status: selesai
tags:
  - tipe/jurnal
  - tahap/09
  - modal/fusion
  - komponen/evaluasi
terkait:
  - "[[kashyap-2026-ringkasan]]"
  - "[[sintesis-akhir]]"
  - "[[peta-proyek]]"
---

# Ringkasan — Mirończuk (2026)

> M. M. Mirończuk, "Document classification pattern recognition via information fusion:
> A systematic review of multimodal and multiview representation approaches,"
> *Information Fusion*, vol. 132, art. 104247, 2026. doi:10.1016/j.inffus.2026.104247.
> PDF: `references/1-s2.0-S1566253526001260-main.pdf`

Dipakai sebagai **pembingkai gap**. Review ini mengukur kelemahan lapangan yang juga
ditemukan proyek kita di Fase 1.

## Masalah

Riset fusion untuk klasifikasi dokumen belum punya kerangka bersama dan belum punya
sintesis kuantitatif tentang seberapa besar sebenarnya keuntungan fusion.

## Kontribusi

1. Kerangka formal representasi, pola, dan model, yang mengaitkan arsitektur deep
   learning dengan teori fusion klasik (Bayesian, Dempster–Shafer).
2. Meta-analisis efek acak atas 139 studi primer — menurut penulis, yang pertama untuk
   klasifikasi dokumen.
3. Pedoman praktis berbasis data.

## Hasil utama

| Temuan | Angka |
|---|---|
| Fusion multimodal menaikkan akurasi | +5,28 poin, p = 0,0016 |
| Efek fusion multimodal terhadap F1 | positif tapi **tidak signifikan** |
| Studi multimodal yang memakai uji statistik | **11,8%** |
| Studi multiview yang memakai uji statistik | 23,3% |

## Yang langsung relevan untuk kita

1. **Kualitas OCR tidak bisa dikontrol dalam meta-analisisnya**, karena studi primer
   jarang melaporkannya. Penulis mengakuinya sebagai sumber heterogenitas yang tidak
   terjelaskan. Gap Fase 2 berangkat dari sini: kita menjadikan mesin OCR variabel yang
   terukur.
2. Kelemahan fusion yang didaftar penulis: **sulit mencapai interaksi lintas-modalitas
   yang sungguh sinergis**, dan **sensitif terhadap kualitas data, misalnya galat OCR**.
   Temuan Fase 1 kita adalah contoh konkret yang pertama: cabang teks fusion praktis
   tidak terpakai ([[11-diagnostik-cabang-teks]]).
3. Pedoman pelaporan penulis: sertakan baseline unimodal, F1 untuk data timpang, ablasi
   lengkap, dan uji statistik. Fase 1 sudah memenuhi tiga yang pertama. Uji statistik
   kita tambahkan di Fase 2.
4. Akurasi naik secara signifikan tapi F1 tidak. Ini sejalan dengan temuan kita bahwa OA
   dan macro F1 bisa bergerak terpisah ([[12-perbaikan-norm-dan-init]]).

## Catatan kritis saya

Basis datanya hanya Scopus, yang diakui penulis sebagai bias seleksi. Meta-analisisnya
menggabungkan dataset, protokol split, dan metrik yang berbeda-beda. Angka +5,28 poin
adalah rata-rata lapangan, bukan prediksi untuk satu dataset.

## Ambiguitas & asumsi

> Detail yang TIDAK disebutkan paper. Jangan ditambal dengan tebakan diam-diam.

Apakah Audebert 2019 atau Tobacco3482 termasuk dalam 139 studi tidak saya periksa di
lampiran.
