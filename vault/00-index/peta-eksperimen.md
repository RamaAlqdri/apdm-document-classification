---
judul: Peta Eksperimen
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
  - "[[peta-metode]]"
---

# Peta Eksperimen

Agregasi otomatis seluruh catatan di `30-eksperimen/`. Butuh plugin **Dataview**.
Tabel kosong berarti belum ada run nyata — itu benar, bukan error. Lihat aturan 3
di `CLAUDE.md`: angka hanya boleh berasal dari eksekusi.

## Semua run

```dataview
TABLE WITHOUT ID
  file.link AS "Eksperimen",
  model AS "Model",
  split AS "Split",
  seed AS "Seed",
  oa AS "OA",
  macro_f1 AS "Macro F1",
  durasi AS "Durasi"
FROM "30-eksperimen"
WHERE tipe = "eksperimen"
SORT model ASC, seed ASC
```

## Yang sudah dijalankan, urut akurasi

```dataview
TABLE WITHOUT ID
  file.link AS "Eksperimen",
  model AS "Model",
  oa AS "OA",
  macro_f1 AS "Macro F1"
FROM "30-eksperimen"
WHERE tipe = "eksperimen" AND oa
SORT oa DESC
```

## Yang belum dijalankan

```dataview
LIST
FROM "30-eksperimen"
WHERE status = "belum-dijalankan"
```

## Temuan yang ditandai

```dataview
TABLE WITHOUT ID
  file.link AS "Catatan",
  filter(file.tags, (t) => startswith(t, "#temuan")) AS "Temuan"
FROM "30-eksperimen" OR "50-hasil"
WHERE contains(string(file.tags), "#temuan")
SORT file.name ASC
```

## Target pembanding (angka paper, Tabel 3)

| Model | OA | F1 |
|---|---|---|
| TEXT | 73,8% | 0,71 |
| IMAGE | 84,5% | 0,82 |
| FUSION | 87,8% | 0,86 |
| Oracle | 92,1% | — |

Selisih 2-4% wajar. Yang wajib konsisten adalah urutannya.

## Hasil akhir, per 2026-09-30

| Model | Kita | Paper |
|---|---|---|
| TEXT (CNN1D) | 0,7266 ± 0,0059 | 0,738 |
| IMAGE | 0,8086 ± 0,0210 | 0,845 |
| FUSION concat | 0,8342 ± 0,0105 | 0,878 |
| FUSION sum | 0,8275 ± 0,0164 | tidak dilaporkan |
| Oracle | 0,8977 | 0,921 |

Urutan terpenuhi. Tapi lihat [[sintesis-akhir]] sebelum menyimpulkan bahwa fusion
memakai kedua modalitas — ia tidak, sampai diperbaiki di
[[12-perbaikan-norm-dan-init]]. Tabel lengkap dengan F1 per kelas: [[tabel-utama]].
