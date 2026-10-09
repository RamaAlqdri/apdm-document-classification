---
judul: Papan Progress
tipe: referensi
tahap: "00"
tanggal: 2026-09-28
status: selesai
tags:
  - tipe/referensi
  - tahap/00
terkait:
  - "[[peta-proyek]]"
  - "[[peta-tag]]"
---

# Papan Progress

Butuh plugin **Dataview**. Checklist manual ada di [[peta-proyek]]; halaman ini
membaca status dari frontmatter tiap catatan, jadi kalau keduanya berbeda,
yang ini lebih dipercaya.

## Catatan per tahap

```dataview
TABLE WITHOUT ID
  tahap AS "Tahap",
  length(rows) AS "Jumlah",
  rows.file.link AS "Catatan"
FROM ""
WHERE tipe AND tahap
GROUP BY tahap
SORT tahap ASC
```

## Status per tahap

```dataview
TABLE WITHOUT ID
  tahap AS "Tahap",
  length(filter(rows.status, (s) => s = "selesai")) AS "Selesai",
  length(filter(rows.status, (s) => s = "wip")) AS "WIP",
  length(filter(rows.status, (s) => s = "todo")) AS "Todo",
  length(filter(rows.status, (s) => s = "belum-dijalankan")) AS "Belum jalan"
FROM ""
WHERE tipe AND tahap
GROUP BY tahap
SORT tahap ASC
```

## Yang tertinggal (wip / todo)

```dataview
TABLE WITHOUT ID
  file.link AS "Catatan",
  tahap AS "Tahap",
  status AS "Status",
  tanggal AS "Tanggal"
FROM ""
WHERE tipe AND (status = "wip" OR status = "todo")
SORT tahap ASC, file.name ASC
```

## Catatan yatim (tidak punya link keluar)

Aturan vault: tiap catatan minimal 2 link keluar dan terhubung ke MOC-nya.

```dataview
LIST
FROM ""
WHERE tipe AND length(file.outlinks) < 2
SORT file.name ASC
```

## Indeks log

Query Dataview di bawah mengambilnya otomatis, tapi wikilink eksplisit diperlukan supaya
log tidak jadi catatan yatim di graf Obsidian:

- [[2026-09-28-tahap-01]]
- [[2026-09-28-tahap-02]]
- [[2026-09-28-tahap-03]]
- [[2026-09-28-tahap-04]]
- [[2026-09-28-tahap-05]]
- [[2026-09-28-tahap-06]]
- [[2026-09-28-tahap-07]]
- [[2026-09-30-migrasi-lingkungan-py312]]
- [[2026-09-30-pengisian-hasil]]
- [[2026-09-30-tahap-08-sintesis]]
- [[2026-10-09-tahap-09-revisi-arah]]
- [[2026-10-09-tahap-10-ocr-multi-mesin]]

## Log terakhir

```dataview
TABLE WITHOUT ID
  file.link AS "Log",
  tahap AS "Tahap",
  tanggal AS "Tanggal"
FROM "40-log"
WHERE tipe = "log"
SORT tanggal DESC
LIMIT 10
```
