# Template LaTeX naskah akademik FILKOM UB

Kerangka kosong yang formatnya identik dengan naskah yang sudah jadi. Salin folder ini ke
proyek baru, ganti data sampul di `main.tex`, lalu tulis isinya.

```bash
cp -r template-latex-ub ~/proyek-baru/naskah
cd ~/proyek-baru/naskah
latexmk main.tex
```

Butuh pdfLaTeX dan biber. Uji kompilasi: 8 halaman, 0 *overfull*, 0 rujukan menggantung.

**Baca [PANDUAN.md](PANDUAN.md)** untuk langkah lengkap, spesifikasi format yang sudah
dikunci, perintah khusus yang disediakan, penanganan masalah, dan daftar periksa sebelum
menyerahkan naskah.
