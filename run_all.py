#!/usr/bin/env python3
"""Run the whole Fase 2 pipeline: notebooks/pipeline.ipynb, executed in place.

    python run_all.py --periksa    # prasyarat + self-check, tidak menjalankan apa pun
    python run_all.py --cepat      # 1 seed, epoch dipangkas: uji pipeline
    python run_all.py              # setelan penuh: 3 seed, 100/200 epoch
    python run_all.py --timpa      # latih ulang walau hasil di disk sudah ada

The notebook is resumable: OCR pages, text features and trained models already on
disk are reused, so after a crash the same command continues where it stopped.
Same as opening the notebook and pressing Run All; this is the terminal route.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "pipeline.ipynb"


def check_prerequisites() -> list[str]:
    """Cheap checks before anything expensive. Returns blocking problems."""
    sys.path.insert(0, str(ROOT))
    from src import setup_env as SE

    problems = []
    if sys.version_info[:2] != (3, 12):
        print(f"  WARN  Python {sys.version_info.major}.{sys.version_info.minor}; 3.12 yang diuji")
    info = SE.torch_info()
    if info is None:
        problems.append("torch belum terpasang: pip install -r requirements.txt")
    elif info["cuda"]:
        print(f"  OK    CUDA: {info['nama']}, {info['vram_gb']} GB VRAM")
    else:
        print("  WARN  tidak ada CUDA: training akan sangat lambat"
              + (" (GPU NVIDIA ada: pasang torch dari indeks CUDA)" if SE.has_nvidia() else ""))
    for mod, pkg in [("pandas", "pandas"), ("sklearn", "scikit-learn"), ("scipy", "scipy"),
                     ("spacy", "spacy"), ("fasttext", "fasttext-wheel"),
                     ("wordfreq", "wordfreq"), ("kagglehub", "kagglehub"),
                     ("nbformat", "jupyter")]:
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"{pkg} belum terpasang: pip install -r requirements.txt")
    if not SE.kaggle_ready():
        problems.append(f"kredensial Kaggle tidak ada: simpan token sebagai {SE.KAGGLE_JSON} "
                        "(kaggle.com -> Settings -> API -> Create New Token)")
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"  {'OK  ' if free >= 25 else 'WARN'}  disk kosong {free:.0f} GB "
          "(butuh ± 25 GB: citra 3,3 + FastText 7 + 4 sekuens teks 4 + venv OCR)")

    print("\nself-check modul:")
    for m, err in SE.run_self_checks().items():
        print(f"  {'OK  ' if err is None else 'GAGAL'}  src/{m}" + (f"  -> {err}" if err else ""))
        if err:
            problems.append(f"self-check src/{m} gagal")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cepat", action="store_true", help="1 seed, epoch dipangkas")
    ap.add_argument("--timpa", action="store_true", help="latih ulang walau hasil sudah ada")
    ap.add_argument("--periksa", action="store_true", help="hanya prasyarat + self-check")
    args = ap.parse_args()

    os.environ["APDM_CEPAT"] = "1" if args.cepat else "0"
    os.environ["APDM_TIMPA"] = "1" if args.timpa else "0"
    print("Memeriksa prasyarat\n")
    problems = check_prerequisites()
    if problems:
        print("\nBelum bisa jalan:\n" + "\n".join(f"  - {p}" for p in problems))
        return 1
    if args.periksa:
        print("\n--periksa: semua siap, tidak menjalankan notebook.")
        return 0

    print(f"\nMenjalankan {NOTEBOOK.relative_to(ROOT)} "
          f"({'cepat' if args.cepat else 'penuh'}{', timpa' if args.timpa else ''})")
    t0 = time.perf_counter()
    # timeout=-1: the default 30 s per cell would kill training at the first epoch.
    # In place, so the executed notebook with its outputs is the deliverable.
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
                        str(NOTEBOOK)], cwd=ROOT)
    hours = (time.perf_counter() - t0) / 3600
    if r.returncode:
        print(f"\nGAGAL setelah {hours:.2f} jam. Buka notebook: sel yang gagal menyimpan "
              "traceback-nya. Setelah diperbaiki, jalankan perintah yang sama lagi; "
              "yang sudah selesai tidak diulang.")
        return 1
    print(f"\nSelesai dalam {hours:.2f} jam. Hasil: reports/fase2/, gambar: reports/figures/fase2_*.")
    print("Langkah manual: salin angka dari sel ringkasan terakhir ke vault/30-eksperimen/ "
          "dan vault/50-hasil/, lalu ubah status belum-dijalankan -> selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
