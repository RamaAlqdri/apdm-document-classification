#!/usr/bin/env python3
"""Run the whole pipeline: notebooks 01 through 06, in order, with outputs saved.

    python run_all.py              # setelan paper: 100/200/200 epoch, 3 seed
    python run_all.py --cepat      # 15/20/20 epoch, 1 seed, untuk uji pipeline
    python run_all.py --dari 04    # lanjut dari notebook tertentu
    python run_all.py --periksa    # hanya cek prasyarat, tidak menjalankan apa pun

Everything the machine can decide for itself is decided for itself: GPU micro-batch,
mixed precision, DataLoader workers, and the spaCy and FastText downloads. The two
things that genuinely need a human are the Kaggle token and a CUDA-enabled torch, and
both are checked before anything long starts.
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
NOTEBOOKS = [
    ("01", "01_data_audit", "audit data + unduh dataset (3,3 GB)"),
    ("02", "02_text_features", "fitur teks + unduh FastText (7 GB)"),
    ("03", "03_baseline_teks", "baseline MLP dan CNN1D"),
    ("04", "04_baseline_citra", "baseline MobileNetV2"),
    ("05", "05_fusion", "fusion concat dan sum + oracle"),
    ("06", "06_ablasi", "ablasi degradasi dan missing modality"),
    ("07", "07_uji_lanjutan", "uji lanjutan: missing modality pada sum + concat tanpa early stopping"),
]

# Hasil yang mahal dan tidak boleh hilang karena run ulang yang tidak sengaja.
HASIL_PENTING = [
    "reports/tabel_utama.csv",
    "reports/ablasi1_degradasi_citra.csv",
]


def _ok(msg): print(f"  \033[32mOK\033[0m   {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m {msg}")
def _bad(msg): print(f"  \033[31mGAGAL\033[0m {msg}")


def periksa_prasyarat() -> list[str]:
    """Check everything cheap before anything expensive. Returns blocking errors."""
    galat: list[str] = []
    print("Memeriksa prasyarat\n")

    if sys.version_info[:2] != (3, 12):
        _warn(f"Python {sys.version_info.major}.{sys.version_info.minor}; "
              f"3.12 yang diuji")
    else:
        _ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    try:
        import torch
    except ImportError:
        galat.append("torch belum terpasang: pip install -r requirements.txt")
        _bad("torch tidak ada")
    else:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            _ok(f"CUDA aktif: {p.name}, {p.total_memory / 1e9:.1f} GB VRAM")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _ok("Apple MPS aktif")
        else:
            _warn("tidak ada GPU terdeteksi — training akan SANGAT lambat. "
                  "Kalau ada GPU NVIDIA, pasang torch dari indeks CUDA: "
                  "pip install torch torchvision "
                  "--index-url https://download.pytorch.org/whl/cu124")

    for modul, paket in [("pandas", "pandas"), ("matplotlib", "matplotlib"),
                         ("sklearn", "scikit-learn"), ("PIL", "pillow"),
                         ("kagglehub", "kagglehub"), ("spacy", "spacy"),
                         ("fasttext", "fasttext-wheel"), ("nbformat", "jupyter")]:
        try:
            __import__(modul)
        except ImportError:
            galat.append(f"{paket} belum terpasang: pip install -r requirements.txt")
            _bad(f"{paket} tidak ada")
    if not galat:
        _ok("semua dependency Python ada")

    # Checked even under --periksa: this is precisely what you want to confirm
    # BEFORE starting a run that takes hours.
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists() or os.environ.get("KAGGLE_KEY"):
        _ok("kredensial Kaggle ditemukan")
    else:
        galat.append(
            f"kredensial Kaggle tidak ada. Buat token di "
            f"kaggle.com -> Settings -> API -> Create New Token, "
            f"simpan sebagai {token}"
        )
        _bad("kredensial Kaggle tidak ada")

    bebas = shutil.disk_usage(ROOT).free / 1e9
    if bebas < 16:
        _warn(f"disk kosong {bebas:.0f} GB; butuh ± 15 GB")
    else:
        _ok(f"disk kosong {bebas:.0f} GB")

    return galat


def jalankan_self_check() -> bool:
    """The ten module self-checks. Cheap, and they run without any dataset."""
    modul = ["config", "data", "splits", "text_features", "datasets", "ablation",
             "models.text_models", "models.image_model", "models.fusion", "train"]
    print("\nMenjalankan self-check modul\n")
    gagal = []
    for m in modul:
        r = subprocess.run([sys.executable, "-m", f"src.{m}"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            _ok(f"src/{m}")
        else:
            _bad(f"src/{m}")
            print("      " + (r.stderr.strip().splitlines() or ["?"])[-1])
            gagal.append(m)
    return not gagal


def jalankan_notebook(nama: str, deskripsi: str) -> bool:
    src = ROOT / "notebooks" / f"{nama}.ipynb"
    print(f"\n{'=' * 70}\n{nama} — {deskripsi}\n{'=' * 70}")
    t0 = time.perf_counter()
    # timeout=-1: the default 30s per cell would kill training at the first epoch.
    # Executed in place so the notebook keeps its outputs, which is the deliverable.
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1", str(src)],
        cwd=ROOT,
    )
    menit = (time.perf_counter() - t0) / 60
    if r.returncode == 0:
        print(f"\n{nama} selesai dalam {menit:.1f} menit")
        return True
    print(f"\n{nama} GAGAL setelah {menit:.1f} menit.")
    print(f"Buka notebooks/{nama}.ipynb — sel yang gagal menyimpan traceback-nya.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cepat", action="store_true",
                    help="epoch dan seed dikurangi; deviasinya dicatat otomatis")
    ap.add_argument("--dari", default="01", help="mulai dari notebook ini (01-06)")
    ap.add_argument("--periksa", action="store_true",
                    help="hanya cek prasyarat dan self-check")
    ap.add_argument("--timpa", action="store_true",
                    help="izinkan menimpa hasil run sebelumnya (checkpoint, log, CSV)")
    args = ap.parse_args()

    # Notebook 01-06 menulis ke nama file yang sama setiap kali dijalankan, jadi run
    # ulang menghapus hasil sebelumnya — termasuk checkpoint dan log yang butuh belasan
    # jam. Notebook 07 memakai nama sendiri (...-noES..., uji_lanjutan.csv) sehingga
    # aman dan tidak dijaga di sini.
    ada = [f for f in HASIL_PENTING if (ROOT / f).exists()]
    if ada and args.dari != "07" and not args.timpa:
        print(f"\n{'=' * 70}")
        print("BERHENTI: hasil run sebelumnya sudah ada di repo ini.")
        print(f"{'=' * 70}\n")
        for f in ada:
            print(f"  {f}")
        print("\nMenjalankan notebook 01-06 lagi akan menimpa file di atas, plus")
        print("seluruh checkpoint di models/, log di reports/, dan output di dalam")
        print("notebook itu sendiri.\n")
        print("Yang mungkin Anda maksud:")
        print("  python run_all.py --dari 07     # uji lanjutan, tidak menimpa apa pun")
        print("  python run_all.py --timpa       # ya, saya memang mau menimpa semuanya")
        print("\nUntuk menyimpan hasil lama sebelum menimpa, salin dulu models/ dan")
        print("reports/ ke folder lain.")
        return 1

    if args.cepat:
        os.environ["APDM_CEPAT"] = "1"

    galat = periksa_prasyarat()
    if not jalankan_self_check():
        print("\nSelf-check gagal. Perbaiki itu dulu — tidak ada gunanya "
              "mengunduh 3 GB kalau modulnya sendiri rusak.")
        return 1
    if galat:
        print("\nPrasyarat belum lengkap:\n")
        for g in galat:
            print(f"  - {g}")
        return 1

    sys.path.insert(0, str(ROOT))
    from src.config import CONFIG, deviasi_runtime

    print(f"\n{'=' * 70}\nKonfigurasi yang dipilih otomatis\n{'=' * 70}")
    print(f"  epoch        : {CONFIG['epochs']}")
    print(f"  seed         : {CONFIG['seeds']}")
    print(f"  batch        : {CONFIG['batch_size']}"
          + (f" (micro-batch {CONFIG['micro_batch_size']}, akumulasi gradien)"
             if CONFIG["micro_batch_size"] else ""))
    print(f"  worker       : {CONFIG['num_workers']}")
    print(f"  mode cepat   : {CONFIG['quick']}")
    dv = deviasi_runtime()
    print("  deviasi      : " + ("tidak ada" if not dv else ""))
    for d in dv:
        print(f"    - {d}")

    if args.periksa:
        print("\n--periksa: berhenti di sini, tidak menjalankan notebook.")
        return 0

    if not CONFIG["quick"]:
        print("\nSetelan paper penuh. Di GPU kelas RTX 3050 perkiraannya 12-18 jam.")
        print("Uji pipeline dulu dengan: python run_all.py --cepat")

    mulai = [i for i, (n, _, _) in enumerate(NOTEBOOKS) if n == args.dari]
    if not mulai:
        print(f"--dari harus salah satu dari {[n for n, _, _ in NOTEBOOKS]}")
        return 1


    t0 = time.perf_counter()
    for _, nama, deskripsi in NOTEBOOKS[mulai[0]:]:
        if not jalankan_notebook(nama, deskripsi):
            print(f"\nLanjutkan setelah diperbaiki dengan: "
                  f"python run_all.py --dari {nama[:2]}"
                  + (" --cepat" if args.cepat else ""))
            return 1

    print(f"\n{'=' * 70}")
    print(f"Semua notebook selesai dalam {(time.perf_counter() - t0) / 3600:.1f} jam.")
    print("Angkanya ada di notebook (sel ringkasan terakhir masing-masing) dan di "
          "reports/.")
    print("Langkah terakhir manual: salin angka itu ke catatan di "
          "vault/30-eksperimen/ dan vault/50-hasil/, lalu ubah statusnya dari "
          "belum-dijalankan menjadi selesai.")
    if dv:
        print("\nJangan lupa catat deviasi runtime di catatan eksperimen:")
        for d in dv:
            print(f"  - {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
