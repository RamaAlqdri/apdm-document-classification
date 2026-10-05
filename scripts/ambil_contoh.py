"""Fetch a few Tobacco3482 samples (image + OCR text) for slides.

Not part of the pipeline: the full dataset still lives on the compute machine.
Needs ~/.kaggle/kaggle.json and data/raw/QS-OCR-small/ already extracted.

    python3 scripts/ambil_contoh.py [--per-kelas 3]
"""
import argparse, csv, io, json, pathlib, shutil, urllib.parse, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data/raw/QS-OCR-small"
OUT = ROOT / "data/contoh"
API = "https://www.kaggle.com/api/v1/datasets/download/patrickaudriaz/tobacco3482jpg/"


def fetch_image(rel_path: pathlib.Path, dest: pathlib.Path, auth: str) -> int:
    """Download one file from the Kaggle dataset. Slashes must be percent-encoded.

    Kaggle serves some single files raw and others wrapped in a one-member zip,
    so handle both rather than trusting the extension.
    """
    url = API + urllib.parse.quote(str(rel_path), safe="")
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req) as r:
        ctype, blob = r.headers.get("Content-Type", ""), r.read()
    if ctype.startswith("image/"):
        dest.write_bytes(blob)
    elif "zip" in ctype:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
            if len(names) != 1:
                raise RuntimeError(f"{rel_path}: zip berisi {len(names)} jpg, bukan 1")
            dest.write_bytes(z.read(names[0]))
    else:
        raise RuntimeError(f"{rel_path}: bukan gambar atau zip ({ctype})")
    if dest.read_bytes()[:2] != b"\xff\xd8":
        raise RuntimeError(f"{rel_path}: hasilnya bukan JPEG")
    return dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kelas", type=int, default=3)
    n = ap.parse_args().per_kelas

    import base64
    cred = json.loads((pathlib.Path.home() / ".kaggle/kaggle.json").read_text())
    auth = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()

    if not TEXT_DIR.is_dir():
        raise SystemExit(f"belum ada {TEXT_DIR} — ekstrak QS-OCR-small.tar.gz dulu")

    rows = []
    for class_dir in sorted(p for p in TEXT_DIR.iterdir() if p.is_dir()):
        dest_dir = OUT / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for txt in sorted(class_dir.glob("*.txt"))[:n]:
            img = dest_dir / f"{txt.stem}.jpg"
            size = img.stat().st_size if img.exists() else fetch_image(
                pathlib.Path("Tobacco3482-jpg") / class_dir.name / f"{txt.stem}.jpg", img, auth)
            shutil.copy2(txt, dest_dir / txt.name)
            body = txt.read_text(errors="replace")
            rows.append({"kelas": class_dir.name, "stem": txt.stem,
                         "citra": img.relative_to(ROOT), "teks": (dest_dir / txt.name).relative_to(ROOT),
                         "kb_citra": round(size / 1024, 1), "karakter_teks": len(body.strip()),
                         "kata_teks": len(body.split())})
            print(f"{class_dir.name:12} {txt.stem:18} {size/1024:7.1f} KB  {len(body.split()):5} kata")

    with (OUT / "indeks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} contoh di {OUT.relative_to(ROOT)}, indeks di {(OUT/'indeks.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
