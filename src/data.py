"""Acquisition, verification and pairing of the two Tobacco3482 modalities.

Nothing here is executed on the authoring machine: download and audit run on the
compute machine. Every function is pure enough to be checked with the synthetic
fixture in __main__.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from .config import CONFIG

KAGGLE_DATASET = "patrickaudriaz/tobacco3482jpg"

# Release v1.0 carries only TWO assets, and "small" is lowercase. The project plan
# said four; it was wrong.
QS_OCR_URL = (
    "https://github.com/QuickSign/ocrized-text-dataset/releases/download/"
    "v1.0/QS-OCR-small.tar.gz"
)

# Image folders use the short code, the text archive follows the README wording.
# Kept explicit on purpose: nothing is normalised silently.
CLASS_ALIASES = {
    "ADVE": "Advertisement",
    "Advertisement": "Advertisement",
    "Email": "Email",
    "Form": "Form",
    "Letter": "Letter",
    "Memo": "Memo",
    "News": "News",
    "Note": "Note",
    "Report": "Report",
    "Resume": "Resume",
    "Scientific": "Scientific",
}


# --------------------------------------------------------------------------- #
# acquisition
# --------------------------------------------------------------------------- #

def download_images(link_into_raw: bool = True) -> Path:
    """Fetch the Kaggle JPG dataset (~3.3GB) and return the folder holding it.

    kagglehub keeps its own cache, so the bytes are not copied into data/raw —
    only a symlink is placed there to keep the documented layout.
    """
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    if link_into_raw:
        CONFIG["data_raw"].mkdir(parents=True, exist_ok=True)
        link = CONFIG["data_raw"] / "tobacco3482-jpg"
        if not link.exists():
            link.symlink_to(path, target_is_directory=True)
    return path


def download_text() -> Path:
    """Fetch and extract QS-OCR-small (2.5MB) into data/raw/. Returns its folder."""
    raw = CONFIG["data_raw"]
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "QS-OCR-small.tar.gz"
    out = raw / "qs-ocr-small"

    if not archive.exists():
        urllib.request.urlretrieve(QS_OCR_URL, archive)
    if not out.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(out, filter="data")  # filter= is required on py>=3.12
    return out


# --------------------------------------------------------------------------- #
# layout discovery
# --------------------------------------------------------------------------- #

def find_class_dirs(root: Path, suffix: str, min_classes: int = 5) -> list[Path]:
    """Return every directory that directly holds >=1 file with `suffix`.

    Archives from Kaggle are often nested twice (foo/foo/<class>/), and some
    mirrors ship a duplicate copy alongside. Rather than guessing, this returns
    all candidate class directories and lets the caller report what it found.
    """
    hits = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(f.suffix.lower() == suffix for f in d.iterdir() if f.is_file()):
            hits.append(d)
    if root.is_dir() and any(
        f.suffix.lower() == suffix for f in root.iterdir() if f.is_file()
    ):
        hits.append(root)
    return hits


def group_by_parent(class_dirs: list[Path]) -> dict[Path, list[Path]]:
    """Group candidate class directories by their parent, so duplicate nesting
    shows up as two parents each owning a full set of classes."""
    groups: dict[Path, list[Path]] = {}
    for d in class_dirs:
        groups.setdefault(d.parent, []).append(d)
    return groups


def pick_dataset_root(class_dirs: list[Path], expected: int) -> tuple[Path, dict]:
    """Choose the parent that owns `expected` class folders.

    Returns (root, report). The report always lists every candidate so a
    duplicate nested copy is visible in the audit instead of being swallowed.
    """
    groups = group_by_parent(class_dirs)
    report = {
        "candidates": {str(p): sorted(d.name for d in ds) for p, ds in groups.items()},
        "n_candidates": len(groups),
    }
    exact = [p for p, ds in groups.items() if len(ds) == expected]
    if len(exact) == 1:
        return exact[0], report
    if len(exact) > 1:
        report["warning"] = (
            f"{len(exact)} folder masing-masing punya {expected} kelas — "
            "kemungkinan arsip bersarang duplikat. Dipilih yang path-nya terpendek."
        )
        return min(exact, key=lambda p: len(p.parts)), report
    report["warning"] = (
        f"tidak ada folder dengan tepat {expected} kelas; "
        f"terbanyak = {max((len(d) for d in groups.values()), default=0)}"
    )
    return max(groups, key=lambda p: len(groups[p])), report


# --------------------------------------------------------------------------- #
# class-name reconciliation
# --------------------------------------------------------------------------- #

def reconcile_classes(image_names: list[str], text_names: list[str]) -> dict:
    """Map both archives' folder names onto one canonical set via CLASS_ALIASES.

    Returns a report rather than a silently-fixed mapping: anything the alias
    table does not cover is listed as unmatched so it can be decided by hand.
    """
    def canon(names):
        out, unknown = {}, []
        for n in names:
            key = CLASS_ALIASES.get(n) or CLASS_ALIASES.get(n.strip().title())
            (out.setdefault(key, n) if key else unknown.append(n))
        return out, unknown

    img, img_unknown = canon(image_names)
    txt, txt_unknown = canon(text_names)
    shared = sorted(set(img) & set(txt))
    return {
        "mapping": {c: {"citra": img.get(c), "teks": txt.get(c)} for c in shared},
        "hanya_di_citra": sorted(set(img) - set(txt)),
        "hanya_di_teks": sorted(set(txt) - set(img)),
        "tidak_dikenali_citra": sorted(img_unknown),
        "tidak_dikenali_teks": sorted(txt_unknown),
        "n_kelas_cocok": len(shared),
    }


# --------------------------------------------------------------------------- #
# pairing + manifest
# --------------------------------------------------------------------------- #

def pair_class(img_dir: Path, txt_dir: Path, img_suffix: str = ".jpg") -> dict:
    """Match image stems against text stems inside one class. Reports orphans."""
    images = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() == img_suffix}
    texts = {f.stem: f for f in txt_dir.iterdir() if f.suffix.lower() == ".txt"}
    both = sorted(images.keys() & texts.keys())
    return {
        "pairs": [(s, images[s], texts[s]) for s in both],
        "citra_yatim": sorted(images.keys() - texts.keys()),
        "teks_yatim": sorted(texts.keys() - images.keys()),
    }


def build_manifest(image_root: Path, text_root: Path, mapping: dict) -> tuple:
    """Build the manifest rows plus a per-class pairing report.

    Returns (rows, report). Rows are dicts, so the notebook owns the DataFrame.
    """
    rows, report = [], {}
    for kelas, src in sorted(mapping.items()):
        res = pair_class(image_root / src["citra"], text_root / src["teks"])
        report[kelas] = {
            "n_pasangan": len(res["pairs"]),
            "n_citra_yatim": len(res["citra_yatim"]),
            "n_teks_yatim": len(res["teks_yatim"]),
            "contoh_citra_yatim": res["citra_yatim"][:5],
            "contoh_teks_yatim": res["teks_yatim"][:5],
        }
        for stem, img, txt in res["pairs"]:
            text = txt.read_text(encoding="utf-8", errors="replace")
            rows.append(
                {
                    "id": stem,
                    "kelas": kelas,
                    "path_citra": str(img),
                    "path_teks": str(txt),
                    "n_kata": len(text.split()),
                    "n_karakter": len(text),
                }
            )
    return rows, report


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    """Synthetic fixture: nested duplicate, an alias, and one orphan each way."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        # images nested twice, class folder named with the short code
        img = tmp / "arsip" / "arsip" / "ADVE"
        img.mkdir(parents=True)
        for stem in ("a", "b", "hanya_citra"):
            (img / f"{stem}.jpg").write_bytes(b"x")
        # text archive uses the long name, and has one text without an image
        txt = tmp / "teks" / "Advertisement"
        txt.mkdir(parents=True)
        for stem in ("a", "b", "hanya_teks"):
            (txt / f"{stem}.txt").write_text("satu dua tiga", encoding="utf-8")

        img_dirs = find_class_dirs(tmp / "arsip", ".jpg")
        img_root, img_report = pick_dataset_root(img_dirs, expected=1)
        assert img_root == img.parent, img_report
        txt_root, _ = pick_dataset_root(find_class_dirs(tmp / "teks", ".txt"), 1)

        rec = reconcile_classes([d.name for d in img_dirs], ["Advertisement"])
        assert rec["n_kelas_cocok"] == 1, rec
        assert rec["mapping"]["Advertisement"] == {
            "citra": "ADVE",
            "teks": "Advertisement",
        }, rec["mapping"]

        rows, report = build_manifest(img_root, txt_root, rec["mapping"])
        assert len(rows) == 2, rows
        assert report["Advertisement"]["n_citra_yatim"] == 1, report
        assert report["Advertisement"]["n_teks_yatim"] == 1, report
        assert rows[0]["n_kata"] == 3 and rows[0]["n_karakter"] == 13, rows[0]
        print("data self-check ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    _self_check()
