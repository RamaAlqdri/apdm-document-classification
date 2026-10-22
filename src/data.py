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
            try:
                link.symlink_to(path, target_is_directory=True)
            except OSError as exc:
                # Windows refuses symlinks without Developer Mode or admin rights.
                # The link is cosmetic — everything downstream uses the returned
                # path — so this must not abort the download.
                print(f"symlink {link} dilewati ({exc}); memakai cache langsung: {path}")
                (CONFIG["data_raw"] / "LOKASI_CITRA.txt").write_text(
                    str(path), encoding="utf-8")
    return path


def _extract_safely(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract `tar` into `dest`, refusing members that escape it.

    The `filter=` argument only exists on Python >=3.11.4/3.12; on older
    interpreters it raises TypeError, so the same protection is applied by hand.
    """
    try:
        tar.extractall(dest, filter="data")
        return
    except TypeError:
        pass

    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if dest != target and dest not in target.parents:
            raise RuntimeError(f"anggota arsip keluar dari {dest}: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"tipe anggota arsip tidak diizinkan: {member.name}")
    tar.extractall(dest)


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
            _extract_safely(tar, out)
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


MANIFEST_FIELDS = ["id", "kelas", "path_citra", "path_teks", "n_kata", "n_karakter"]


def write_manifest(rows: list[dict], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def audit_and_pair(image_path: Path, text_path: Path, expected_classes: int = 10) -> dict:
    """Locate both archives' class folders, reconcile names, pair files, and report.

    Raises when not every class can be mapped: an alias missing from CLASS_ALIASES
    must be added there by hand, never guessed.
    """
    from collections import Counter

    image_root, img_rep = pick_dataset_root(find_class_dirs(image_path, ".jpg"),
                                            expected_classes)
    text_root, txt_rep = pick_dataset_root(find_class_dirs(text_path, ".txt"),
                                           expected_classes)
    rec = reconcile_classes([d.name for d in image_root.iterdir() if d.is_dir()],
                            [d.name for d in text_root.iterdir() if d.is_dir()])
    if rec["n_kelas_cocok"] != expected_classes:
        raise RuntimeError(f"hanya {rec['n_kelas_cocok']} kelas terpetakan: {rec}. "
                           "Tambahkan alias ke CLASS_ALIASES di src/data.py.")
    rows, pair_report = build_manifest(image_root, text_root, rec["mapping"])

    def stems_in_many_classes(root, suffix):
        c = Counter(f.stem for d in root.iterdir() if d.is_dir()
                    for f in d.iterdir() if f.suffix.lower() == suffix)
        return sorted(s for s, n in c.items() if n > 1)

    return {
        "rows": rows,
        "image_root": image_root, "text_root": text_root,
        "kandidat_citra": img_rep, "kandidat_teks": txt_rep,
        "rekonsiliasi": rec, "pairing": pair_report,
        "stem_ganda_citra": stems_in_many_classes(image_root, ".jpg"),
        "stem_ganda_teks": stems_in_many_classes(text_root, ".txt"),
    }


def prepare_dataset(out: Path | None = None) -> dict:
    """Download both modalities, audit them, and write manifest.csv. Idempotent."""
    out = CONFIG["data_processed"] / "manifest.csv" if out is None else out
    report = audit_and_pair(download_images(), download_text())
    report["manifest"] = write_manifest(report["rows"], out)
    return report


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

        # the one-call audit agrees with the step-by-step one and writes a CSV
        rep = audit_and_pair(tmp / "arsip", tmp / "teks", expected_classes=1)
        assert len(rep["rows"]) == 2 and rep["stem_ganda_citra"] == [], rep
        path = write_manifest(rep["rows"], tmp / "m.csv")
        assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(MANIFEST_FIELDS)
        try:
            audit_and_pair(tmp / "arsip", tmp / "teks", expected_classes=2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("kelas yang tidak lengkap harus ditolak")
        print("data self-check ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    _self_check()
