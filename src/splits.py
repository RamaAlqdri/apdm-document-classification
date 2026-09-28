"""Train/val/test splits, frozen to disk so every experiment sees the same rows.

DEVIATION FROM THE PAPER: the authors use k-fold cross-validation with 800
training documents. We use three stratified random splits (seeds 42/43/44) and
carve a small validation set out of the 800 for early stopping, which the paper
does not do — it trains for a fixed number of epochs. Both deviations are listed
in vault/10-jurnal/1907.06370-spesifikasi-implementasi.md.

Splits are stored as indices into manifest.csv plus a fingerprint of that file, so
applying a split to a differently-ordered manifest fails loudly instead of
silently scrambling the labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import CONFIG


def manifest_fingerprint(labels: list[str]) -> str:
    """Hash of the label column, in order. Cheap guard against a reordered manifest."""
    return hashlib.sha256("\n".join(labels).encode()).hexdigest()[:16]


def make_split(labels: list[str], seed: int, n_train: int | None = None,
               val_fraction: float | None = None) -> dict:
    """Stratified n_train / val / test split over `labels`.

    `n_train` counts train+val together, matching the paper's "800 documents for
    training"; the validation rows are taken out of those 800, not added on top.
    """
    n_train = CONFIG["n_train"] if n_train is None else n_train
    val_fraction = CONFIG["val_fraction"] if val_fraction is None else val_fraction
    idx = list(range(len(labels)))

    fit, test = train_test_split(
        idx, train_size=n_train, stratify=labels, random_state=seed
    )
    train, val = train_test_split(
        fit,
        test_size=val_fraction,
        stratify=[labels[i] for i in fit],
        random_state=seed,
    )
    return {
        "seed": seed,
        "fingerprint": manifest_fingerprint(labels),
        "n_total": len(labels),
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "catatan_deviasi": (
            "3 random split terstratifikasi, bukan k-fold cross-validation seperti "
            "paper. Validation dipotong dari 800 train untuk early stopping, yang "
            "juga tidak dilakukan paper."
        ),
    }


def write_splits(labels: list[str], seeds: list[int] | None = None,
                 out_dir: Path | None = None) -> list[Path]:
    """Write one split_seed{N}.json per seed. Returns the paths."""
    seeds = CONFIG["seeds"] if seeds is None else seeds
    out_dir = CONFIG["data_processed"] if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for seed in seeds:
        path = out_dir / f"split_seed{seed}.json"
        path.write_text(json.dumps(make_split(labels, seed), indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def load_split(seed: int, labels: list[str], out_dir: Path | None = None) -> dict:
    """Load a split and refuse it if the manifest no longer matches."""
    out_dir = CONFIG["data_processed"] if out_dir is None else out_dir
    split = json.loads((out_dir / f"split_seed{seed}.json").read_text(encoding="utf-8"))
    actual = manifest_fingerprint(labels)
    if split["fingerprint"] != actual:
        raise ValueError(
            f"manifest berubah sejak split dibuat "
            f"({split['fingerprint']} != {actual}). Buat ulang splitnya, jangan "
            f"dipakai — label akan tertukar."
        )
    return split


def describe(split: dict, labels: list[str]) -> dict:
    """Per-class counts for each part, to check stratification actually held."""
    out = {}
    for part in ("train", "val", "test"):
        counts: dict[str, int] = {}
        for i in split[part]:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        out[part] = dict(sorted(counts.items()))
    return out


def _self_check() -> None:
    import tempfile

    # deliberately imbalanced, like Tobacco3482
    labels = ["A"] * 1500 + ["B"] * 1200 + ["C"] * 500 + ["D"] * 282
    assert len(labels) == 3482

    s = make_split(labels, seed=42)
    assert len(s["train"]) + len(s["val"]) == 800, (len(s["train"]), len(s["val"]))
    assert len(s["val"]) == 80, len(s["val"])
    assert len(s["test"]) == 2682, len(s["test"])

    parts = [set(s[p]) for p in ("train", "val", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) \
        and not (parts[1] & parts[2]), "bagian split tumpang tindih"
    assert set().union(*parts) == set(range(3482)), "ada indeks yang hilang"

    # stratification: train proportions track the corpus within a few percent
    d = describe(s, labels)
    for kelas in "ABCD":
        got = d["train"][kelas] / len(s["train"])
        want = labels.count(kelas) / len(labels)
        assert abs(got - want) < 0.02, (kelas, got, want)

    # same seed -> identical split; different seed -> different split
    assert make_split(labels, 42) == s
    assert make_split(labels, 43)["train"] != s["train"]

    # fingerprint guard actually fires
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_splits(labels, seeds=[42], out_dir=tmp)
        assert load_split(42, labels, out_dir=tmp)["seed"] == 42
        try:
            load_split(42, labels[::-1], out_dir=tmp)
        except ValueError:
            pass
        else:
            raise AssertionError("fingerprint tidak menolak manifest yang berubah")

    print("splits self-check ok")


if __name__ == "__main__":
    _self_check()
