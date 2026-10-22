"""Exploratory data analysis and curation helpers (pipeline notebook, section 4 and 6.1).

Everything here is a pure function over arrays, strings or one image, so it can be
checked without the dataset. The notebook does the looping, plotting and saving.

Curation flags documents rather than dropping them: OCR noise is the object of study,
so a noisy page stays in the data and only gets a marker.
"""

from __future__ import annotations

import re

import numpy as np

LETTER_WIDTH_IN = 8.5   # Tobacco3482 pages are US letter; used to estimate DPI


# --------------------------------------------------------------------------- #
# Image statistics

def image_stats(gray: np.ndarray, width_px: int) -> dict:
    """Simple per-page statistics from a 2-D uint8 grayscale array.

    `gray` should be resized to a fixed width by the caller so that sharpness is
    comparable across pages; `width_px` is the ORIGINAL width, used for DPI.
    """
    g = gray.astype(np.float32)
    # 4-neighbour Laplacian; its variance is the usual blur proxy (low = blurry)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return {
        "dpi_perkiraan": width_px / LETTER_WIDTH_IN,
        "kecerahan": float(g.mean()),
        "kontras": float(g.std()),
        "rasio_tinta": float((g < 128).mean()),     # share of dark pixels
        "ketajaman": float(lap.var()),
    }


def thumbnail_vector(gray: np.ndarray) -> np.ndarray:
    """Z-scored, flattened small grayscale page (the caller resizes, e.g. 64x64).

    Z-scoring removes the white-background offset that makes every document page
    look alike to coarse hashes; what is left is the layout of the ink.
    """
    v = gray.astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def near_duplicates(thumbs: np.ndarray, min_corr: float = 0.98,
                    chunk: int = 512) -> list[tuple[int, int, float]]:
    """All pairs (i, j, corr) of pages whose thumbnails correlate >= min_corr, i < j.

    Candidates only: pages built from the same printed template can also correlate
    highly, so the notebook confirms each pair with the OCR text.
    ponytail: dense O(n^2) correlation in chunks, fine for 3482 pages (~12M pairs);
    switch to approximate nearest neighbours if the dataset grows past ~50k.
    """
    t = np.asarray(thumbs, dtype=np.float32)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    pairs = []
    for start in range(0, len(t), chunk):
        sim = t[start:start + chunk] @ t.T
        for a, j in zip(*np.nonzero(sim >= min_corr)):
            i = start + int(a)
            if i < j:
                pairs.append((i, int(j), float(sim[a, j])))
    return pairs


def split_leaks(pairs: list[tuple[int, int, int]], split: dict) -> list[tuple[int, int]]:
    """Duplicate pairs with one member in train/val and the other in test."""
    fit = set(split["train"]) | set(split["val"])
    test = set(split["test"])
    return [(i, j) for i, j, _ in pairs
            if (i in fit and j in test) or (j in fit and i in test)]


# --------------------------------------------------------------------------- #
# Archive ID codes (Larson et al., 2025)

_ID = re.compile(r"(?<![\d.,/:-])\d{6,}(?![\d.,/:])")


def id_codes(text: str) -> list[str]:
    """Long digit runs that look like archive ID codes stamped on the page.

    ponytail: heuristic, runs of >= 6 digits not part of a decimal, date or time.
    Phone numbers and dates are split by separators so they fall below 6 digits.
    Larson et al. annotated codes by hand; this only approximates that.
    """
    return _ID.findall(text)


def id_prefix(text: str, n: int = 3) -> str:
    """First `n` digits of the longest ID code, or 'tidak-ada'. A categorical
    feature for the "ID code only" shortcut classifier."""
    codes = id_codes(text)
    return max(codes, key=len)[:n] if codes else "tidak-ada"


# --------------------------------------------------------------------------- #
# Flags and feature summaries

def flag_tail(values: np.ndarray, pct: float, upper: bool = False) -> tuple[np.ndarray, float]:
    """Flag the lower (or upper) `pct` percent tail. Returns (mask, threshold).

    Thresholds come from the data, not from numbers chosen in advance. NaN values
    are never flagged.
    """
    v = np.asarray(values, dtype=float)
    thr = float(np.nanpercentile(v, 100 - pct if upper else pct))
    mask = (v > thr) if upper else (v < thr)
    return mask & ~np.isnan(v), thr


def mean_nonzero_rows(seq: np.ndarray) -> np.ndarray:
    """Mean of the non-padding rows of each (n_words, dim) sequence.

    Turns the stored FastText sequences into one document vector without loading
    the 15 GB FastText model again. An empty document gives the zero vector.
    """
    seq = np.asarray(seq, dtype=np.float32)
    keep = np.any(seq != 0, axis=-1, keepdims=True)
    n = keep.sum(axis=-2)
    return (seq * keep).sum(axis=-2) / np.maximum(n, 1)


def centroid_cosine(x: np.ndarray, labels: np.ndarray, classes: list) -> np.ndarray:
    """Cosine similarity between class centroids. High off-diagonal = classes that
    look alike in this feature space."""
    c = np.stack([x[labels == k].mean(axis=0) for k in classes])
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    return c @ c.T


# --------------------------------------------------------------------------- #
# Loops over the corpus (formerly cells of notebook 11). sklearn and pandas are
# imported inside, so the pure helpers above stay importable on their own.

def scan_images(paths: list[str], stats_width: int = 1000, thumb: int = 64,
                every: int = 500) -> tuple[list[dict], np.ndarray]:
    """image_stats for every page (resized to a common width) + 64x64 thumbnails."""
    import time

    from PIL import Image

    rows, thumbs, t0 = [], [], time.perf_counter()
    for n, path in enumerate(paths, 1):
        with Image.open(path) as im:
            w, h = im.size
            g = im.convert("L")
            small = np.asarray(g.resize((stats_width, round(h * stats_width / w))))
            thumbs.append(thumbnail_vector(np.asarray(g.resize((thumb, thumb)))))
        rows.append({"lebar": w, "tinggi": h, **image_stats(small, w)})
        if every and n % every == 0:
            print(f"{n}/{len(paths)}  {time.perf_counter() - t0:.0f} dtk", flush=True)
    return rows, np.stack(thumbs)


def text_stats(texts: list[str], known) -> list[dict]:
    """Length, alphabetic words, OOV share and archive-ID count per document."""
    from .ocr import oov_share, words

    return [{"n_kata": len(t.split()), "n_kata_alfabet": len(words(t)),
             "oov": oov_share(t, known), "n_kode_id": len(id_codes(t))} for t in texts]


def probe(features: np.ndarray, y: np.ndarray, fit_idx, test_idx,
          classes: list) -> tuple[dict, np.ndarray]:
    """Linear probe: how separable are the classes in this feature space?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(features[fit_idx])
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(features[fit_idx]), y[fit_idx])
    pred = clf.predict(sc.transform(features[test_idx]))
    return ({"oa_probe": float((pred == y[test_idx]).mean()),
             "macro_f1_probe": float(f1_score(y[test_idx], pred, average="macro"))},
            f1_score(y[test_idx], pred, labels=classes, average=None, zero_division=0))


def tsne_2d(features: np.ndarray, seed: int = 42, perplexity: float = 30) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    z = StandardScaler().fit_transform(features)
    z = PCA(min(50, *z.shape), random_state=seed).fit_transform(z)
    return TSNE(2, perplexity=min(perplexity, (len(z) - 1) / 3), random_state=seed,
                init="pca").fit_transform(z)


def top_terms(texts: list[str], y: np.ndarray, fit_idx, classes: list,
              k: int = 10, min_df: int = 3) -> dict:
    """Highest mean TF-IDF words per class, fitted on the training part only."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", lowercase=True,
                          stop_words="english", min_df=min_df, sublinear_tf=True)
    x = vec.fit_transform([texts[i] for i in fit_idx])
    vocab = np.array(vec.get_feature_names_out())
    yf = y[fit_idx]
    return {c: ", ".join(vocab[np.argsort(np.asarray(x[yf == c].mean(axis=0)).ravel())
                               [::-1][:k]]) for c in classes}


def shortcut_probe(values: np.ndarray, y: np.ndarray, fit_idx, test_idx) -> dict:
    """Classify from one categorical value only — how much a shortcut can explain."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown="ignore").fit(values[fit_idx].reshape(-1, 1))
    clf = LogisticRegression(max_iter=3000).fit(
        enc.transform(values[fit_idx].reshape(-1, 1)), y[fit_idx])
    pred = clf.predict(enc.transform(values[test_idx].reshape(-1, 1)))
    return {"oa": float((pred == y[test_idx]).mean()),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro"))}


def pretrained_image_features(paths: list[str], batch: int = 16,
                              num_workers: int = 0) -> np.ndarray:
    """1280-d MobileNetV2 (ImageNet, untrained on our data) vector per page."""
    import torch

    from .datasets import ImageDataset
    from .models import ImageBranch, build_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageBranch(pretrained=True).to(device).eval()
    ds = ImageDataset(paths, np.zeros(len(paths), dtype=int), list(range(len(paths))),
                      build_transform())
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for x, _ in torch.utils.data.DataLoader(ds, batch_size=batch,
                                                num_workers=num_workers):
            out.append(model.features(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def split_table(y: np.ndarray, splits: dict, classes: list):
    """Per-class counts of every part of every seed's split."""
    import pandas as pd

    rows = []
    for s, sp in splits.items():
        for part in ("train", "val", "test"):
            counts = pd.Series(y[sp[part]]).value_counts().reindex(classes, fill_value=0)
            rows.append({"seed": s, "bagian": part, **counts.to_dict(),
                         "total": len(sp[part])})
    return pd.DataFrame(rows)


def curate(df, thumbs: np.ndarray, texts: list[str], splits: dict, pct: float = 5,
           min_corr: float = 0.98, min_jaccard: float = 0.8):
    """Flag (never drop) problem documents and find near-duplicate leaks.

    `df` needs n_kata_alfabet, oov, dpi_perkiraan, ketajaman, in manifest order.
    Returns (df with flag_* and bocor_s* columns, info dict).
    """
    from .ocr import jaccard

    df = df.copy()
    df["flag_teks_kosong"] = df["n_kata_alfabet"] == 0
    nonempty = np.where(df["flag_teks_kosong"], np.nan, df["n_kata_alfabet"])
    df["flag_teks_pendek"], t_short = flag_tail(nonempty, pct)
    df["flag_oov_tinggi"], t_oov = flag_tail(df["oov"].to_numpy(float), pct, upper=True)
    df["flag_resolusi_rendah"], t_dpi = flag_tail(df["dpi_perkiraan"].to_numpy(), pct)
    df["flag_buram"], t_sharp = flag_tail(df["ketajaman"].to_numpy(), pct)

    candidates = near_duplicates(thumbs, min_corr)

    def text_agrees(i, j):
        jac = jaccard(texts[i], texts[j])
        return jac is None or jac >= min_jaccard      # None = both pages without text

    pairs = [p for p in candidates if text_agrees(p[0], p[1])]
    dup = np.zeros(len(df), bool)
    for i, j, _ in pairs:
        dup[[i, j]] = True
    df["flag_duplikat"] = dup
    leaks = {}
    for s, sp in splits.items():
        found = split_leaks(pairs, sp)
        leak = np.zeros(len(df), bool)
        for i, j in found:
            leak[[i, j]] = True
        df[f"bocor_s{s}"] = leak
        leaks[s] = len(found)
    info = {"ambang": {"teks_pendek_maks_kata": t_short, "oov_tinggi_min": t_oov,
                       "resolusi_rendah_maks_dpi": t_dpi, "buram_maks_ketajaman": t_sharp},
            "kandidat_citra": len(candidates), "pasangan_duplikat": pairs,
            "pasangan_bocor": leaks}
    return df, info


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    rng = np.random.default_rng(0)

    # image stats: a sharp checkerboard beats its blurred version; DPI from width
    board = (np.indices((64, 64)).sum(0) % 2 * 255).astype(np.uint8)
    blurred = np.full_like(board, 128)
    s1, s2 = image_stats(board, 1700), image_stats(blurred, 1700)
    assert s1["ketajaman"] > s2["ketajaman"] == 0.0, (s1, s2)
    assert s1["dpi_perkiraan"] == 200.0 and abs(s1["rasio_tinta"] - 0.5) < 1e-6

    # near duplicates: two mostly-white pages with DIFFERENT ink must not pair up
    # (the failure of a coarse hash); a re-scan with slight noise must
    def page(seed):
        g = np.full((64, 64), 250, np.uint8)
        r = np.random.default_rng(seed)
        for _ in range(12):
            y0, x0 = r.integers(4, 56, 2)
            g[y0:y0 + 2, x0:x0 + 8] = 20                 # short ink strokes
        return g
    p1, p2 = page(1), page(2)
    p1_rescan = np.clip(p1.astype(int) + rng.integers(-8, 8, p1.shape), 0, 255)
    th = np.stack([thumbnail_vector(x) for x in (p1, p2, p1_rescan)])
    pairs = near_duplicates(th, min_corr=0.95, chunk=2)
    assert [(i, j) for i, j, _ in pairs] == [(0, 2)], pairs
    assert split_leaks([(0, 1, 0)], {"train": [0], "val": [], "test": [1]}) == [(0, 1)]
    assert split_leaks([(0, 1, 0)], {"train": [0, 1], "val": [], "test": []}) == []

    # ID codes: Bates-style numbers yes; phone, date, money, short numbers no
    text = ("CTR CONTRACTS 023247\n2085601691\nTel (212) 403-1254, March 1, 2001, "
            "$80,000.00 for 1987, ref 50343351-3351")
    assert id_codes(text) == ["023247", "2085601691", "50343351"], id_codes(text)
    assert id_prefix(text) == "208" and id_prefix("no codes 12345") == "tidak-ada"

    # flags: data-driven tail, NaN never flagged
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan], dtype=float)
    m, thr = flag_tail(v, 20)
    assert m.tolist() == [True, True] + [False] * 9 and 2 < thr < 3, (m, thr)
    m_up, _ = flag_tail(v, 20, upper=True)
    assert m_up[-1] == False and m_up[9] and m_up[8]   # noqa: E712

    # mean over non-padding rows; empty doc -> zero vector
    seq = np.zeros((2, 4, 3), dtype=np.float16)
    seq[0, :2] = [[1, 2, 3], [3, 2, 1]]
    out = mean_nonzero_rows(seq)
    assert np.allclose(out[0], [2, 2, 2]) and np.all(out[1] == 0), out

    # centroid cosine: two classes along orthogonal axes -> identity matrix
    x = np.array([[1, 0], [2, 0], [0, 1], [0, 3]], dtype=float)
    cc = centroid_cosine(x, np.array(["a", "a", "b", "b"]), ["a", "b"])
    assert np.allclose(cc, np.eye(2)), cc

    # corpus loops on synthetic data
    import tempfile
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for n, g in enumerate((p1, p2)):
            paths.append(str(Path(tmp) / f"{n}.png"))
            Image.fromarray(g).resize((170, 220)).save(paths[-1])
        rows, th = scan_images(paths, stats_width=100, thumb=16, every=0)
        assert len(rows) == 2 and th.shape == (2, 256) and rows[0]["lebar"] == 170

    known = {"dear", "tobacco"}.__contains__
    ts = text_stats(["Dear Sirr tobacco 2085601691", ""], known)
    assert ts[0]["n_kode_id"] == 1 and ts[1]["oov"] is None and ts[1]["n_kata"] == 0, ts

    yy = np.array(["a", "b"] * 30)
    feats = np.where(yy[:, None] == "a", 1.0, -1.0) + rng.normal(0, 0.1, (60, 3))
    fit, test = np.arange(40), np.arange(40, 60)
    res, per_class = probe(feats, yy, fit, test, ["a", "b"])
    assert res["oa_probe"] == 1.0 and per_class.shape == (2,), res
    assert shortcut_probe(yy.copy(), yy, fit, test)["oa"] == 1.0
    assert tsne_2d(feats, perplexity=5).shape == (60, 2)
    tt = top_terms(["tobacco filter smoke"] * 20 + ["memo meeting agenda"] * 20,
                   np.array(["a"] * 20 + ["b"] * 20), np.arange(40), ["a", "b"], k=2)
    assert "memo" in tt["b"] and "tobacco" in tt["a"], tt

    splits = {1: {"train": [0, 1], "val": [], "test": [2, 3]}}
    st = split_table(np.array(["a", "b", "a", "b"]), splits, ["a", "b"])
    assert st.loc[st.bagian == "test", "total"].item() == 2
    frame = pd.DataFrame({"n_kata_alfabet": [0, 5, 50, 60], "oov": [np.nan, .9, .1, .1],
                          "dpi_perkiraan": [100, 200, 200, 200],
                          "ketajaman": [1, 50, 50, 50]})
    th4 = np.stack([thumbnail_vector(x) for x in (p1, p2, p1, p2)])
    cur, info = curate(frame, th4, ["", "quarterly sales report", "",
                                    "smoking health research"], splits, pct=30)
    assert cur["flag_teks_kosong"].tolist() == [True, False, False, False]
    assert cur["flag_duplikat"].tolist() == [True, False, True, False], info
    assert info["pasangan_bocor"] == {1: 1}, info
    print("eda self-check ok")


if __name__ == "__main__":
    _self_check()
