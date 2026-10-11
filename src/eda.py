"""Exploratory data analysis and curation helpers for notebook 11.

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
    print("eda self-check ok")


if __name__ == "__main__":
    _self_check()
