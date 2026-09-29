"""Two text representations, precomputed to disk once.

1. A 500x300 zero-padded sequence of FastText word vectors, for the CNN1D.
2. A single SIF document vector (weighted average + first-component removal),
   for the MLP baseline.

The heavy pieces — spaCy and the FastText model — are injected as plain callables
rather than imported at module level, for two reasons. It lets the logic below be
checked on a machine that holds neither model, and it makes the one-pass rule
explicit: the caller owns the model's lifetime and drops it before training.

RAM warning: cc.en.300.bin is ~7GB on disk and ~15GB loaded. fasttext.util's
reduce_model does NOT help, because it has to load the full 300-dim model first.
Run the extraction once, write to disk, release the model.
"""

from __future__ import annotations

import difflib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .config import CONFIG

Tokenize = Callable[[str], list[str]]
Embed = Callable[[str], np.ndarray]


# --------------------------------------------------------------------------- #
# model loaders (only these two touch the big dependencies)
# --------------------------------------------------------------------------- #

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"


def download_fasttext(path: Path | None = None) -> Path:
    """Fetch and gunzip cc.en.300.bin if it is not already there. ~7GB, one time.

    Downloaded straight to the final location so a second run is a no-op, and the
    .gz is deleted afterwards rather than leaving 4GB of dead weight behind.
    """
    import gzip
    import shutil
    import urllib.request

    path = Path(path or CONFIG["models"] / CONFIG["fasttext_model"])
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    gz = path.with_suffix(path.suffix + ".gz")

    if not gz.exists():
        print(f"mengunduh {FASTTEXT_URL} (~4,2GB terkompresi, sekali saja)...")
        tmp = gz.with_suffix(".part")          # so an interrupted download is not
        urllib.request.urlretrieve(FASTTEXT_URL, tmp)   # mistaken for a complete one
        tmp.rename(gz)

    print(f"mengekstrak ke {path} (~7GB)...")
    with gzip.open(gz, "rb") as src, path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    gz.unlink()
    return path


def spacy_tokenizer(model: str = "en_core_web_sm") -> Tokenize:
    """spaCy tokenisation with punctuation and whitespace dropped, as the paper does.

    Downloads the model on first use: one less manual step to forget.
    """
    import spacy

    try:
        nlp = spacy.load(model, disable=["parser", "ner", "tagger", "lemmatizer"])
    except OSError:
        print(f"model spaCy {model} belum ada, mengunduh...")
        from spacy.cli import download as spacy_download

        spacy_download(model)
        nlp = spacy.load(model, disable=["parser", "ner", "tagger", "lemmatizer"])
    nlp.max_length = 2_000_000  # some OCR outputs are long

    def tokenize(text: str) -> list[str]:
        return [t.text for t in nlp(text) if not (t.is_punct or t.is_space)]

    return tokenize


def fasttext_embedder(path: Path | None = None) -> Embed:
    """Word vectors from a FastText .bin model.

    Must be .bin, never .vec: only the binary model carries the subword n-grams
    that let an out-of-vocabulary word get a plausible vector, which is the whole
    argument of the paper. See vault/20-metode/fasttext-subword.md.
    """
    import fasttext

    path = Path(path or CONFIG["models"] / CONFIG["fasttext_model"])
    if path.suffix != ".bin":
        raise ValueError(f"butuh model .bin, dapat {path.name} — .vec tidak bisa OOV")
    if not path.exists():
        download_fasttext(path)
    model = fasttext.load_model(str(path))

    @lru_cache(maxsize=200_000)
    def embed(word: str) -> np.ndarray:
        return model.get_word_vector(word)

    return embed


# --------------------------------------------------------------------------- #
# tokenisation pass
# --------------------------------------------------------------------------- #

def tokenize_corpus(texts: Iterable[str], tokenize: Tokenize) -> list[list[str]]:
    """Tokenise once; both representations are built from the result."""
    return [tokenize(t) for t in texts]


# --------------------------------------------------------------------------- #
# representation 1 — padded sequences for the CNN1D
# --------------------------------------------------------------------------- #

def build_sequences(token_lists: Sequence[Sequence[str]], embed: Embed,
                    out_path: Path, max_words: int | None = None,
                    dim: int | None = None, dtype=np.float16) -> Path:
    """Write an (N, max_words, dim) .npy memmap of zero-padded word vectors.

    Documents shorter than max_words are zero-padded, as the paper says. Longer
    documents are truncated — the paper never mentions this case; it is our
    assumption, recorded in the spec note.

    float16 halves the file (3482x500x300 is ~1.0GB instead of ~2.1GB) at a
    precision loss that is irrelevant next to OCR noise.
    """
    max_words = CONFIG["max_words"] if max_words is None else max_words
    dim = CONFIG["embedding_dim"] if dim is None else dim
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(len(token_lists), max_words, dim)
    )
    for i, tokens in enumerate(token_lists):
        for j, word in enumerate(tokens[:max_words]):
            arr[i, j] = embed(word)
    arr.flush()
    return out_path


# --------------------------------------------------------------------------- #
# representation 2 — SIF document vectors for the MLP
# --------------------------------------------------------------------------- #

def word_probabilities(token_lists: Sequence[Sequence[str]]) -> dict[str, float]:
    """Unigram probabilities over the corpus, for the SIF weights."""
    counts = Counter(w for tokens in token_lists for w in tokens)
    total = sum(counts.values()) or 1
    return {w: c / total for w, c in counts.items()}


def weighted_average(token_lists: Sequence[Sequence[str]], embed: Embed,
                     probs: dict[str, float], a: float = 1e-3,
                     dim: int | None = None) -> np.ndarray:
    """SIF step 1: average word vectors weighted by a / (a + p(w)).

    Frequent words get a small weight, rare words a large one — the same instinct
    as IDF, but derived from a generative model.
    """
    dim = CONFIG["embedding_dim"] if dim is None else dim
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        if not tokens:
            continue  # empty OCR output stays a zero vector, on purpose
        w = np.array([a / (a + probs.get(t, 0.0)) for t in tokens], dtype=np.float32)
        vecs = np.stack([embed(t) for t in tokens]).astype(np.float32)
        out[i] = (w[:, None] * vecs).sum(0) / w.sum()
    return out


def fit_principal_direction(vectors: np.ndarray) -> np.ndarray:
    """SIF step 2, fitted part: the first principal direction.

    Must be fitted on the TRAIN rows only. Fitting it over the whole corpus leaks
    test information — subtle, and easy to get wrong.
    """
    centred = vectors - vectors.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def remove_component(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project the given direction out of every row."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return vectors - np.outer(vectors @ d, d)


# --------------------------------------------------------------------------- #
# Fig. 4b reproduction helpers
# --------------------------------------------------------------------------- #

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def find_misspelling_pairs(token_lists: Sequence[Sequence[str]],
                           min_ref_freq: int = 50, max_cand_freq: int = 2,
                           min_len: int = 4, cutoff: float = 0.85,
                           limit: int = 40) -> list[tuple[str, str, int]]:
    """Candidate (misspelling, likely original, freq) triples found in our own corpus.

    No external dictionary is used. Words the corpus repeats many times are taken
    as the reference vocabulary, and near-unique words that are spelled almost like
    one of them are treated as OCR misspellings. That is a heuristic, not ground
    truth, so the notebook prints the pairs for a human to sanity-check.
    """
    counts = Counter(w for tokens in token_lists for w in tokens)
    reference = [
        w for w, c in counts.items()
        if c >= min_ref_freq and len(w) >= min_len and w.isalpha()
    ]
    pairs = []
    for word, c in counts.most_common()[::-1]:
        if c > max_cand_freq or len(word) < min_len or not word.isalpha():
            continue
        if word in reference:
            continue
        match = difflib.get_close_matches(word, reference, n=1, cutoff=cutoff)
        if match and match[0].lower() != word.lower():
            pairs.append((word, match[0], c))
        if len(pairs) >= limit:
            break
    return pairs


def oov_rate(token_lists: Sequence[Sequence[str]], min_ref_freq: int = 50,
             min_len: int = 4) -> np.ndarray:
    """Per-document share of tokens outside the corpus reference vocabulary.

    A stand-in for the paper's Fig. 4a, which used an English dictionary. The
    numbers are therefore comparable in shape, not in value.
    """
    counts = Counter(w for tokens in token_lists for w in tokens)
    reference = {
        w for w, c in counts.items()
        if c >= min_ref_freq and len(w) >= min_len and w.isalpha()
    }
    rates = []
    for tokens in token_lists:
        kept = [w for w in tokens if len(w) >= min_len]
        rates.append(
            0.0 if not kept else sum(w not in reference for w in kept) / len(kept)
        )
    return np.array(rates, dtype=np.float32)


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    """Checked with a fake tokenizer and fake embedder, so no spaCy or FastText
    model is needed. Validates the logic, not the real vectors."""
    import tempfile

    import zlib

    rng = np.random.default_rng(0)

    @lru_cache(maxsize=None)
    def embed(word: str) -> np.ndarray:
        # crc32, not hash(): str hashing is salted per process, and a test that
        # changes between runs is not a test. Nothing here mimics subword
        # similarity — that is the real model's job and cannot be checked without it.
        seed = zlib.crc32(word.encode())
        return np.random.default_rng(seed).standard_normal(8).astype(np.float32)

    docs = [
        "the filter was the best filter",      # 6 tokens
        " ".join(["word"] * 12),               # longer than max_words below
        "",                                    # empty OCR output
    ]
    toks = tokenize_corpus(docs, str.split)
    assert [len(t) for t in toks] == [6, 12, 0], toks

    # sequences: padding, truncation, and the empty document
    with tempfile.TemporaryDirectory() as tmp:
        p = build_sequences(toks, embed, Path(tmp) / "seq.npy",
                            max_words=10, dim=8, dtype=np.float32)
        arr = np.load(p, mmap_mode="r")
        assert arr.shape == (3, 10, 8), arr.shape
        assert np.any(arr[0, :6] != 0) and np.all(arr[0, 6:] == 0), "padding salah"
        assert np.all(arr[1, 9] != 0), "truncation memotong terlalu awal"
        assert np.all(arr[2] == 0), "dokumen kosong seharusnya nol seluruhnya"

    # SIF weighting: "the" is frequent, so it must be pushed down hard
    probs = word_probabilities(toks)
    assert probs["the"] > probs["best"], probs
    a = 1e-3
    assert (a / (a + probs["the"])) < (a / (a + probs["best"])), "bobot SIF terbalik"

    vecs = weighted_average(toks, embed, probs, a=a, dim=8)
    assert vecs.shape == (3, 8)
    assert np.all(vecs[2] == 0), "dokumen kosong seharusnya vektor nol"

    # removing the fitted direction kills the variance along it
    many = rng.standard_normal((40, 8)).astype(np.float32)
    many[:, 0] *= 30  # a dominant direction to find
    d = fit_principal_direction(many)
    after = remove_component(many, d)
    assert abs(float(np.abs(after @ d).max())) < 1e-3, "komponen utama tidak terbuang"

    # misspelling discovery on a corpus where the answer is known
    corpus = [["filter"] * 60, ["fiilter"], ["zzzz"]]
    pairs = find_misspelling_pairs(corpus, min_ref_freq=50, max_cand_freq=2)
    assert ("fiilter", "filter", 1) in pairs, pairs
    assert all(p[0] != "zzzz" for p in pairs), "kata acak tak boleh dipasangkan"

    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine(v, v * 3) == 1.0, "cosine tidak invarian terhadap skala"
    assert abs(cosine(v, np.array([0.0, 1.0, 0.0], dtype=np.float32))) < 1e-6
    assert cosine(v, -v) == -1.0

    rates = oov_rate(corpus, min_ref_freq=50)
    assert rates[0] == 0.0 and rates[1] == 1.0, rates

    print("text_features self-check ok")


if __name__ == "__main__":
    _self_check()
