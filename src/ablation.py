"""Inference-time perturbations for the ablation study.

Nothing here retrains anything: every function perturbs an input, so the trained
checkpoints from Tahap 5-6 are reused as they are.

This stage is not part of the paper. The authors admit in their limitations that
Tobacco3482 is uniformly well-oriented and professionally scanned, so it does not
represent real conditions — but they never test that. Ablations 1 and 3 are the
empirical test of a limitation the authors named and left alone.

IMPORTANT CAVEAT, and it belongs in the write-up: degrading the image while the text
still comes from OCR of the *clean* original measures the robustness of the
ARCHITECTURE, not of the end-to-end system. In a real deployment a blurred scan
produces worse OCR too, so both branches would degrade together. See
vault/20-metode/taksonomi-multimodal.md.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from .config import CONFIG
from .models.image_model import build_transform

# Degradation levels, mild to severe. Level 0 of every kind is the identity, so the
# first point of every curve is the untouched baseline.
IMAGE_LEVELS: dict[str, list[float]] = {
    "rotate": [0, 2, 5, 10, 20, 45],        # degrees
    "blur": [0, 1, 2, 4, 8],                # gaussian radius, px
    "jpeg": [100, 50, 20, 10, 5],           # quality, descending = worse
    "noise": [0.0, 0.02, 0.05, 0.1, 0.2],   # gaussian sigma in [0,1] pixel space
}

TEXT_LEVELS: dict[str, list[float]] = {
    "drop_words": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "typos": [0.0, 0.1, 0.25, 0.5],
}


# --------------------------------------------------------------------------- #
# image degradation
# --------------------------------------------------------------------------- #

def _rotate(level: float):
    # fill=255 (white), not the default black: documents are dark text on white, so
    # black corners would inject a huge artificial signal and the ablation would
    # measure "can the CNN see black triangles", not "can it read a tilted page"
    return lambda im: im.rotate(level, resample=Image.BILINEAR, expand=False,
                                fillcolor=255)


def _blur(level: float):
    return lambda im: im.filter(ImageFilter.GaussianBlur(radius=level))


def _jpeg(level: float):
    def apply(im):
        buf = io.BytesIO()
        mode = im.mode
        im.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        # keep the mode: collapsing channels here would break Normalize downstream
        return Image.open(buf).convert(mode)
    return apply


class AddNoise:
    """Gaussian noise in [0,1] pixel space, before normalisation."""

    def __init__(self, sigma: float, seed: int = 0):
        self.sigma = sigma
        self.seed = seed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return x
        g = torch.Generator().manual_seed(self.seed)
        noise = torch.randn(x.shape, generator=g) * self.sigma
        return (x + noise).clamp(0.0, 1.0)


def degraded_transform(kind: str = "none", level: float = 0, seed: int = 0):
    """The standard eval transform with one degradation inserted.

    PIL-level degradations go FIRST, on the incoming single-channel scan at its
    original resolution — the way a real bad scan would arrive, before any of our
    preprocessing. Noise goes after ToTensor but before Normalize, so sigma is
    interpretable in [0,1] pixel units.
    """
    steps = list(build_transform().transforms)   # Grayscale, Resize, ToTensor, Normalize
    if kind == "none" or level in (0, 0.0) or (kind == "jpeg" and level >= 100):
        return transforms.Compose(steps)

    if kind == "rotate":
        steps.insert(0, _rotate(level))
    elif kind == "blur":
        steps.insert(0, _blur(level))
    elif kind == "jpeg":
        steps.insert(0, _jpeg(level))
    elif kind == "noise":
        steps.insert(3, AddNoise(level, seed=seed))   # after ToTensor, before Normalize
    else:
        raise ValueError(f"kind tidak dikenal: {kind!r}. Pilihan: {list(IMAGE_LEVELS)}")
    return transforms.Compose(steps)


# --------------------------------------------------------------------------- #
# text degradation
# --------------------------------------------------------------------------- #

def drop_words(seq: torch.Tensor, fraction: float, seed: int = 0) -> torch.Tensor:
    """Delete a fraction of the real word positions and re-pad at the end.

    seq is (dim, length), channel-first as the CNN1D expects.

    Deleting-and-re-padding rather than zeroing in place, on purpose: during training
    zeros only ever appear as a trailing pad, so punching holes in the middle of a
    sequence is out-of-distribution in a way that conflates "less text" with "input
    the model has never seen". Re-padding keeps the perturbation to the one variable
    we mean to vary.
    """
    if fraction <= 0:
        return seq
    real = int((seq.abs().sum(0) > 0).sum())   # trailing zeros are padding
    if real == 0:
        return seq

    keep_n = int(round(real * (1.0 - fraction)))
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(real, size=keep_n, replace=False)) if keep_n else []

    out = torch.zeros_like(seq)
    if keep_n:
        out[:, :keep_n] = seq[:, torch.as_tensor(np.asarray(keep), dtype=torch.long)]
    return out


def inject_typos(tokens: list[str], fraction: float, seed: int = 0) -> list[str]:
    """Corrupt a fraction of tokens with single-character OCR-like errors.

    Needs the FastText model to re-embed afterwards, which is why this one cannot run
    on a machine that only holds the precomputed features.

    The corruptions mimic what Tesseract actually does: doubling a letter, dropping
    one, or swapping a neighbour — the same family as the paper's own examples
    (specifically/Specificalily, filter/fiilter, Largely/Largly).
    """
    if fraction <= 0:
        return list(tokens)
    rng = np.random.default_rng(seed)
    out = []
    for tok in tokens:
        if len(tok) < 4 or rng.random() >= fraction:
            out.append(tok)
            continue
        i = int(rng.integers(1, len(tok) - 1))
        mode = int(rng.integers(0, 3))
        if mode == 0:                                   # double a letter
            out.append(tok[:i] + tok[i] + tok[i:])
        elif mode == 1:                                 # drop a letter
            out.append(tok[:i] + tok[i + 1:])
        else:                                           # swap neighbours
            out.append(tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:])
    return out


# --------------------------------------------------------------------------- #
# missing modality
# --------------------------------------------------------------------------- #

def zero_modality(batch, which: str):
    """Blank one modality of a (image, text) batch.

    For text, zeros are exactly what padding looks like, so a zeroed sequence is a
    genuinely in-distribution "empty document" — the dataset already contains such
    documents where OCR returned nothing.

    For the image it is not so clean: the tensor is already normalised, so zeros mean
    the ImageNet MEAN image, not black. That is the least surprising choice (it is
    what a missing input contributes nothing relative to), but it is a choice, and it
    has to be stated in the write-up.
    """
    image, text = batch
    if which == "text":
        return image, torch.zeros_like(text)
    if which == "image":
        return torch.zeros_like(image), text
    raise ValueError(f"which harus 'text' atau 'image', dapat {which!r}")


@torch.no_grad()
def evaluate_zeroed(model, loader, which: str, device) -> dict:
    """Evaluate a fusion model with one modality blanked, batch by batch.

    Same return shape as train.evaluate, so results are directly comparable.

    NOTE: `notebooks/06_ablasi.ipynb` has an equivalent loop written inline, from
    before this helper existed. The duplication is deliberate: regenerating that
    notebook would wipe the outputs of a 12-hour run. Notebook 07 re-measures the
    concat number with THIS function and asserts it matches the inline version's
    0.8203, so the two cannot silently disagree.
    """
    from sklearn.metrics import f1_score

    model.eval()
    trues, preds = [], []
    for (img, txt), y in loader:
        batch = zero_modality((img.to(device), txt.to(device)), which)
        preds.append(model(batch).argmax(1).cpu().numpy())
        trues.append(y.numpy())
    yt, yp = np.concatenate(trues), np.concatenate(preds)
    return {
        "oa": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "y_true": yt,
        "y_pred": yp,
    }


class DegradedPairDataset(torch.utils.data.Dataset):
    """Wraps a PairDataset and perturbs the text side on the fly."""

    def __init__(self, pair_dataset, text_fraction: float, seed: int = 0):
        self.inner = pair_dataset
        self.text_fraction = text_fraction
        self.seed = seed

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int):
        (img, txt), y = self.inner[i]
        return (img, drop_words(txt, self.text_fraction, seed=self.seed + i)), y


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    size, dim, length = CONFIG["image_size"], 8, 10

    # --- image degradations ---------------------------------------------------
    # a document-like image: white page with a dark bar, so blur and rotation have
    # something to destroy
    page = Image.new("L", (300, 900), color=255)
    page.paste(0, (50, 100, 250, 140))

    clean = degraded_transform("none")(page)
    assert clean.shape == (3, size, size), clean.shape

    # level 0 of every kind must be a no-op, or the first point of every curve lies
    for kind, levels in IMAGE_LEVELS.items():
        out = degraded_transform(kind, levels[0])(page)
        assert torch.allclose(out, clean, atol=1e-6), f"{kind} level {levels[0]} bukan no-op"

    # each degradation must actually change the image, and more must change it more
    for kind in ("rotate", "blur", "jpeg", "noise"):
        levels = IMAGE_LEVELS[kind][1:]
        deltas = [float((degraded_transform(kind, lv, seed=0)(page) - clean).abs().mean())
                  for lv in levels]
        assert all(d > 0 for d in deltas), (kind, deltas)
        assert deltas[-1] > deltas[0], (f"{kind}: level terkuat tidak lebih merusak",
                                        levels, deltas)

    # rotation must fill with white, not black: a black fill would dwarf the content
    rot = degraded_transform("rotate", 45)(page)
    corner = rot[0, :20, :20].mean()
    white = clean[0].max()
    assert corner > 0.5 * white, ("sudut hasil rotasi terlalu gelap — fillcolor salah",
                                  float(corner), float(white))

    assert degraded_transform("noise", 0.1, seed=1)(page).shape == (3, size, size)
    try:
        degraded_transform("sepia", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("kind tak dikenal seharusnya ditolak")

    # --- text degradation -----------------------------------------------------
    seq = torch.zeros(dim, length)
    seq[:, :6] = torch.arange(1, 7).float()          # 6 real words, 4 padding
    assert int((seq.abs().sum(0) > 0).sum()) == 6

    assert torch.equal(drop_words(seq, 0.0), seq), "fraction 0 harus no-op"

    half = drop_words(seq, 0.5, seed=0)
    assert int((half.abs().sum(0) > 0).sum()) == 3, "harus tersisa 3 dari 6 kata"
    # survivors are packed at the front, nothing punched out in the middle
    occupied = (half.abs().sum(0) > 0).nonzero().flatten().tolist()
    assert occupied == [0, 1, 2], ("kata sisa tidak dirapatkan ke depan", occupied)
    # order is preserved, not shuffled
    vals = half[0, :3].tolist()
    assert vals == sorted(vals), ("urutan kata tidak dipertahankan", vals)

    assert int((drop_words(seq, 1.0).abs().sum(0) > 0).sum()) == 0, \
        "fraction 1.0 harus menghapus semua kata"
    # an already-empty document must survive untouched, not crash
    assert torch.equal(drop_words(torch.zeros(dim, length), 0.5),
                       torch.zeros(dim, length))

    toks = ["specifically", "filter", "of", "largely"]
    same = inject_typos(toks, 0.0)
    assert same == toks, same
    broken = inject_typos(toks, 1.0, seed=3)
    assert broken[2] == "of", "token pendek tidak boleh dirusak"
    assert sum(a != b for a, b in zip(toks, broken)) >= 2, (toks, broken)
    # a single-character corruption shifts the length by at most one, and never
    # empties a token — anything else is a mangling, not an OCR-like typo
    for a, b in zip(toks, broken):
        assert b, (a, b)
        assert abs(len(b) - len(a)) <= 1, (a, b)
    assert inject_typos(toks, 1.0, seed=3) == broken, "injeksi typo tidak deterministik"

    # --- missing modality -----------------------------------------------------
    img_b, txt_b = torch.randn(2, 3, 8, 8), torch.randn(2, dim, length)
    i2, t2 = zero_modality((img_b, txt_b), "text")
    assert torch.equal(i2, img_b) and torch.all(t2 == 0)
    i3, t3 = zero_modality((img_b, txt_b), "image")
    assert torch.all(i3 == 0) and torch.equal(t3, txt_b)
    try:
        zero_modality((img_b, txt_b), "keduanya")
    except ValueError:
        pass
    else:
        raise AssertionError("which tak dikenal seharusnya ditolak")

    print("ablation self-check ok")


if __name__ == "__main__":
    _self_check()
