"""Fase 2 experiment: the OCR engine is the variable, two fusion rules sit on the
same unimodal models.

    per seed            IMAGE  = MobileNetV2                         (trained once)
    per seed x OCR      TEXT   = FastText + CNN1D on that engine's text
    per seed x OCR      LATE   = argmax((z_img + z_txt) / 2)          Kashyap et al.
                        CONCAT = ConcatHead on the frozen features     Audebert-style
                        ORACLE = right whenever IMAGE or TEXT is right

Every unimodal model is trained once and its outputs for ALL documents are cached in
one .npz: features, logits, and the outputs for a blank input (empty text / mean
image), which is what the missing-modality test feeds in. Everything downstream —
both fusion rules, oracle, missing modality, McNemar — reads that cache, so it takes
seconds and can be rerun at will. An existing .npz is reused unless overwrite is set,
which makes the multi-hour training part resumable after a crash.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from . import datasets as DS
from . import train as T
from .config import CONFIG, set_seed
from .models.fusion import ConcatHead

IMAGE = "IMAGE"


def text_name(condition: str) -> str:
    return f"TEXT-{condition}"


def output_path(name: str, seed: int) -> Path:
    return CONFIG["fase2_data"] / "outputs" / f"{name}_seed{seed}.npz"


# --------------------------------------------------------------------------- #
# unimodal: train once, cache outputs for every document

@torch.no_grad()
def extract_outputs(model, loader, blank_input: torch.Tensor, device) -> dict:
    """Features and logits for every row of `loader`, plus those of a blank input.

    The model must expose `features(x)` and `classifier(features)`, as ImageBranch and
    CNN1D both do.
    """
    model.to(device).eval()
    feats, logits = [], []
    for x, _ in loader:
        f = model.features(x.to(device))
        feats.append(f.float().cpu().numpy())
        logits.append(model.classifier(f).float().cpu().numpy())
    bf = model.features(blank_input.to(device))
    return {"feat": np.concatenate(feats), "logit": np.concatenate(logits),
            "blank_feat": bf.float().cpu().numpy()[0],
            "blank_logit": model.classifier(bf).float().cpu().numpy()[0]}


def load_outputs(name: str, seed: int) -> dict:
    with np.load(output_path(name, seed)) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def train_unimodal(name: str, seed: int, split: dict, build_model: Callable,
                   build_dataset: Callable, blank_input: torch.Tensor, *, epochs: int,
                   n_total: int, device=None, overwrite: bool | None = None,
                   num_workers: int | None = None, verbose: bool = True,
                   **train_kwargs) -> dict:
    """Train one unimodal model on `split` and cache its outputs for all documents.

    Returns the run metadata. When the cache already exists and overwrite is off,
    nothing is trained and the stored metadata comes back with dari_cache=True.
    """
    path = output_path(name, seed)
    overwrite = CONFIG["overwrite"] if overwrite is None else overwrite
    if path.exists() and not overwrite:
        return {**load_outputs(name, seed)["meta"], "dari_cache": True}

    device = T.pick_device() if device is None else device
    workers = CONFIG["num_workers"] if num_workers is None else num_workers
    set_seed(seed)
    loaders = DS.make_loaders(build_dataset, split, num_workers=workers)
    model = build_model()
    res = T.train_model(
        model, loaders, epochs=epochs, device=device, verbose=verbose,
        log_csv=CONFIG["fase2_reports"] / "log" / f"{name}_seed{seed}.csv",
        ckpt_path=CONFIG["fase2_models"] / f"{name}_seed{seed}.pt", **train_kwargs)
    DS.close_datasets(*(ld.dataset for ld in loaders.values()))

    full = build_dataset(list(range(n_total)))
    batch = CONFIG.get("micro_batch_size") or CONFIG["batch_size"]
    out = extract_outputs(model, DataLoader(full, batch_size=batch, shuffle=False,
                                            num_workers=workers), blank_input, device)
    DS.close_datasets(full)

    meta = {"nama": name, "seed": seed,
            **{k: res[k] for k in ("best_epoch", "best_val_oa", "epochs_run",
                                   "durasi_detik", "device", "micro_batch",
                                   "accum_steps", "amp")}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")      # atomic: no half-written cache
    np.savez(tmp, **out, meta=json.dumps(meta))
    tmp.replace(path)
    return {**meta, "dari_cache": False}


# --------------------------------------------------------------------------- #
# fusion rules

def late_fusion(logit_img: np.ndarray, logit_txt: np.ndarray) -> np.ndarray:
    """Kashyap et al.: unweighted average of the two streams' logits."""
    return (logit_img + logit_txt) / 2


class _FeaturePairs(Dataset):
    def __init__(self, fi: torch.Tensor, ft: torch.Tensor, y: torch.Tensor, idx):
        self.fi, self.ft, self.y, self.idx = fi, ft, y, list(idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        r = self.idx[i]
        return (self.fi[r], self.ft[r]), self.y[r]


def train_concat_head(img: dict, txt: dict, y: np.ndarray, split: dict, seed: int,
                      epochs: int | None = None, verbose: bool = False,
                      **train_kwargs) -> ConcatHead:
    """ConcatHead on the cached features, early-stopped on val. CPU: inputs are tiny.

    Caveat for the write-up: the train features come from models that were fitted on
    those same documents, so the head judges each branch by its (optimistic) train
    behaviour — the same thing end-to-end training does.
    """
    set_seed(seed)
    fi, ft = torch.from_numpy(img["feat"]), torch.from_numpy(txt["feat"])
    yy = torch.from_numpy(np.asarray(y, dtype=np.int64))
    loaders = DS.make_loaders(lambda idx: _FeaturePairs(fi, ft, yy, idx), split,
                              num_workers=0)
    head = ConcatHead(img_dim=fi.shape[1], txt_dim=ft.shape[1],
                      n_classes=img["logit"].shape[1])
    T.train_model(head, loaders, device=torch.device("cpu"), amp=False,
                  epochs=CONFIG["epochs"]["head"] if epochs is None else epochs,
                  verbose=verbose, **train_kwargs)
    return head.eval()


@torch.no_grad()
def head_predict(head: ConcatHead, fi: np.ndarray, ft: np.ndarray) -> np.ndarray:
    return head((torch.from_numpy(np.ascontiguousarray(fi)),
                 torch.from_numpy(np.ascontiguousarray(ft)))).argmax(1).numpy()


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"oa": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0))}


def evaluate_cell(img: dict, txt: dict, y: np.ndarray, split: dict, seed: int,
                  head_epochs: int | None = None, **head_kwargs) -> tuple[list, dict]:
    """Every Fase 2 measurement for one (OCR condition, seed) on the test split.

    Returns (rows, preds): one score row per model, and the test predictions per
    model (plus "y_true") for McNemar. "_tanpa_teks" / "_tanpa_citra" replace that
    modality by the model's own output for a blank input.
    """
    te = np.asarray(split["test"])
    yt = np.asarray(y)[te]
    li, lt = img["logit"][te], txt["logit"][te]
    fi, ft = img["feat"][te], txt["feat"][te]

    def blank(v):
        return np.repeat(v[None], len(te), axis=0)

    head = train_concat_head(img, txt, y, split, seed, epochs=head_epochs, **head_kwargs)
    preds = {
        "IMAGE": li.argmax(1),
        "TEXT": lt.argmax(1),
        "LATE": late_fusion(li, lt).argmax(1),
        "LATE_tanpa_teks": late_fusion(li, blank(txt["blank_logit"])).argmax(1),
        "LATE_tanpa_citra": late_fusion(blank(img["blank_logit"]), lt).argmax(1),
        "CONCAT": head_predict(head, fi, ft),
        "CONCAT_tanpa_teks": head_predict(head, fi, blank(txt["blank_feat"])),
        "CONCAT_tanpa_citra": head_predict(head, blank(img["blank_feat"]), ft),
    }
    # tie-break to IMAGE, the stronger baseline in Fase 1 (see train.oracle)
    preds["ORACLE"] = T.oracle(yt, preds["TEXT"], preds["IMAGE"])["y_pred"]
    rows = [{"model": k, **scores(yt, p)} for k, p in preds.items()]
    return rows, {**preds, "y_true": yt}


def run_grid(y: np.ndarray, splits: dict, conditions: list[str], seeds: list[int],
             cell: Callable = evaluate_cell, **kwargs) -> tuple[pd.DataFrame, dict]:
    """`cell` over every condition x seed. Returns (scores, preds[(cond, seed)])."""
    rows, preds = [], {}
    for seed in seeds:
        img = load_outputs(IMAGE, seed)
        for cond in conditions:
            r, p = cell(img, load_outputs(text_name(cond), seed), y, splits[seed], seed,
                        **kwargs)
            rows += [{"ocr": cond, "seed": seed, **x} for x in r]
            preds[(cond, seed)] = p
    return pd.DataFrame(rows), preds


# --------------------------------------------------------------------------- #
# follow-up to 16-fusi-per-ocr finding 2: is the missing-modality collapse an
# out-of-distribution artefact, and did the head learn "empty text => Note"?

def empty_text_mask(txt: dict, tol: float = 1e-3) -> np.ndarray:
    """Documents whose text features equal the blank-input features.

    An empty page and a page whose tokens were all dropped both become the all-zero
    sequence, i.e. exactly the blank input, so this is "empty" as the model sees it.
    `tol` absorbs batch-vs-single numeric noise on GPU.
    """
    scale = max(1.0, float(np.abs(txt["blank_feat"]).max()))
    return np.abs(txt["feat"] - txt["blank_feat"]).max(1) <= tol * scale


def evaluate_followup(img: dict, txt: dict, y: np.ndarray, split: dict, seed: int,
                      head_epochs: int | None = None, **head_kwargs) -> tuple[list, dict]:
    """Missing modality with an in-distribution substitute, and a head without empties.

    "_rerata": the missing modality is replaced by its mean feature (CONCAT) or mean
    logit (LATE) over the documents the head was trained on, instead of the blank
    input's output. "CONCAT_nonkosong*": a second head trained with the empty-text
    documents (empty_text_mask) left out of its train and val; test is unchanged.
    Rows also carry which class takes the largest share of the predictions.
    """
    te = np.asarray(split["test"])
    yt = np.asarray(y)[te]
    li, lt = img["logit"][te], txt["logit"][te]
    fi, ft = img["feat"][te], txt["feat"][te]
    empty = empty_text_mask(txt)
    keep = {p: [i for i in split[p] if not empty[i]] for p in ("train", "val")}

    def fill(out, key, idx):
        return np.repeat(out[key][np.asarray(idx)].mean(0)[None], len(te), axis=0)

    def blank(v):
        return np.repeat(v[None], len(te), axis=0)

    tr = split["train"]
    head = train_concat_head(img, txt, y, split, seed, epochs=head_epochs, **head_kwargs)
    head_ne = train_concat_head(img, txt, y, {**split, **keep}, seed, epochs=head_epochs,
                                **head_kwargs)
    preds = {
        "IMAGE": li.argmax(1),
        "TEXT": lt.argmax(1),
        "LATE_tanpa_teks_rerata": late_fusion(li, fill(txt, "logit", tr)).argmax(1),
        "LATE_tanpa_citra_rerata": late_fusion(fill(img, "logit", tr), lt).argmax(1),
        "CONCAT": head_predict(head, fi, ft),
        "CONCAT_tanpa_teks": head_predict(head, fi, blank(txt["blank_feat"])),
        "CONCAT_tanpa_teks_rerata": head_predict(head, fi, fill(txt, "feat", tr)),
        "CONCAT_tanpa_citra_rerata": head_predict(head, fill(img, "feat", tr), ft),
        "CONCAT_nonkosong": head_predict(head_ne, fi, ft),
        "CONCAT_nonkosong_tanpa_teks": head_predict(head_ne, fi, blank(txt["blank_feat"])),
        "CONCAT_nonkosong_tanpa_teks_rerata": head_predict(
            head_ne, fi, fill(txt, "feat", keep["train"])),
    }
    n_cls = img["logit"].shape[1]
    rows = []
    for k, p in preds.items():
        share = np.bincount(p, minlength=n_cls) / len(p)
        rows.append({"model": k, **scores(yt, p), "kelas_terbanyak": int(share.argmax()),
                     "porsi_kelas_terbanyak": float(share.max())})
    info = {"n_kosong_latih": len(split["train"]) - len(keep["train"]),
            "n_kosong_val": len(split["val"]) - len(keep["val"]),
            "n_kosong_uji": int(empty[te].sum())}
    rows = [{**r, **info} for r in rows]
    return rows, {**preds, "y_true": yt}


# --------------------------------------------------------------------------- #
# summaries and tests

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and population std over seeds, one row per (ocr, model)."""
    g = df.groupby(["ocr", "model"], sort=False)
    return pd.DataFrame({
        "oa_mean": g["oa"].mean(), "oa_std": g["oa"].std(ddof=0),
        "f1_mean": g["macro_f1"].mean(), "f1_std": g["macro_f1"].std(ddof=0),
        "n_seed": g["oa"].size()}).reset_index()


def modality_use(df: pd.DataFrame) -> pd.DataFrame:
    """How much each fusion rule leans on each modality, per (ocr, seed).

    turun_tanpa_teks = OA(rule) - OA(rule with blank text). Q3/H3: does this grow
    with the text branch's own accuracy?
    """
    w = df.pivot_table(index=["ocr", "seed"], columns="model", values="oa")
    out = pd.DataFrame({"oa_text": w["TEXT"], "oa_image": w["IMAGE"],
                        "oa_oracle": w["ORACLE"]})
    for rule in ("LATE", "CONCAT"):
        out[f"oa_{rule.lower()}"] = w[rule]
        out[f"{rule.lower()}_turun_tanpa_teks"] = w[rule] - w[f"{rule}_tanpa_teks"]
        out[f"{rule.lower()}_turun_tanpa_citra"] = w[rule] - w[f"{rule}_tanpa_citra"]
    return out.reset_index()


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test (two-sided binomial on the discordant pairs)."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"hanya_a_benar": b, "hanya_b_benar": c, "p": p}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, same order as given."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def mcnemar_conditions(preds: dict, model: str, conditions: list[str],
                       seeds: list[int]) -> pd.DataFrame:
    """H1: every pair of OCR conditions for one model, Holm-corrected within a seed."""
    rows = []
    for seed in seeds:
        fam = []
        for a, b in combinations(conditions, 2):
            pa, pb = preds[(a, seed)], preds[(b, seed)]
            yt = pa["y_true"]
            assert np.array_equal(yt, pb["y_true"]), "set uji berbeda antar-kondisi"
            fam.append({"seed": seed, "model": model, "a": a, "b": b,
                        "oa_a": float((pa[model] == yt).mean()),
                        "oa_b": float((pb[model] == yt).mean()),
                        **mcnemar(yt, pa[model], pb[model])})
        for r, q in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = float(q)
            r["signifikan"] = bool(q < CONFIG["alpha"])
        rows += fam
    return pd.DataFrame(rows)


def mcnemar_models(preds: dict, model_a: str, model_b: str, conditions: list[str],
                   seeds: list[int]) -> pd.DataFrame:
    """H3: model_a vs model_b inside each condition, Holm-corrected within a seed."""
    rows = []
    for seed in seeds:
        fam = []
        for cond in conditions:
            p = preds[(cond, seed)]
            yt = p["y_true"]
            fam.append({"seed": seed, "ocr": cond, "a": model_a, "b": model_b,
                        "oa_a": float((p[model_a] == yt).mean()),
                        "oa_b": float((p[model_b] == yt).mean()),
                        **mcnemar(yt, p[model_a], p[model_b])})
        for r, q in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = float(q)
            r["signifikan"] = bool(q < CONFIG["alpha"])
        rows += fam
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    """End to end on a synthetic task where each modality only knows half the classes,
    so fusion has something real to gain and missing modality something to lose."""
    import tempfile

    import torch.nn as nn

    from . import splits as S

    class Tiny(nn.Module):
        def __init__(self, d_in, n):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU())
            self.classifier = nn.Linear(16, n)

        def features(self, x):
            return self.f(x)

        def forward(self, x):
            return self.classifier(self.features(x))

    class Rows(Dataset):
        def __init__(self, x, y, idx):
            self.x, self.y, self.idx = x, y, list(idx)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            r = self.idx[i]
            return self.x[r], int(self.y[r])

    rng = np.random.default_rng(0)
    n, k = 400, 4
    y = np.arange(n) % k
    # image separates classes {0,1} from the rest and 0 from 1, text does the same
    # for {2,3}; each is blind inside the other pair
    x_img = rng.normal(0, 0.1, (n, 6)).astype(np.float32)
    x_txt = rng.normal(0, 0.1, (n, 6)).astype(np.float32)
    for c in range(k):
        x_img[y == c, min(c, 2)] += 1.0          # 2 and 3 collapse onto one axis
        x_txt[y == c, max(c, 1) + 2] += 1.0      # 0 and 1 collapse onto one axis
    # condition "c": a third of class 3 has empty text, i.e. exactly the blank input
    is_empty = (y == 3) & (np.arange(n) % 3 == 0)
    x_txt_empty = np.where(is_empty[:, None], 0, x_txt).astype(np.float32)
    labels = [str(c) for c in y]
    splits = {s: S.make_split(labels, s, n_train=240, val_fraction=0.25) for s in (1, 2)}

    saved = {k_: CONFIG[k_] for k_ in ("fase2_data", "fase2_models", "fase2_reports",
                                         "micro_batch_size")}
    with tempfile.TemporaryDirectory() as tmp:
        CONFIG.update(fase2_data=Path(tmp) / "d", fase2_models=Path(tmp) / "m",
                      fase2_reports=Path(tmp) / "r", micro_batch_size=None)
        try:
            for seed in (1, 2):
                for name, x in ((IMAGE, x_img), (text_name("a"), x_txt),
                                (text_name("b"), x_txt[:, ::-1].copy()),
                                (text_name("c"), x_txt_empty)):
                    meta = train_unimodal(
                        name, seed, splits[seed], lambda: Tiny(6, k),
                        lambda idx, x=x: Rows(torch.from_numpy(x), y, idx),
                        torch.zeros(1, 6), epochs=60, n_total=n,
                        device=torch.device("cpu"), num_workers=0, lr=0.1,
                        patience=0, verbose=False, amp=False)
                    assert not meta["dari_cache"]
            again = train_unimodal(IMAGE, 1, splits[1], lambda: 1 / 0, None, None,
                                   epochs=1, n_total=n)
            assert again["dari_cache"], "cache tidak dipakai ulang"
            out = load_outputs(IMAGE, 1)
            assert out["feat"].shape == (n, 16) and out["logit"].shape == (n, k)
            assert out["blank_logit"].shape == (k,)

            df, preds = run_grid(y, splits, ["a", "b"], [1, 2], head_epochs=80,
                                 lr=0.1, patience=0)
            s = summarize(df).set_index(["ocr", "model"])
            for cond in ("a", "b"):
                oa = s.loc[cond, "oa_mean"]
                best_uni = max(oa["IMAGE"], oa["TEXT"])
                assert 0.6 < best_uni < 0.85, ("unimodal harus buta separuh kelas", oa)
                assert oa["ORACLE"] >= best_uni
                assert oa["LATE"] > best_uni + 0.1, ("late fusion tidak menolong", oa)
                assert oa["CONCAT"] > best_uni + 0.1, ("concat head tidak menolong", oa)
            use = modality_use(df)
            assert (use["late_turun_tanpa_teks"] > 0.1).all(), use
            assert (use["concat_turun_tanpa_citra"] > 0.1).all(), use

            h1 = mcnemar_conditions(preds, "TEXT", ["a", "b"], [1, 2])
            assert len(h1) == 2 and {"p", "p_holm", "signifikan"} <= set(h1.columns)
            h3 = mcnemar_models(preds, "LATE", "CONCAT", ["a", "b"], [1, 2])
            assert len(h3) == 4

            fu, fpreds = run_grid(y, splits, ["a", "c"], [1, 2], cell=evaluate_followup,
                                  head_epochs=80, lr=0.1, patience=0)
            for seed in (1, 2):
                txt_c = load_outputs(text_name("c"), seed)
                assert np.array_equal(empty_text_mask(txt_c), is_empty), "mask kosong salah"
                assert not empty_text_mask(load_outputs(text_name("a"), seed)).any()
                r = fu[(fu.seed == seed) & (fu.ocr == "c")].iloc[0]
                sp_ = splits[seed]
                assert (r.n_kosong_latih, r.n_kosong_val, r.n_kosong_uji) == tuple(
                    int(is_empty[sp_[p]].sum()) for p in ("train", "val", "test")), r
                assert r.n_kosong_latih > 0
                # same seed, same data, same head as the grid; and with nothing to drop
                # the "non-empty" head is that same head again
                assert np.array_equal(fpreds[("a", seed)]["CONCAT"], preds[("a", seed)]["CONCAT"])
                fa = fpreds[("a", seed)]
                assert np.array_equal(fa["CONCAT_nonkosong"], fa["CONCAT"])
            w = fu.pivot_table(index=["ocr", "seed"], columns="model", values="oa")
            # the mean substitute is in distribution: losing text leaves roughly IMAGE
            assert ((w["LATE_tanpa_teks_rerata"] - w["IMAGE"]).abs() < 0.1).all(), w
            assert (w["CONCAT_tanpa_teks_rerata"] > w["IMAGE"] - 0.15).all(), w
            assert fu["porsi_kelas_terbanyak"].between(1 / k, 1).all()
        finally:
            CONFIG.update(saved)

    # McNemar against a hand count, Holm against a hand computation
    yt = np.zeros(10, dtype=int)
    pa = np.array([0] * 8 + [1] * 2)
    pb = np.array([0] * 2 + [1] * 8)
    m = mcnemar(yt, pa, pb)
    assert (m["hanya_a_benar"], m["hanya_b_benar"]) == (6, 0), m
    assert abs(m["p"] - binomtest(6, 6, 0.5).pvalue) < 1e-12
    assert mcnemar(yt, pa, pa)["p"] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]), holm([0.01, 0.04, 0.03])
    assert np.allclose(late_fusion(np.array([[2.0, 0.0]]), np.array([[0.0, 4.0]])),
                       [[1.0, 2.0]])
    print("experiment self-check ok")


if __name__ == "__main__":
    _self_check()
