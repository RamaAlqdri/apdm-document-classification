"""Generic training loop shared by every experiment.

Optimiser settings come from the paper (SGD with momentum 0.9, lr 0.01, batch 40).
Early stopping and the best-on-val checkpoint are OUR additions: the paper trains a
fixed number of epochs with hyperparameters tuned on a separate subset. Recorded as
a deviation in the spec note.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import CONFIG


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda, then Apple mps, then cpu. Explicit so the log records what ran."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch, device):
    """Batches are either a tensor or a tuple of tensors (the fusion case)."""
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    return batch.to(device)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred). Kept separate because Tahap 6's oracle needs the
    per-sample predictions of each unimodal baseline, not just their accuracy."""
    model.eval()
    trues, preds = [], []
    for x, y in loader:
        logits = model(_to_device(x, device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    """Overall accuracy and macro F1 — the two metrics the paper reports."""
    y_true, y_pred = predict(model, loader, device)
    return {
        "oa": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_model(model: nn.Module, loaders: dict[str, DataLoader], *,
                epochs: int, lr: float | None = None, momentum: float | None = None,
                device=None, log_csv: Path | None = None,
                ckpt_path: Path | None = None, patience: int | None = None,
                verbose: bool = True) -> dict:
    """Train, keeping the best-on-val weights. Returns history plus timings.

    The returned model has the best-on-val weights loaded, not the last epoch's.
    """
    lr = CONFIG["lr"] if lr is None else lr
    momentum = CONFIG["momentum"] if momentum is None else momentum
    patience = CONFIG["patience"] if patience is None else patience
    device = pick_device() if device is None else device

    model.to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    handle = writer = None
    if log_csv:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        handle = log_csv.open("w", newline="")
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_oa", "val_macro_f1",
                        "detik"],
        )
        writer.writeheader()

    history, best_val, best_epoch, best_state = [], -1.0, -1, None
    t_start = time.perf_counter()
    try:

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            total, seen = 0.0, 0
            for x, y in loaders["train"]:
                x, y = _to_device(x, device), y.to(device)
                optimiser.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(y)
                seen += len(y)
            train_loss = total / max(seen, 1)

            model.eval()
            with torch.no_grad():
                vtotal, vseen = 0.0, 0
                for x, y in loaders["val"]:
                    x, y = _to_device(x, device), y.to(device)
                    vtotal += criterion(model(x), y).item() * len(y)
                    vseen += len(y)
            val_loss = vtotal / max(vseen, 1)
            val = evaluate(model, loaders["val"], device)

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_oa": round(val["oa"], 5),
                "val_macro_f1": round(val["macro_f1"], 5),
                "detik": round(time.perf_counter() - t0, 2),
            }
            history.append(row)
            if writer is not None:
                writer.writerow(row)
                handle.flush()  # a crash at epoch 150 must not lose epochs 1-149

            if val["oa"] > best_val:
                best_val, best_epoch = val["oa"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_path:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"epoch": epoch, "val_oa": best_val, "state_dict": best_state},
                               ckpt_path)

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"  epoch {epoch:3d}/{epochs}  train_loss {train_loss:.4f}  "
                      f"val_oa {val['oa']:.4f}  ({row['detik']:.1f}s)")

            if patience and epoch - best_epoch >= patience:
                if verbose:
                    print(f"  early stopping di epoch {epoch}; terbaik epoch {best_epoch}")
                break

    finally:
        if handle is not None:
            handle.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_oa": best_val,
        "epochs_run": len(history),
        # 3 dp, not 1: estimate_duration divides this, and a fast probe epoch
        # rounded to 0.0 would extrapolate to "0 minutes for 200 epochs"
        "durasi_detik": round(time.perf_counter() - t_start, 3),
        "device": str(device),
    }


def estimate_duration(model: nn.Module, loaders: dict[str, DataLoader],
                      target_epochs: int, probe_epochs: int = 2, **kwargs) -> dict:
    """Run a couple of epochs to measure seconds/epoch, then extrapolate.

    The plan's ">15 minutes, ask first" rule is useless if nothing measures the
    time first — 200 epochs of MobileNetV2 at 384x384 will trip it every run.
    """
    res = train_model(model, loaders, epochs=probe_epochs, patience=0,
                      verbose=False, **kwargs)
    per_epoch = res["durasi_detik"] / max(res["epochs_run"], 1)
    total = per_epoch * target_epochs
    # seconds are reported too: a fast model rounds to "0.0 menit", which reads as
    # a broken estimate rather than a quick one
    return {
        "detik_per_epoch": round(per_epoch, 4),
        "target_epochs": target_epochs,
        "perkiraan_detik": round(total, 1),
        "perkiraan_menit": round(total / 60, 2),
        "perkiraan_jam": round(total / 3600, 2),
        "device": res["device"],
    }


def report(result: dict, class_names: list[str] | None = None) -> str:
    """Per-class table, for pasting into the experiment note."""
    class_names = CONFIG["classes"] if class_names is None else class_names
    return classification_report(result["y_true"], result["y_pred"],
                                 target_names=class_names, digits=3, zero_division=0)


def plot_confusion(result: dict, title: str, class_names: list[str] | None = None):
    """Normalised confusion matrix figure. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    class_names = CONFIG["classes"] if class_names is None else class_names
    cm = confusion_matrix(result["y_true"], result["y_pred"],
                          labels=range(len(class_names)))
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("prediksi")
    ax.set_ylabel("sebenarnya")
    ax.set_title(f"{title}\nOA {result['oa']:.3f} · macro F1 {result['macro_f1']:.3f}")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if norm[i, j] > 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def _self_check() -> None:
    import tempfile

    from torch.utils.data import TensorDataset

    torch.manual_seed(0)
    device = torch.device("cpu")

    # a task that is trivially learnable, so a loop that does not learn is a bug
    n, dim, n_classes = 240, 8, 3
    y = torch.arange(n) % n_classes
    x = torch.nn.functional.one_hot(y, n_classes).float()
    x = torch.cat([x, torch.randn(n, dim - n_classes) * 0.01], dim=1)

    def loader(lo, hi, shuffle):
        return DataLoader(TensorDataset(x[lo:hi], y[lo:hi]), batch_size=16,
                          shuffle=shuffle)

    loaders = {"train": loader(0, 180, True), "val": loader(180, 210, False),
               "test": loader(210, n, False)}
    model = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, n_classes))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        res = train_model(model, loaders, epochs=12, lr=0.1, device=device,
                          log_csv=tmp / "log.csv", ckpt_path=tmp / "best.pt",
                          patience=0, verbose=False)

        assert res["epochs_run"] == 12, res["epochs_run"]
        assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"], \
            "loss tidak turun — loop training tidak belajar"
        assert res["best_val_oa"] > 0.8, res["best_val_oa"]

        rows = list(csv.DictReader((tmp / "log.csv").open()))
        assert len(rows) == 12, len(rows)
        assert set(rows[0]) == {"epoch", "train_loss", "val_loss", "val_oa",
                               "val_macro_f1", "detik"}, set(rows[0])

        ckpt = torch.load(tmp / "best.pt", weights_only=True)
        assert ckpt["epoch"] == res["best_epoch"], (ckpt["epoch"], res["best_epoch"])
        # the model in hand must be the best epoch, not the last
        for k, v in ckpt["state_dict"].items():
            assert torch.allclose(model.state_dict()[k].cpu(), v), \
                "bobot terbaik tidak dimuat kembali setelah training"

        ev = evaluate(model, loaders["test"], device)
        assert 0.0 <= ev["oa"] <= 1.0 and len(ev["y_true"]) == 30
        assert report(ev, [f"k{i}" for i in range(n_classes)]).count("\n") > 3

        # early stopping fires on a model that cannot improve (lr=0)
        frozen = nn.Sequential(nn.Linear(dim, n_classes))
        res2 = train_model(frozen, loaders, epochs=50, lr=0.0, device=device,
                           patience=3, verbose=False)
        assert res2["epochs_run"] <= 5, res2["epochs_run"]

        est = estimate_duration(nn.Sequential(nn.Linear(dim, n_classes)), loaders,
                               target_epochs=200, probe_epochs=2, device=device)
        assert est["target_epochs"] == 200, est
        assert est["detik_per_epoch"] > 0, ("probe tidak terukur sama sekali", est)
        # the extrapolation must actually be per_epoch * target, not a stale constant
        assert abs(est["perkiraan_detik"] - est["detik_per_epoch"] * 200) < 0.5, est

    print("train self-check ok")
    print(f"  device terpilih di mesin ini: {pick_device()}")


if __name__ == "__main__":
    _self_check()
