"""Single source of truth for paths, hyperparameters and seeding."""

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# DataLoader workers. Windows spawns rather than forks, and inside a Jupyter kernel
# that regularly hangs or crashes, so default to single-process there. Override by
# setting CONFIG["num_workers"] in the notebook if you know your setup is fine.
_DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4

CONFIG = {
    # --- paths ---
    "root": ROOT,
    "data_raw": ROOT / "data" / "raw",
    "data_interim": ROOT / "data" / "interim",
    "data_processed": ROOT / "data" / "processed",
    "models": ROOT / "models",
    "figures": ROOT / "reports" / "figures",
    "paper_pdf": ROOT / "references" / "1907.06370v1.pdf",
    # --- dataset ---
    # Canonical class names, matching the "kelas" column of manifest.csv. The image
    # archive uses the short code ADVE for Advertisement; data.CLASS_ALIASES maps it.
    # Label index = position in this list, so the order must never change.
    "classes": [
        "Advertisement", "Email", "Form", "Letter", "Memo",
        "News", "Note", "Report", "Resume", "Scientific",
    ],
    "n_samples_expected": 3482,
    # --- split (deviasi: paper pakai k-fold, kita 3 random split) ---
    "n_train": 800,
    "val_fraction": 0.1,          # dipotong dari n_train, deviasi dari paper
    "seeds": [42, 43, 44],
    # --- text branch (paper sec. 3.2 & 4.2) ---
    "max_words": 500,
    "embedding_dim": 300,
    "fasttext_model": "cc.en.300.bin",
    "cnn1d": {"n_layers": 4, "kernel_size": 12, "channels": 512, "pool_stride": 2},
    # n_hidden is OUR assumption: the paper fixes the width at 2048 but never says
    # how many layers there are. See the spec note's ambiguity table.
    "mlp": {"width": 2048, "n_hidden": 2, "out_dim": 128},
    "dropout": 0.5,  # assumption, paper does not state the rate
    # --- image branch (paper sec. 3.1 & 4.2) ---
    "image_size": 384,            # aspect ratio sengaja di-warp, tanpa padding
    "imagenet_mean": [0.485, 0.456, 0.406],   # asumsi, tidak disebut paper
    "imagenet_std": [0.229, 0.224, 0.225],    # asumsi, tidak disebut paper
    # --- fusion (paper sec. 3.3) ---
    "fusion_dim": 128,
    "image_feature_dim": 1280,
    "fusion_strategy": "concat",  # "concat" | "sum"
    # --- optimisation (paper sec. 4.2, batch 40 for every model) ---
    "optimizer": "sgd",
    "lr": 0.01,
    "momentum": 0.9,
    "batch_size": 40,
    "epochs": {"text": 100, "image": 200, "fusion": 200},
    # Early stopping is OUR deviation: the paper trains a fixed number of epochs.
    "patience": 15,
    "num_workers": _DEFAULT_WORKERS,
    # Micro-batch for gradient accumulation. None = use batch_size directly.
    # Set this on a small GPU: MobileNetV2 at 384x384 with batch 40 needs well over
    # 4GB of VRAM. The optimiser still steps on 40 samples, so the paper's batch
    # size is preserved; only BatchNorm sees the smaller group.
    "micro_batch_size": None,
}


def set_seed(seed: int = 42) -> int:
    """Seed python, numpy and torch. Returns the seed for logging."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


if __name__ == "__main__":
    # the reproducibility claim is the only non-trivial thing here, so check it
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    assert a == [random.random() for _ in range(5)], "set_seed is not reproducible"
    assert len(CONFIG["classes"]) == 10
    print("config self-check ok")
