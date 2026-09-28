"""Multimodal fusion model, paper section 3.3.

Image branch: MobileNetV2 -> global average pooling -> 1280 -> FC -> 128.
Text branch: CNN1D -> 128.
Both 128-d vectors are combined and fed to an MLP -> softmax, trained end to end.

Two combination strategies, switchable:

- "concat"  -> [v_img ; v_txt], 256-d. What the paper settled on.
- "sum"     -> w0*v_img + w1*v_txt, 128-d. The paper's "adaptive averaging", which
               it reports failed badly. We test that ourselves rather than
               inheriting the conclusion.

THREE ambiguities the paper leaves open are decided here, not silently:

1. What makes the summation "adaptive" is never stated. We use one learned scalar
   per branch, passed through a softmax so the two weights sum to 1 — the reading
   that actually matches the phrase "adaptive averaging". A plain unweighted sum, or
   a per-dimension gate, would behave differently, so a negative result for this
   strategy is a result about OUR choice, not a replication of the paper's finding.
2. The fusion MLP's depth and width are not given. One hidden layer of 256 with
   BN/ReLU/Dropout.
3. Whether the branches start from separately-trained baseline weights or from the
   usual initialisation is not stated. We read "trained end-to-end" as the latter:
   the image branch starts from ImageNet weights exactly as the IMAGE baseline does,
   and the text branch starts from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CONFIG
from .image_model import ImageBranch
from .text_models import CNN1D

STRATEGIES = ("concat", "sum")


class FusionModel(nn.Module):
    def __init__(self, n_classes: int = 10, strategy: str | None = None,
                 pretrained_image: bool = True, fusion_width: int = 256,
                 dropout: float | None = None):
        super().__init__()
        strategy = CONFIG["fusion_strategy"] if strategy is None else strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy harus salah satu dari {STRATEGIES}, "
                             f"dapat {strategy!r}")
        dropout = CONFIG["dropout"] if dropout is None else dropout
        dim = CONFIG["fusion_dim"]

        self.strategy = strategy

        # Both branches are reused as-is, with their classifiers replaced by
        # Identity: the paper cuts the final layers off, and leaving dead Linear
        # layers in would ship unused weights inside every checkpoint.
        self.image = ImageBranch(n_classes=n_classes, pretrained=pretrained_image)
        self.image.classifier = nn.Identity()
        self.project = nn.Linear(self.image.feature_dim, dim)   # 1280 -> 128

        self.text = CNN1D(n_classes=n_classes, out_dim=dim)
        self.text.classifier = nn.Identity()

        if strategy == "sum":
            # two logits -> softmax -> weights that sum to 1
            self.branch_logits = nn.Parameter(torch.zeros(2))

        head_in = 2 * dim if strategy == "concat" else dim
        self.head = nn.Sequential(
            nn.Linear(head_in, fusion_width),
            nn.BatchNorm1d(fusion_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, n_classes),
        )

    def branch_weights(self) -> torch.Tensor:
        """The learned averaging weights, for logging. Only meaningful for "sum"."""
        if self.strategy != "sum":
            raise AttributeError('branch_weights hanya ada untuk strategy="sum"')
        return torch.softmax(self.branch_logits, dim=0)

    def features(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        v_img = self.project(self.image.features(image))
        v_txt = self.text.features(text)
        if self.strategy == "concat":
            return torch.cat([v_img, v_txt], dim=1)
        w = torch.softmax(self.branch_logits, dim=0)
        return w[0] * v_img + w[1] * v_txt

    def forward(self, batch) -> torch.Tensor:
        """`batch` is the (image, text) tuple produced by PairDataset."""
        image, text = batch
        return self.head(self.features(image, text))


def _self_check() -> None:
    torch.manual_seed(0)
    b, size, dim, length, n_classes = 2, CONFIG["image_size"], 300, 500, 10
    img = torch.randn(b, 3, size, size)
    txt = torch.randn(b, dim, length)

    for strategy in STRATEGIES:
        model = FusionModel(n_classes=n_classes, strategy=strategy,
                            pretrained_image=False)
        feats = model.features(img, txt)
        expected = 256 if strategy == "concat" else 128
        assert feats.shape == (b, expected), (strategy, feats.shape)
        assert model((img, txt)).shape == (b, n_classes), strategy

        # The wiring check that matters: a fusion model where one branch is detached
        # trains perfectly happily while being secretly unimodal. Both branches must
        # receive gradient.
        model.zero_grad()
        model((img, txt)).sum().backward()
        img_conv = next(p for p in model.image.trunk.parameters() if p.ndim == 4)
        txt_conv = next(p for p in model.text.conv.parameters() if p.ndim == 3)
        assert img_conv.grad is not None and img_conv.grad.abs().sum() > 0, \
            f"{strategy}: gradien tidak mencapai cabang citra"
        assert txt_conv.grad is not None and txt_conv.grad.abs().sum() > 0, \
            f"{strategy}: gradien tidak mencapai cabang teks"
        assert model.project.weight.grad.abs().sum() > 0, strategy

        # no dead classifier weights left inside the checkpoint
        state = model.state_dict()
        assert not any(k.startswith("image.classifier") for k in state), \
            "classifier cabang citra masih ada di state_dict"
        assert not any(k.startswith("text.classifier") for k in state), \
            "classifier cabang teks masih ada di state_dict"

    # the "adaptive" weights really are learnable and really normalised
    model = FusionModel(strategy="sum", pretrained_image=False)
    w = model.branch_weights()
    assert torch.allclose(w.sum(), torch.tensor(1.0)), w
    assert torch.allclose(w, torch.tensor([0.5, 0.5])), ("inisialisasi harus seimbang", w)
    model.zero_grad()
    model((img, txt)).sum().backward()
    assert model.branch_logits.grad is not None \
        and model.branch_logits.grad.abs().sum() > 0, \
        "bobot penjumlahan tidak ikut dilatih — itu bukan 'adaptive' averaging"

    # concat must NOT carry branch weights, and asking for them must fail loudly
    concat = FusionModel(strategy="concat", pretrained_image=False)
    assert not hasattr(concat, "branch_logits")
    try:
        concat.branch_weights()
    except AttributeError:
        pass
    else:
        raise AssertionError("branch_weights seharusnya tidak tersedia untuk concat")

    try:
        FusionModel(strategy="rata-rata")
    except ValueError:
        pass
    else:
        raise AssertionError("strategy tak dikenal seharusnya ditolak")

    print("fusion self-check ok")
    n = sum(p.numel() for p in FusionModel(strategy="concat",
                                           pretrained_image=False).parameters())
    print(f"  parameter (concat): {n:,}")


if __name__ == "__main__":
    _self_check()
