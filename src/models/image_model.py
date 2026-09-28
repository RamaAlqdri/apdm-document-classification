"""MobileNetV2 image branch, paper sections 3.1 and 4.2.

`features(x)` returns the 1280-dim global-average-pooled vector; the fusion model
projects that down to 128 itself, exactly as the paper describes. The standalone
IMAGE baseline classifies straight off the 1280 vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from ..config import CONFIG


def build_transform() -> transforms.Compose:
    """Resize to 384x384 WITHOUT padding, warping the aspect ratio on purpose.

    The paper cites Tensmeyer & Martinez reporting better accuracy this way than
    with padding at the same resolution. Grayscale is duplicated into 3 channels so
    the ImageNet weights apply.

    One transform for train and eval alike: the paper found augmentation slightly
    HURT (83.9% with vs 84.5% without), because every document is dark text on
    white with horizontal lines, so colour and geometric jitter buys nothing.

    The normalisation statistics are our assumption — the paper never states them.
    ImageNet values are the consistent choice given ImageNet-pretrained weights.
    """
    size = CONFIG["image_size"]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),  # tuple => warp, not aspect-preserving
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["imagenet_mean"], CONFIG["imagenet_std"]),
    ])


class ImageBranch(nn.Module):
    """MobileNetV2, ImageNet-pretrained, fine-tuned end to end.

    The whole network is trainable: the paper fine-tunes rather than freezing the
    backbone. Freezing would cost several points of accuracy and the gap would be
    misread as a replication failure.
    """

    def __init__(self, n_classes: int = 10, pretrained: bool = True,
                 dropout: float | None = None):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        self.trunk = backbone.features           # conv stack, no classifier
        self.pool = nn.AdaptiveAvgPool2d(1)      # global average pooling per channel
        self.feature_dim = backbone.last_channel  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(CONFIG["dropout"] if dropout is None else dropout),
            nn.Linear(self.feature_dim, n_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.trunk(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _self_check() -> None:
    """pretrained=False on purpose: a self-check must not download weights."""
    torch.manual_seed(0)
    model = ImageBranch(n_classes=10, pretrained=False)
    x = torch.randn(2, 3, CONFIG["image_size"], CONFIG["image_size"])

    model.eval()
    with torch.no_grad():
        feats, logits = model.features(x), model(x)
    assert feats.shape == (2, 1280), feats.shape
    assert logits.shape == (2, 10), logits.shape
    assert model.feature_dim == 1280, model.feature_dim

    # the backbone must be trainable, not frozen
    assert all(p.requires_grad for p in model.trunk.parameters()), \
        "backbone dibekukan — paper melakukan fine-tuning penuh"

    # gradients reach the very first conv
    model.train()
    model(x).sum().backward()
    first = next(p for p in model.trunk.parameters() if p.ndim == 4)
    assert first.grad is not None and first.grad.abs().sum() > 0

    # the transform warps rather than pads: a very non-square input must come out
    # square, with 3 channels, and no grey border
    from PIL import Image
    from torchvision import transforms as tv

    size = CONFIG["image_size"]
    tall = Image.new("L", (300, 900), color=255)
    out = build_transform()(tall)
    assert out.shape == (3, size, size), out.shape

    # Check duplication and the absence of padding BEFORE normalisation. After it,
    # the three channels legitimately differ: ImageNet mean/std are per-channel, so
    # an identical grayscale triple comes out as three different constants.
    pre = tv.Compose(build_transform().transforms[:-1])(tall)
    assert torch.allclose(pre[0], pre[1]) and torch.allclose(pre[1], pre[2]), \
        "kanal grayscale tidak diduplikasi identik sebelum normalisasi"
    assert pre.std() < 1e-5, "citra polos seharusnya seragam — ada padding?"
    # and each channel is still uniform after normalisation
    assert all(out[c].std() < 1e-5 for c in range(3)), "normalisasi merusak keseragaman"

    norm = build_transform().transforms[-1]
    assert isinstance(norm, tv.Normalize) and list(norm.mean) == CONFIG["imagenet_mean"]

    print("image_model self-check ok")
    print(f"  parameter: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    _self_check()
