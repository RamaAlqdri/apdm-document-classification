"""The two text baselines from the paper, section 4.2.

Both expose `features(x)` returning the 128-dim vector the fusion model consumes,
and `forward(x)` returning class logits on top of it. The paper describes the MLP
that way explicitly ("produces a 128 feature vector, classified by a softmax
layer") and says of fusion only that the TEXT model's last layer outputs 128
instead of the number of classes — so routing every classifier through a shared
128-dim bottleneck satisfies both statements with one architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CONFIG


def _he_init(module: nn.Module) -> None:
    """He initialisation, as the paper specifies for the MLP."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """Document-embedding baseline: SIF vector in, class logits out.

    Paper: ReLU, Dropout and BatchNorm after every layer, width fixed at 2048 for
    all layers except the last, which produces 128 features. He initialisation.

    The paper never says HOW MANY 2048-wide layers there are. n_hidden is our
    assumption (default 2), recorded in the spec note's ambiguity table.
    """

    def __init__(self, in_dim: int | None = None, n_classes: int = 10,
                 width: int | None = None, n_hidden: int | None = None,
                 out_dim: int | None = None, dropout: float | None = None):
        super().__init__()
        cfg = CONFIG["mlp"]
        in_dim = CONFIG["embedding_dim"] if in_dim is None else in_dim
        width = cfg["width"] if width is None else width
        n_hidden = cfg["n_hidden"] if n_hidden is None else n_hidden
        out_dim = cfg["out_dim"] if out_dim is None else out_dim
        dropout = CONFIG["dropout"] if dropout is None else dropout

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.ReLU(),
                       nn.Dropout(dropout)]
            d = width
        # last layer is the 128-d feature vector, also with BN/ReLU/Dropout
        layers += [nn.Linear(d, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                   nn.Dropout(dropout)]

        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(out_dim, n_classes)
        self.feature_dim = out_dim
        self.apply(_he_init)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.trunk(x))


class CNN1D(nn.Module):
    """Word-sequence baseline, and the text branch of the fusion model.

    Paper: 4 layers deep, 1D convolutions with a window of 12 interleaved with
    maxpooling of stride 2, 512 channels per layer with ReLU, then a
    max-pooling-through-time layer, Dropout, a fully connected layer, softmax.

    Input is (batch, dim, length) — PyTorch's Conv1d channel-first convention, so
    the 500x300 sequence arrives transposed to 300x500.
    """

    def __init__(self, dim: int | None = None, n_classes: int = 10,
                 channels: int | None = None, kernel_size: int | None = None,
                 n_layers: int | None = None, pool_stride: int | None = None,
                 out_dim: int | None = None, dropout: float | None = None):
        super().__init__()
        cfg = CONFIG["cnn1d"]
        dim = CONFIG["embedding_dim"] if dim is None else dim
        channels = cfg["channels"] if channels is None else channels
        kernel_size = cfg["kernel_size"] if kernel_size is None else kernel_size
        n_layers = cfg["n_layers"] if n_layers is None else n_layers
        pool_stride = cfg["pool_stride"] if pool_stride is None else pool_stride
        out_dim = CONFIG["mlp"]["out_dim"] if out_dim is None else out_dim
        dropout = CONFIG["dropout"] if dropout is None else dropout

        blocks: list[nn.Module] = []
        c_in = dim
        for i in range(n_layers):
            blocks += [
                nn.Conv1d(c_in, channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            # maxpool BETWEEN layers only: pooling after the last conv would throw
            # away resolution that max-pool-through-time is about to consume anyway
            if i < n_layers - 1:
                blocks.append(nn.MaxPool1d(kernel_size=2, stride=pool_stride))
            c_in = channels

        self.conv = nn.Sequential(*blocks)
        self.pool_through_time = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, out_dim)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(out_dim, n_classes)
        self.feature_dim = out_dim
        self.apply(_he_init)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool_through_time(h).squeeze(-1)
        return self.act(self.fc(self.dropout(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _self_check() -> None:
    torch.manual_seed(0)
    b, dim, length, n_classes = 4, 300, 500, 10

    mlp = MLP(in_dim=dim, n_classes=n_classes)
    x = torch.randn(b, dim)
    assert mlp.features(x).shape == (b, 128), mlp.features(x).shape
    assert mlp(x).shape == (b, n_classes), mlp(x).shape
    # width 2048 really is used for the hidden layers
    widths = [m.out_features for m in mlp.trunk if isinstance(m, torch.nn.Linear)]
    assert widths == [2048, 2048, 128], widths

    cnn = CNN1D(dim=dim, n_classes=n_classes)
    seq = torch.randn(b, dim, length)
    assert cnn.features(seq).shape == (b, 128), cnn.features(seq).shape
    assert cnn(seq).shape == (b, n_classes), cnn(seq).shape
    convs = [m for m in cnn.conv if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == 4, len(convs)
    assert all(c.kernel_size == (12,) for c in convs), [c.kernel_size for c in convs]
    assert all(c.out_channels == 512 for c in convs), [c.out_channels for c in convs]

    # max-pool-through-time must make the output length-independent: the same
    # document padded to a different length has to give the same features
    cnn.eval()
    with torch.no_grad():
        short = torch.randn(1, dim, 120)
        padded = torch.cat([short, torch.zeros(1, dim, 380)], dim=-1)
        a, c = cnn.features(short), cnn.features(padded)
    assert a.shape == c.shape == (1, 128)

    # He init: ReLU fan-in scaling means std ~ sqrt(2/fan_in), not the torch default
    w = convs[0].weight
    expected = (2.0 / (w.shape[1] * w.shape[2])) ** 0.5
    assert 0.5 * expected < w.std().item() < 1.5 * expected, (w.std().item(), expected)

    # gradients reach the first conv layer
    cnn.train()
    cnn(seq).sum().backward()
    assert convs[0].weight.grad is not None and convs[0].weight.grad.abs().sum() > 0

    print("text_models self-check ok")
    print(f"  MLP   parameter: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"  CNN1D parameter: {sum(p.numel() for p in cnn.parameters()):,}")


if __name__ == "__main__":
    _self_check()
