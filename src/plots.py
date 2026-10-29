"""Figures for the pipeline notebook. Each function returns a matplotlib figure; the
notebook decides whether to show and save it (src.utils.save_fig)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BLUE, GREEN, RED = "#4C78A8", "#54A24B", "#E45756"


def _plt():
    import matplotlib.pyplot as plt
    return plt


def bar_counts(counts: pd.Series, title: str, ylabel: str = "jumlah dokumen"):
    fig, ax = _plt().subplots(figsize=(8, 3.5))
    ax.bar(counts.index.astype(str), counts.values, color=BLUE)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    return fig


def boxes_by_class(df: pd.DataFrame, cols: list[str], classes: list[str],
                   by: str = "kelas", log_cols: tuple = (), title: str = ""):
    """One boxplot panel per column, one box per class."""
    fig, axes = _plt().subplots(1, len(cols), figsize=(4.5 * len(cols), 4), squeeze=False)
    for ax, col in zip(axes[0], cols):
        ax.boxplot([df.loc[df[by] == k, col].dropna() for k in classes], showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1), classes, rotation=60, fontsize=8)
        ax.set_title(col)
        ax.grid(alpha=0.3)
        if col in log_cols:
            ax.set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def scatter_2d(panels: dict[str, np.ndarray], y: np.ndarray, classes: list[str],
               ncols: int | None = None):
    """2-D embeddings (e.g. t-SNE) in a grid of `ncols` columns, coloured by class."""
    plt = _plt()
    colors = plt.cm.tab10(np.arange(len(classes)))
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 6.5 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (title, z) in zip(axes.flat, panels.items()):
        for k, c in zip(classes, colors):
            m = y == k
            ax.scatter(z[m, 0], z[m, 1], s=5, color=c, label=k, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, ncols - 1].legend(markerscale=3, fontsize=8, loc="center left",
                              bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig


def heatmap(matrix, xlabels, ylabels, title: str, vmin=0.0, vmax=1.0,
            cmap: str = "viridis", fmt: str = "{:.2f}"):
    m = np.asarray(matrix, dtype=float)
    fig, ax = _plt().subplots(figsize=(1.0 + 0.7 * len(xlabels), 0.8 + 0.6 * len(ylabels)))
    im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    mid = (vmin + vmax) / 2
    for (r, c), v in np.ndenumerate(m):
        if not np.isnan(v):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v < mid) == (cmap == "viridis") else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def signed_bars(values: pd.Series, title: str, xlabel: str):
    """Horizontal bars, green when positive, red when negative."""
    v = values.sort_values()
    fig, ax = _plt().subplots(figsize=(8, 0.4 * len(v) + 1.2))
    ax.barh(v.index.astype(str), v.values, color=np.where(v.values > 0, GREEN, RED))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def samples(items: list[tuple[str, str, str]], n_chars: int = 400):
    """(image_path, text, title) per column: the page above, its OCR text below."""
    from PIL import Image

    plt = _plt()
    fig, axes = plt.subplots(2, len(items), figsize=(4 * len(items), 9), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 2]})
    for i, (path, text, title) in enumerate(items):
        with Image.open(path) as im:
            axes[0, i].imshow(im.convert("L"), cmap="gray")
        axes[0, i].set_title(title, fontsize=9)
        axes[1, i].text(0, 1, text[:n_chars] or "(kosong)", va="top", fontsize=6,
                        family="monospace", wrap=True)
        for r in (0, 1):
            axes[r, i].axis("off")
    fig.tight_layout()
    return fig


def grouped_bars(summary: pd.DataFrame, models: list[str], conditions: list[str],
                 value: str = "oa", title: str = ""):
    """Mean +/- std per (OCR condition, model) from experiment.summarize()."""
    s = summary.set_index(["ocr", "model"])
    x = np.arange(len(conditions))
    width = 0.8 / len(models)
    fig, ax = _plt().subplots(figsize=(2.2 * len(conditions) + 2, 4))
    for k, m in enumerate(models):
        mean = [s.loc[(c, m), f"{value}_mean"] for c in conditions]
        std = [s.loc[(c, m), f"{value}_std"] for c in conditions]
        ax.bar(x + (k - (len(models) - 1) / 2) * width, mean, width, yerr=std,
               capsize=3, label=m)
    ax.set_xticks(x, conditions)
    ax.set_ylabel(value)
    lo = min(s[f"{value}_mean"].min() - 0.05, 0.9)
    ax.set_ylim(max(0, lo), 1)
    ax.legend(fontsize=8, ncol=len(models))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def modality_use_plot(use: pd.DataFrame):
    """H3: text accuracy vs the accuracy lost when the text is blanked, per rule."""
    fig, ax = _plt().subplots(figsize=(6.5, 4.5))
    for rule, color in (("late", BLUE), ("concat", RED)):
        ax.scatter(use["oa_text"], use[f"{rule}_turun_tanpa_teks"], color=color,
                   label=rule.upper())
        for _, r in use.iterrows():
            ax.annotate(r["ocr"], (r["oa_text"], r[f"{rule}_turun_tanpa_teks"]),
                        fontsize=6, alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("OA cabang teks (TEXT)")
    ax.set_ylabel("OA turun saat teks dikosongkan")
    ax.set_title("Seberapa banyak fusi memakai teks")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _self_check() -> None:
    import matplotlib
    matplotlib.use("Agg")

    classes = ["a", "b"]
    df = pd.DataFrame({"kelas": ["a", "b"] * 5, "v": np.arange(10.0) + 1})
    figs = [
        bar_counts(pd.Series([3, 5], index=classes), "t"),
        boxes_by_class(df, ["v"], classes, log_cols=("v",)),
        scatter_2d({"x": np.random.default_rng(0).normal(size=(10, 2))},
                   df["kelas"].to_numpy(), classes),
        scatter_2d({t: np.random.default_rng(0).normal(size=(10, 2)) for t in "pqr"},
                   df["kelas"].to_numpy(), classes, ncols=2),
        heatmap([[1, 0.5], [0.5, np.nan]], classes, classes, "h"),
        signed_bars(pd.Series([0.1, -0.2], index=classes), "s", "x"),
    ]
    summary = pd.DataFrame({"ocr": ["q", "q"], "model": ["TEXT", "LATE"],
                            "oa_mean": [0.7, 0.85], "oa_std": [0.01, 0.02]})
    figs.append(grouped_bars(summary, ["TEXT", "LATE"], ["q"]))
    use = pd.DataFrame({"ocr": ["q"], "oa_text": [0.7], "late_turun_tanpa_teks": [0.02],
                        "concat_turun_tanpa_teks": [0.0]})
    figs.append(modality_use_plot(use))
    assert all(f.axes for f in figs)
    print("plots self-check ok")


if __name__ == "__main__":
    _self_check()
