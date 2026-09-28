"""Small helpers shared by the notebooks."""

from pathlib import Path

from .config import CONFIG


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure into reports/figures/ and return its path."""
    CONFIG["figures"].mkdir(parents=True, exist_ok=True)
    path = CONFIG["figures"] / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def relpath(path) -> str:
    """Path relative to the project root, for pasting into vault notes."""
    return str(Path(path).resolve().relative_to(CONFIG["root"]))
