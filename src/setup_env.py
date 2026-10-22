"""Environment checks for the pipeline notebook and run_all.py.

Everything that must survive a package reinstall in the running kernel (torch
version, CUDA) is asked in a fresh subprocess, never through the kernel's own
already-imported modules.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from .config import ROOT

# Modules whose __main__ is a self-check that runs without any dataset.
SELF_CHECKS = ["config", "data", "splits", "text_features", "datasets",
               "models.text_models", "models.image_model", "models.fusion", "train",
               "ocr", "eda", "experiment", "plots", "setup_env"]

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def torch_info() -> dict | None:
    """torch version and CUDA device, measured in a new process."""
    code = ("import torch,json;d={'versi':torch.__version__,'cuda':torch.cuda.is_available()};"
            "p=torch.cuda.get_device_properties(0) if d['cuda'] else None;"
            "d['nama']=p.name if p else None;"
            "d['vram_gb']=round(p.total_memory/1e9,1) if p else None;print(json.dumps(d))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None


def has_nvidia() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _lock(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)        # 600; ignored on Windows
    except OSError:
        pass


def kaggle_ready() -> bool:
    return KAGGLE_JSON.exists() or bool(os.environ.get("KAGGLE_KEY"))


def adopt_kaggle_json(search=(Path.home() / "Downloads", Path.home() / "Unduhan",
                              ROOT)) -> Path | None:
    """Copy a downloaded kaggle.json into ~/.kaggle. A copy inside the repo is then
    deleted so the token can never be committed. Returns where it came from."""
    for d in search:
        src = Path(d) / "kaggle.json"
        if src.exists():
            KAGGLE_JSON.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, KAGGLE_JSON)
            _lock(KAGGLE_JSON)
            if src.resolve().parent == ROOT.resolve():
                src.unlink()
            return src
    return None


def write_kaggle_token(username: str, key: str) -> Path:
    KAGGLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    KAGGLE_JSON.write_text(json.dumps({"username": username, "key": key}),
                           encoding="utf-8")
    _lock(KAGGLE_JSON)
    return KAGGLE_JSON


def verify_kaggle() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-c",
                        "from kagglehub.auth import whoami;print(whoami().get('username'))"],
                       capture_output=True, text=True)
    msg = r.stdout.strip() or (r.stderr.strip().splitlines() or ["?"])[-1]
    return r.returncode == 0, msg


def pin_file(path: Path, packages=("numpy", "torch", "torchvision")) -> Path:
    """pip constraints pinning what is installed now, so an OCR library cannot
    silently upgrade numpy or torch under the training code."""
    from importlib import metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p}=={metadata.version(p)}\n" for p in packages))
    return path


def run_self_checks(modules=SELF_CHECKS, skip=()) -> dict[str, str | None]:
    """{module: None if ok else last error line}. Each runs in its own process."""
    out = {}
    for m in modules:
        if m in skip:
            continue
        r = subprocess.run([sys.executable, "-m", f"src.{m}"], cwd=ROOT,
                           capture_output=True, text=True)
        out[m] = None if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1]
    return out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = pin_file(Path(tmp) / "c.txt", packages=("numpy",))
        assert p.read_text().startswith("numpy=="), p.read_text()
    assert set(SELF_CHECKS) - {"setup_env"} <= {
        str(f.relative_to(ROOT / "src").with_suffix("")).replace("/", ".")
        for f in (ROOT / "src").rglob("*.py")}, "SELF_CHECKS menyebut modul yang tidak ada"
    assert isinstance(kaggle_ready(), bool)
    print("setup_env self-check ok")
