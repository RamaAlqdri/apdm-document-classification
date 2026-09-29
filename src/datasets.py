"""Torch datasets over the precomputed features and the raw images.

Kept out of src/data.py on purpose: that module's audit self-check must keep
running on a machine without torch installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG


def label_indices(labels: Sequence[str]) -> np.ndarray:
    """Class name -> integer, by position in CONFIG['classes'].

    Fails loudly on an unknown name rather than inventing a new index, because a
    silently added class would shift every label after it.
    """
    lookup = {c: i for i, c in enumerate(CONFIG["classes"])}
    unknown = sorted(set(labels) - set(lookup))
    if unknown:
        raise ValueError(f"kelas tidak dikenal: {unknown}. Perbaiki CONFIG['classes'] "
                         f"atau CLASS_ALIASES, jangan tambah indeks baru diam-diam.")
    return np.array([lookup[l] for l in labels], dtype=np.int64)


class _MemmapBacked:
    """Shared open/close for a dataset served from a memory-mapped .npy.

    Windows refuses to delete or overwrite a file while a mapping on it is open.
    CPython would drop the mapping when the dataset is garbage-collected, but a
    notebook keeps the DataLoader — and through it the dataset — alive well past
    the loop body, so the release has to be explicit. Without it, deleting a
    throwaway .npy fails with WinError 32.
    """

    def _open(self, path: Path) -> None:
        self.path = Path(path)
        self.arr = np.load(self.path, mmap_mode="r")

    def close(self) -> None:
        """Release the mapping. Indexing the dataset afterwards raises ValueError."""
        arr, self.arr = getattr(self, "arr", None), None
        mmap = getattr(arr, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SequenceDataset(_MemmapBacked, Dataset):
    """500x300 word-vector sequences, served as (300, 500) for Conv1d.

    The .npy stays memory-mapped: the file is ~1GB and only a batch is needed at a
    time. float16 on disk, float32 in the batch.
    """

    def __init__(self, path: Path, y: np.ndarray, indices: Sequence[int]):
        self._open(path)
        self.y = y
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.indices[i]
        # np.array, not asarray: asarray on a memmap of matching dtype hands back a
        # read-only view, and torch.from_numpy then warns about non-writable tensors
        seq = np.array(self.arr[row], dtype=np.float32)   # (500, 300)
        return torch.from_numpy(seq.T).contiguous(), int(self.y[row])  # (300, 500)


class VectorDataset(_MemmapBacked, Dataset):
    """One precomputed vector per document — the SIF representation for the MLP."""

    def __init__(self, path: Path, y: np.ndarray, indices: Sequence[int]):
        self._open(path)
        self.y = y
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.indices[i]
        vec = np.array(self.arr[row], dtype=np.float32)
        return torch.from_numpy(vec), int(self.y[row])


class ImageDataset(Dataset):
    """JPG documents, loaded and transformed on the fly."""

    def __init__(self, paths: Sequence[str], y: np.ndarray, indices: Sequence[int],
                 transform):
        self.paths = list(paths)
        self.y = y
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.indices[i]
        with Image.open(self.paths[row]) as im:
            return self.transform(im.convert("L")), int(self.y[row])


class PairDataset(Dataset):
    """Image and text for the same document, for the fusion model in Tahap 6."""

    def __init__(self, image_ds: Dataset, text_ds: Dataset):
        if len(image_ds) != len(text_ds):
            raise ValueError("kedua dataset harus punya jumlah dan urutan indeks sama")
        self.image_ds, self.text_ds = image_ds, text_ds

    def __len__(self) -> int:
        return len(self.image_ds)

    def __getitem__(self, i: int):
        img, y_img = self.image_ds[i]
        txt, y_txt = self.text_ds[i]
        assert y_img == y_txt, f"label tidak cocok di posisi {i}: {y_img} != {y_txt}"
        return (img, txt), y_img


def make_loaders(build_dataset, split: dict, batch_size: int | None = None,
                 num_workers: int | None = None) -> dict[str, DataLoader]:
    """One DataLoader per split part. `build_dataset(indices)` returns a Dataset.

    The TRAIN loader yields micro-batches of `CONFIG["micro_batch_size"]` when that
    is set, and train_model accumulates gradients back up to `batch_size`. Eval
    loaders use the micro size too — they only need to fit in memory, and batching
    does not change their result.
    """
    batch_size = CONFIG["batch_size"] if batch_size is None else batch_size
    num_workers = CONFIG["num_workers"] if num_workers is None else num_workers
    micro = CONFIG.get("micro_batch_size") or batch_size
    return {
        part: DataLoader(
            build_dataset(split[part]),
            batch_size=micro,
            shuffle=(part == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(part == "train"),  # BatchNorm1d needs >1 sample per batch
        )
        for part in ("train", "val", "test")
    }


def close_datasets(*datasets) -> None:
    """Close every memory-mapped dataset reachable from those given.

    PairDataset and DataLoader.dataset are accepted directly: the walk follows the
    wrapped datasets so the caller does not have to remember which of a pair holds
    the .npy.
    """
    for ds in datasets:
        if ds is None:
            continue
        close = getattr(ds, "close", None)
        if callable(close):
            close()
        for attr in ("dataset", "image_ds", "text_ds"):
            child = getattr(ds, attr, None)
            if child is not None and child is not ds:
                close_datasets(child)


def _drop_leftover(path: Path) -> None:
    """Delete a leftover from an interrupted run, naming the fix if it is locked."""
    try:
        path.unlink(missing_ok=True)
    except PermissionError as exc:        # Windows only; POSIX unlinks regardless
        raise PermissionError(
            f"{path.name} dari run sebelumnya masih dipetakan oleh proses lain "
            f"(biasanya kernel Jupyter yang belum di-restart). Restart kernel, "
            f"lalu jalankan sel ini lagi."
        ) from exc


@contextmanager
def scratch_npy(path: Path) -> Iterator[Path]:
    """Yield `path` for a throwaway .npy and delete it on the way out.

    Deletion also runs when the body raises, so an ablation level that fails
    halfway does not leave a multi-hundred-MB file behind.

    A file still mapped at exit is reported, but never at the cost of the body's
    own exception. When the body failed, that error is the one worth reading — an
    undefined name, a missing checkpoint — and raising a cleanup complaint over it
    points the traceback at this function instead of at the real line.
    """
    path = Path(path)
    _drop_leftover(path)
    body_failed = False
    try:
        yield path
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            path.unlink(missing_ok=True)
        except PermissionError as exc:
            pesan = (f"{path.name} masih dipetakan ke memori: panggil "
                     f"close_datasets(...) atas setiap dataset di atas file ini "
                     f"sebelum keluar dari scratch_npy()")
            if body_failed:
                print(f"peringatan: {pesan} (file dibiarkan; error sebenarnya "
                      f"ada di traceback di bawah)")
            else:
                raise PermissionError(pesan) from exc


def _self_check() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = 20
        labels = [CONFIG["classes"][i % 10] for i in range(n)]
        y = label_indices(labels)
        assert y.tolist() == [i % 10 for i in range(n)], y[:12]

        # a sequence whose first axis is time, so the transpose is observable
        seq = np.zeros((n, 5, 3), dtype=np.float16)
        seq[:, 0, :] = 1.0          # first time step only
        np.save(tmp / "seq.npy", seq)
        ds = SequenceDataset(tmp / "seq.npy", y, range(n))
        x, lab = ds[3]
        assert x.shape == (3, 5), x.shape                 # (dim, length)
        assert x.dtype == torch.float32, x.dtype
        assert torch.all(x[:, 0] == 1) and torch.all(x[:, 1:] == 0), \
            "transpose salah: sumbu waktu dan dimensi tertukar"
        assert lab == 3

        np.save(tmp / "sif.npy", np.arange(n * 4, dtype=np.float32).reshape(n, 4))
        v, lab = VectorDataset(tmp / "sif.npy", y, range(n))[2]
        assert v.tolist() == [8.0, 9.0, 10.0, 11.0], v.tolist()
        assert lab == 2

        # loaders: train shuffles and drops the tail, eval does neither
        split = {"train": list(range(12)), "val": list(range(12, 16)),
                 "test": list(range(16, 20))}
        loaders = make_loaders(
            lambda idx: VectorDataset(tmp / "sif.npy", y, idx), split, batch_size=5
        )
        assert len(loaders["train"].dataset) == 12
        assert loaders["train"].drop_last and not loaders["test"].drop_last
        # the effective batch is CONFIG["micro_batch_size"] when that is set, and it
        # is derived from the GPU, so the expected count is computed rather than
        # hardcoded — a literal here fails on any machine with a different micro size
        micro = loaders["train"].batch_size
        assert sum(len(b[1]) for b in loaders["train"]) == (12 // micro) * micro, \
            "drop_last tidak jalan"
        assert sum(len(b[1]) for b in loaders["test"]) == 4

        # pairing refuses mismatched label order
        a = VectorDataset(tmp / "sif.npy", y, range(4))
        b = VectorDataset(tmp / "sif.npy", y, list(range(4))[::-1])
        try:
            PairDataset(a, b)[0]
        except AssertionError:
            pass
        else:
            raise AssertionError("PairDataset tidak mendeteksi label tertukar")

        # a live memory map must not block deletion once closed: this is the
        # exact shape of the typo-injection loop in Tahap 7
        with scratch_npy(tmp / "throwaway.npy") as scratch:
            np.save(scratch, np.zeros((4, 3), dtype=np.float32))
            loader = DataLoader(
                PairDataset(VectorDataset(scratch, y, range(4)),
                            VectorDataset(scratch, y, range(4))),
                batch_size=2,
            )
            assert sum(len(b[1]) for b in loader) == 4
            close_datasets(loader.dataset)
        assert not (tmp / "throwaway.npy").exists(), "scratch_npy tidak menghapus file"

        # and the guard must fire when the caller forgets to close
        try:
            with scratch_npy(tmp / "locked.npy") as scratch:
                np.save(scratch, np.zeros((4, 3), dtype=np.float32))
                held = VectorDataset(scratch, y, range(4))
                held[0]                       # force the mapping to be touched
        except PermissionError as exc:
            assert "close_datasets" in str(exc), exc
            held.close()
            (tmp / "locked.npy").unlink(missing_ok=True)
        # POSIX unlinks a mapped file without complaint, so no failure is expected
        # there; on Windows the branch above is the one that runs.

        # a cleanup failure must never replace the body's own exception: that is
        # what turned an undefined name in the Tahap 7 loop into a misleading
        # complaint about memory maps
        try:
            with scratch_npy(tmp / "masked.npy") as scratch:
                np.save(scratch, np.zeros((4, 3), dtype=np.float32))
                leaked = VectorDataset(scratch, y, range(4))
                leaked[0]
                raise RuntimeError("kesalahan asli")
        except RuntimeError as exc:
            assert str(exc) == "kesalahan asli", f"error asli tertukar: {exc}"
        except PermissionError as exc:
            raise AssertionError(f"scratch_npy menelan error asli: {exc}") from exc
        finally:
            leaked.close()
            (tmp / "masked.npy").unlink(missing_ok=True)

        # unknown class must fail loudly
        try:
            label_indices(["ADVE"])
        except ValueError:
            pass
        else:
            raise AssertionError("label_indices menerima kelas tak dikenal")

        # the tempdir cannot be removed on Windows while these are still mapped
        close_datasets(ds, a, b, *(l.dataset for l in loaders.values()))

    print("datasets self-check ok")


if __name__ == "__main__":
    _self_check()
