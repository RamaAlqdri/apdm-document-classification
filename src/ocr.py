"""Multi-engine OCR for Tahap 10: same images, several engines, one output format.

Every engine writes one JSON per document:

    {"id": ..., "engine": ..., "seconds": float, "size": [w, h],
     "text": str,                                   # in reading order
     "boxes": [{"text": str, "box": [[x, y] x4], "conf": float | None}, ...]}

The boxes are kept for Tahap 14 (text as objects), so nothing needs re-OCR later.

Run one engine per process, so a crash or a CUDA/DLL clash in one engine cannot
take the others (or the notebook kernel) down with it:

    python -m src.ocr --engine easyocr --manifest data/processed/manifest.csv \
        --out data/interim/ocr --gpu

This module imports only the standard library at the top: PaddleOCR runs from a
separate venv that has no torch, pandas or numpy pin. Engine libraries are
imported inside their factory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

# Keras-OCR is not here: it needs TensorFlow with the Keras 2 API, which TF only
# offers below 2.16, and TF supports Python 3.12 only from 2.16. Its imgaug 0.4.0
# dependency also still uses np.bool. See vault/30-eksperimen/13-ocr-multi-mesin.md.
ENGINES = ("tesseract", "easyocr", "paddleocr")
TESSERACT_CONFIG = "--oem 1 --psm 3"   # QS-OCR's configuration, kept for comparability

Item = dict          # {"text": str, "box": [[x, y] x4], "conf": float | None}
Engine = Callable[[Path], tuple[list[Item], str]]


# --------------------------------------------------------------------------- #
# Reading order

def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def reading_order(items: list[Item], line_tol: float = 0.5) -> str:
    """Join box-level detections into text: top-to-bottom lines, left-to-right words.

    Box engines (EasyOCR, PaddleOCR) return detections in no guaranteed order. A box
    joins the current line when its vertical centre is within `line_tol` x the median
    box height of that line's running centre.

    ponytail: single-column heuristic. Two-column pages interleave their columns line
    by line; Tobacco3482 is mostly single-column, and Tahap 14 has the boxes if a
    layout-aware order is ever needed.
    """
    geo = []
    for it in items:
        if not it["text"].strip():
            continue
        xs = [p[0] for p in it["box"]]
        ys = [p[1] for p in it["box"]]
        geo.append((min(xs), (min(ys) + max(ys)) / 2, max(ys) - min(ys), it["text"]))
    if not geo:
        return ""
    tol = line_tol * max(statistics.median(g[2] for g in geo), 1.0)

    lines: list[list[tuple]] = []
    for g in sorted(geo, key=lambda g: g[1]):
        if lines:
            centre = statistics.fmean(x[1] for x in lines[-1])
            if abs(g[1] - centre) <= tol:
                lines[-1].append(g)
                continue
        lines.append([g])
    return "\n".join(" ".join(g[3] for g in sorted(line)) for line in lines)


def tesseract_items(data: dict) -> tuple[list[Item], str]:
    """Words and text from pytesseract.image_to_data(output_type=DICT).

    Keeps Tesseract's own block/paragraph/line order rather than re-sorting, so the
    text matches what `image_to_string` (and therefore QS-OCR) would produce.
    """
    items, lines, current, key_prev = [], [], [], None
    for i, word in enumerate(data["text"]):
        if not str(word).strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key != key_prev and current:
            lines.append(" ".join(current))
            current = []
        key_prev = key
        current.append(str(word))
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        conf = float(data["conf"][i])
        items.append({"text": str(word), "box": _rect(x, y, x + w, y + h),
                      "conf": conf / 100 if conf >= 0 else None})
    if current:
        lines.append(" ".join(current))
    return items, "\n".join(lines)


# --------------------------------------------------------------------------- #
# Engines. Each factory loads its model once and returns path -> (items, text).

# PaddleOCR's default models change between releases (3.7 silently moved to
# PP-OCRv6_medium, ~3x slower on CPU than the v5 mobile pair), so they are named.
PADDLE_MODELS = {
    "mobile": ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"),
    "default": (None, None),   # whatever the installed paddleocr picks
}


def make_engine(name: str, gpu: bool = False, tesseract_cmd: str | None = None,
                device: str | None = None, paddle_models: str = "mobile") -> Engine:
    if name == "tesseract":
        import pytesseract
        from PIL import Image
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        def run(path: Path):
            with Image.open(path) as im:
                data = pytesseract.image_to_data(
                    im, lang="eng", config=TESSERACT_CONFIG,
                    output_type=pytesseract.Output.DICT)
            return tesseract_items(data)
        return run

    if name == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

        def run(path: Path):
            items = [{"text": t, "box": [[float(x), float(y)] for x, y in box],
                      "conf": float(c)}
                     for box, t, c in reader.readtext(str(path), detail=1)]
            return items, reading_order(items)
        return run

    if name == "paddleocr":
        from paddleocr import PaddleOCR
        det, rec = PADDLE_MODELS[paddle_models]
        named = ({} if det is None else
                 {"text_detection_model_name": det, "text_recognition_model_name": rec})
        # Page-level orientation/unwarping off: Tobacco3482 pages are upright scans,
        # and Tesseract's --psm 3 does no orientation detection either.
        ocr = PaddleOCR(lang=None if named else "en",
                        device=device or ("gpu" if gpu else "cpu"),
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False, **named)

        def run(path: Path):
            res = ocr.predict(str(path))[0]
            items = [{"text": t, "box": [[float(x), float(y)] for x, y in poly],
                      "conf": float(s)}
                     for t, s, poly in zip(res["rec_texts"], res["rec_scores"],
                                           res["rec_polys"])]
            return items, reading_order(items)
        return run

    raise ValueError(f"mesin tidak dikenal: {name!r}, pilih dari {ENGINES}")


# --------------------------------------------------------------------------- #
# Batch runner: resumable, atomic, errors logged rather than written

def _image_size(path: Path) -> list[int] | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return list(im.size)
    except Exception:
        return None


def engine_meta(name: str, **opts) -> dict:
    """Library versions and options, so a result can be traced to what produced it."""
    from importlib import metadata
    pkgs = {"tesseract": ["pytesseract"], "easyocr": ["easyocr", "torch"],
            "paddleocr": ["paddleocr", "paddlepaddle", "paddlepaddle-gpu"]}[name]
    meta = {"engine": name, "python": sys.version.split()[0], "opsi": opts}
    for p in pkgs:
        try:
            meta[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    if name == "tesseract":
        import pytesseract
        if opts.get("tesseract_cmd"):
            pytesseract.pytesseract.tesseract_cmd = opts["tesseract_cmd"]
        meta["tesseract_binary"] = str(pytesseract.get_tesseract_version())
        meta["config"] = TESSERACT_CONFIG
    if name == "paddleocr":
        meta["models"] = PADDLE_MODELS[opts.get("paddle_models", "mobile")]
    return meta


def run_engine(engine: Engine, name: str, docs: Iterable[tuple[str, Path]],
               out_dir: Path, every: int = 50, meta: dict | None = None) -> dict:
    """OCR every (id, image_path) not already done. Safe to interrupt and rerun.

    A document that raises is NOT written, only appended to _errors.txt, so a rerun
    retries it instead of silently keeping an empty result.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is not None:
        (out_dir / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    docs = list(docs)
    todo = [(i, p) for i, p in docs if not (out_dir / f"{i}.json").exists()]
    done, errors, spent = 0, 0, 0.0
    print(f"[{name}] {len(docs)} dokumen, {len(docs) - len(todo)} sudah ada, "
          f"{len(todo)} dikerjakan", flush=True)
    for k, (doc_id, path) in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            items, text = engine(Path(path))
        except Exception as e:  # one bad page must not stop a multi-hour run
            errors += 1
            with open(out_dir / "_errors.txt", "a", encoding="utf-8") as f:
                f.write(f"{doc_id}\t{type(e).__name__}: {e}\n")
            continue
        sec = time.perf_counter() - t0
        spent += sec
        record = {"id": doc_id, "engine": name, "seconds": round(sec, 3),
                  "size": _image_size(Path(path)), "text": text, "boxes": items}
        tmp = out_dir / f"{doc_id}.json.part"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_dir / f"{doc_id}.json")
        done += 1
        if k % every == 0 or k == len(todo):
            rate = spent / max(done, 1)
            print(f"[{name}] {k}/{len(todo)} | {rate:.2f} dtk/halaman | "
                  f"sisa ± {rate * (len(todo) - k) / 60:.0f} menit | galat {errors}",
                  flush=True)
    return {"engine": name, "dikerjakan": done, "galat": errors,
            "detik_per_halaman": round(spent / max(done, 1), 3)}


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["id"], Path(r["path_citra"])) for r in csv.DictReader(f)]


def load_texts(out_dir: Path, name: str) -> dict[str, str]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))["text"]
            for p in sorted((Path(out_dir) / name).glob("*.json"))
            if not p.name.startswith("_")}      # _meta.json is not a page


# --------------------------------------------------------------------------- #
# Quality proxies. Tobacco3482 has no ground-truth transcription, so CER is not
# computable; these describe the output, they do not score it.

def words(text: str, min_len: int = 1) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= min_len]


def oov_share(text: str, known: Callable[[str], bool], min_len: int = 4) -> float | None:
    """Share of alphabetic words (len >= min_len) not in an external word list.

    None for a page with no such words: "no text" is not "0% errors".
    """
    ws = words(text, min_len)
    return None if not ws else sum(not known(w) for w in ws) / len(ws)


def jaccard(a: str, b: str, min_len: int = 3) -> float | None:
    """Overlap of the two word *sets*. Order-free on purpose: box engines and
    Tesseract disagree on reading order even when they read the same words."""
    sa, sb = set(words(a, min_len)), set(words(b, min_len))
    return None if not (sa or sb) else len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    import tempfile

    # reading order: two lines given shuffled, with jitter inside a line
    def box(x, y, w=40, h=20):
        return _rect(x, y, x + w, y + h)
    items = [{"text": "world", "box": box(60, 12)}, {"text": "line", "box": box(60, 60)},
             {"text": "hello", "box": box(5, 10)}, {"text": "second", "box": box(5, 58)},
             {"text": " ", "box": box(200, 200)}]
    assert reading_order(items) == "hello world\nsecond line", reading_order(items)
    assert reading_order([]) == ""

    # tesseract: native line grouping, non-word rows skipped, conf -1 -> None
    data = {"text": ["", "Dear", "Sir", "", "Thanks"], "block_num": [1, 1, 1, 1, 2],
            "par_num": [0, 1, 1, 1, 1], "line_num": [0, 1, 1, 1, 1],
            "left": [0, 10, 60, 0, 10], "top": [0, 10, 10, 0, 50],
            "width": [0, 40, 30, 0, 50], "height": [0, 12, 12, 0, 12],
            "conf": [-1, 96.0, 91.5, -1, -1]}
    its, text = tesseract_items(data)
    assert text == "Dear Sir\nThanks", text
    assert its[0]["conf"] == 0.96 and its[2]["conf"] is None, its
    assert its[1]["box"] == [[60, 10], [90, 10], [90, 22], [60, 22]], its[1]

    # proxies
    known = {"dear", "thanks", "tobacco"}.__contains__
    # "4" and "the" are below min_len, "sirr" is length 4 and unknown
    assert oov_share("Dear Sirr, thanks 4 the tobacco", known) == 0.25
    assert oov_share("Dear Siirr tobacco", known) == 1 / 3
    assert oov_share("12 34 ok", known) is None
    assert jaccard("the cat sat", "sat the cat") == 1.0
    assert jaccard("the cat", "the dog") == 1 / 3
    assert jaccard("", "") is None

    # runner: resumable, atomic, a failing page is logged and retried next time
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        calls = []

        def fake(path: Path):
            calls.append(path.name)
            if path.name == "bad.jpg" and calls.count("bad.jpg") == 1:
                raise RuntimeError("halaman rusak")
            return [{"text": path.stem, "box": box(0, 0), "conf": 1.0}], path.stem

        docs = [("a", tmp / "a.jpg"), ("bad", tmp / "bad.jpg")]
        r1 = run_engine(fake, "palsu", docs, tmp / "out", every=10, meta={"x": 1})
        assert r1["dikerjakan"] == 1 and r1["galat"] == 1, r1
        assert not (tmp / "out" / "palsu" / "bad.json").exists()
        assert "bad\tRuntimeError" in (tmp / "out" / "palsu" / "_errors.txt").read_text()
        r2 = run_engine(fake, "palsu", docs, tmp / "out", every=10)
        assert r2["dikerjakan"] == 1 and calls == ["a.jpg", "bad.jpg", "bad.jpg"], calls
        assert load_texts(tmp / "out", "palsu") == {"a": "a", "bad": "bad"}
        assert not list((tmp / "out" / "palsu").glob("*.part"))

        man = tmp / "manifest.csv"
        man.write_text("id,kelas,path_citra\nx1,Memo,/d/x1.jpg\n", encoding="utf-8")
        assert read_manifest(man) == [("x1", Path("/d/x1.jpg"))]

    try:
        make_engine("tidak-ada")
    except ValueError:
        pass
    else:
        raise AssertionError("mesin tak dikenal harus ditolak")
    print("ocr self-check ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=ENGINES)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, help="hanya N dokumen pertama (uji coba)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device", help="paddleocr: 'cpu' atau 'gpu:0'")
    ap.add_argument("--tesseract-cmd", help="path tesseract.exe bila tidak di PATH")
    ap.add_argument("--paddle-models", choices=sorted(PADDLE_MODELS), default="mobile")
    args = ap.parse_args(argv)
    if not args.engine:
        _self_check()
        return 0
    docs = read_manifest(args.manifest)[: args.limit]
    opts = {"gpu": args.gpu, "tesseract_cmd": args.tesseract_cmd, "device": args.device,
            "paddle_models": args.paddle_models}
    engine = make_engine(args.engine, **opts)
    summary = run_engine(engine, args.engine, docs, args.out,
                         meta=engine_meta(args.engine, **opts))
    print("RINGKASAN " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
