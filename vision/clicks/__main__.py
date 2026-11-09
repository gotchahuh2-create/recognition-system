"""CLI for the Click gallery and recogniser.

    # 1. build the gallery once (catalog images -> embeddings + index)
    python -m vision.clicks build --images catalog_images/ --model models/dinov2s_reg_224.onnx \
        --out gallery/ --augment 12 [--real photos/train/] [--backgrounds bg/]

    # add a new Click without touching the model
    python -m vision.clicks add --gallery gallery/ --model models/dinov2s_reg_224.onnx \
        --pid MIKROE-5678 --name "Thermo 30 Click" img1.jpg img2.jpg

    # recognise (one Click per photo, or --box x,y,w,h per Click)
    python -m vision.clicks query --gallery gallery/ --model models/dinov2s_reg_224.onnx photo.jpg

    python -m vision.clicks info --gallery gallery/

``--model`` takes an exported .onnx (runtime) or, on the dev machine with
torch + transformers, a local DINOv2 folder / HF id (with --pooling/--size).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..catalog import Catalog
from ..loader import iter_photos
from .build import add_images, build_gallery
from .embedder import get_embedder
from .gallery import Gallery
from .recognizer import ClickConfig, ClickRecognizer


def _embedder(a):
    return get_embedder(a.model, pooling=a.pooling, size=a.size, clahe=a.clahe or None,
                        providers=a.providers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision.clicks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", default="models/dinov2s_reg_224.onnx")
        p.add_argument("--pooling", default="cls", choices=["cls", "cls_gem", "cls_mean"],
                       help="torch models only (ONNX files carry their pooling)")
        p.add_argument("--size", type=int, default=224, help="torch models only")
        p.add_argument("--clahe", action="store_true")
        p.add_argument("--providers", default="cpu", help="cpu | auto | <ORT provider>")

    b = sub.add_parser("build", help="embed catalog (+ real) images into a new gallery")
    model_args(b)
    b.add_argument("--images", required=True)
    b.add_argument("--real", default=None, help="real photos from the TRAIN split only")
    b.add_argument("--out", default="gallery")
    b.add_argument("--augment", type=int, default=12, help="augmented copies per catalog image")
    b.add_argument("--backgrounds", nargs="*", default=[])
    b.add_argument("--catalog", default=None, help="data folder / products.db for Click names")
    b.add_argument("--seed", type=int, default=0)

    d = sub.add_parser("add", help="append one Click's images to an existing gallery")
    model_args(d)
    d.add_argument("--gallery", default="gallery")
    d.add_argument("--pid", required=True)
    d.add_argument("--name", default=None)
    d.add_argument("--source", default="catalog", choices=["catalog", "real"])
    d.add_argument("--augment", type=int, default=12)
    d.add_argument("images", nargs="+")

    q = sub.add_parser("query", help="recognise Clicks in photos")
    model_args(q)
    q.add_argument("--gallery", default="gallery")
    q.add_argument("--box", action="append", default=None,
                   help="x,y,w,h of one Click (repeat per Click); default: whole photo")
    q.add_argument("--backend", default="auto", help="OCR backend for the re-rank")
    q.add_argument("--no-ocr", action="store_true")
    q.add_argument("--no-tta", action="store_true")
    q.add_argument("--verify", action="store_true", help="local feature check on hard pairs")
    q.add_argument("--catalog", default=None)
    q.add_argument("inputs", nargs="+")

    i = sub.add_parser("info", help="gallery summary")
    i.add_argument("--gallery", default="gallery")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        g = Gallery.load(a.gallery)
        src = {}
        for m in g.meta:
            src[m["source"]] = src.get(m["source"], 0) + 1
        print(json.dumps({"vectors": len(g), "pids": len(g.pids), "dim": g.dim,
                          "model_version": g.model_version,
                          "preprocess_version": g.preprocess_version,
                          "sources": src, "index": g.index.backend}, indent=1))
        return 0

    emb = _embedder(a)
    if a.cmd == "build":
        names = Catalog.load(a.catalog).click_names()
        g = build_gallery(emb, a.images, a.out, a.augment, a.real, names,
                          a.backgrounds, a.seed)
        print(f"gallery: {len(g)} vectors, {len(g.pids)} Clicks -> {a.out}")
        return 0
    if a.cmd == "add":
        g = Gallery.load(a.gallery, emb.model_version, emb.preprocess_version)
        n = add_images(g, emb, a.pid.upper(), [Path(p) for p in a.images], a.source,
                       a.augment, name=a.name)
        g.save(a.gallery)
        print(f"added {n} vectors for {a.pid}; gallery now {len(g)} vectors")
        return 0

    cfg = ClickConfig.from_file(Path(a.gallery) / "thresholds.json",
                                tta=False if a.no_tta else None,
                                ocr=False if a.no_ocr else None,
                                verify=True if a.verify else None)
    rec = ClickRecognizer(emb, a.gallery, None if a.no_ocr else a.backend,
                          Catalog.load(a.catalog), cfg)
    boxes = [[float(v) for v in b.split(",")] for b in a.box] if a.box else [None]
    photos = (ph for inp in a.inputs for ph in iter_photos(inp))
    for photo in photos:
        for box in boxes:
            r = rec.recognise(photo.image, box)
            top = ", ".join(f"{c.name} {c.score:.2f}" for c in r.recognition.candidates)
            print(f"{photo.name}{' ' + str(box) if box else ''}: {r.outcome.upper()} "
                  f"| {r.recognition.reason} | top: {top} | {r.timings_ms['total']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
