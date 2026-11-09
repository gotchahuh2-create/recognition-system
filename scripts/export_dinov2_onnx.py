"""Export the DINOv2 descriptor to ONNX (run once, on the development machine).

    pip install torch transformers onnx            # dev machine only
    python scripts/get_dinov2.py
    python scripts/export_dinov2_onnx.py --model models/hf/dinov2-with-registers-small \
        --pooling cls --size 224 --verify-images catalog_images/
    # optional INT8 weights (re-embed the whole gallery afterwards!)
    python scripts/export_dinov2_onnx.py ... --int8

Steps:
  1. load the backbone in Transformers (local files only);
  2. wrap it (``vision.clicks.torch_model.Descriptor``) so the graph outputs the
     final normalised descriptor (CLS, or CLS + GeM), so the app never
     has to deal with tokens;
  3. ``torch.onnx.export`` with opset 17, dynamic batch axis, fixed spatial
     size; ``onnx.checker``; onnx-simplifier if installed;
  4. verify: cosine between PyTorch and ONNX Runtime outputs >= 0.9999 on
     50 images (your catalog images, or random noise if none are given).

Writes ``models/dinov2s_reg_<size>[_gem][_int8].onnx`` plus a ``.json`` sidecar
(model_version, pooling, size) that the runtime and the gallery check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.clicks.embedder import model_version  # noqa: E402
from vision.clicks.preprocess import PrepConfig, prepare_batch  # noqa: E402


def _images(folder: Path | None, n: int) -> list[np.ndarray]:
    if folder is None:
        return []
    from vision.clicks.build import scan_images
    from vision.loader import load_photo
    paths = [p for ps in scan_images(folder).values() for p in ps]
    if not paths:
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in
                       (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    return [load_photo(p).image for p in paths[:n]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/hf/dinov2-with-registers-small")
    ap.add_argument("--pooling", default="cls", choices=["cls", "cls_gem", "cls_mean"])
    ap.add_argument("--size", type=int, default=224, help="224 / 336 / 448 (multiple of 14)")
    ap.add_argument("--clahe", action="store_true", help="record CLAHE in the preprocessing")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify-images", type=Path, default=None)
    ap.add_argument("--verify-n", type=int, default=50)
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--int8", action="store_true",
                    help="also write a dynamically quantised INT8 copy")
    a = ap.parse_args(argv)

    import torch
    from vision.clicks.torch_model import Descriptor, load_backbone

    if a.size % 14:
        ap.error("--size must be a multiple of 14")
    net, model_id = load_backbone(a.model, local_files_only=True)
    desc = Descriptor(net, a.pooling).eval()
    suffix = {"cls": "", "cls_gem": "_gem", "cls_mean": "_mean"}[a.pooling]
    name = "dinov2b_reg" if "base" in model_id else "dinov2s_reg" if "small" in model_id \
        else Path(str(model_id)).name.replace("-", "_")
    out = a.out or Path("models") / f"{name}_{a.size}{suffix}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, a.size, a.size)
    t0 = time.perf_counter()
    kw = dict(input_names=["pixel_values"], output_names=["embedding"],
              dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
              opset_version=a.opset, do_constant_folding=True)
    try:  # newer torch uses the dynamo exporter by default; we want TorchScript
        torch.onnx.export(desc, (dummy,), str(out), dynamo=False, **kw)
    except TypeError:   # torch < 2.5 has no ``dynamo`` argument
        torch.onnx.export(desc, (dummy,), str(out), **kw)
    print(f"exported {out} in {time.perf_counter() - t0:.1f} s")

    import onnx
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    try:
        from onnxsim import simplify  # type: ignore
        m2, ok = simplify(m)
        if ok:
            onnx.save(m2, str(out))
            print("onnx-simplifier: ok")
    except ImportError:
        print("onnx-simplifier not installed (optional)")

    # ---- Check that PyTorch and ONNX Runtime give the same output, using the app's own
    # preprocessing ----
    import onnxruntime as ort
    prep = PrepConfig(size=a.size, clahe=a.clahe)
    imgs = _images(a.verify_images, a.verify_n)
    if imgs:
        x = prepare_batch(imgs, prep)
    else:
        print("no --verify-images: verifying on random inputs")
        x = np.random.default_rng(0).normal(0, 1, (a.verify_n, 3, a.size, a.size)).astype(np.float32)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    cos = []
    for i in range(0, len(x), 8):
        xb = x[i:i + 8]
        with torch.inference_mode():
            ref = desc(torch.from_numpy(xb)).numpy()
        got = sess.run(None, {"pixel_values": xb})[0]
        cos += list((ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1)))
    cmin = float(np.min(cos))
    print(f"PyTorch vs ONNX cosine on {len(cos)} inputs: min {cmin:.6f}, mean {np.mean(cos):.6f}")
    if cmin < a.min_cos:
        print(f"FAIL: min cosine {cmin:.6f} < {a.min_cos}", file=sys.stderr)
        return 2

    def sidecar(path: Path, quant: str):
        path.with_suffix(".json").write_text(json.dumps({
            "model_version": model_version(model_id, a.pooling, a.size, quant),
            "model_id": str(model_id), "pooling": a.pooling, "size": a.size,
            "clahe": a.clahe, "opset": a.opset, "quant": quant,
            "verify_min_cos": round(cmin, 6), "exported": time.strftime("%Y-%m-%d")},
            indent=1), encoding="utf-8")

    sidecar(out, "fp32")
    print(f"model_version: {model_version(model_id, a.pooling, a.size)}")

    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = out.with_name(out.stem + "_int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
        sidecar(q, "int8")
        print(f"INT8 copy: {q} — rebuild the gallery with it and re-check hard pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
