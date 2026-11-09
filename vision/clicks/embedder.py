"""Turn an image into a normalised descriptor vector.

There are two interchangeable versions with the same interface:

* ``OnnxEmbedder``: what ships with the app. Runs an exported DINOv2 model
  with ONNX Runtime (on the CPU by default, or DirectML / CUDA if that
  version of ONNX Runtime is installed). Only needs ``onnxruntime`` and
  NumPy.
* ``TorchEmbedder``: for the development machine (``torch`` +
  ``transformers``). Used to build galleries for experiments and compare
  settings, and it's what ``scripts/export_dinov2_onnx.py`` exports.

Both use ``preprocess.prepare`` (not the Hugging Face image processor), so
a gallery built with one and searched with the other only differs by tiny
rounding differences.

``model_version`` names the weights, pooling, input size (and quantisation).
The gallery stores it and refuses to load with a different one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .preprocess import PrepConfig, prepare_batch

DEFAULT_MODEL_ID = "facebook/dinov2-with-registers-small"
POOLINGS = ("cls", "cls_gem", "cls_mean")


def l2n(v: np.ndarray, axis: int = -1) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def model_version(model_id: str, pooling: str, size: int, quant: str = "fp32") -> str:
    name = str(model_id).rstrip("/").split("/")[-1]
    return f"{name}:{pooling}:{size}:{quant}"


class Embedder:
    """``embed(list of BGR crops)`` returns float32 [N, dim], each row normalised to length 1."""

    model_version: str = "base"
    prep: PrepConfig = PrepConfig()
    dim: int = 0
    batch_size: int = 16

    @property
    def preprocess_version(self) -> str:
        return self.prep.version

    def _run(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(crops), self.batch_size):
            x = prepare_batch(crops[i:i + self.batch_size], self.prep)
            out.append(np.asarray(self._run(x), np.float32))
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return l2n(np.concatenate(out))


# ---- ONNX Runtime ----
def ort_providers(prefer: str = "cpu") -> list[str]:
    """Choose ONNX Runtime providers.

    'cpu' (the default, use this for shipping), 'auto' (DirectML/CUDA if that
    version of ONNX Runtime is installed), or a provider name.
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if prefer == "cpu":
        return ["CPUExecutionProvider"]
    if prefer == "auto":
        for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider"):
            if p in avail:
                return [p, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    return [prefer, "CPUExecutionProvider"]


def read_sidecar(model_path: Path) -> dict:
    side = Path(model_path).with_suffix(".json")
    if side.is_file():
        return json.loads(side.read_text(encoding="utf-8"))
    return {}


class OnnxEmbedder(Embedder):
    def __init__(self, path: str | Path, providers: str = "cpu", clahe: bool | None = None,
                 threads: int | None = None):
        import onnxruntime as ort
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"{self.path} not found - export it once on the dev machine: "
                "python scripts/get_dinov2.py && python scripts/export_dinov2_onnx.py")
        meta = read_sidecar(self.path)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(self.path), so, providers=ort_providers(providers))
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.sess.get_outputs()[0].name
        size = inp.shape[-1] if isinstance(inp.shape[-1], int) else int(meta.get("size", 224))
        self.prep = PrepConfig(size=int(size),
                               clahe=bool(meta.get("clahe", False) if clahe is None else clahe))
        if meta.get("model_version"):
            self.model_version = meta["model_version"]
        else:   # unknown export: tie the gallery to these exact weights
            sha = hashlib.sha256(self.path.read_bytes()).hexdigest()[:12]
            self.model_version = f"{self.path.stem}:onnx-{sha}:{size}"
        out_dim = self.sess.get_outputs()[0].shape[-1]
        self.dim = out_dim if isinstance(out_dim, int) else \
            int(self._run(np.zeros((1, 3, size, size), np.float32)).shape[-1])
        self.providers = self.sess.get_providers()

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.output_name], {self.input_name: x})[0]


# ---- PyTorch / transformers (development machine) ----
class TorchEmbedder(Embedder):
    def __init__(self, model: str | Path = DEFAULT_MODEL_ID, pooling: str = "cls",
                 size: int = 224, clahe: bool = False, device: str = "cpu",
                 gem_p: float = 3.0, local_files_only: bool = True):
        import torch
        from .torch_model import Descriptor, load_backbone
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        if size % 14:
            raise ValueError("DINOv2 input size must be a multiple of 14 (224, 336, 448)")
        self.torch = torch
        net, model_id = load_backbone(model, local_files_only=local_files_only)
        self.module = Descriptor(net, pooling, gem_p).eval().to(device)
        self.device = device
        self.prep = PrepConfig(size=size, clahe=bool(clahe))
        self.model_version = model_version(model_id, pooling, size)
        with torch.inference_mode():
            self.dim = int(self.module(torch.zeros(1, 3, size, size, device=device)).shape[-1])

    def _run(self, x: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self.module(self.torch.from_numpy(x).to(self.device)).float().cpu().numpy()


def get_embedder(spec: str | Path, **kw) -> Embedder:
    """Return an OnnxEmbedder for ``*.onnx`` files, otherwise a TorchEmbedder (Hugging Face id
    or local folder).
    """
    if str(spec).lower().endswith(".onnx"):
        return OnnxEmbedder(spec, **{k: v for k, v in kw.items()
                                     if k in ("providers", "clahe", "threads")})
    return TorchEmbedder(spec, **{k: v for k, v in kw.items()
                                  if k in ("pooling", "size", "clahe", "device", "gem_p",
                                           "local_files_only")})
