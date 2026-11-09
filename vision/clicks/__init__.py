"""Recognise Click boards by image search with DINOv2.

    crop -> preprocess (pad to a square, never stretch) -> DINOv2 ViT-S/14
    (with registers) image descriptor via ONNX Runtime -> exact cosine
    search in a gallery of images tagged with PIDs -> re-rank (flipped/cropped
    views, OCR, board size, local features) -> accept | ask | unknown

Modules: preprocess, embedder (ONNX + PyTorch), torch_model (for exporting),
augment, gallery (storage + index), build, rerank (OCR), recognizer.
Command line: ``python -m vision.clicks build|add|query|info``.
"""

from .embedder import Embedder, OnnxEmbedder, TorchEmbedder, get_embedder  # noqa: F401
from .gallery import FlatIndex, Gallery, GalleryMismatch  # noqa: F401
from .preprocess import PrepConfig, prepare  # noqa: F401
from .recognizer import ClickConfig, ClickRecognizer, ClickResult, assign_slots  # noqa: F401
