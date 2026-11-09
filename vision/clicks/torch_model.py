"""PyTorch side of the DINOv2 descriptor (development machine only: torch + transformers).

``Descriptor`` turns the model's tokens into the final normalised vector,
so the exported ONNX model already outputs the descriptor and the app
never has to deal with tokens.

Token layout of the register model at 224 px with 14 px patches:
1 CLS + 4 registers + 256 patches = 261 tokens. Register tokens hold little
information about the image itself, so they are never used in the descriptor.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_backbone(model: str | Path, local_files_only: bool = True):
    """Load a DINOv2 model from a local folder (see ``scripts/get_dinov2.py``).

    With ``local_files_only=False`` it can also download from the Hugging Face hub.
    """
    from transformers import AutoModel
    net = AutoModel.from_pretrained(str(model), local_files_only=local_files_only)
    net.eval()
    model_id = getattr(net.config, "_name_or_path", None) or str(model)
    return net, model_id


class Descriptor(nn.Module):
    """Turn pixel_values [B,3,S,S] into embeddings [B,D], normalised to length 1.

    pooling:
      cls       the CLS token (describes the whole image; a good default)
      cls_gem   CLS + GeM(patch tokens) joined, each part normalised, then the whole
      cls_mean  CLS + mean(patch tokens), a backup in case GeM misbehaves
    """

    def __init__(self, backbone: nn.Module, pooling: str = "cls", gem_p: float = 3.0):
        super().__init__()
        self.m = backbone
        self.pooling = pooling
        self.gem_p = float(gem_p)
        self.n_reg = int(getattr(backbone.config, "num_register_tokens", 0) or 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.m(pixel_values=pixel_values).last_hidden_state     # [B, 1+R+P, D]
        cls = h[:, 0]
        if self.pooling == "cls":
            return F.normalize(cls, dim=-1)
        patches = h[:, 1 + self.n_reg:]  # skip CLS and the registers
        if self.pooling == "cls_gem":
            # clamp keeps pow() from producing NaN on negative values
            pooled = patches.clamp(min=1e-6).pow(self.gem_p).mean(1).pow(1.0 / self.gem_p)
        else:
            pooled = patches.mean(1)
        v = torch.cat([F.normalize(cls, dim=-1), F.normalize(pooled, dim=-1)], -1)
        return F.normalize(v, dim=-1)
