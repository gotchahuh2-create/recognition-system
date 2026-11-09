"""Download the DINOv2 weights once into models/hf/.

    pip install huggingface_hub
    python scripts/get_dinov2.py                 # ViT-S/14 + registers (default)
    python scripts/get_dinov2.py --variant base  # ViT-B/14 + registers (more accurate, slower)

After that everything loads with ``local_files_only=True``, so the app never
needs the internet. Both variants are Apache-2.0 licensed (see the model
cards). DINOv3 is left out on purpose: it has a custom licence and a gated
download, so keep it as an experiment until MIKROE legal approves it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

VARIANTS = {
    "small": "facebook/dinov2-with-registers-small",  # 21M parameters, 384-d
    "base": "facebook/dinov2-with-registers-base",  # 86M parameters, 768-d
    "large": "facebook/dinov2-large",  # only for offline gallery experiments
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="small", choices=sorted(VARIANTS))
    ap.add_argument("--out", type=Path, default=Path("models/hf"))
    a = ap.parse_args(argv)
    from huggingface_hub import snapshot_download
    repo = VARIANTS[a.variant]
    dst = a.out / repo.split("/")[-1]
    path = snapshot_download(repo_id=repo, local_dir=str(dst),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
    print(f"{repo} -> {path}")
    print("next: python scripts/export_dinov2_onnx.py --model", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
