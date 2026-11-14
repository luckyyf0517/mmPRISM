"""Packaged numerical recipes for reconstruction and sign-language understanding."""

from __future__ import annotations

import json

import yaml

from mmprism.release.paths import PACKAGE_ROOT


def recipe(name: str) -> dict:
    return yaml.safe_load((PACKAGE_ROOT / "configs" / f"{name}.yaml").read_text())


def write_report(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def release_device_memory(device: str) -> None:
    """Release completed signers' model cycles before loading the next model."""
    import gc

    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
