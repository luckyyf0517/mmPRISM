"""Complex-radar reconstruction and the frame-aligned pose interface to the reader."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from mmprism.release.data import CellDataset, Segment, calibration_ids, cell_segments, person_data
from mmprism.release.paths import resolve
from mmprism.release.settings import recipe
from mmprism.training.reconstruction.trainer import build_model

WINDOW = 32
STRIDE = 24
TOKENS = 19


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_reconstruction(
    config: Mapping[str, Any] | None = None,
    *,
    device: torch.device,
    signer: str | None = None,
    adapted: bool = False,
) -> Any:
    """Load the original payload, optionally followed by a personal adaptation."""
    config = recipe("reconstruction") if config is None else config
    torch.manual_seed(int(config["optimization"]["seed"]))
    model, _ = build_model(config, load_official_dino=False)
    payload = torch.load(
        resolve("weights.reconstruction"), map_location="cpu", mmap=True, weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if adapted:
        if signer is None:
            raise ValueError("An adapted reconstruction requires a signer")
        path = resolve("outputs.reconstruction_checkpoint", signer=signer)
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def plan_windows(frames: int, *, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    if frames <= window:
        return [0]
    starts = list(range(0, frames - window + 1, stride))
    if starts[-1] != frames - window:
        starts.append(frames - window)
    return starts


def owner_windows(frames: int, starts: list[int], *, window: int = WINDOW) -> np.ndarray:
    """Assign each frame to the window with the largest margin from either edge."""
    owner = np.full(frames, -1, dtype=np.int64)
    best = np.full(frames, -1, dtype=np.int64)
    for index, start in enumerate(starts):
        stop = min(start + window, frames)
        for t in range(start, stop):
            margin = min(t - start, start + window - 1 - t)
            if margin > best[t]:
                best[t] = margin
                owner[t] = index
    assert bool((owner >= 0).all()), "a frame is covered by no window"
    return owner


def export_sequence(model, raw: np.ndarray, device) -> dict[str, np.ndarray]:
    """Export overlapping 32-frame windows at stride 24 with the original owner rule."""
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    assert raw.ndim == 5 and raw.shape[1:] == (64, 116, 32, 2), raw.shape
    frames = int(raw.shape[0])
    starts = plan_windows(frames)
    owner = owner_windows(frames, starts)
    pose = np.zeros((frames, 2, 24, 3), dtype=np.float32)
    feature = np.zeros((frames, TOKENS, int(model.decoder.hidden_channels)), dtype=np.float32)
    error = np.zeros((frames, 2, 24), dtype=np.float32)
    error_hand_rel = np.zeros((frames, 2, 20), dtype=np.float32)
    window_start = np.zeros(frames, dtype=np.int32)
    model.eval()
    with torch.inference_mode():
        for index, start in enumerate(starts):
            count = min(WINDOW, frames - start)
            batch = np.zeros((1, WINDOW, 64, 116, 32, 2), dtype=np.float32)
            batch[0, :count] = raw[start : start + count]
            mask = np.zeros((1, WINDOW), dtype=bool)
            mask[0, :count] = True
            output = model(
                torch.from_numpy(batch).to(device), valid_mask=torch.from_numpy(mask).to(device)
            )
            tokens = output.normalized_reader_tokens
            assert int(tokens.shape[-2]) == TOKENS, tokens.shape
            log_error = model.error_head_log_estimate
            assert log_error is not None and log_error.shape == (1, WINDOW, 2, 24, 2)
            full24 = output.direct.full24_m[0].float().cpu().numpy()
            tok = tokens[0].float().cpu().numpy()
            err_all = (torch.exp(log_error[0].float()) * 100.0).cpu().numpy()
            err = err_all[..., 0]
            err_rel = err_all[:, :, 4:, 1]
            for local in range(count):
                t = start + local
                if owner[t] == index:
                    pose[t] = full24[local]
                    feature[t] = tok[local]
                    error[t] = err[local]
                    error_hand_rel[t] = err_rel[local]
                    window_start[t] = start
    return {
        "pose": pose,
        "feature": feature,
        "error": error,
        "error_hand_rel": error_hand_rel,
        "window_start": window_start,
    }


def reconstruct_signer(signer: str, *, mode: str, device: str) -> None:
    configure_determinism()
    target_device = torch.device(device)
    model = load_reconstruction(device=target_device, signer=signer, adapted=mode == "adapted")
    rows, _, _ = person_data(signer)
    evaluation = cell_segments("D", signer)
    calibration = [Segment(rows[sid], signer, "C") for sid in calibration_ids(signer)]
    dataset = CellDataset([*calibration, *evaluation])
    for index, segment in enumerate(dataset.segments):
        arrays = export_sequence(model, dataset.whole_segment(index), target_device)
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"{segment.sample_id}: reconstruction contains nonfinite values")
        path = resolve(
            "outputs.reconstruction", mode=mode, signer=signer, segment=segment.sample_id
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        if (index + 1) % 25 == 0:
            print(f"{signer}: reconstructed {index + 1}/{len(dataset)} segments", flush=True)


def reconstruct(
    config: Mapping[str, Any], *, input_key: str, output_key: str, device: str = "cpu"
) -> None:
    """Reconstruct one configured radar archive through the same frame-owner interface."""
    target_device = torch.device(device)
    model = load_reconstruction(config, device=target_device)
    with np.load(resolve(input_key), allow_pickle=False) as source:
        arrays = export_sequence(model, source["raw_radar"], target_device)
    output = resolve(output_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
