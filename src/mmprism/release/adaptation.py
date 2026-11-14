"""Fifty-segment personal adaptation of reconstruction and sign-language understanding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from mmprism.release.data import SIGNERS, CellDataset, Segment, calibration_ids, person_data
from mmprism.release.inference import load_reconstruction
from mmprism.release.paths import resolve
from mmprism.release.settings import recipe, release_device_memory, write_report
from mmprism.training.adaptation.pose_loss import pose_only_terms
from mmprism.training.reconstruction.trainer import _new_fk, decode_complete_mhr_full24

WINDOW = 32


def learning_rate(step: int, config: dict) -> float:
    peak = float(config["learning_rate"])
    start = float(config["initial_learning_rate"])
    end = float(config["final_learning_rate"])
    warmup = int(config["warmup_steps"])
    steps = int(config["steps"])
    if step == warmup:
        return peak
    if step <= warmup:
        return start + (peak - start) * (step - 1) / (warmup - 1)
    progress = (step - warmup) / (steps - warmup)
    return end + (peak - end) * (1 + math.cos(math.pi * progress)) / 2


def signer_seed(signer: str, segments: int) -> int:
    value = "|".join(map(str, (signer, segments)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 2**31


def adapt_reconstruction(signer: str, *, device: str) -> dict:
    config = recipe("adaptation")["reconstruction"]
    target_device = torch.device(device)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = load_reconstruction(device=target_device)
    torch.manual_seed(int(config["seed"]))
    fk = _new_fk(recipe("reconstruction"), device=target_device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("error_head."))
    parameters = [value for value in model.parameters() if value.requires_grad]
    rows, _, _ = person_data(signer)
    selected = calibration_ids(signer)
    if any(sid not in rows for sid in selected):
        raise ValueError(f"{signer}: a selected calibration segment is missing")
    dataset = CellDataset([Segment(rows[sid], signer, "C") for sid in selected])
    dataset = MemoryWindows(dataset)
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
    rng = np.random.default_rng([config["seed"], 11, signer_seed(signer, len(selected))])
    started = time.monotonic()
    losses = []
    # Personal adaptation retains evaluation mode, including normalization and dropout behavior.
    model.eval()
    for step in range(1, int(config["steps"]) + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, config)
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _micro in range(4):
            items = []
            for _ in range(2):
                index = int(rng.integers(len(dataset)))
                frames = dataset.segments[index].frame_count
                start = int(rng.integers(max(1, frames - 32 + 1)))
                items.append(dataset.fixed_window(index, start))
            batch = to_batch(items, target_device)
            prediction = model(batch["raw_radar"], valid_mask=batch["temporal_mask"])
            decoded = decode_complete_mhr_full24(
                prediction.mhr_state.state, temporal_mask=batch["temporal_mask"], fk=fk
            )
            terms = pose_only_terms(
                prediction=prediction,
                fk_full24=decoded,
                target_global=batch["target_global_pose"],
                valid_mask=batch["target_mask"],
            )
            loss = terms["total"]
            assert torch.isfinite(loss), "nonfinite loss"
            (loss / 4).backward()
            accumulated += float(loss.detach()) / 4
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(accumulated)
        if step == 1 or step % 10 == 0 or step == config["steps"]:
            print(
                json.dumps(
                    {
                        "signer": signer,
                        "step": step,
                        "loss": accumulated,
                        "grad_norm": float(norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                ),
                flush=True,
            )
    path = resolve("outputs.reconstruction_checkpoint", signer=signer)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "calibration_segments": selected,
            "completed_optimizer_steps": config["steps"],
        },
        path,
    )
    result = {
        "signer": signer,
        "calibration_segments": selected,
        "configuration": config,
        "loss_first20": float(np.mean(losses[:20])),
        "loss_last20": float(np.mean(losses[-20:])),
        "seconds": time.monotonic() - started,
    }
    write_report(resolve("outputs.adaptation_report", signer=signer, side="reconstruction"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="mmprism-adapt", description=__doc__)
    _common(parser)

def _common(parser: argparse.ArgumentParser) -> None:
    """The two options the mmprism command parses before the sub-command."""
    parser.add_argument("--data-root", help="root of the supplied data pack (default from config)")
    parser.add_argument("--config-dir", help="directory holding paths.toml (default: configs/)")

    parser.add_argument("--signer", action="append", choices=SIGNERS)
    parser.add_argument("--side", choices=("reconstruction", "reader"), default="reconstruction")
    parser.add_argument("--mode", choices=("zero-shot", "adapted"), default="adapted")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for signer in dict.fromkeys(args.signer or SIGNERS):
        if args.side == "reader":
            from mmprism.release.reader import adapt_reader

            adapt_reader(signer, mode=args.mode, device=args.device)
        else:
            adapt_reconstruction(signer, device=args.device)
        release_device_memory(args.device)
    return 0


class MemoryWindows:
    """Calibration windows cached in host memory."""

    window_frames = WINDOW

    def __init__(self, base, budget_gb=60.0):
        self.base = base
        self.segments = base.segments
        frames = sum(s.frame_count for s in base.segments)
        self.gb = frames * 64 * 116 * 32 * 2 * 4 / 1000000000.0
        assert self.gb <= budget_gb, (
            f"{self.gb:.1f} GB of IQ exceeds the in-memory budget {budget_gb} GB"
        )
        self.raw = [base.whole_segment(i) for i in range(len(base.segments))]
        self.pose, self.mask = ([], [])
        for i in range(len(base.segments)):
            p, m = base.targets(i)
            self.pose.append(np.asarray(p))
            self.mask.append(np.asarray(m))

    def __len__(self):
        return len(self.segments)

    def fixed_window(self, i, start):
        s = self.segments[i]
        n = min(WINDOW, s.frame_count - start)
        if n == WINDOW:
            raw = self.raw[i][start : start + WINDOW]
        else:
            raw = np.zeros((WINDOW, 64, 116, 32, 2), np.float32)
            raw[:n] = self.raw[i][start : start + n]
        pose = np.zeros((WINDOW, 2, 24, 3), np.float32)
        pose[:n] = self.pose[i][start : start + n]
        tm = np.zeros(WINDOW, bool)
        tm[:n] = True
        km = np.zeros(WINDOW, bool)
        km[:n] = self.mask[i][start : start + n]
        return {
            "raw_radar": raw,
            "temporal_mask": tm,
            "target_mask": km,
            "target_global_pose": pose,
        }


_PINNED = {}


def to_batch(items, device):
    """Stack into alternating pinned buffers and copy to the target device."""
    import torch

    out = {}
    for k in ("raw_radar", "target_global_pose", "target_mask", "temporal_mask"):
        first = np.asarray(items[0][k])
        shape = (len(items), *first.shape)
        slot = _PINNED.setdefault((k, shape, first.dtype.str), [None, None, 0])
        if slot[0] is None:
            slot[0] = (
                torch.empty(shape, dtype=torch.from_numpy(first[:1]).dtype, pin_memory=True)
                if device.type == "cuda"
                else torch.empty(shape, dtype=torch.from_numpy(first[:1]).dtype)
            )
            slot[1] = torch.empty_like(slot[0], pin_memory=device.type == "cuda")
        buf = slot[slot[2] % 2]
        slot[2] += 1
        arr = buf.numpy()
        for j, it in enumerate(items):
            arr[j] = it[k]
        out[k] = buf.to(device, non_blocking=True)
    return out
