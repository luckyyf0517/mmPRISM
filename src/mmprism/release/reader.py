"""Pose-conditioned Chinese translation, personal reader adaptation and paper metrics."""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch

from mmprism.data.ce_frozen_daily_predpose import normalize_ce_daily_pose
from mmprism.data.sign_language_translation import (
    SignLanguageTranslationSample,
    collate_sign_language_translation_samples,
)
from mmprism.models.stgcn_legacy import LegacyWaveLLMPoseEncoder
from mmprism.models.translation import GeometryGuidedMT5
from mmprism.release.data import SIGNERS, calibration_ids, cell_segments, person_data
from mmprism.release.paths import PACKAGE_ROOT, resolve
from mmprism.release.settings import recipe, release_device_memory, write_report
from mmprism.training.translation.pose_source import PoseSource
from mmprism.training.translation.wavellm_lightning import WaveLLMLightningModule, _tensor_batch
from mmprism.training.translation.wavellm_run_config import WaveLLMRunConfig

WINDOW = 256


def normalization():
    profile = json.loads((PACKAGE_ROOT / "configs/pose_normalization.json").read_text())
    return tuple(np.asarray(profile[key], dtype=np.float32) for key in ("mean", "std"))


def load_reader(device, *, signer=None, mode="adapted", adapted=False, groups=None):
    from transformers import AutoTokenizer, MT5Config, MT5ForConditionalGeneration

    config = WaveLLMRunConfig.from_mapping(recipe("reader"))
    base = resolve("pack.reader_base")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    language = MT5ForConditionalGeneration(MT5Config.from_pretrained(base, local_files_only=True))
    model = GeometryGuidedMT5(
        language,
        hidden_size=768,
        input_mode="pose_only",
        joint_count=24,
        coordinate_dim=3,
        pose_channels=(32, 64),
        temporal_kernel_size=5,
        dropout=0.0,
        label_smoothing=0.2,
        pose_encoder=LegacyWaveLLMPoseEncoder(root_centering=True),
    )
    module = WaveLLMLightningModule(model, tokenizer, config, train_group_ids=groups or {})
    path = (
        resolve("outputs.reader_checkpoint", signer=signer, mode=mode)
        if adapted
        else resolve("weights.reader")
    )
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.to(device), config


def build_sample(sample_id, pose, mask, mean, std):
    pose = np.asarray(pose, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    pose = normalize_ce_daily_pose(pose, mask, mean, std)
    n = pose.shape[0]
    if n > WINDOW:
        start = (n - WINDOW) // 2
        pose = pose[start : start + WINDOW]
        mask = mask[start : start + WINDOW]
    elif n < WINDOW:
        pad = WINDOW - n
        pose = np.concatenate([pose, np.repeat(pose[-1:], pad, axis=0)], axis=0)
        mask = np.concatenate([mask, np.zeros(pad, dtype=bool)], axis=0)
    confidence = np.broadcast_to(mask[:, None, None], pose.shape[:-1]).astype(np.float32).copy()
    return SignLanguageTranslationSample(
        sample_id=sample_id,
        pose=pose,
        pose_confidence=confidence,
        radar_feature=None,
        frame_mask=mask,
        caption="",
        coordinate_frame="own_yoke_per_side_root_fixed_direct_affine",
        pose_units="train_standardized",
    )


def prediction_clip(signer, sid, mode):
    rows, _, _ = person_data(signer)
    row = rows[sid]
    offset = row["frame_offset"]
    path = resolve("outputs.reconstruction", signer=signer, segment=sid, mode=mode)
    with np.load(path, allow_pickle=False) as payload:
        pose = payload["pose"].copy()
    # the inference mask comes from the reconstruction itself: a frame is usable when
    # every reconstructed joint is finite. No annotation is read on this path.
    mask = np.isfinite(pose).all(axis=(-2, -1))
    return pose, mask, np.arange(row["frames"], dtype=np.float64) * 0.1


def adapt_reader(signer: str, *, mode: str, device: str) -> dict:
    from dataclasses import replace

    torch.set_num_threads(2)
    settings = recipe("adaptation")["reader"]
    batch_size = int(settings["batch_size"])
    if batch_size <= 0:
        raise ValueError("Reader adaptation batch_size must be positive")
    seed = int(settings["seed"])
    rows, _, _ = person_data(signer)
    selected = calibration_ids(signer)
    records = [
        {
            "sample_id": sid,
            "segment_id": sid,
            "caption": rows[sid]["caption"],
            "group_id": rows[sid]["sentence_key"],
            "frames": rows[sid]["frames"],
        }
        for sid in selected
    ]
    groups = {row["sample_id"]: row["group_id"] for row in records}
    target_device = torch.device(device)
    module, config = load_reader(target_device, groups=groups)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    module.config = replace(
        config,
        optimization=replace(
            config.optimization,
            learning_rate=settings["learning_rate"],
            language_model_learning_rate=None,
            pose_branch_learning_rate=None,
            max_steps=settings["steps"],
        ),
    )
    module.model.requires_grad_(True)
    module.train()
    # The paper's stage-3 recipe: AdamW at its own defaults over the whole reader, a cosine
    # annealing schedule to zero, and the cross-entropy objective alone. The conditional
    # discrimination term belongs to the first two stages of reader training.
    optimizer = torch.optim.AdamW(
        [value for value in module.parameters() if value.requires_grad],
        lr=settings["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["steps"], eta_min=0.0
    )
    mean, std = normalization()
    source = PoseSource(
        records,
        lambda row: prediction_clip(signer, row["sample_id"], mode),
        mean,
        std,
        training=True,
    )
    rng = random.Random(seed)
    step, losses = 0, []
    per_epoch = math.ceil(len(records) / batch_size)
    while step < settings["steps"]:
        source.set_epoch(step // per_epoch)
        order = list(range(len(records)))
        rng.shuffle(order)
        for offset in range(0, len(order), batch_size):
            if step >= settings["steps"]:
                break
            step += 1
            optimizer.zero_grad(set_to_none=True)
            samples = [source.load_sample(index) for index in order[offset : offset + batch_size]]
            tensor = _tensor_batch(
                collate_sign_language_translation_samples(samples, max_frames=WINDOW),
                target_device,
                input_mode=config.input_mode,
            )
            output, *_ = module._forward_output(tensor)
            loss = output.loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [value for value in module.parameters() if value.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "signer": signer,
                            "step": step,
                            "loss": losses[-1],
                            "grad_norm": float(norm),
                        }
                    ),
                    flush=True,
                )
    path = resolve("outputs.reader_checkpoint", mode=mode, signer=signer)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
            "calibration_segments": selected,
            "completed_updates": step,
        },
        path,
    )
    result = {
        "signer": signer,
        "calibration_segments": selected,
        "steps": step,
        "configuration": settings,
        "final_loss_mean_last25": float(np.mean(losses[-25:])),
    }
    write_report(resolve("outputs.reader_report", signer=signer, mode=mode), result)
    return result


def translate_signer(signer, *, mode, reader, device):
    target_device = torch.device(device)
    module, config = load_reader(
        target_device, signer=signer, mode=mode, adapted=reader == "adapted"
    )
    module.eval()
    mean, std = normalization()
    path = resolve("outputs.translation", mode=mode, reader=reader, signer=signer)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream, torch.inference_mode():
        for segment in cell_segments("D", signer):
            pose, mask, _ = prediction_clip(signer, segment.sample_id, mode)
            sample = build_sample(segment.sample_id, pose, mask, mean, std)
            collated = collate_sign_language_translation_samples([sample], max_frames=WINDOW)
            tensor = _tensor_batch(collated, target_device, input_mode=config.input_mode)
            tokens = module._tokens([config.data.prompt], max_length=config.data.max_prompt_length)
            output = module.model.generate(
                pose=tensor.pose,
                pose_confidence=tensor.pose_confidence,
                frame_attention_mask=tensor.frame_attention_mask,
                prompt_input_ids=tokens["input_ids"].to(target_device),
                prompt_attention_mask=tokens["attention_mask"].to(target_device),
                max_new_tokens=config.generation.max_new_tokens,
                num_beams=config.generation.num_beams,
            )
            prediction = module.tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)[0]
            stream.write(
                json.dumps(
                    {
                        "segment_id": segment.sample_id,
                        "person": signer,
                        "reference": segment.row["caption"],
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(prog="mmprism-translate", description=__doc__)
    _common(parser)

def _common(parser: argparse.ArgumentParser) -> None:
    """The two options the mmprism command parses before the sub-command."""
    parser.add_argument("--data-root", help="root of the supplied data pack (default from config)")
    parser.add_argument("--config-dir", help="directory holding paths.toml (default: configs/)")

    parser.add_argument("--signer", action="append", choices=SIGNERS)
    parser.add_argument("--mode", choices=("zero-shot", "adapted"), default="adapted")
    parser.add_argument("--reader", choices=("base", "adapted"), default="adapted")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    signers = tuple(dict.fromkeys(args.signer or SIGNERS))
    for signer in signers:
        translate_signer(signer, mode=args.mode, reader=args.reader, device=args.device)
        release_device_memory(args.device)
    from mmprism.release.translation_metrics import score_translation

    result = score_translation(signers, mode=args.mode, reader=args.reader, device=args.device)
    print(json.dumps(result["pooled"], indent=2))
    return 0
