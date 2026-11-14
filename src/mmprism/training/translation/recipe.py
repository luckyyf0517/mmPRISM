"""Public-pose reader training: sentence-disjoint membership and a fixed 100-epoch endpoint."""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from mmprism.models.stgcn_legacy import LegacyWaveLLMPoseEncoder
from mmprism.models.translation import GeometryGuidedMT5
from mmprism.release.paths import resolve
from mmprism.release.settings import recipe
from mmprism.training.translation.pose_source import PoseSource
from mmprism.training.translation.wavellm_lightning import (
    WaveLLMDataModule,
    WaveLLMLightningModule,
    build_wavellm_lightning_trainer,
)
from mmprism.training.translation.wavellm_run_config import WaveLLMRunConfig


def normalize_sentence(text):
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def train_reader() -> Path:
    """Read the complete public-pose training recipe; no console command is provided."""
    from mmprism.release.reader import normalization

    seed = 20261101
    config = WaveLLMRunConfig.from_mapping(recipe("reader"))
    root = resolve("pack.daily_annotations")
    split = json.loads(resolve("pack.bigsplit").read_text())
    train_texts, test_texts = set(split["big_train"]), set(split["big_test"])
    if train_texts & test_texts:
        raise ValueError("Training and test sentence texts overlap")
    records = sorted(
        (
            row
            for line in (root / "episodes.jsonl").read_text().splitlines()
            if (row := json.loads(line)) and normalize_sentence(row["caption"]) in train_texts
        ),
        key=lambda row: row["sample_id"],
    )
    if any(normalize_sentence(row["caption"]) in test_texts for row in records):
        raise ValueError("A test sentence entered reader training")
    validation = random.Random(seed).sample(records, 300)
    pose = np.load(root / "arrays/gt_full24_m.npy", mmap_mode="r")
    mask = np.load(root / "arrays/frame_mask.npy", mmap_mode="r")
    stamps = np.load(root / "arrays/target_timestamps_s.npy", mmap_mode="r")

    def raw_clip(row):
        start, count = row["range"]["start"], row["range"]["frame_count"]
        return (
            np.asarray(pose[start : start + count], dtype=np.float32),
            np.asarray(mask[start : start + count], dtype=bool),
            np.asarray(stamps[start : start + count], dtype=np.float64),
        )

    mean, std = normalization()
    train = PoseSource(records, raw_clip, mean, std, training=True, speed_range=(0.6, 1.0))
    held = PoseSource(validation, raw_clip, mean, std, training=False, speed_range=(0.6, 1.0))
    L.seed_everything(seed, workers=True, verbose=False)
    language_path = resolve("training.reader_initial_language")
    tokenizer = AutoTokenizer.from_pretrained(language_path, local_files_only=True)
    language = MT5ForConditionalGeneration.from_pretrained(
        language_path,
        local_files_only=True,
        dtype=torch.float32,
    )
    encoder = LegacyWaveLLMPoseEncoder(root_centering=True)
    encoder.load_state_dict(
        torch.load(
            resolve("training.reader_initial_pose"),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
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
        pose_encoder=encoder,
    )
    identity = {
        "training_takes": len(records),
        "validation_takes": 300,
        "validation_role": "diagnostic subset of the training membership",
    }
    data = WaveLLMDataModule(train, held, config, seed=seed, source_identity=identity)
    module = WaveLLMLightningModule(
        model,
        tokenizer,
        config,
        source_identity=identity,
        train_group_ids=data.train_group_ids,
    )
    output = resolve("outputs.reader_training")
    output.mkdir(parents=True, exist_ok=True)
    module.configure_progress_output(output)
    trainer = build_wavellm_lightning_trainer(
        config,
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        accumulate_grad_batches=16,
        output_root=output,
        training_batches=570,
    )
    trainer.callbacks = [
        callback
        for callback in trainer.callbacks
        if not isinstance(callback, ModelCheckpoint) or callback.monitor is None
    ]
    trainer.fit(module, datamodule=data)
    return output
