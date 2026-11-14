"""Pose normalization, joint-style perturbation and tempo augmentation for reader training."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from mmprism.data.ce_frozen_daily_predpose import normalize_ce_daily_pose
from mmprism.data.daily_pose_rotation import sample_rotation
from mmprism.data.daily_style_packages import transform
from mmprism.data.sam3d_full24_yoke import yoke_normalize
from mmprism.data.sign_language_translation import SignLanguageTranslationSample


class PoseSource:
    FIXED_FRAMES = 256
    JOINT_STYLE_SEED = 20260920

    def __init__(self, records, raw_loader, mean, std, *, training, speed_range=(0.8, 1.2)):
        self.records, self.raw_loader = records, raw_loader
        self.mean, self.std = mean, std
        self.training = training
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.SPEED_MIN, self.SPEED_MAX = speed_range
        self.sample_group_ids = {row["sample_id"]: row["group_id"] for row in records}
        self.sample_frame_counts = tuple(
            row.get("frames", row.get("range", {}).get("frame_count")) for row in records
        )

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch[0] = int(epoch)

    def raw_clip(self, record):
        return self.raw_loader(record)

    def load_sample(self, index):
        record = self.records[index]
        raw, mask, stamps = self.raw_clip(record)
        pose = None
        if self.training:
            key = (
                self.JOINT_STYLE_SEED,
                int(self.epoch[0].item()),
                int.from_bytes(
                    hashlib.blake2b(record["sample_id"].encode(), digest_size=4).digest(), "little"
                ),
            )
            if sample_rotation(*key)["enabled"]:
                common, _, _ = yoke_normalize(raw)
                perturbed, mapping, _ = transform(common, stamps, key, "joint")
                mask = mask[mapping]
                pose = (
                    ((perturbed.reshape(len(perturbed), -1) - self.mean) / self.std)
                    .reshape(perturbed.shape)
                    .astype(np.float32)
                )
                pose[~mask] = 0.0
        if pose is None:
            pose = normalize_ce_daily_pose(raw, mask, self.mean, self.std)
        speed = (
            float(self.SPEED_MIN + (self.SPEED_MAX - self.SPEED_MIN) * torch.rand(1).item())
            if self.training
            else 1.0
        )
        if speed != 1.0:
            n = pose.shape[0]
            target = max(2, round(n / speed))
            positions = np.linspace(0.0, n - 1.0, target, dtype=np.float32)
            lo = np.floor(positions).astype(np.int64)
            hi = np.minimum(lo + 1, n - 1)
            weight = (positions - lo)[:, None, None, None]
            pose = pose[lo] * (1.0 - weight) + pose[hi] * weight
            mask = mask[lo]
        n = pose.shape[0]
        if n > self.FIXED_FRAMES:
            begin = int(torch.randint(0, n - self.FIXED_FRAMES + 1, (1,)).item())
            pose = pose[begin : begin + self.FIXED_FRAMES]
            mask = mask[begin : begin + self.FIXED_FRAMES]
        assert 0 < pose.shape[0] <= self.FIXED_FRAMES, pose.shape
        confidence = np.broadcast_to(mask[:, None, None], pose.shape[:-1]).astype(np.float32).copy()
        return SignLanguageTranslationSample(
            sample_id=record["sample_id"],
            pose=pose,
            pose_confidence=confidence,
            radar_feature=None,
            frame_mask=mask,
            caption=record["caption"],
            coordinate_frame="own_yoke_per_side_root_fixed_direct_affine",
            pose_units="train_standardized",
        )
