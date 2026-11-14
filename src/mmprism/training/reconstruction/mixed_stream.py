"""One real and one freshly simulated window per rank and optimizer update."""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from mmprism.data.csl_daily_mhr_windows import load_csl_daily_mhr_trajectory
from mmprism.simulation import renderer
from mmprism.simulation.mixed_domain import (
    DonorStats,
    calibrate_iq,
    calibrate_iq_with_donor_stats,
    donor_statistics,
)
from mmprism.training.reconstruction.dataset import collate_windows

WINDOW_FRAMES = 32
HELD_OUT_SIGNERS = ("p01", "p03", "p06", "p10", "p11", "p17", "p36", "p37")
_RENDER_LOCK = threading.Lock()


class MixedStreamError(RuntimeError):
    """A mixed window or donor draw violates the training contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MixedStreamError(message)


def simulation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    block = config.get("simulation_stream")
    _require(isinstance(block, Mapping), "simulation_stream section is missing")
    membership = block.get("membership")
    _require(isinstance(membership, Mapping), "simulation_stream.membership is missing")
    scene = block.get("scene")
    _require(isinstance(scene, Mapping), "simulation_stream.scene is missing")
    return {
        "data": {
            "source_root": str(membership["source_root"]),
            "strict_split_assignments": str(membership["strict_split_assignments"]),
            "expected_episodes": dict(membership["expected_episodes"]),
            "allowed_unassigned_sample_ids": list(membership["allowed_unassigned_sample_ids"]),
            "source_frame_stride": int(block["source_frame_stride"]),
            "target_fps": float(block["target_fps"]),
            "train_window_frames": WINDOW_FRAMES,
        },
        "simulation": dict(scene),
        "upstream": dict(config["upstream"]),
        "optimization": {"sampler_seed": int(block["seed"])},
    }


def background(z: np.ndarray) -> np.ndarray:
    return (np.median(z.real, axis=0) + 1j * np.median(z.imag, axis=0)).astype(np.complex64)


def calibrate_with_canonical_solver(
    raw_sim: np.ndarray, raw_real: np.ndarray, *, alpha: float, device: Any = None
) -> tuple[np.ndarray, dict[str, float]]:
    _require(raw_sim.shape[-1] == 2 and raw_real.shape[-1] == 2, "background transplant needs I/Q")
    return calibrate_iq(raw_sim, raw_real, alpha=float(alpha), device=device)


@dataclass(frozen=True)
class SimDraw:
    item: dict[str, Any]
    record: dict[str, Any]


class DailySimulationStream:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        device: Any,
        donor_dataset: Any,
        donor_indices: Sequence[int],
    ) -> None:
        block = config["simulation_stream"]
        self.sim = renderer
        self.cfg = simulation_config(config)
        self.device = device
        self.seed = int(block["seed"])
        self.stride = int(block["source_frame_stride"])
        self.alpha = float(block["background"]["alpha"])
        self.denoiser_pool = str(block["background"]["donor_pool"])
        _require(block["background"]["donor_pool"] == "old", "background donor pool differs")
        _require(
            tuple(str(value) for value in block["background"]["heldout_signers"])
            == HELD_OUT_SIGNERS,
            "declared held-out signer list differs",
        )
        self.adapter = self.sim._build_adapter(self.cfg, device=device)
        self.simulator = self.sim._build_simulator(self.cfg)
        self.specs = self.sim._membership(self.cfg).role("train")
        _require(self.specs, "Daily train membership is empty")
        self.donor_dataset = donor_dataset
        self.donor_indices = tuple(int(index) for index in donor_indices)
        _require(self.donor_indices, "background donor pool is empty")
        self.donor_sample_ids = tuple(
            donor_dataset.segments[index].sample_id for index in self.donor_indices
        )
        leaked = [
            sample_id
            for sample_id in self.donor_sample_ids
            if sample_id.split(":")[0] in HELD_OUT_SIGNERS
        ]
        _require(not leaked, f"held-out signer entered the background donor pool: {leaked[:3]}")
        by_person: dict[str, list[int]] = {}
        for index in self.donor_indices:
            segment = donor_dataset.segments[index]
            if int(segment.frame_count) >= WINDOW_FRAMES:
                by_person.setdefault(str(segment.sample_id).split(":")[0], []).append(index)
        self.donor_persons = tuple(sorted(by_person))
        self.donor_by_person = {person: tuple(by_person[person]) for person in self.donor_persons}
        _require(bool(self.donor_persons), "no donor person has a full 32-frame segment")
        self.donor_windows = int(block["background"].get("windows_per_draw", 3))
        _require(self.donor_windows >= 1, "donor windows per draw must be positive")
        anchor = self.cfg["simulation"]["shoulder_anchor"]
        self.radius_range = tuple(float(v) for v in anchor["train_radar_distance_range_m"])
        self.lateral_range = tuple(float(v) for v in anchor["train_lateral_range_m"])
        self.height_range = tuple(float(v) for v in anchor["train_height_range_m"])
        self.rendered_windows = 0

    def placement(self, *, global_draw: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, 31337, int(global_draw)])
        radius = float(rng.uniform(*self.radius_range))
        lateral = float(rng.uniform(*self.lateral_range))
        height = float(rng.uniform(*self.height_range))
        forward_squared = radius * radius - lateral * lateral - height * height
        _require(forward_squared > 0.0, "shoulder placement exceeds the sampled radius")
        return np.asarray((lateral, height, math.sqrt(forward_squared)), dtype=np.float32)

    def donor_window(self, *, step: int, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng([self.seed, 7919, int(step), int(rank)])
        person = self.donor_persons[int(rng.integers(0, len(self.donor_persons)))]
        pool = self.donor_by_person[person]
        count = min(self.donor_windows, len(pool))
        chosen = rng.choice(len(pool), size=count, replace=False)
        windows = []
        for position in chosen:
            segment_index = pool[int(position)]
            segment = self.donor_dataset.segments[segment_index]
            start = int(rng.integers(0, segment.frame_count - WINDOW_FRAMES + 1))
            item = self.donor_dataset.fixed_window(segment_index, start)
            _require(
                bool(np.asarray(item["temporal_mask"], dtype=bool).all()), "donor window is short"
            )
            windows.append(np.asarray(item["raw_radar"], dtype=np.float32))
        return (np.concatenate(windows, axis=0), {"donor_person": person, "donor_windows": count})

    def draw(
        self,
        *,
        global_draw: int,
        step: int,
        rank: int,
        donor_stats: tuple[DonorStats, dict[str, Any]] | None = None,
    ) -> SimDraw:
        started = time.perf_counter()
        spec = self.sim._spec_for_global_draw(
            self.specs, sampler_seed=self.seed, global_draw=int(global_draw)
        )
        trajectory = load_csl_daily_mhr_trajectory(spec, source_frame_stride=self.stride)
        window = self.sim._window_for_draw(
            trajectory,
            sampler_seed=self.seed,
            global_draw=int(global_draw),
            source_frame_stride=self.stride,
        )
        scene = self.sim._scene_for_draw(self.cfg, role="train", draw_index=int(global_draw))
        scene = dataclasses.replace(
            scene, desired_shoulder_midpoint_m=self.placement(global_draw=int(global_draw))
        )
        with _RENDER_LOCK:
            prepared = self.sim._materialize_window(
                self.cfg,
                window=window,
                scene=scene,
                adapter=self.adapter,
                simulator=self.simulator,
                device=self.device,
                annotation_read_seconds=0.0,
            )
        render_seconds = time.perf_counter() - started
        raw = prepared.raw_iq.numpy()
        count = int(raw.shape[0])
        _require(count > 0, "rendered simulation window is empty")
        if donor_stats is None:
            (donor, donor_info) = self.donor_window(step=step, rank=rank)
            (raw, transplant) = calibrate_with_canonical_solver(
                raw, donor, alpha=self.alpha, device=self.device
            )
        else:
            (stats, donor_info) = donor_stats
            (raw, transplant) = calibrate_iq_with_donor_stats(
                raw, stats, alpha=self.alpha, device=self.device
            )
        raw_padded = np.zeros((WINDOW_FRAMES, 64, 116, 32, 2), dtype=np.float32)
        pose_padded = np.zeros((WINDOW_FRAMES, 2, 24, 3), dtype=np.float32)
        state_padded = np.zeros((WINDOW_FRAMES, 191), dtype=np.float32)
        mask = np.zeros(WINDOW_FRAMES, dtype=bool)
        raw_padded[:count] = raw
        pose_padded[:count] = prepared.target_global.numpy()
        state_padded[:count] = prepared.target_state.numpy()
        mask[:count] = True
        self.rendered_windows += 1
        item = {
            "raw_radar": raw_padded,
            "target_global_pose": pose_padded,
            "target_mhr191": state_padded,
            "temporal_mask": mask,
            "target_mask": mask.copy(),
            "sample_id": f"reconstruction:{spec.sample_id}",
            "reference_id": f"reconstruction:{spec.sample_id}",
            "source_id": "daily_simulation",
            "segment_index": -1,
            "window_start": int(window.target_start),
        }
        midpoint = np.asarray(scene.desired_shoulder_midpoint_m, dtype=np.float64)
        record = {
            "step": int(step),
            "rank": int(rank),
            "global_draw": int(global_draw),
            "daily_sample_id": str(spec.sample_id),
            "valid_frames": count,
            "shoulder_radius_m": float(np.linalg.norm(midpoint)),
            "shoulder_lateral_m": float(midpoint[0]),
            "shoulder_height_m": float(midpoint[1]),
            "viewpoint_jitter_rad": [float(value) for value in scene.rotation_axis_angle_rad],
            "array_geometry": "inherited_unchanged",
            "render_seconds": render_seconds,
            **donor_info,
            **transplant,
        }
        return SimDraw(item=item, record=record)


def _ordered_parallel(
    count: int, build: Any, *, workers: int, depth: int, device_index: int | None
) -> Iterator[Any]:
    if workers <= 0:
        for index in range(count):
            yield build(index, 0)
        return
    local = threading.local()
    slots = iter(range(workers))
    slot_lock = threading.Lock()

    def task(index: int) -> Any:
        if not hasattr(local, "slot"):
            with slot_lock:
                local.slot = next(slots)
            if device_index is not None:
                torch.cuda.set_device(device_index)
        return build(index, local.slot)

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mixed-prefetch")
    pending: deque[Any] = deque()
    submitted = 0
    try:
        while submitted < min(count, depth):
            pending.append(pool.submit(task, submitted))
            submitted += 1
        for _ in range(count):
            batch = pending.popleft().result()
            if submitted < count:
                pending.append(pool.submit(task, submitted))
                submitted += 1
            yield batch
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


class DonorSampler:
    def __init__(
        self, *, seed: int, dataset: Any, by_person: Mapping[str, Sequence[int]], windows: int
    ) -> None:
        self.seed = int(seed)
        self.dataset = dataset
        self.persons = tuple(sorted(by_person))
        self.by_person = {person: tuple(by_person[person]) for person in self.persons}
        self.windows = int(windows)

    def sample(self, *, step: int, rank: int) -> tuple[DonorStats, dict[str, Any]]:
        rng = np.random.default_rng([self.seed, 7919, int(step), int(rank)])
        person = self.persons[int(rng.integers(0, len(self.persons)))]
        pool = self.by_person[person]
        count = min(self.windows, len(pool))
        chosen = rng.choice(len(pool), size=count, replace=False)
        windows = []
        for position in chosen:
            segment_index = pool[int(position)]
            segment = self.dataset.segments[segment_index]
            start = int(rng.integers(0, segment.frame_count - WINDOW_FRAMES + 1))
            item = self.dataset.fixed_window(segment_index, start)
            _require(
                bool(np.asarray(item["temporal_mask"], dtype=bool).all()), "donor window is short"
            )
            windows.append(np.asarray(item["raw_radar"], dtype=np.float32))
        stats = donor_statistics(np.concatenate(windows, axis=0), device="cpu")
        return (stats, {"donor_person": person, "donor_windows": count})


class WorkerItems(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        real_dataset: Any,
        batches: Sequence[Sequence[int]],
        real_count: int,
        sampler_cycle: int,
        start_batch: int,
        rank: int,
        donor_sampler: DonorSampler,
    ) -> None:
        self.real_dataset = real_dataset
        self.batches = batches
        self.real_count = int(real_count)
        self.sampler_cycle = int(sampler_cycle)
        self.start_batch = int(start_batch)
        self.rank = int(rank)
        self.donor_sampler = donor_sampler

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, offset: int) -> dict[str, Any]:
        indices = tuple(self.batches[offset])[: self.real_count]
        real_items = []
        for index in indices:
            item = dict(self.real_dataset[int(index), self.sampler_cycle])
            item["raw_radar"] = torch.from_numpy(np.ascontiguousarray(item["raw_radar"]))
            real_items.append(item)
        (stats, info) = self.donor_sampler.sample(step=self.start_batch + offset, rank=self.rank)
        return {
            "real_items": real_items,
            "indices": [int(i) for i in indices],
            "donor": (stats, info),
        }


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def _identity(sample: Any) -> Any:
    return sample


class MixedCycleLoader:
    def __init__(
        self,
        *,
        real_dataset: Any,
        sim_streams: Sequence[DailySimulationStream],
        world_size: int,
        rank: int,
        per_device_batch: int,
        real_count: int,
        sim_count: int,
        base_seed: int,
        sampler_cycle: int,
        start_batch: int,
        inline: bool = False,
        data_workers: int = 0,
    ) -> None:
        from mmprism.data.reconstruction_membership import deterministic_cycle_batches

        _require(
            (real_count, sim_count, per_device_batch) == (1, 1, 2),
            "each update requires one real and one simulated window",
        )
        self.real_dataset = real_dataset
        self.sim_streams = tuple(sim_streams)
        _require(bool(self.sim_streams), "the mixed loader needs at least one simulation stream")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.real_count = int(real_count)
        self.sim_count = int(sim_count)
        self.sampler_cycle = int(sampler_cycle)
        self.start_batch = int(start_batch)
        self._batches = deterministic_cycle_batches(
            segment_count=len(real_dataset),
            world_size=self.world_size,
            per_device_batch=int(per_device_batch),
            base_seed=int(base_seed),
            sampler_cycle=self.sampler_cycle,
        )[self.rank][self.start_batch :]
        self.global_step = 0
        self.real_share: list[float] = []
        self.prefetch_depth = len(self.sim_streams) + 1
        self.inline = bool(inline)
        self.data_workers = int(data_workers)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.data_workers > 0:
            return self._iter_with_data_workers()
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
        return _ordered_parallel(
            len(self._batches),
            self._build,
            workers=0 if self.inline else len(self.sim_streams),
            depth=self.prefetch_depth,
            device_index=device_index,
        )

    def _iter_with_data_workers(self) -> Iterator[dict[str, Any]]:
        stream = self.sim_streams[0]
        sampler = DonorSampler(
            seed=stream.seed,
            dataset=stream.donor_dataset,
            by_person=stream.donor_by_person,
            windows=stream.donor_windows,
        )
        items = WorkerItems(
            real_dataset=self.real_dataset,
            batches=self._batches,
            real_count=self.real_count,
            start_batch=self.start_batch,
            sampler_cycle=self.sampler_cycle,
            rank=self.rank,
            donor_sampler=sampler,
        )
        loader = torch.utils.data.DataLoader(
            items,
            batch_size=None,
            shuffle=False,
            num_workers=self.data_workers,
            prefetch_factor=2,
            collate_fn=_identity,
            worker_init_fn=_worker_init,
            pin_memory=False,
        )
        for offset, sample in enumerate(loader):
            yield self._build(offset, 0, sample)

    def _build(
        self, offset: int, slot: int, sample: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        indices = self._batches[offset]
        stream = self.sim_streams[slot]
        cycle_step = self.start_batch + offset
        global_draw = self.sampler_cycle * 1000003 + cycle_step * self.world_size + self.rank
        self.global_step = cycle_step
        if sample is None:
            real_items = [
                self.real_dataset[int(index), self.sampler_cycle]
                for index in tuple(indices)[: self.real_count]
            ]
            donor_stats = None
        else:
            real_items = []
            for item in sample["real_items"]:
                item = dict(item)
                item["raw_radar"] = item["raw_radar"].numpy()
                real_items.append(item)
            donor_stats = sample["donor"]
        draws = [
            stream.draw(
                global_draw=global_draw * self.sim_count + k,
                step=cycle_step,
                rank=self.rank,
                donor_stats=donor_stats,
            )
            for k in range(self.sim_count)
        ]
        real_batch = collate_windows(real_items)
        sim_batch = collate_windows([draw.item for draw in draws])
        batch: dict[str, Any] = {}
        for key, value in real_batch.items():
            if isinstance(value, torch.Tensor):
                merged = torch.cat((value, sim_batch[key]), dim=0)
                batch[key] = (
                    merged.to(dtype=torch.float32) if merged.is_floating_point() else merged
                )
            else:
                batch[key] = [*value, *sim_batch[key]]
        batch["mixed_batch"] = {
            "real_count": len(real_items),
            "sim_count": len(draws),
            "real_indices": [int(index) for index in tuple(indices)[: self.real_count]],
            "sim_records": [draw.record for draw in draws],
        }
        return batch


COORDINATE_SCALE_METRES = 0.1
WRIST_JOINT = 2
HAND_FIRST_JOINT = 4


class ErrorEstimationHead(nn.Module):
    def __init__(
        self,
        *,
        token_channels: int,
        token_count: int,
        hidden_channels: int = 192,
        joints: int = 24,
        sides: int = 2,
        initial_error_metres: float = 0.03,
    ) -> None:
        super().__init__()
        self.token_channels = int(token_channels)
        self.token_count = int(token_count)
        self.hidden_channels = int(hidden_channels)
        self.joints = int(joints)
        self.sides = int(sides)
        self.feature_projection = nn.Linear(self.token_channels, self.hidden_channels)
        self.pose_projection = nn.Linear(3, self.hidden_channels)
        self.relative_pose_projection = nn.Linear(3, self.hidden_channels)
        self.joint_embedding = nn.Parameter(
            torch.zeros(self.sides * self.joints, self.hidden_channels)
        )
        self.activation = nn.GELU()
        self.output = nn.Linear(self.hidden_channels, 2)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias[0] = float(np.log(initial_error_metres / COORDINATE_SCALE_METRES))
            self.output.bias[1] = float(
                np.log(0.5 * initial_error_metres / COORDINATE_SCALE_METRES)
            )

    def forward(self, tokens: Tensor, direct_full24_m: Tensor) -> Tensor:
        batch, frames = tokens.shape[:2]
        _require(
            tokens.shape == (batch, frames, self.token_count, self.token_channels),
            "error head token feature shape differs",
        )
        _require(
            direct_full24_m.shape == (batch, frames, self.sides, self.joints, 3),
            "error head direct pose shape differs",
        )
        pooled = tokens.mean(dim=2)
        feature = self.feature_projection(pooled)[:, :, None, :]
        pose = self.pose_projection(
            direct_full24_m.reshape(batch, frames, self.sides * self.joints, 3)
        )
        wrist = direct_full24_m[:, :, :, WRIST_JOINT : WRIST_JOINT + 1, :]
        relative = self.relative_pose_projection(
            (direct_full24_m - wrist).reshape(batch, frames, self.sides * self.joints, 3)
        )
        hidden = self.activation(feature + pose + relative + self.joint_embedding[None, None])
        return self.output(hidden).reshape(batch, frames, self.sides, self.joints, 2)


def install_error_head(model: Any, *, config: Mapping[str, Any]) -> dict[str, object]:
    block = config.get("error_head")
    _require(isinstance(block, Mapping), "error_head section is missing")
    model_config = config["model"]
    token_count = 19
    token_channels = int(model_config["hidden_channels"])
    head = ErrorEstimationHead(
        token_channels=token_channels,
        token_count=token_count,
        initial_error_metres=float(block["initial_error_metres"]),
    )
    model.error_head = head
    model.error_head_log_estimate = None
    model.error_head_token_feature = None

    def hook(module: Any, inputs: Any, output: Any) -> None:
        tokens = output.normalized_reader_tokens
        _require(int(tokens.shape[-2]) == 19, "reader token feature must be [B,T,19,H]")
        model.error_head_token_feature = tokens.detach()
        model.error_head_log_estimate = model.error_head(tokens, output.direct.full24_m)

    model.decoder.register_forward_hook(hook)
    model.error_head_contract = {
        "feature": "decoder output.normalized_reader_tokens, the [B,T,19,384] tensor also exported",
        "feature_reduction": (
            "mean over the 19 reader tokens per frame, then a shared 384->192 projection"
        ),
        "extra_input": "direct.full24_m (the emitted Direct prediction), per joint 3 -> 192",
        "readout": (
            "per-joint embedding(48,192) + feature + pose + wrist-relative pose, GELU, 192->2"
        ),
        "output": (
            "log error in 0.1 m units, shape [B,T,2,24,2] "
            "(two hands x 24 joints x {absolute, wrist-relative}); mm = exp(output)*100; "
            "the wrist-relative channel is supervised and exported for the 20 hand points only"
        ),
        "parameters": int(sum(parameter.numel() for parameter in head.parameters())),
        "initialisation": (
            f"output layer zero weight, bias log({block['initial_error_metres']}/0.1) absolute "
            f"and log({0.5 * float(block['initial_error_metres'])}/0.1) wrist-relative"
        ),
    }
    return model.error_head_contract


def error_head_loss(
    *,
    log_error: Tensor,
    prediction_full24_m: Tensor,
    target_full24_m: Tensor,
    target_mask: Tensor,
    eps: float = 0.001,
) -> tuple[Tensor, Tensor, Tensor]:
    _require(
        log_error.shape == (*prediction_full24_m.shape[:-1], 2)
        and prediction_full24_m.shape == target_full24_m.shape,
        "error loss shapes differ",
    )
    prediction = prediction_full24_m.detach()
    target = target_full24_m.detach()
    with torch.no_grad():
        absolute = (prediction - target).norm(dim=-1)
        relative = (
            prediction[..., HAND_FIRST_JOINT:, :]
            - prediction[..., WRIST_JOINT : WRIST_JOINT + 1, :]
            - (target[..., HAND_FIRST_JOINT:, :] - target[..., WRIST_JOINT : WRIST_JOINT + 1, :])
        ).norm(dim=-1)
        physical = target_full24_m[..., 2, 1] <= target_full24_m[..., 1, 1]
        weights = (physical & target_mask[:, :, None]).to(dtype=torch.float32)
        absolute_log = torch.log(absolute / COORDINATE_SCALE_METRES + eps)
        relative_log = torch.log(relative / COORDINATE_SCALE_METRES + eps)
    _require(float(weights.sum()) > 0.0, "error head supervision support is empty")
    abs_term = ((log_error[..., 0] - absolute_log).abs() * weights[..., None]).sum() / (
        weights.sum() * float(log_error.shape[-2])
    )
    rel_term = (
        (log_error[..., HAND_FIRST_JOINT:, 1] - relative_log).abs() * weights[..., None]
    ).sum() / (weights.sum() * float(log_error.shape[-2] - HAND_FIRST_JOINT))
    return (0.5 * (abs_term + rel_term), abs_term, rel_term)


def window_metrics(
    prediction_full24_m: Tensor, target_full24_m: Tensor, target_mask: Tensor
) -> Tensor:
    with torch.no_grad():
        error = (prediction_full24_m.detach() - target_full24_m.detach()).norm(dim=-1)
        physical = target_full24_m[..., 2, 1] <= target_full24_m[..., 1, 1]
        weights = (physical & target_mask[:, :, None]).to(dtype=torch.float32)
        relative = (
            prediction_full24_m.detach()[..., HAND_FIRST_JOINT:, :]
            - prediction_full24_m.detach()[..., WRIST_JOINT : WRIST_JOINT + 1, :]
            - (
                target_full24_m.detach()[..., HAND_FIRST_JOINT:, :]
                - target_full24_m.detach()[..., WRIST_JOINT : WRIST_JOINT + 1, :]
            )
        ).norm(dim=-1)
        count = weights.sum(dim=(1, 2))
        full = (error * weights[..., None]).sum(dim=(1, 2, 3)) / (count * error.shape[-1])
        hand = (relative * weights[..., None]).sum(dim=(1, 2, 3)) / (count * relative.shape[-1])
        out = torch.stack((full, hand), dim=1) * 1000.0
        out[count == 0] = float("nan")
        return out


class AugmentedTerms:
    def __init__(self, terms: Any, total: Tensor, auxiliary: Tensor, error: Tensor) -> None:
        self._terms = terms
        self.total = total
        self.auxiliary = auxiliary
        self.error = error

    def __getattr__(self, name: str) -> Any:
        return getattr(self._terms, name)


def mixed_batch_terms(
    *,
    model: Any,
    batch: Mapping[str, Any],
    normalization: Any,
    fk: Any,
    device: Any,
    contribution_weights: Mapping[str, float],
    n_real: int | None = None,
    capture_activations: bool = False,
) -> tuple[Any, Any, Any]:
    from mmprism.training.reconstruction.dense_token_dual_supervision import (
        masked_dual_supervision_loss,
    )
    from mmprism.training.reconstruction.hand_palm_frame_torch import (
        intrinsic_unit_bone_cosine_loss,
    )
    from mmprism.training.reconstruction.trainer import decode_complete_mhr_full24

    raw = batch["raw_radar"].to(device=device, dtype=torch.float32, non_blocking=True)
    temporal = batch["temporal_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    target_mask = batch["target_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    target_global = batch["target_global_pose"].to(device=device, dtype=torch.float32)
    target_state = batch["target_mhr191"].to(device=device, dtype=torch.float32)
    prediction = model(raw, valid_mask=temporal, capture_activations=capture_activations)
    state = prediction.mhr_state.state
    total_rows = int(state.shape[0])
    split = total_rows if n_real is None else int(n_real)
    _require(0 <= split <= total_rows, "mixed real-row count is outside the batch")
    if split == total_rows:
        fk_full24 = decode_complete_mhr_full24(state, temporal_mask=temporal, fk=fk)
    elif split == 0:
        fk_full24 = decode_complete_mhr_full24(
            state, temporal_mask=temporal, fk=fk, fixed_standard_body=True
        )
    else:
        real_decoded = decode_complete_mhr_full24(
            state[:split], temporal_mask=temporal[:split], fk=fk
        )
        sim_decoded = decode_complete_mhr_full24(
            state[split:], temporal_mask=temporal[split:], fk=fk, fixed_standard_body=True
        )
        fk_full24 = torch.cat((real_decoded, sim_decoded), dim=0)
    terms = masked_dual_supervision_loss(
        direct_coarse18_m=prediction.direct.coarse18_m.reshape(*raw.shape[:2], 18),
        direct_hand120_wrist_relative_m=prediction.direct.hand120_wrist_relative_m.reshape(
            *raw.shape[:2], 120
        ),
        fk_decoded_full24_m=fk_full24,
        predicted_mhr191=state,
        target_global_m=target_global,
        target_mhr191=target_state,
        valid_mask=target_mask,
        capacity_frames=int(temporal.shape[1]),
        normalization=normalization,
    )
    auxiliary = intrinsic_unit_bone_cosine_loss(
        prediction.direct.full24_m, target_global, target_mask
    )
    log_error = getattr(model, "module", model).error_head_log_estimate
    _require(log_error is not None, "error head did not run inside the forward pass")
    error, error_abs, error_rel = error_head_loss(
        log_error=log_error,
        prediction_full24_m=prediction.direct.full24_m,
        target_full24_m=target_global,
        target_mask=target_mask,
    )
    base = getattr(model, "module", model)
    base.error_head_log_estimate = None
    base.intrinsic_auxiliary_last = auxiliary.detach()
    base.error_head_last_error = error.detach()
    base.mixed_row_metrics = window_metrics(prediction.direct.full24_m, target_global, target_mask)
    with torch.no_grad():
        estimate = torch.exp(log_error[..., 0].detach()) * COORDINATE_SCALE_METRES * 1000.0
        physical_rows = (target_global[..., 2, 1] <= target_global[..., 1, 1]) & target_mask[
            :, :, None
        ]
        wsum = physical_rows.to(torch.float32)
        base.mixed_row_estimate_mm = (estimate * wsum[..., None]).sum(dim=(1, 2, 3)) / (
            wsum.sum(dim=(1, 2)) * estimate.shape[-1]
        ).clamp_min(1.0)
    base.error_head_last_parts = (error_abs.detach(), error_rel.detach())
    total = (
        terms.total
        + float(contribution_weights["auxiliary_weight"]) * auxiliary
        + float(contribution_weights["error_head_weight"]) * error
    )
    _require(bool(total.isfinite()), "mixed total loss is non-finite")
    return (AugmentedTerms(terms, total, auxiliary, error), prediction, temporal)


BAD_FULL24_MM = 50.0

BAD_HAND_REL_MM = 35.0

SIM_VALIDATION_WINDOWS = 32

SIM_VALIDATION_DRAW_BASE = 3_000_000_000


def summarise_windows(
    values: np.ndarray, *, estimate: np.ndarray | None = None
) -> dict[str, float]:
    """Mean / p90 / badcase share over windows; ``values`` is ``[n, 2]`` (full24, hand-rel).

    A window is a *badcase* when its full24 MPJPE exceeds ``BAD_FULL24_MM`` (hand
    relative: ``BAD_HAND_REL_MM``).  The thresholds are fixed, not tuned.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] == 0:
        return {"n": 0}
    out: dict[str, float] = {
        "n": int(values.shape[0]),
        "full24_mean_mm": float(values[:, 0].mean()),
        "full24_p90_mm": float(np.quantile(values[:, 0], 0.9)),
        "full24_bad_pct": float(100.0 * (values[:, 0] > BAD_FULL24_MM).mean()),
        "hand_rel_mean_mm": float(values[:, 1].mean()),
        "hand_rel_p90_mm": float(np.quantile(values[:, 1], 0.9)),
        "hand_rel_bad_pct": float(100.0 * (values[:, 1] > BAD_HAND_REL_MM).mean()),
    }
    if estimate is not None and len(estimate):
        out["error_head_est_mm"] = float(np.mean(estimate))
    return out


def simulation_validation(
    *, model: Any, stream: Any, device: Any, rank: int, world_size: int, all_reduce: Any
) -> dict[str, float]:
    """Held-out simulated windows (Daily validation split), fixed draw ids, calibrated
    exactly like training windows; direct full24 and wrist-relative hand error per
    window.  Every rank renders and scores its share; the per-window values are
    summed across ranks into one vector so all ranks return the same summary."""
    count = SIM_VALIDATION_WINDOWS
    local = torch.zeros(count, 2, dtype=torch.float64, device=device)
    present = torch.zeros(count, dtype=torch.float64, device=device)
    module = getattr(model, "module", model)
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode():
            for index in range(rank, count, world_size):
                draw = stream.draw(
                    global_draw=SIM_VALIDATION_DRAW_BASE + index,
                    step=SIM_VALIDATION_DRAW_BASE + index,
                    rank=0,
                )
                batch = collate_windows([draw.item])
                raw = batch["raw_radar"].to(device=device, dtype=torch.float32)
                temporal = batch["temporal_mask"].to(device=device, dtype=torch.bool)
                target = batch["target_global_pose"].to(device=device, dtype=torch.float32)
                mask = batch["target_mask"].to(device=device, dtype=torch.bool)
                prediction = module(raw, valid_mask=temporal)
                metrics = window_metrics(prediction.direct.full24_m, target, mask)
                local[index] = metrics[0].to(torch.float64)
                present[index] = 1.0
    except Exception as error:  # noqa: BLE001 - all ranks must reach the collectives below
        print(f"Simulation validation error on rank {rank}: {error!r}", flush=True)
    finally:
        module.train(was_training)
        module.error_head_log_estimate = None
    local = torch.nan_to_num(local, nan=-1.0) * present[:, None]
    all_reduce(local)
    all_reduce(present)
    values = local.cpu().numpy()
    values[values < 0] = np.nan
    return summarise_windows(values[present.cpu().numpy() > 0])
