"""Render independent Daily pose windows with one velocity successor."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from mmprism.data.csl_daily_mhr_windows import (
    CslDailyMhrEpisodeSpec,
    CslDailyMhrMembership,
    CslDailyMhrTrajectory,
    CslDailyMhrWindow,
    build_csl_daily_strict_mhr_membership,
    sample_csl_daily_mhr_window,
)
from mmprism.release.paths import resolve
from mmprism.simulation.mesh_surface import SurfaceSamplingSpec
from mmprism.simulation.mhr_earlyfusion_simulation import (
    MHR_STATE_DIMENSION,
    MhrEarlyFusionComplexSimulator,
    MhrEarlyFusionSurfaceAdapter,
)

ALL_ROLES = ("train", "validation", "test")
WINDOW_FRAMES = 32


class RenderingError(RuntimeError):
    """A rendered window does not satisfy its input or target contract."""


def _path(config, section, name):
    return resolve(config[section][name])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RenderingError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True)
class PreparedWindow:
    window: CslDailyMhrWindow
    raw_iq: torch.Tensor
    target_global: torch.Tensor
    target_state: torch.Tensor
    scene: Any
    signal: Mapping[str, float]
    annotation_read_seconds: float
    simulation_seconds: float


@dataclass(frozen=True)
class ScenePlacement:
    desired_shoulder_midpoint_m: np.ndarray
    rotation_axis_angle_rad: np.ndarray
    identity45: np.ndarray
    morphology28: np.ndarray
    surface_scale: float
    noise_seed: int
    draw_index: int


def _membership(config: Mapping[str, Any]) -> CslDailyMhrMembership:
    data = _mapping(config.get("data"), "data")
    return build_csl_daily_strict_mhr_membership(
        source_root=_path(config, "data", "source_root"),
        split_assignments_path=_path(config, "data", "strict_split_assignments"),
        expected_counts=cast(Mapping[str, int], data["expected_episodes"]),
        allowed_unassigned_sample_ids=cast(Sequence[str], data["allowed_unassigned_sample_ids"]),
    )


def _build_adapter(
    config: Mapping[str, Any], *, device: torch.device
) -> MhrEarlyFusionSurfaceAdapter:
    return MhrEarlyFusionSurfaceAdapter(
        mhr_rig=_path(config, "upstream", "mhr_rig"),
        fk_constants=_path(config, "upstream", "mhr_fk_constants"),
        device=device,
        scene_translation_m=(0.0, 0.0, 0.0),
    )


def _build_simulator(config: Mapping[str, Any]) -> MhrEarlyFusionComplexSimulator:
    simulation = _mapping(config.get("simulation"), "simulation")
    return MhrEarlyFusionComplexSimulator(
        sampling=SurfaceSamplingSpec(
            sample_count=int(simulation["scatter_sample_count"]),
            seed=int(simulation["scatter_sampling_seed"]),
        ),
        radar_position=tuple(float(value) for value in simulation["radar_position_m"]),
        path_chunk_size=int(simulation["complex_path_chunk_size"]),
    )


def _scene_for_draw(config: Mapping[str, Any], *, role: str, draw_index: int) -> ScenePlacement:
    _require(role in ALL_ROLES, "reconstruction scene has unsupported role")
    simulation = _mapping(config.get("simulation"), "simulation")
    anchor = _mapping(simulation.get("shoulder_anchor"), "simulation.shoulder_anchor")
    scatter = _mapping(
        simulation.get("scattering_augmentation"), "simulation.scattering_augmentation"
    )
    noise = _mapping(simulation.get("observation_noise"), "simulation.observation_noise")
    base_seed = int(_mapping(config.get("optimization"), "optimization")["sampler_seed"])
    if role == "train":
        rng = np.random.default_rng(base_seed + 1000003 * (draw_index + 1))
        (low, high) = (float(value) for value in anchor["train_radar_distance_range_m"])
        radius = float(rng.uniform(low, high))
        lateral = float(
            rng.uniform(
                -float(anchor["train_lateral_jitter_m"]), float(anchor["train_lateral_jitter_m"])
            )
        )
        height = float(
            rng.uniform(
                -float(anchor["train_height_jitter_m"]), float(anchor["train_height_jitter_m"])
            )
        )
        forward_squared = radius * radius - lateral * lateral - height * height
        _require(forward_squared > 0.0, "reconstruction shoulder jitter exceeds sampled radius")
        desired = np.asarray((lateral, height, math.sqrt(forward_squared)), dtype=np.float32)
        bound = np.asarray(anchor["train_rotation_axis_angle_jitter_rad"], dtype=np.float32)
        rotation = rng.uniform(-bound, bound).astype(np.float32)
        identity = rng.normal(0.0, float(scatter["identity_std"]), size=45).astype(np.float32)
        morphology = rng.normal(0.0, float(scatter["morphology_std"]), size=28).astype(np.float32)
        (scale_low, scale_high) = (float(value) for value in scatter["surface_scale_range"])
        scale = float(rng.uniform(scale_low, scale_high))
    else:
        radius = float(anchor["canonical_radar_distance_m"])
        lateral = float(anchor["canonical_lateral_m"])
        height = float(anchor["canonical_height_m"])
        forward = math.sqrt(radius * radius - lateral * lateral - height * height)
        desired = np.asarray((lateral, height, forward), dtype=np.float32)
        rotation = np.zeros(3, dtype=np.float32)
        identity = np.zeros(45, dtype=np.float32)
        morphology = np.zeros(28, dtype=np.float32)
        scale = 1.0
    return ScenePlacement(
        desired_shoulder_midpoint_m=desired,
        rotation_axis_angle_rad=rotation,
        identity45=identity,
        morphology28=morphology,
        surface_scale=scale,
        noise_seed=int(noise["seed"]) + 1000003 * (draw_index + 1),
        draw_index=draw_index,
    )


def _apply_observation_noise(
    raw_iq: torch.Tensor, *, relative_rms: float, seed: int
) -> torch.Tensor:
    _require(
        raw_iq.dtype == torch.float32 and raw_iq.shape[-1] == 2, "reconstruction raw I/Q differs"
    )
    _require(relative_rms >= 0.0, "reconstruction observation noise differs")
    if relative_rms == 0.0:
        return raw_iq
    generator = torch.Generator(device=raw_iq.device)
    generator.manual_seed(seed)
    rms = raw_iq.square().mean().sqrt()
    return raw_iq + torch.randn(
        raw_iq.shape, generator=generator, device=raw_iq.device, dtype=raw_iq.dtype
    ) * (rms * relative_rms)


def _materialize_window(
    config: Mapping[str, Any],
    *,
    window: CslDailyMhrWindow,
    scene: Any,
    adapter: MhrEarlyFusionSurfaceAdapter,
    simulator: MhrEarlyFusionComplexSimulator,
    device: torch.device,
    annotation_read_seconds: float,
) -> PreparedWindow:
    simulation = _mapping(config.get("simulation"), "simulation")
    noise = _mapping(simulation.get("observation_noise"), "simulation.observation_noise")
    started = time.perf_counter()
    scene_rotation = torch.from_numpy(scene.rotation_axis_angle_rad).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        targets = adapter.canonical_targets(
            body_pose133=torch.from_numpy(window.body_target133).to(device=device),
            hand_pose108=torch.from_numpy(window.hand_target108).to(device=device),
            global_rotation3=torch.from_numpy(window.global_target3).to(device=device),
            source_frame_indices=torch.from_numpy(window.source_frame_indices).to(device=device),
            scene_rotation_axis_angle_rad=scene_rotation,
            shoulder_midpoint_target_m=torch.from_numpy(scene.desired_shoulder_midpoint_m).to(
                device=device, dtype=torch.float32
            ),
        )
        camera_translation = targets.state191[0, 159:162]
        _require(
            bool(torch.allclose(targets.state191[:, 159:162], camera_translation[None])),
            "reconstruction MHR camera translation changed inside selected window",
        )
        surfaces = adapter.scattering_vertices(
            body_pose133=torch.from_numpy(window.body_surface133).to(device=device),
            hand_pose108=torch.from_numpy(window.hand_surface108).to(device=device),
            global_rotation3=torch.from_numpy(window.global_surface3).to(device=device),
            identity45=torch.from_numpy(scene.identity45).to(device=device),
            morphology28=torch.from_numpy(scene.morphology28).to(device=device),
            scene_translation_m=targets.scene_translation_m,
            scene_rotation_axis_angle_rad=scene_rotation,
            surface_scale=scene.surface_scale,
        )
        radar = simulator.synthesize_sequence(
            vertices=surfaces,
            faces=adapter.faces,
            delta_t_s=1.0 / float(_mapping(config.get("data"), "data")["target_fps"]),
            hand_face_mask=adapter.hand_face_mask,
            apply_temporal_static_subtraction=False,
        )
        raw = _apply_observation_noise(
            radar.raw_iq,
            relative_rms=float(noise["relative_rms_per_component"]),
            seed=int(scene.noise_seed),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    raw_cpu = raw.detach().cpu()
    target_global = targets.full24_m.detach().cpu()
    target_state = targets.state191.detach().cpu()
    _require(
        raw_cpu.shape[0] == target_global.shape[0] == target_state.shape[0] == window.valid_frames,
        "reconstruction simulated window alignment differs",
    )
    _require(
        radar.generated_physical_frames == window.valid_frames,
        "reconstruction window rendered frames differ",
    )
    _require(
        radar.range_diagnostics["temporal_static_subtraction_applied"] == 0.0,
        "reconstruction simulator applied forbidden static subtraction",
    )
    hand_distance = target_global[:, :, 4:].norm(dim=-1)
    return PreparedWindow(
        window=window,
        raw_iq=raw_cpu.pin_memory() if torch.cuda.is_available() else raw_cpu,
        target_global=target_global.pin_memory() if torch.cuda.is_available() else target_global,
        target_state=target_state.pin_memory() if torch.cuda.is_available() else target_state,
        scene=scene,
        signal={
            "range_rms": radar.range_rms,
            "visible_sample_fraction": radar.visible_sample_fraction,
            "hand_sample_fraction": radar.hand_sample_fraction,
            "visible_hand_sample_fraction": radar.visible_hand_sample_fraction,
            "target_hand_distance_min_m": float(hand_distance.min().item()),
            "target_hand_distance_p95_m": float(torch.quantile(hand_distance, 0.95).item()),
            "target_hand_distance_max_m": float(hand_distance.max().item()),
            **radar.range_diagnostics,
        },
        annotation_read_seconds=annotation_read_seconds,
        simulation_seconds=elapsed,
    )


def _prepared_window_batch(
    windows: Sequence[PreparedWindow], *, device: torch.device
) -> tuple[dict[str, torch.Tensor], float]:
    _require(windows, "reconstruction batch has no windows")
    started = time.perf_counter()
    raw_rows: list[torch.Tensor] = []
    pose_rows: list[torch.Tensor] = []
    state_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for item in windows:
        valid = item.window.valid_frames
        raw = torch.zeros((WINDOW_FRAMES, 64, 116, 32, 2), dtype=torch.float32)
        pose = torch.zeros((WINDOW_FRAMES, 2, 24, 3), dtype=torch.float32)
        state = torch.zeros((WINDOW_FRAMES, MHR_STATE_DIMENSION), dtype=torch.float32)
        mask = torch.zeros(WINDOW_FRAMES, dtype=torch.bool)
        raw[:valid] = item.raw_iq
        pose[:valid] = item.target_global
        state[:valid] = item.target_state
        mask[:valid] = True
        raw_rows.append(raw)
        pose_rows.append(pose)
        state_rows.append(state)
        mask_rows.append(mask)
    batch = {
        "raw_radar": torch.stack(raw_rows).to(device=device, non_blocking=True),
        "target_global": torch.stack(pose_rows).to(device=device, non_blocking=True),
        "target_state": torch.stack(state_rows).to(device=device, non_blocking=True),
        "mask": torch.stack(mask_rows).to(device=device, non_blocking=True),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (batch, time.perf_counter() - started)


def _spec_for_global_draw(
    specs: Sequence[CslDailyMhrEpisodeSpec], *, sampler_seed: int, global_draw: int
) -> CslDailyMhrEpisodeSpec:
    _require(specs and global_draw >= 0, "reconstruction source draw is invalid")
    (cycle, position) = divmod(global_draw, len(specs))
    order = np.random.default_rng(sampler_seed + cycle).permutation(len(specs))
    return specs[int(order[position])]


def _window_for_draw(
    trajectory: CslDailyMhrTrajectory,
    *,
    sampler_seed: int,
    global_draw: int,
    source_frame_stride: int,
) -> CslDailyMhrWindow:
    return sample_csl_daily_mhr_window(
        trajectory,
        window_frames=WINDOW_FRAMES,
        random_seed=sampler_seed + 10000019 * (global_draw + 1),
        source_frame_stride=source_frame_stride,
    )
