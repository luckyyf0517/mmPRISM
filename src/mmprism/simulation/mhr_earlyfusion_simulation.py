"""Surface-based complex echo synthesis from articulated body poses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from torch import Tensor

from mmprism.models.mhr_param_fk import MhrParamFK, map_keypoints_70_to_full24_torch
from mmprism.simulation.mesh_surface import MeshSurfacePathBackend, SurfaceSamplingSpec
from mmprism.simulation.path_batch import FmcwPathBatch
from mmprism.simulation.radar_config import (
    ACTIVE_MMWCAS_TDM_CONFIG_ID,
    RadarConfig,
    get_radar_config,
)
from mmprism.simulation.simulator import (
    MEASURED_LINK_AMPLITUDE_CALIBRATION_V1,
    SIMULATION_LIGHT_SPEED,
)
from mmprism.training.reconstruction.dense_token_deterministic_mhr import UPPER_BODY_INDICES

MHR_STATE_DIMENSION: Final[int] = 191


class MhrEarlyFusionSimulationError(RuntimeError):
    """Raised when the MHR-to-complex-EarlyFusion simulation contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MhrEarlyFusionSimulationError(message)


_COMPILE_COMPLEX_PATH_KERNEL = True

_ComplexPathComponents = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, float], tuple[Tensor, Tensor]
]


def _complex_path_components(
    amplitude: Tensor,
    delay_s: Tensor,
    radial_mps: Tensor,
    time_steps: Tensor,
    fs: Tensor,
    start_freq: float,
) -> tuple[Tensor, Tensor]:
    """Pure real-valued compute region; validation stays in the eager wrapper."""

    tau_velocity = radial_mps * 2.0 / SIMULATION_LIGHT_SPEED
    tau_chirp = tau_velocity[:, None] * time_steps[None, :, None, None]
    tau_chirp = (delay_s[:, None] + tau_chirp)[:, :, :, None, :]
    phase = 2.0 * np.pi * (fs[None, None, None, :, None] + start_freq) * tau_chirp
    phase = torch.remainder(phase, 2.0 * np.pi)
    weighted = amplitude[:, None, :, None, :]
    return (weighted * phase.cos()).sum(dim=-1), (weighted * phase.sin()).sum(dim=-1)


@lru_cache(maxsize=1)
def _compiled_complex_path_components() -> _ComplexPathComponents:
    """Compile once on first CUDA use, never at import or on CPU."""

    return cast(
        _ComplexPathComponents,
        torch.compile(_complex_path_components, fullgraph=True, dynamic=True),
    )


def _synthesize_complex_path_tensors_unchecked(
    amplitude: Tensor,
    delay_s: Tensor,
    radial_mps: Tensor,
    time_steps: Tensor,
    fs: Tensor,
    start_freq: float,
) -> Tensor:
    """Inner reduction; the caller validates the complete tensors once.

    Per-chunk finite checks force device-to-host synchronization after every
    reduction. The public tensor entry point keeps its guards; the bounded
    synthesizer instead checks inputs before and the final output after its loop.
    """

    if _COMPILE_COMPLEX_PATH_KERNEL and amplitude.is_cuda:
        real, imag = _compiled_complex_path_components()(
            amplitude, delay_s, radial_mps, time_steps, fs, start_freq
        )
        values = torch.complex(real, imag)
    else:
        tau_velocity = radial_mps * 2.0 / SIMULATION_LIGHT_SPEED
        tau_chirp = tau_velocity[:, None] * time_steps[None, :, None, None]
        tau_chirp = (delay_s[:, None] + tau_chirp)[:, :, :, None, :]
        phase = 2.0 * np.pi * (fs[None, None, None, :, None] + start_freq) * tau_chirp
        phase = torch.remainder(phase, 2.0 * np.pi)
        values = torch.polar(amplitude[:, None, :, None, :], phase).sum(dim=-1)
    return values


class ComplexFmcwPathSynthesizer:
    """Memory-bounded complex FMCW synthesis for one or more path batches."""

    def __init__(
        self,
        radar_config: RadarConfig | None = None,
        *,
        path_chunk_size: int = 32,
        amplitude_link_scale: float = MEASURED_LINK_AMPLITUDE_CALIBRATION_V1,
    ) -> None:
        _require(path_chunk_size > 0, "path_chunk_size must be positive")
        _require(np.isfinite(amplitude_link_scale) and amplitude_link_scale > 0.0, "link scale")
        self.radar_config = (
            radar_config
            if radar_config is not None
            else get_radar_config(ACTIVE_MMWCAS_TDM_CONFIG_ID)
        )
        self.path_chunk_size = int(path_chunk_size)
        self.amplitude_link_scale = float(amplitude_link_scale)
        self.start_freq = float(self.radar_config.start_freq)

    def _axes(self, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        fs = (
            self.radar_config.adc_start_time
            + torch.arange(self.radar_config.num_adc_samples, dtype=dtype, device=device)
            / self.radar_config.adc_sample_rate
        ) * self.radar_config.freq_slope
        time_steps = (
            torch.arange(self.radar_config.num_chirps, dtype=dtype, device=device)
            * self.radar_config.chirp_period
        )
        return time_steps, fs

    def _compact_active_paths(
        self, amplitude: Tensor, delay_s: Tensor, radial_mps: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Drop surface samples whose amplitude is exactly zero on every channel.

        A hidden or area-degenerate sample carries an exact zero amplitude, so
        removing it changes the path sum by exactly nothing while shrinking the
        synthesis to the visible subset.  A running count assigns each kept path
        its own destination, so two paths are never merged and the reduction sees
        the same terms; only the chunk boundaries implied by the shorter axis
        differ.  The returned axis is the widest per-batch active count, and the
        padded slots hold exact zeros.
        """

        _require(
            amplitude.shape == delay_s.shape == radial_mps.shape,
            "path tensors differ before compaction",
        )
        active = amplitude.amax(dim=1).gt(0.0)
        capacity = int(active.sum(dim=1).max().item())
        if capacity >= amplitude.shape[-1]:
            return amplitude, delay_s, radial_mps
        shape = (amplitude.shape[0], amplitude.shape[1], capacity)
        compacted_amplitude = torch.zeros(shape, dtype=amplitude.dtype, device=amplitude.device)
        compacted_delay = torch.zeros_like(compacted_amplitude)
        compacted_radial = torch.zeros_like(compacted_amplitude)
        if capacity == 0:  # Every path is hidden; the reduction is exactly zero.
            return compacted_amplitude, compacted_delay, compacted_radial
        destinations = (active.cumsum(dim=1) - 1).clamp_(min=0, max=capacity - 1)
        indices = destinations[:, None, :].expand(-1, amplitude.shape[1], -1)
        mask = active[:, None, :].to(dtype=amplitude.dtype)
        compacted_amplitude.scatter_add_(2, indices, amplitude * mask)
        compacted_delay.scatter_add_(2, indices, delay_s * mask)
        compacted_radial.scatter_add_(2, indices, radial_mps * mask)
        return compacted_amplitude, compacted_delay, compacted_radial

    def synthesize_path_batch(self, paths: FmcwPathBatch) -> Tensor:
        """Return complex ``[B,64,116,num_adc_samples]`` raw FMCW frames."""

        _require(paths.receiver_count == 116, "complex synthesis requires the full array")
        expected = (
            paths.batch_size,
            self.radar_config.num_chirps,
            116,
            self.radar_config.num_adc_samples,
        )
        amplitude, delay_s, radial_mps = self._compact_active_paths(
            paths.amplitude * self.amplitude_link_scale, paths.delay_s, paths.radial_mps
        )
        if amplitude.shape[-1] == 0:
            return torch.zeros(expected, dtype=torch.complex64, device=amplitude.device)
        _require(bool(torch.isfinite(amplitude).all()), "path amplitude is non-finite")
        _require(bool(torch.isfinite(delay_s).all()), "path delay is non-finite")
        _require(bool(torch.isfinite(radial_mps).all()), "path radial velocity is non-finite")
        time_steps, fs = self._axes(device=amplitude.device, dtype=amplitude.dtype)
        # Compact and validate once for the whole batch, not once per frame
        # tile. No scalar device reads occur inside either reduction loop.
        # ``path_chunk_size`` still bounds the tile's transient phase plane:
        # [tile_frames, 64, 116, samples, path_chunk].
        raw_tiles: list[Tensor] = []
        for first in range(0, paths.batch_size, 4):
            last = min(first + 4, paths.batch_size)
            chunk = max(1, self.path_chunk_size // (last - first))
            tile_output: Tensor | None = None
            for start in range(0, amplitude.shape[-1], chunk):
                stop = min(start + chunk, amplitude.shape[-1])
                current = _synthesize_complex_path_tensors_unchecked(
                    amplitude[first:last, :, start:stop],
                    delay_s[first:last, :, start:stop],
                    radial_mps[first:last, :, start:stop],
                    time_steps,
                    fs,
                    self.start_freq,
                )
                tile_output = current if tile_output is None else tile_output + current
            if tile_output is None:  # The empty active set returns above.
                raise MhrEarlyFusionSimulationError("non-empty paths yielded no complex raw output")
            raw_tiles.append(tile_output)
        output = raw_tiles[0] if len(raw_tiles) == 1 else torch.cat(raw_tiles)
        _require(output.shape == expected, "complex FMCW output shape differs")
        _require(bool(torch.isfinite(output.real).all()), "complex FMCW real part is non-finite")
        _require(
            bool(torch.isfinite(output.imag).all()), "complex FMCW imaginary part is non-finite"
        )
        return output


def range_fft_complex_adc(raw_complex: Tensor, *, range_bin_count: int | None = 32) -> Tensor:
    """Apply ADC preprocessing and optionally retain the full range spectrum.

    The current model receives only bins ``[0, 32)``. The simulator keeps all
    100 bins until optional temporal processing is complete so a scene
    placement error cannot be hidden by an early crop.
    """

    _require(torch.is_complex(raw_complex), "raw complex radar must use a complex dtype")
    _require(raw_complex.ndim == 4 and raw_complex.shape[1:3] == (64, 116), "raw layout")
    samples = int(raw_complex.shape[-1])
    _require(range_bin_count is None or 0 < range_bin_count <= samples, "range crop differs")
    demeaned = raw_complex - raw_complex.mean(dim=-1, keepdim=True)
    window = torch.from_numpy(np.hanning(samples + 2)[1:-1].astype(np.float32)).to(
        device=raw_complex.device, dtype=raw_complex.real.dtype
    )
    spectrum = torch.fft.fft(demeaned * window.view(1, 1, 1, -1), dim=-1)
    result = spectrum if range_bin_count is None else spectrum[..., :range_bin_count]
    _require(bool(torch.isfinite(result.real).all()), "range FFT real output is non-finite")
    _require(bool(torch.isfinite(result.imag).all()), "range FFT imaginary output is non-finite")
    return result


def subtract_episode_static(range_frames: Tensor) -> Tensor:
    """Subtract one complex background across a supplied continuous sequence.

    Mixed training transfers window-local background through its calibration
    function; this helper supplies sequence-wide mean subtraction when requested.
    """

    _require(torch.is_complex(range_frames), "range input must be complex")
    _require(
        range_frames.ndim == 4
        and range_frames.shape[1:3] == (64, 116)
        and range_frames.shape[-1] > 0,
        "range layout",
    )
    _require(
        range_frames.shape[0] >= 2,
        "sequence needs two physical frames for static subtraction",
    )
    result = range_frames - range_frames.mean(dim=0, keepdim=True)
    _require(bool(torch.isfinite(result.real).all()), "static-subtracted real output is non-finite")
    _require(
        bool(torch.isfinite(result.imag).all()), "static-subtracted imaginary output is non-finite"
    )
    return result


@dataclass(frozen=True)
class MhrEarlyFusionEpisodeTargets:
    """One standard-body trajectory and its complete MHR supervision state."""

    state191: Tensor
    full24_m: Tensor
    scene_translation_m: Tensor
    source_frame_indices: Tensor


@dataclass(frozen=True)
class MhrEarlyFusionRadarEpisode:
    """Transient complex range input and compact facts from one source sequence."""

    raw_iq: Tensor
    visible_sample_fraction: float
    hand_sample_fraction: float
    visible_hand_sample_fraction: float
    range_rms: float
    generated_physical_frames: int
    range_diagnostics: dict[str, float]


@dataclass(frozen=True)
class MhrEarlyFusionPairRadarBatch:
    """Per-item complex range input and compact facts for independent pairs."""

    raw_iq: Tensor
    visible_sample_fraction: Tensor
    hand_sample_fraction: Tensor
    visible_hand_sample_fraction: Tensor
    range_rms: Tensor
    generated_physical_frames: int
    range_diagnostics: tuple[dict[str, float], ...]


def _axis_angle_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    """Return row-coordinate rotation matrices for finite ``[..., 3]`` values."""

    _require(axis_angle.shape[-1] == 3 and bool(torch.isfinite(axis_angle).all()), "axis angle")
    values = axis_angle.to(dtype=torch.float32)
    angle = values.norm(dim=-1, keepdim=True)
    squared = angle.square()
    sine_over_angle = torch.where(
        angle > 1.0e-5,
        torch.sin(angle) / angle,
        1.0 - squared / 6.0 + squared.square() / 120.0,
    )
    one_minus_cosine_over_angle_squared = torch.where(
        angle > 1.0e-5,
        (1.0 - torch.cos(angle)) / squared.clamp_min(1.0e-20),
        0.5 - squared / 24.0 + squared.square() / 720.0,
    )
    x, y, z = values.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )
    identity = torch.eye(3, dtype=values.dtype, device=values.device).expand(
        values.shape[:-1] + (3, 3)
    )
    return (
        identity
        + sine_over_angle[..., None] * skew
        + (one_minus_cosine_over_angle_squared[..., None] * (skew @ skew))
    )


def _rotation_matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert a proper rotation matrix to axis angle without a SciPy dependency."""

    _require(matrix.shape[-2:] == (3, 3) and bool(torch.isfinite(matrix).all()), "rotation")
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    angle = torch.acos(cosine)
    vector = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    result = vector * (angle / (2.0 * sine).clamp_min(1.0e-8))[..., None]
    small = angle <= 1.0e-4
    result = torch.where(small[..., None], vector * 0.5, result)
    _require(bool(torch.isfinite(result).all()), "axis angle conversion is non-finite")
    return result


def _mhr_xyz_euler_to_rotation_matrix(euler_xyz: Tensor) -> Tensor:
    """Convert MHR's global ``xyz`` parameter vector to rotation matrices.

    The MHR parameter/FK interface consumes ``[x, y, z]`` in the equivalent
    extrinsic-``xyz`` order ``Rz(z) @ Ry(y) @ Rx(x)``.  Scene placement itself
    remains a separate axis-angle rigid transform.
    """

    _require(
        euler_xyz.shape[-1] == 3 and bool(torch.isfinite(euler_xyz).all()),
        "MHR xyz Euler rotation",
    )
    x, y, z = euler_xyz.to(dtype=torch.float32).unbind(dim=-1)
    cosine_z, sine_z = torch.cos(z), torch.sin(z)
    cosine_y, sine_y = torch.cos(y), torch.sin(y)
    cosine_x, sine_x = torch.cos(x), torch.sin(x)
    return torch.stack(
        (
            torch.stack(
                (
                    cosine_z * cosine_y,
                    cosine_z * sine_y * sine_x - sine_z * cosine_x,
                    cosine_z * sine_y * cosine_x + sine_z * sine_x,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    sine_z * cosine_y,
                    sine_z * sine_y * sine_x + cosine_z * cosine_x,
                    sine_z * sine_y * cosine_x - cosine_z * sine_x,
                ),
                dim=-1,
            ),
            torch.stack((-sine_y, cosine_y * sine_x, cosine_y * cosine_x), dim=-1),
        ),
        dim=-2,
    )


def _rotation_matrix_to_mhr_xyz_euler(matrix: Tensor) -> Tensor:
    """Return one finite MHR ``xyz`` parameter vector for ``matrix``."""

    _require(matrix.shape[-2:] == (3, 3) and bool(torch.isfinite(matrix).all()), "rotation")
    y = torch.asin((-matrix[..., 2, 0]).clamp(min=-1.0, max=1.0))
    cosine_y = torch.cos(y)
    non_singular = cosine_y.abs() > 1.0e-6
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    # At gimbal lock, use the equivalent representation with the final X
    # angle set to zero. The active placement range is far from this branch, but
    # retaining a finite representation keeps the accepted MHR interface total.
    z_singular = torch.atan2(-matrix[..., 0, 1], matrix[..., 1, 1])
    z = torch.where(non_singular, z, z_singular)
    x = torch.where(non_singular, x, torch.zeros_like(x))
    result = torch.stack((x, y, z), dim=-1)
    _require(bool(torch.isfinite(result).all()), "MHR xyz Euler conversion is non-finite")
    return result


def _compose_mhr_global_rotation_for_scene(
    global_rotation3: Tensor,
    *,
    scene_rotation: Tensor,
    axis_flip: Tensor,
) -> Tensor:
    """Express a physical scene rotation in MHR's reflected model axes."""

    _require(global_rotation3.ndim == 2 and global_rotation3.shape[1] == 3, "global rotation")
    _require(scene_rotation.shape == (3, 3), "scene rotation differs")
    _require(axis_flip.shape == (3,), "axis flip differs")
    physical_flip = torch.diag(axis_flip.to(device=global_rotation3.device, dtype=torch.float32))
    model_scene_rotation = physical_flip @ scene_rotation @ physical_flip
    original = _mhr_xyz_euler_to_rotation_matrix(global_rotation3)
    return _rotation_matrix_to_mhr_xyz_euler(model_scene_rotation[None] @ original)


def transform_global_rotation_for_scene_per_item(
    global_rotation3: Tensor,
    *,
    scene_rotation_axis_angle_rad: Tensor,
    axis_flip: Tensor,
) -> Tensor:
    """Compose one independent physical scene rotation per item.

    ``global_rotation3`` and ``scene_rotation_axis_angle_rad`` are ``[B,3]`` and
    the result is ``[B,3]``.  For a batch whose rows share one rotation this is
    identical to :func:`transform_global_rotation_for_scene`.
    """

    _require(global_rotation3.ndim == 2 and global_rotation3.shape[1] == 3, "global rotation")
    _require(
        scene_rotation_axis_angle_rad.shape == global_rotation3.shape,
        "per-item scene axis angle differs",
    )
    _require(axis_flip.shape == (3,), "axis flip differs")
    scene_rotation = _axis_angle_to_rotation_matrix(scene_rotation_axis_angle_rad)
    physical_flip = torch.diag(axis_flip.to(device=global_rotation3.device, dtype=torch.float32))
    model_scene_rotation = physical_flip[None] @ scene_rotation @ physical_flip[None]
    original = _mhr_xyz_euler_to_rotation_matrix(global_rotation3)
    return _rotation_matrix_to_mhr_xyz_euler(model_scene_rotation @ original)


class MhrEarlyFusionSurfaceAdapter:
    """Decode MHR parameters to one standard target and a full surface.

    The MHR TorchScript body is frozen and only used to prepare simulator
    inputs and targets. Training gradients begin at the radar encoder.
    """

    def __init__(
        self,
        *,
        mhr_rig: Path | str,
        fk_constants: Path | str,
        device: torch.device | str,
        scene_translation_m: tuple[float, float, float],
        scene_rotation_axis_angle_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.device = torch.device(device)
        self.fk = MhrParamFK(mhr_rig, fk_constants, device=self.device)
        self.scene_translation = torch.tensor(
            scene_translation_m, dtype=torch.float32, device=self.device
        )
        _require(self.scene_translation.shape == (3,), "scene translation must have three values")
        _require(bool(torch.isfinite(self.scene_translation).all()), "scene translation non-finite")
        self.scene_rotation_axis_angle = torch.tensor(
            scene_rotation_axis_angle_rad, dtype=torch.float32, device=self.device
        )
        _require(
            self.scene_rotation_axis_angle.shape == (3,), "scene rotation must have three values"
        )
        self.scene_rotation = _axis_angle_to_rotation_matrix(self.scene_rotation_axis_angle)
        self.faces = self.fk.mhr.character_torch.mesh.faces.to(
            device=self.device, dtype=torch.int64
        )
        self.hand_face_mask = self._build_hand_face_mask()

    def _scene_parameters(
        self,
        *,
        scene_translation_m: Tensor | None,
        scene_rotation_axis_angle_rad: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        translation = (
            self.scene_translation
            if scene_translation_m is None
            else scene_translation_m.to(device=self.device, dtype=torch.float32)
        )
        axis_angle = (
            self.scene_rotation_axis_angle
            if scene_rotation_axis_angle_rad is None
            else scene_rotation_axis_angle_rad.to(device=self.device, dtype=torch.float32)
        )
        _require(
            translation.shape == (3,) and bool(torch.isfinite(translation).all()),
            "scene translation differs",
        )
        _require(
            axis_angle.shape == (3,) and bool(torch.isfinite(axis_angle).all()),
            "scene rotation differs",
        )
        return translation, _axis_angle_to_rotation_matrix(axis_angle)

    def _scene_adjusted_global_rotation(
        self, global_rotation3: Tensor, *, scene_rotation: Tensor
    ) -> Tensor:
        """Express physical scene rotation in the MHR model's reflected axes."""

        return _compose_mhr_global_rotation_for_scene(
            global_rotation3,
            scene_rotation=scene_rotation,
            axis_flip=self.fk.flip,
        )

    def _build_hand_face_mask(self) -> Tensor:
        """Classify faces whose vertices are hand-rig influenced for reporting."""

        lbs = self.fk.mhr.character_torch.linear_blend_skinning
        vertices = lbs.vert_indices_flattened.to(device=self.device, dtype=torch.long)
        joints = lbs.skin_indices_flattened.to(device=self.device, dtype=torch.long)
        weights = lbs.skin_weights_flattened.to(device=self.device)
        hand_joints = ((joints >= 42) & (joints <= 64)) | ((joints >= 78) & (joints <= 100))
        hand_vertex = torch.zeros(18439, dtype=torch.bool, device=self.device)
        hand_vertex[vertices[hand_joints & (weights > 0.25)]] = True
        result = hand_vertex[self.faces].all(dim=1)
        _require(bool(result.any()), "MHR full surface has no hand faces")
        return result

    def _model_parameters(
        self,
        *,
        body_pose133: Tensor,
        hand_pose108: Tensor,
        global_rotation3: Tensor,
        morphology28: Tensor,
    ) -> Tensor:
        """Construct the official MHR 204-D body parameter vector."""

        frames = int(body_pose133.shape[0])
        _require(body_pose133.shape == (frames, 133), "body pose layout differs")
        _require(hand_pose108.shape == (frames, 108), "hand pose layout differs")
        _require(global_rotation3.shape == (frames, 3), "global rotation layout differs")
        _require(morphology28.shape == (frames, 28), "morphology layout differs")
        body = torch.zeros((frames, 133), dtype=torch.float32, device=self.device)
        body = body.index_copy(
            1, self.fk.upper_body_dims, body_pose133[:, list(UPPER_BODY_INDICES)]
        )
        scales = self.fk.scale_mean[None] + morphology28 @ self.fk.scale_comps
        translation = torch.zeros((frames, 3), dtype=torch.float32, device=self.device)
        full_pose = torch.cat((translation, global_rotation3, body[:, :130]), dim=1)
        hand_cont = torch.cat((hand_pose108[:, :54], hand_pose108[:, 54:]), dim=0)
        hand_model = self.fk._hand_cont_to_model(
            self.fk.hand_pose_mean[None] + hand_cont @ self.fk.hand_pose_comps
        )
        full_pose = full_pose.index_copy(1, self.fk.hand_joint_idxs_left, hand_model[:frames])
        full_pose = full_pose.index_copy(1, self.fk.hand_joint_idxs_right, hand_model[frames:])
        result = torch.cat((full_pose, scales), dim=1)
        _require(result.shape == (frames, 204), "MHR model parameter width differs")
        _require(bool(torch.isfinite(result).all()), "MHR model parameters non-finite")
        return result

    def canonical_targets(
        self,
        *,
        body_pose133: Tensor,
        hand_pose108: Tensor,
        global_rotation3: Tensor,
        source_frame_indices: Tensor,
        scene_translation_m: Tensor | None = None,
        scene_rotation_axis_angle_rad: Tensor | None = None,
        shoulder_midpoint_target_m: Tensor | None = None,
    ) -> MhrEarlyFusionEpisodeTargets:
        """Create standard-body targets in the fixed radar scene.

        The standard morphology is the maintained MHR mean (28 zero
        coefficients) and the standard-person lambda is exactly one.  Source
        video camera translation is intentionally not reused as a radar scene
        placement; ``scene_translation`` is the explicit simulation scene.
        """

        frames = int(body_pose133.shape[0])
        morphology = torch.zeros((frames, 28), dtype=torch.float32, device=self.device)
        lambda_values = torch.ones((frames, 1), dtype=torch.float32, device=self.device)
        upper = body_pose133[:, list(UPPER_BODY_INDICES)]
        scene_translation, scene_rotation = self._scene_parameters(
            scene_translation_m=scene_translation_m,
            scene_rotation_axis_angle_rad=scene_rotation_axis_angle_rad,
        )
        source_points70 = self.fk.decode_points70_complete(
            hand_pose108, upper, global_rotation3, morphology
        )
        source_full24 = map_keypoints_70_to_full24_torch(source_points70)
        if shoulder_midpoint_target_m is not None:
            desired = shoulder_midpoint_target_m.to(device=self.device, dtype=torch.float32)
            _require(
                desired.shape == (3,) and bool(torch.isfinite(desired).all()),
                "shoulder midpoint target differs",
            )
            scene_translation = desired - (
                source_full24[:, :, 0].mean(dim=(0, 1)) @ scene_rotation.transpose(0, 1)
            )
        adjusted_global_rotation = self._scene_adjusted_global_rotation(
            global_rotation3, scene_rotation=scene_rotation
        )
        adjusted_points70 = self.fk.decode_points70_complete(
            hand_pose108, upper, adjusted_global_rotation, morphology
        )
        full24 = map_keypoints_70_to_full24_torch(adjusted_points70)
        expected_shoulder = (
            source_full24[:, :, 0].mean(dim=(0, 1)) @ scene_rotation.transpose(0, 1)
            + scene_translation
        )
        # MHR applies its global Euler state about a fixed model root.  A
        # scene rotation changes the equivalent parameter-space pivot, so the
        # state camera translation differs from the physical scene translation
        # used for the scattering surface.
        camera_translation = expected_shoulder - full24[:, :, 0].mean(dim=(0, 1))
        state = torch.cat(
            (
                upper,
                hand_pose108[:, :54],
                hand_pose108[:, 54:],
                adjusted_global_rotation,
                camera_translation[None].expand(frames, -1),
                morphology,
                lambda_values,
            ),
            dim=1,
        )
        _require(state.shape == (frames, MHR_STATE_DIMENSION), "standard MHR state differs")
        full24 = full24 + camera_translation[None, None]
        _require(full24.shape == (frames, 2, 24, 3), "canonical full24 layout differs")
        _require(bool(torch.isfinite(full24).all()), "canonical target non-finite")
        return MhrEarlyFusionEpisodeTargets(
            state191=state,
            full24_m=full24,
            scene_translation_m=scene_translation,
            source_frame_indices=source_frame_indices.to(device=self.device, dtype=torch.long),
        )

    def _item_scene_rows(self, value: Tensor, *, items: int, width: int, name: str) -> Tensor:
        """Broadcast one declared row or accept one row per item."""

        tensor = value.to(device=self.device, dtype=torch.float32)
        if tensor.shape == (width,):
            tensor = tensor.expand(items, width)
        _require(
            tensor.shape == (items, width) and bool(torch.isfinite(tensor).all()),
            f"{name} differs",
        )
        return tensor

    def _item_adjusted_global_rotation(
        self, global_rotation3: Tensor, *, scene_rotation: Tensor
    ) -> Tensor:
        """Express one independent physical scene rotation per item."""

        _require(
            scene_rotation.shape == (global_rotation3.shape[0], 3, 3),
            "per-item scene rotation differs",
        )
        axis_angle = _rotation_matrix_to_axis_angle(scene_rotation)
        return transform_global_rotation_for_scene_per_item(
            global_rotation3,
            scene_rotation_axis_angle_rad=axis_angle,
            axis_flip=self.fk.flip,
        )

    def canonical_targets_per_item(
        self,
        *,
        body_pose133: Tensor,
        hand_pose108: Tensor,
        global_rotation3: Tensor,
        source_frame_indices: Tensor,
        shoulder_midpoint_target_m: Tensor,
        scene_rotation_axis_angle_rad: Tensor,
    ) -> MhrEarlyFusionEpisodeTargets:
        """Create one independently placed standard-body target per item.

        Every item carries its own scene rotation and its own shoulder anchor,
        which is what independent single-frame draws need.  Because no item
        shares a window, the scene rotation enters batched MHR decodes instead
        of one decode per item, and the shared-window temporal mean taken by
        :meth:`canonical_targets` does not apply.

        Both decodes are required and are kept: the MHR global Euler state is
        applied about a model pivot, so the scene-rotated body differs from the
        rotated source body by a constant offset that cancels in the label but
        not in the physical scattering placement.
        """

        items = int(body_pose133.shape[0])
        morphology = torch.zeros((items, 28), dtype=torch.float32, device=self.device)
        lambda_values = torch.ones((items, 1), dtype=torch.float32, device=self.device)
        upper = body_pose133[:, list(UPPER_BODY_INDICES)]
        desired = self._item_scene_rows(
            shoulder_midpoint_target_m, items=items, width=3, name="per-item shoulder target"
        )
        rotation_rows = self._item_scene_rows(
            scene_rotation_axis_angle_rad, items=items, width=3, name="per-item scene rotation"
        )
        scene_rotation = _axis_angle_to_rotation_matrix(rotation_rows)
        rotate = scene_rotation.transpose(1, 2)
        source_points70 = self.fk.decode_points70_complete(
            hand_pose108, upper, global_rotation3, morphology
        )
        source_full24 = map_keypoints_70_to_full24_torch(source_points70)
        source_shoulder = source_full24[:, :, 0].mean(dim=1)
        rotated_source_shoulder = torch.bmm(source_shoulder[:, None, :], rotate).squeeze(1)
        scene_translation = desired - rotated_source_shoulder
        adjusted_global_rotation = self._item_adjusted_global_rotation(
            global_rotation3, scene_rotation=scene_rotation
        )
        adjusted_points70 = self.fk.decode_points70_complete(
            hand_pose108, upper, adjusted_global_rotation, morphology
        )
        full24 = map_keypoints_70_to_full24_torch(adjusted_points70)
        camera_translation = desired - full24[:, :, 0].mean(dim=1)
        state = torch.cat(
            (
                upper,
                hand_pose108[:, :54],
                hand_pose108[:, 54:],
                adjusted_global_rotation,
                camera_translation,
                morphology,
                lambda_values,
            ),
            dim=1,
        )
        _require(state.shape == (items, MHR_STATE_DIMENSION), "per-item MHR state differs")
        full24 = full24 + camera_translation[:, None, None]
        _require(full24.shape == (items, 2, 24, 3), "per-item full24 layout differs")
        _require(bool(torch.isfinite(full24).all()), "per-item target non-finite")
        return MhrEarlyFusionEpisodeTargets(
            state191=state,
            full24_m=full24,
            scene_translation_m=scene_translation,
            source_frame_indices=source_frame_indices.to(device=self.device, dtype=torch.long),
        )

    def scattering_vertices(
        self,
        *,
        body_pose133: Tensor,
        hand_pose108: Tensor,
        global_rotation3: Tensor,
        identity45: Tensor | None = None,
        morphology28: Tensor | None = None,
        scene_translation_m: Tensor | None = None,
        scene_rotation_axis_angle_rad: Tensor | None = None,
        surface_scale: float = 1.0,
    ) -> Tensor:
        """Return a complete full-MHR scattering surface in the radar scene."""

        frames = int(body_pose133.shape[0])
        if identity45 is None:
            identity45 = torch.zeros((45,), dtype=torch.float32, device=self.device)
        if morphology28 is None:
            morphology28 = torch.zeros((28,), dtype=torch.float32, device=self.device)
        _require(identity45.shape == (45,), "scattering identity must have 45 values")
        _require(morphology28.shape == (28,), "scattering morphology must have 28 values")
        _require(np.isfinite(surface_scale) and surface_scale > 0.0, "surface scale differs")
        translation, scene_rotation = self._scene_parameters(
            scene_translation_m=scene_translation_m,
            scene_rotation_axis_angle_rad=scene_rotation_axis_angle_rad,
        )
        params = self._model_parameters(
            body_pose133=body_pose133,
            hand_pose108=hand_pose108,
            global_rotation3=global_rotation3,
            morphology28=morphology28[None].expand(frames, -1),
        )
        expressions = torch.zeros((frames, 72), dtype=torch.float32, device=self.device)
        vertices_cm, _skeleton = self.fk.mhr(
            identity45[None].expand(frames, -1), params, expressions, True
        )
        vertices_m = vertices_cm.to(torch.float32) / 100.0
        vertices_m = vertices_m * self.fk.flip[None, None]
        vertices_m = float(surface_scale) * (vertices_m @ scene_rotation.transpose(0, 1))
        vertices_m = vertices_m + translation[None, None]
        _require(vertices_m.shape == (frames, 18439, 3), "complete MHR surface layout differs")
        _require(bool(torch.isfinite(vertices_m).all()), "scattering surface non-finite")
        return vertices_m

    def scattering_vertices_per_item(
        self,
        *,
        body_pose133: Tensor,
        hand_pose108: Tensor,
        global_rotation3: Tensor,
        identity45: Tensor,
        morphology28: Tensor,
        scene_translation_m: Tensor,
        scene_rotation_axis_angle_rad: Tensor,
        surface_scale: Tensor,
    ) -> Tensor:
        """Return one independently placed scattering surface per item.

        Every item carries its own identity, morphology, scale and scene, which
        is what independent single-frame draws need.  The scene transform is
        applied per item rather than collapsing the batch, so a shared rotation
        is expressed by broadcasting one row instead of by narrowing the call.
        """

        items = int(body_pose133.shape[0])
        identity = self._item_scene_rows(
            identity45, items=items, width=45, name="per-item identity"
        )
        morphology = self._item_scene_rows(
            morphology28, items=items, width=28, name="per-item morphology"
        )
        translation = self._item_scene_rows(
            scene_translation_m, items=items, width=3, name="per-item scene translation"
        )
        rotation_rows = self._item_scene_rows(
            scene_rotation_axis_angle_rad, items=items, width=3, name="per-item scene rotation"
        )
        scale = self._item_scene_rows(
            surface_scale, items=items, width=1, name="per-item surface scale"
        )
        _require(bool((scale > 0.0).all()), "per-item surface scale must be positive")
        scene_rotation = _axis_angle_to_rotation_matrix(rotation_rows)
        params = self._model_parameters(
            body_pose133=body_pose133,
            hand_pose108=hand_pose108,
            global_rotation3=global_rotation3,
            morphology28=morphology,
        )
        expressions = torch.zeros((items, 72), dtype=torch.float32, device=self.device)
        vertices_cm, _skeleton = self.fk.mhr(identity, params, expressions, True)
        vertices_m = vertices_cm.to(torch.float32) / 100.0
        vertices_m = vertices_m * self.fk.flip[None, None]
        vertices_m = scale.reshape(items, 1, 1) * torch.bmm(
            vertices_m, scene_rotation.transpose(1, 2)
        )
        vertices_m = vertices_m + translation[:, None, :]
        _require(vertices_m.shape == (items, 18439, 3), "per-item surface layout differs")
        _require(bool(torch.isfinite(vertices_m).all()), "per-item surface non-finite")
        return vertices_m


class MhrEarlyFusionComplexSimulator:
    """Full-surface MHR sequence -> post-range complex EarlyFusion input."""

    def __init__(
        self,
        *,
        sampling: SurfaceSamplingSpec,
        radar_config: RadarConfig | None = None,
        radar_position: tuple[float, float, float] = (0.0, 0.0, -1.0),
        path_chunk_size: int = 32,
    ) -> None:
        self.radar_config = (
            radar_config
            if radar_config is not None
            else get_radar_config(ACTIVE_MMWCAS_TDM_CONFIG_ID)
        )
        _require(
            (self.radar_config.num_chirps, self.radar_config.num_adc_samples) == (64, 100),
            "simulation requires the delivered 64x100 FMCW geometry",
        )
        self.surface_backend = MeshSurfacePathBackend(
            self.radar_config,
            sampling=sampling,
            radar_position=radar_position,
        )
        self.synthesizer = ComplexFmcwPathSynthesizer(
            self.radar_config, path_chunk_size=path_chunk_size
        )

    def synthesize_sequence(
        self,
        *,
        vertices: Tensor,
        faces: Tensor,
        delta_t_s: float,
        hand_face_mask: Tensor,
        apply_temporal_static_subtraction: bool = True,
    ) -> MhrEarlyFusionRadarEpisode:
        """Generate a contiguous surface sequence into complex EarlyFusion frames.

        Callers that reproduce the active window route retain the raw per-frame
        range spectrum. Sequence-wide mean subtraction is controlled by
        apply_temporal_static_subtraction, which defaults to true.
        """

        _require(
            isinstance(apply_temporal_static_subtraction, bool),
            "static subtraction flag differs",
        )
        _require(vertices.ndim == 3 and vertices.shape[1:] == (18439, 3), "surface sequence")
        minimum_surface_frames = 3 if apply_temporal_static_subtraction else 2
        _require(
            vertices.shape[0] >= minimum_surface_frames,
            "surface sequence must include a successor frame",
        )
        _require(faces.ndim == 2 and faces.shape[1] == 3, "surface faces differ")
        _require(hand_face_mask.shape == (faces.shape[0],), "hand face mask differs")
        _require(np.isfinite(delta_t_s) and delta_t_s > 0.0, "surface timing differs")
        # Each row is paired with its own physical successor, never with a
        # neighbouring sample. Geometry and the range transform run once for
        # the window; synthesis uses small frame tiles under its path budget.
        frame_count = int(vertices.shape[0] - 1)
        path_result = self.surface_backend.compute_paths_from_surface_pair(
            vertices[:-1], vertices[1:], faces, delta_t_s=delta_t_s
        )
        table = path_result.sample_table
        paths = path_result.paths
        raw = self.synthesizer.synthesize_path_batch(paths)
        complete_spectrum = range_fft_complex_adc(raw, range_bin_count=None)
        selected_hand = hand_face_mask.to(device=vertices.device)[
            table.face_indices.to(device=vertices.device)
        ]
        visible = path_result.visible_mask
        _require(visible.shape == (frame_count, table.sample_count), "surface visibility differs")
        selected_total = frame_count * selected_hand.numel()
        # Integer counts stay on the device until one window-level transfer.
        hand_total, visible_total, visible_hand_total = torch.stack(
            (
                selected_hand.sum() * frame_count,
                visible.sum(),
                (visible & selected_hand[None]).sum(),
            )
        ).tolist()
        first_receiver_ranges = paths.delay_s[:, 0] * (SIMULATION_LIGHT_SPEED * 0.5)
        visible_path_ranges = first_receiver_ranges[visible]
        processed_spectrum = (
            subtract_episode_static(complete_spectrum)
            if apply_temporal_static_subtraction
            else complete_spectrum
        )
        crop = processed_spectrum[..., :32]
        raw_iq = torch.view_as_real(crop).to(dtype=torch.float32)
        _require(raw_iq.shape[1:] == (64, 116, 32, 2), "raw I/Q output layout differs")
        range_rms = float(torch.sqrt(crop.abs().square().mean()).item())
        _require(range_rms > 0.0, "range preprocessing produced an empty signal")
        _require(visible_path_ranges.numel() > 0, "scene has no visible scattering paths")
        full_energy_before_static = complete_spectrum.abs().square().sum()
        crop_energy_before_static = complete_spectrum[..., :32].abs().square().sum()
        full_energy_after_static = processed_spectrum.abs().square().sum()
        crop_energy_after_static = crop.abs().square().sum()
        complete_bin_power = processed_spectrum.abs().square().mean(dim=(0, 1, 2))
        range_diagnostics = {
            "temporal_static_subtraction_applied": float(apply_temporal_static_subtraction),
            "visible_path_range_p01_m": float(torch.quantile(visible_path_ranges, 0.01).item()),
            "visible_path_range_p50_m": float(torch.quantile(visible_path_ranges, 0.50).item()),
            "visible_path_range_p95_m": float(torch.quantile(visible_path_ranges, 0.95).item()),
            "visible_path_range_p99_m": float(torch.quantile(visible_path_ranges, 0.99).item()),
            "visible_path_range_max_m": float(visible_path_ranges.max().item()),
            "complete_spectrum_peak_bin_after_static": float(complete_bin_power.argmax().item()),
            "complete_spectrum_energy_before_static": float(full_energy_before_static.item()),
            "cropped_spectrum_energy_before_static": float(crop_energy_before_static.item()),
            "cropped_energy_fraction_before_static": float(
                (crop_energy_before_static / full_energy_before_static.clamp_min(1.0e-20)).item()
            ),
            "complete_spectrum_energy_after_static": float(full_energy_after_static.item()),
            "cropped_spectrum_energy_after_static": float(crop_energy_after_static.item()),
            "cropped_energy_fraction_after_static": float(
                (crop_energy_after_static / full_energy_after_static.clamp_min(1.0e-20)).item()
            ),
            "complete_spectrum_peak_bin_output": float(complete_bin_power.argmax().item()),
            "cropped_energy_fraction_output": float(
                (crop_energy_after_static / full_energy_after_static.clamp_min(1.0e-20)).item()
            ),
        }
        return MhrEarlyFusionRadarEpisode(
            raw_iq=raw_iq,
            visible_sample_fraction=visible_total / max(selected_total, 1),
            hand_sample_fraction=hand_total / max(selected_total, 1),
            visible_hand_sample_fraction=visible_hand_total / max(selected_total, 1),
            range_rms=range_rms,
            generated_physical_frames=frame_count,
            range_diagnostics=range_diagnostics,
        )

    def synthesize_independent_pairs(
        self,
        *,
        vertices: Tensor,
        faces: Tensor,
        delta_t_s: float | Tensor,
        hand_face_mask: Tensor,
    ) -> MhrEarlyFusionPairRadarBatch:
        """Render independent surface pairs, exactly one pair per item.

        ``vertices`` is ``[items, 2, 18439, 3]``: each item's own target frame
        followed by that same item's successor.  Velocity is always the
        difference inside one item.  A concatenated frame axis must never reach
        this method, because pairing adjacent frames across item boundaries
        would invent motion that no sample has; the shape guard below is what
        enforces that boundary.

        The three heavy stages -- surface paths, complex synthesis and range FFT
        -- run once for the whole batch.  They are kernel-launch bound at batch
        one, where a one-item call costs the same as a sixteen-item call.
        """

        _require(
            vertices.ndim == 4 and vertices.shape[1] == 2 and vertices.shape[2:] == (18439, 3),
            "independent surface pairs must be [items, 2, 18439, 3]",
        )
        _require(faces.ndim == 2 and faces.shape[1] == 3, "surface faces differ")
        _require(hand_face_mask.shape == (faces.shape[0],), "hand face mask differs")
        items = int(vertices.shape[0])
        if isinstance(delta_t_s, Tensor):
            _require(
                delta_t_s.shape in {torch.Size(()), torch.Size((items,))}
                and bool(torch.isfinite(delta_t_s).all())
                and bool((delta_t_s > 0.0).all()),
                "surface pair timing differs",
            )
        else:
            _require(np.isfinite(delta_t_s) and delta_t_s > 0.0, "surface timing differs")
        path_result = self.surface_backend.compute_paths_from_surface_pair(
            vertices[:, 0],
            vertices[:, 1],
            faces,
            delta_t_s=delta_t_s,
            sample_table=None,
        )
        complete_spectrum = range_fft_complex_adc(
            self.synthesizer.synthesize_path_batch(path_result.paths), range_bin_count=None
        )
        crop = complete_spectrum[..., :32]
        raw_iq = torch.view_as_real(crop).to(dtype=torch.float32)
        _require(raw_iq.shape[1:] == (64, 116, 32, 2), "raw I/Q output layout differs")
        table = path_result.sample_table
        selected_hand = hand_face_mask[table.face_indices.to(device=hand_face_mask.device)]
        visible = path_result.visible_mask
        _require(visible.shape == (items, table.face_indices.numel()), "pair visibility differs")
        selected_total = max(int(selected_hand.numel()), 1)
        hand_fraction = float(selected_hand.sum().item()) / selected_total
        visible_fraction = visible.sum(dim=1) / selected_total
        visible_hand_fraction = (visible & selected_hand[None]).sum(dim=1) / selected_total
        first_receiver_ranges = path_result.paths.delay_s[:, 0] * (SIMULATION_LIGHT_SPEED * 0.5)
        range_rms = torch.sqrt(crop.abs().square().mean(dim=(1, 2, 3)))
        _require(bool((range_rms > 0.0).all()), "range preprocessing produced an empty signal")
        full_energy = complete_spectrum.abs().square().sum(dim=(1, 2, 3))
        crop_energy = crop.abs().square().sum(dim=(1, 2, 3))
        complete_bin_power = complete_spectrum.abs().square().mean(dim=(1, 2))
        diagnostics: list[dict[str, float]] = []
        for item in range(items):
            masked = first_receiver_ranges[item][visible[item]]
            _require(bool(masked.numel()), "pair has no visible scattering paths")
            peak_bin = float(complete_bin_power[item].argmax().item())
            cropped_fraction = float(
                (crop_energy[item] / full_energy[item].clamp_min(1.0e-20)).item()
            )
            diagnostics.append(
                {
                    "temporal_static_subtraction_applied": 0.0,
                    "visible_path_range_p01_m": float(torch.quantile(masked, 0.01).item()),
                    "visible_path_range_p50_m": float(torch.quantile(masked, 0.50).item()),
                    "visible_path_range_p95_m": float(torch.quantile(masked, 0.95).item()),
                    "visible_path_range_p99_m": float(torch.quantile(masked, 0.99).item()),
                    "visible_path_range_max_m": float(masked.max().item()),
                    "complete_spectrum_peak_bin_after_static": peak_bin,
                    "complete_spectrum_energy_before_static": float(full_energy[item].item()),
                    "cropped_spectrum_energy_before_static": float(crop_energy[item].item()),
                    "cropped_energy_fraction_before_static": cropped_fraction,
                    "complete_spectrum_energy_after_static": float(full_energy[item].item()),
                    "cropped_spectrum_energy_after_static": float(crop_energy[item].item()),
                    "cropped_energy_fraction_after_static": cropped_fraction,
                    "complete_spectrum_peak_bin_output": peak_bin,
                    "cropped_energy_fraction_output": cropped_fraction,
                }
            )
        return MhrEarlyFusionPairRadarBatch(
            raw_iq=raw_iq,
            visible_sample_fraction=visible_fraction,
            hand_sample_fraction=torch.full((items,), hand_fraction, dtype=torch.float32),
            visible_hand_sample_fraction=visible_hand_fraction,
            range_rms=range_rms,
            generated_physical_frames=1,
            range_diagnostics=tuple(diagnostics),
        )
