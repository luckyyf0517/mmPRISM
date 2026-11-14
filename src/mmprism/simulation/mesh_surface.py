"""Deterministic surface paths for mesh-based radar simulation.

The module accepts reconstructed vertex pairs and fixed mesh topology from an
explicit runtime adapter, then emits the typed propagation paths consumed by
the radar simulator.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mmprism.simulation.path_batch import FmcwPathBatch
from mmprism.simulation.radar_config import RadarConfig
from mmprism.simulation.simulator import SIMULATION_LIGHT_SPEED, RadarArrayGeometry

SURFACE_SAMPLING_PROTOCOL = "mmprism.mesh_surface_topology_uniform_barycentric.v1"

SURFACE_SCATTERING_PROTOCOL = "mmprism.mesh_surface_direct_visibility_v1"


class MeshSurfaceError(ValueError):
    """Raised when a deterministic surface-path request is invalid."""


def _finite_positive(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeshSurfaceError(f"{location} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MeshSurfaceError(f"{location} must be a finite positive number")
    return result


def _finite_nonnegative(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeshSurfaceError(f"{location} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise MeshSurfaceError(f"{location} must be a finite non-negative number")
    return result


def _pair_delta_t(
    value: float | Tensor, *, batch: int, device: torch.device, dtype: torch.dtype
) -> float | Tensor:
    """Validate one shared or one-per-pair physical time interval."""

    if not isinstance(value, Tensor):
        return _finite_positive(value, "delta_t_s")
    result = value.to(device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(batch)
    if (
        result.shape != (batch,)
        or not bool(torch.isfinite(result).all())
        or not bool((result > 0.0).all())
    ):
        raise MeshSurfaceError("delta_t_s pair layout differs")
    return result.reshape(batch, 1, 1)


@dataclass(frozen=True, slots=True)
class SurfaceSamplingSpec:
    """Topology-bound, deterministic triangle and barycentric sampling policy."""

    sample_count: int
    seed: int = 0
    protocol: str = SURFACE_SAMPLING_PROTOCOL

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise MeshSurfaceError("sample_count must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise MeshSurfaceError("seed must be an integer")
        if self.protocol != SURFACE_SAMPLING_PROTOCOL:
            raise MeshSurfaceError(f"protocol must be {SURFACE_SAMPLING_PROTOCOL}")


@dataclass(frozen=True, slots=True)
class SurfaceScatteringSpec:
    """Declared simplified direct-surface response, not a calibrated RCS model."""

    response_coefficient: float = 1.0
    distance_epsilon: float = 1e-6
    visibility_epsilon: float = 0.0
    protocol: str = SURFACE_SCATTERING_PROTOCOL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "response_coefficient",
            _finite_positive(self.response_coefficient, "response_coefficient"),
        )
        object.__setattr__(
            self,
            "distance_epsilon",
            _finite_positive(self.distance_epsilon, "distance_epsilon"),
        )
        object.__setattr__(
            self,
            "visibility_epsilon",
            _finite_nonnegative(self.visibility_epsilon, "visibility_epsilon"),
        )
        if self.protocol != SURFACE_SCATTERING_PROTOCOL:
            raise MeshSurfaceError(f"protocol must be {SURFACE_SCATTERING_PROTOCOL}")


@dataclass(frozen=True, slots=True)
class SurfaceSampleTable:
    """Fixed topology-indexed samples shared by the two adjacent mesh frames."""

    topology_digest: str
    face_indices: Tensor
    barycentric: Tensor

    def __post_init__(self) -> None:
        if len(self.topology_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.topology_digest
        ):
            raise MeshSurfaceError("topology_digest must be a lowercase SHA-256 hex digest")
        if self.face_indices.ndim != 1 or self.face_indices.numel() < 1:
            raise MeshSurfaceError("face_indices must be a non-empty one-dimensional tensor")
        if self.face_indices.dtype != torch.int64:
            raise MeshSurfaceError("face_indices must use int64")
        if self.barycentric.shape != (self.face_indices.numel(), 3):
            raise MeshSurfaceError("barycentric must have shape [sample_count, 3]")
        if not self.barycentric.is_floating_point() or not bool(
            torch.isfinite(self.barycentric).all()
        ):
            raise MeshSurfaceError("barycentric coordinates must be finite floating point values")
        if not bool((self.barycentric >= 0.0).all()) or not bool(
            torch.allclose(
                self.barycentric.sum(dim=1),
                torch.ones(self.barycentric.shape[0], dtype=self.barycentric.dtype),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise MeshSurfaceError("barycentric coordinates must be non-negative and sum to one")

    @property
    def sample_count(self) -> int:
        """Number of deterministic surface slots."""

        return int(self.face_indices.numel())


@dataclass(frozen=True, slots=True)
class MeshSurfacePathResult:
    """Shared paths plus bounded diagnostics for one surface-pair evaluation."""

    paths: FmcwPathBatch
    sample_table: SurfaceSampleTable
    sample_positions: Tensor
    sample_velocities: Tensor
    sample_normals: Tensor
    represented_areas: Tensor
    valid_mask: Tensor
    visible_mask: Tensor


def _validate_faces(faces: Tensor) -> None:
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
        raise MeshSurfaceError(
            f"faces must have non-empty shape [face, 3], got {tuple(faces.shape)}"
        )
    if faces.dtype not in integer_dtypes:
        raise MeshSurfaceError("faces must use an integer dtype")
    if bool((faces < 0).any()):
        raise MeshSurfaceError("faces must not contain negative indices")


def topology_digest(faces: Tensor) -> str:
    """Return a device-independent SHA-256 identity for a triangle topology."""

    _validate_faces(faces)
    canonical = faces.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    payload = canonical.shape.__repr__().encode("ascii") + canonical.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def deterministic_surface_sample_table(
    faces: Tensor,
    sampling: SurfaceSamplingSpec,
    *,
    device: torch.device | None = None,
) -> SurfaceSampleTable:
    """Derive fixed topology-uniform triangle samples and barycentric locations.

    The table is independent of instantaneous vertex positions so a sample
    keeps the same triangle/barycentric identity at both frames.  Area is
    accounted for later by the direct scattering response.
    """

    _validate_faces(faces)
    digest = topology_digest(faces)
    face_count = int(faces.shape[0])
    generator = torch.Generator(device="cpu")
    digest_seed = int.from_bytes(bytes.fromhex(digest)[:8], byteorder="big")
    generator.manual_seed((sampling.seed ^ digest_seed) % ((1 << 63) - 1))
    permutations: list[Tensor] = []
    remaining = sampling.sample_count
    while remaining > 0:
        permutation = torch.randperm(face_count, generator=generator)
        permutations.append(permutation[: min(remaining, face_count)])
        remaining -= face_count
    face_indices = torch.cat(permutations, dim=0).to(dtype=torch.int64)
    draws = torch.rand((sampling.sample_count, 2), generator=generator)
    sqrt_u = torch.sqrt(draws[:, 0])
    barycentric = torch.stack(
        (1.0 - sqrt_u, sqrt_u * (1.0 - draws[:, 1]), sqrt_u * draws[:, 1]), dim=1
    )
    if device is not None:
        face_indices = face_indices.to(device=device)
        barycentric = barycentric.to(device=device)
    return SurfaceSampleTable(digest, face_indices, barycentric)


def _validate_coordinate_points(points: Tensor, location: str) -> None:
    if points.ndim != 3 or points.shape[1] < 1 or points.shape[-1] != 3:
        raise MeshSurfaceError(f"{location} must have non-empty shape [batch, item, 3]")
    if not points.is_floating_point() or not bool(torch.isfinite(points).all()):
        raise MeshSurfaceError(f"{location} must be finite floating point values")


class MeshSurfacePathBackend(nn.Module):
    """Convert paired surface vertices into deterministic direct FMCW paths."""

    def __init__(
        self,
        radar_config: RadarConfig | None = None,
        *,
        sampling: SurfaceSamplingSpec,
        scattering: SurfaceScatteringSpec | None = None,
        dtype: torch.dtype = torch.float32,
        radar_position: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.array_geometry = RadarArrayGeometry(
            radar_config,
            radar_position=radar_position,
        )
        self.sampling = sampling
        self.scattering = scattering if scattering is not None else SurfaceScatteringSpec()
        self.dtype = dtype

    def compute_paths_from_surface_pair(
        self,
        vertices_t: Tensor,
        vertices_t_plus_1: Tensor,
        faces: Tensor,
        *,
        delta_t_s: float | Tensor,
        sample_table: SurfaceSampleTable | None = None,
    ) -> MeshSurfacePathResult:
        """Emit direct paths from a corresponded adjacent surface pair.

        ``vertices_t`` and ``vertices_t_plus_1`` use ``[batch, vertex, 3]``.
        The same sampled triangles and barycentric locations are evaluated in
        both frames, making surface velocity a paired finite difference.
        """

        _validate_coordinate_points(vertices_t, "vertices_t")
        _validate_coordinate_points(vertices_t_plus_1, "vertices_t_plus_1")
        if (
            vertices_t.shape != vertices_t_plus_1.shape
            or vertices_t.device != vertices_t_plus_1.device
        ):
            raise MeshSurfaceError("surface frame pairs must share shape and device")
        delta_t = _pair_delta_t(
            delta_t_s,
            batch=int(vertices_t.shape[0]),
            device=vertices_t.device,
            dtype=vertices_t.dtype,
        )
        _validate_faces(faces)
        if int(faces.max().item()) >= vertices_t.shape[1]:
            raise MeshSurfaceError("faces reference a vertex outside the surface pair")
        table = (
            deterministic_surface_sample_table(faces, self.sampling)
            if sample_table is None
            else sample_table
        )
        expected_digest = topology_digest(faces)
        if table.topology_digest != expected_digest:
            raise MeshSurfaceError("surface sample table topology does not match faces")
        if table.sample_count != self.sampling.sample_count:
            raise MeshSurfaceError("surface sample table does not match sampling.sample_count")
        face_indices = table.face_indices.to(device=vertices_t.device)
        if bool((face_indices < 0).any()) or bool((face_indices >= faces.shape[0]).any()):
            raise MeshSurfaceError("surface sample table contains an out-of-range face index")
        barycentric = table.barycentric.to(device=vertices_t.device, dtype=vertices_t.dtype)
        indexed_faces = faces.to(device=vertices_t.device, dtype=torch.int64)[face_indices]
        triangles_t = vertices_t[:, indexed_faces, :]
        triangles_next = vertices_t_plus_1[:, indexed_faces, :]
        weights = barycentric[None, :, :, None]
        sample_positions = (triangles_t * weights).sum(dim=2)
        next_positions = (triangles_next * weights).sum(dim=2)
        sample_velocities = (next_positions - sample_positions) / delta_t

        edges_first = triangles_t[:, :, 1, :] - triangles_t[:, :, 0, :]
        edges_second = triangles_t[:, :, 2, :] - triangles_t[:, :, 0, :]
        normal_raw = torch.linalg.cross(edges_first, edges_second, dim=-1)
        normal_length = torch.linalg.vector_norm(normal_raw, dim=-1)
        eps = self.scattering.distance_epsilon
        sample_normals = normal_raw / normal_length.clamp_min(eps)[..., None]
        triangle_areas = normal_length * 0.5
        # Topology-uniform samples are a Monte-Carlo surface integral.  This
        # factor gives each selected face its area-weighted representation.
        represented_areas = triangle_areas * (faces.shape[0] / table.sample_count)

        tx = self.array_geometry.tx_position.to(device=vertices_t.device, dtype=vertices_t.dtype)
        rx = self.array_geometry.rx_positions.to(device=vertices_t.device, dtype=vertices_t.dtype)
        to_tx = tx[None, None, :] - sample_positions
        tx_distance = torch.linalg.vector_norm(to_tx, dim=-1)
        to_rx = rx[None, :, None, :] - sample_positions[:, None, :, :]
        rx_distance = torch.linalg.vector_norm(to_rx, dim=-1)
        finite_surface = (
            torch.isfinite(sample_positions).all(dim=-1)
            & torch.isfinite(sample_velocities).all(dim=-1)
            & torch.isfinite(sample_normals).all(dim=-1)
            & (triangle_areas > eps)
            & (tx_distance > eps)
            & (rx_distance > eps).all(dim=1)
        )
        incidence = (sample_normals * (to_tx / tx_distance.clamp_min(eps)[..., None])).sum(dim=-1)
        visible_mask = finite_surface & (incidence > self.scattering.visibility_epsilon)
        denominator = tx_distance[:, None, :] * rx_distance
        amplitude = (
            self.scattering.response_coefficient
            * represented_areas[:, None, :]
            * incidence.clamp_min(0.0)[:, None, :]
            / denominator.clamp_min(eps)
        )
        amplitude = torch.where(visible_mask[:, None, :], amplitude, torch.zeros_like(amplitude))
        delay_s = (tx_distance[:, None, :] + rx_distance) / SIMULATION_LIGHT_SPEED
        incident_direction = -to_tx / tx_distance.clamp_min(eps)[..., None]
        scattered_direction = to_rx / rx_distance.clamp_min(eps)[..., None]
        radial_mps = (
            (sample_velocities * incident_direction).sum(dim=-1)[:, None, :]
            - (sample_velocities[:, None, :, :] * scattered_direction).sum(dim=-1)
        ) / 2.0
        paths = FmcwPathBatch(
            amplitude=amplitude.to(dtype=self.dtype),
            delay_s=delay_s.to(dtype=self.dtype),
            radial_mps=radial_mps.to(dtype=self.dtype),
            sample_mask=finite_surface,
        )
        return MeshSurfacePathResult(
            paths=paths,
            sample_table=table,
            sample_positions=sample_positions,
            sample_velocities=sample_velocities,
            sample_normals=sample_normals,
            represented_areas=represented_areas,
            valid_mask=finite_surface,
            visible_mask=visible_mask,
        )
