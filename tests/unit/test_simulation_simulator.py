from __future__ import annotations

import numpy as np
import torch

from mmprism.simulation.simulator import (
    SIMULATION_LIGHT_SPEED,
    PointReflectorSimulator,
    Simulation,
    get_index_full,
    get_index_large,
    get_index_middle,
    get_index_small,
)


def _seeded_cloud(batch: int = 2, points: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    points_3d = torch.randn(batch, points, 3, generator=generator) * 0.2
    points_3d[..., 2] = points_3d[..., 2].abs() + 1.0
    velocities = torch.randn(batch, points, 3, generator=generator) * 0.5
    return points_3d, velocities


def test_virtual_array_layouts() -> None:
    full_d, full_idx = get_index_full()
    assert len(full_d) == 116
    assert full_idx.tolist() == list(range(116))
    for getter, count in (
        (get_index_large, 44),
        (get_index_middle, 24),
        (get_index_small, 10),
    ):
        sub_d, sub_idx = getter()
        assert len(sub_d) == count
        assert [full_d[i] for i in sub_idx] == sub_d


def test_paths_have_expected_shapes_and_monostatic_limit() -> None:
    simulator = PointReflectorSimulator()
    points, velocities = _seeded_cloud(batch=1)

    paths = simulator.compute_paths_from_points(points[0], velocities[0])

    assert paths["a"].shape == (116, 8)
    assert paths["tau"].shape == (116, 8)
    assert paths["vel"].shape == (116, 8)
    assert paths["a"].dtype == torch.float32
    # Amplitude is 1 / (d/2)^2 with d the two-way path length: positive.
    assert bool((paths["a"] > 0).all())
    # tau is the two-way light-travel time: at least 2 * range / c.
    ranges = torch.norm(points[0] - simulator.tx_position.to(points.dtype), dim=-1)
    assert bool((paths["tau"] >= (2 * ranges / SIMULATION_LIGHT_SPEED).min() * 0.9).all())


def test_simulation_frame_shape_dtype_and_real_cast_quirk() -> None:
    simulation = Simulation()
    points, velocities = _seeded_cloud()

    frames = simulation(points, velocities)

    # Legacy quirk: complex echo is cast to the real default dtype.
    assert frames.shape == (2, 64, 116, 256)
    assert frames.dtype == torch.float32
    assert not frames.is_complex()


def test_simulation_is_deterministic() -> None:
    points, velocities = _seeded_cloud()
    first = Simulation()
    second = Simulation()
    assert torch.equal(first(points, velocities), second(points, velocities))


def test_simulation_real_output_matches_echo_real_part() -> None:
    simulation = Simulation()
    points, velocities = _seeded_cloud()
    paths = simulation.simulator.compute_paths_from_points(points[0], velocities[0])

    frame = simulation.get_raw_radar_frame(paths)

    # Recompute the complex echo without the legacy cast for comparison.
    a = paths["a"]
    tau = paths["tau"]
    vel = paths["vel"]
    tau_chirp = (vel * 2 / SIMULATION_LIGHT_SPEED) * simulation.time_steps[:, None, None]
    tau_chirp = (tau.unsqueeze(0) + tau_chirp)[:, :, None, :]
    frequencies = simulation.fs[None, None, :, None]
    phase = 2 * np.pi * (frequencies + simulation.start_freq) * tau_chirp
    phase %= 2 * np.pi
    complex_echo = (a[None, :, None, :] * torch.exp(1j * phase)).sum(dim=-1)

    assert torch.equal(frame, complex_echo.real.to(torch.float32))
