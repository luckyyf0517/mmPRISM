"""Pin ``mmprism.simulation`` to the frozen legacy-equivalence fixture.

The fixture is generated ONCE by
``scripts/simulation/freeze_legacy_equivalence_fixture.py`` from the legacy
modules; regenerating it requires a deliberate decision because it changes
the reference numerics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mmprism.simulation import (
    PointReflectorSimulator,
    Processor,
    Simulation,
    densify_body_hand_frames,
    densify_dual_hand_pose,
    temporal_smooth_decimate,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "simulation"
    / "legacy_equivalence.npz"
)

# Achieved agreement against the legacy fixture on the reference CPU
# environment (torch 2.11, numpy 2.5): BIT-EXACT (max abs diff 0.0) for
# every frozen output — densified clouds, path geometry, raw frames, power
# cubes, and temporal preprocessing. Tolerances below are kept loose only
# for cross-platform BLAS/FFT variation and still beat the 1e-5 target.
RTOL = 1e-6
ATOL = 1e-6

pytestmark = pytest.mark.filterwarnings(
    "ignore:Casting complex values to real:UserWarning"
)


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE_PATH) as data:
        return {key: data[key] for key in data.files}


def test_fixture_seed_is_pinned(fixture: dict[str, np.ndarray]) -> None:
    assert int(fixture["seed"]) == 20260812


def test_dual_hand_densification_matches_legacy(fixture: dict[str, np.ndarray]) -> None:
    poses = torch.from_numpy(fixture["input_poses"])

    points, mask = densify_dual_hand_pose(poses)

    np.testing.assert_array_equal(mask.numpy(), fixture["legacy_densified_mask"])
    np.testing.assert_allclose(
        points.numpy(), fixture["legacy_densified_points"], rtol=RTOL, atol=ATOL
    )


def test_path_geometry_matches_legacy(fixture: dict[str, np.ndarray]) -> None:
    simulator = PointReflectorSimulator()
    points = torch.from_numpy(fixture["input_points"])
    velocities = torch.from_numpy(fixture["input_velocities"])

    paths = simulator.compute_paths_from_points(points[0], velocities[0])

    np.testing.assert_allclose(
        paths["a"].numpy(), fixture["legacy_path_a"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        paths["tau"].numpy(), fixture["legacy_path_tau"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        paths["vel"].numpy(), fixture["legacy_path_vel"], rtol=RTOL, atol=ATOL
    )


def test_raw_frames_match_legacy(fixture: dict[str, np.ndarray]) -> None:
    simulation = Simulation()
    points = torch.from_numpy(fixture["input_points"])
    velocities = torch.from_numpy(fixture["input_velocities"])

    frames = simulation(points, velocities)

    np.testing.assert_allclose(
        frames.numpy(), fixture["legacy_raw_frames"], rtol=RTOL, atol=ATOL
    )


def test_power_cube_matches_legacy(fixture: dict[str, np.ndarray]) -> None:
    processor = Processor(process_range=True)
    frames = torch.from_numpy(fixture["legacy_raw_frames"])

    with torch.no_grad():
        cube = processor(frames)

    np.testing.assert_allclose(
        cube.numpy(), fixture["legacy_power_cube"], rtol=RTOL, atol=ATOL
    )


def test_numpy_densification_matches_legacy(fixture: dict[str, np.ndarray]) -> None:
    densified = densify_body_hand_frames(fixture["input_full_poses"])
    np.testing.assert_allclose(
        densified, fixture["legacy_numpy_densified"], rtol=RTOL, atol=ATOL
    )


def test_temporal_preprocessing_matches_legacy(fixture: dict[str, np.ndarray]) -> None:
    points_ds, velocities_ds = temporal_smooth_decimate(fixture["legacy_numpy_densified"])
    np.testing.assert_allclose(
        points_ds, fixture["legacy_points_decimated"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        velocities_ds, fixture["legacy_velocities_decimated"], rtol=RTOL, atol=ATOL
    )
