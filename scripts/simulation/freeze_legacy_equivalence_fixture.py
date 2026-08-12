"""Freeze a numerical-equivalence fixture from the LEGACY simulator modules.

This is the ONLY place where legacy imports (``config.radar``,
``src.fmcw.simulator``, ``run_simulation``, resolved from ``legacy/``) are
permitted. It runs the legacy
code on fixed seeded synthetic inputs and stores the outputs in
``tests/fixtures/simulation/legacy_equivalence.npz`` so that
``tests/unit/test_simulation_equivalence.py`` can pin the rebuilt
``mmprism.simulation`` package to legacy numerics without importing legacy
code.

Legacy repair note: ``src/fmcw/simulator.py::mmSimulator.init`` calls an
undefined ``get_index()`` (documented in TENSOR_CONTRACTS evidence
conflicts). The only executable interpretation is the full virtual array, so
this script monkeypatches ``get_index`` to ``get_index_full()[0]``.

Run: ``uv run python scripts/simulation/freeze_legacy_equivalence_fixture.py``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Legacy forensic code lives under ``legacy/`` (relocated 2026-08-12). Putting
# that directory on ``sys.path`` lets the historical absolute imports
# (``src.fmcw.*``, ``config.radar.*``, ``run_simulation``) resolve unchanged.
LEGACY_ROOT = REPO_ROOT / "legacy"
sys.path.insert(0, str(LEGACY_ROOT))

import numpy as np  # noqa: E402
import src.fmcw.simulator as legacy_sim  # noqa: E402
import torch  # noqa: E402
from run_simulation import process_point_cloud as legacy_densify_numpy  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "simulation" / "legacy_equivalence.npz"
SEED = 20260812


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    legacy_sim.get_index = lambda: legacy_sim.get_index_full()[0]  # type: ignore[attr-defined]

    generator = torch.Generator().manual_seed(SEED)

    # --- Dual-hand pose densification input: [T=12, 2, 24, 3] with NaNs ---
    poses = torch.randn(12, 2, 24, 3, generator=generator) * 0.15
    poses[..., 2] = poses[..., 2].abs() + 1.0  # keep depths radar-plausible
    poses[3, 0, 5] = float("nan")  # one left hand joint, frame 3
    poses[7, 1, 2, 0] = float("nan")  # one right arm joint, frame 7
    poses[9] = float("nan")  # whole frame invalid

    legacy_points, legacy_mask = legacy_sim.process_point_cloud(poses)

    # --- Point-reflector frame simulation: [B=2, N=8, 3] ---
    points = torch.randn(2, 8, 3, generator=generator) * 0.2
    points[..., 2] = points[..., 2].abs() + 1.0
    velocities = torch.randn(2, 8, 3, generator=generator) * 0.5

    legacy_simulation = legacy_sim.Simulation()
    legacy_paths = legacy_simulation.simulator.compute_paths_from_points(
        points[0], velocities[0]
    )
    legacy_frames = torch.stack(
        [
            legacy_simulation.simulate_batch(points[b], velocities[b])
            for b in range(points.shape[0])
        ],
        dim=0,
    )

    # --- Processor: raw frames -> power cube ---
    legacy_processor = legacy_sim.Processor(learnable_weights=False)
    legacy_processor.if_process_range = True
    with torch.no_grad():
        legacy_cube = legacy_processor(legacy_frames)

    # --- Numpy full-body sequence: densify + temporal preprocessing ---
    rng = np.random.default_rng(SEED)
    full_poses = rng.normal(0.0, 0.15, size=(12, 54, 3)).astype(np.float64)
    full_poses[..., 2] = np.abs(full_poses[..., 2]) + 1.0
    legacy_densified = legacy_densify_numpy(full_poses)
    smoothed = gaussian_filter1d(legacy_densified, sigma=1, axis=0)
    legacy_velocities = (smoothed[1:] - smoothed[:-1]) * 10
    legacy_points_ds = smoothed[:-1][::3]
    legacy_velocities_ds = legacy_velocities[::3]

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXTURE_PATH,
        seed=np.array(SEED),
        # dual-hand densification
        input_poses=poses.numpy(),
        legacy_densified_points=legacy_points.numpy(),
        legacy_densified_mask=legacy_mask.numpy(),
        # path geometry for batch 0
        input_points=points.numpy(),
        input_velocities=velocities.numpy(),
        legacy_path_a=legacy_paths["a"].numpy(),
        legacy_path_tau=legacy_paths["tau"].numpy(),
        legacy_path_vel=legacy_paths["vel"].numpy(),
        # raw frames (real-only after the legacy complex->float cast quirk)
        legacy_raw_frames=legacy_frames.numpy(),
        # processor power cube
        legacy_power_cube=legacy_cube.numpy(),
        # numpy densification + temporal preprocessing
        input_full_poses=full_poses,
        legacy_numpy_densified=legacy_densified,
        legacy_points_decimated=legacy_points_ds,
        legacy_velocities_decimated=legacy_velocities_ds,
    )
    print(f"Wrote {FIXTURE_PATH}")
    print(f"  densified: {tuple(legacy_points.shape)} mask={tuple(legacy_mask.shape)}")
    print(f"  raw frames: {tuple(legacy_frames.shape)} {legacy_frames.dtype}")
    print(f"  power cube: {tuple(legacy_cube.shape)} {legacy_cube.dtype}")
    print(f"  decimated: {tuple(legacy_points_ds.shape)}")


if __name__ == "__main__":
    main()
