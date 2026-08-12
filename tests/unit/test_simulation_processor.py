from __future__ import annotations

import numpy as np
import pytest
import torch

from mmprism.contracts.tensors import validate_radar_cube
from mmprism.simulation.processor import (
    Processor,
    build_steering_vector,
    build_steering_vector_1d,
)

# The legacy complex->float cast in Simulation emits a UserWarning; the
# simulator tests for that quirk live in test_simulation_simulator.py.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Casting complex values to real:UserWarning"
)


def _seeded_frame(batch: int = 2, antennas: int = 116) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    real = torch.randn(batch, 64, antennas, 256, generator=generator)
    imag = torch.randn(batch, 64, antennas, 256, generator=generator)
    return torch.complex(real, imag)


def test_processor_output_shape_dtype_and_contract() -> None:
    processor = Processor(process_range=True)
    with torch.no_grad():
        cube = processor(_seeded_frame())

    assert cube.shape == (2, 64, 32, 32, 32)
    assert cube.dtype == torch.float32
    assert bool((cube >= 0).all())
    metadata = validate_radar_cube(cube.numpy(), leading_axes=("batch",))
    assert metadata.shape == (2, 64, 32, 32, 32)
    assert metadata.units == "power"


def test_processor_is_deterministic() -> None:
    frame = _seeded_frame()
    first = Processor(process_range=True)
    second = Processor(process_range=True)
    with torch.no_grad():
        assert torch.equal(first(frame), second(frame))


def test_processor_learnable_weights_toggle() -> None:
    assert not Processor().bm_weights.requires_grad
    assert Processor(learnable_weights=True).bm_weights.requires_grad


def test_processor_rejects_unknown_array_size() -> None:
    with pytest.raises(ValueError, match="array_size"):
        Processor(array_size="tiny")


def test_processor_sub_array_selects_antennas() -> None:
    processor = Processor(array_size="small", process_range=True)
    assert processor.D_antennas == 10
    with torch.no_grad():
        cube = processor(_seeded_frame())
    assert cube.shape == (2, 64, 32, 32, 32)


def test_steering_vector_shapes_and_dtype() -> None:
    azi_ele_id = torch.tensor([[0, 0], [3, 2]])
    azi_grid = torch.linspace(-np.pi / 6, np.pi / 6, 8)
    ele_grid = torch.linspace(-np.pi / 6, np.pi / 6, 4)

    steering = build_steering_vector(azi_ele_id, azi_grid, ele_grid)
    assert steering.shape == (2, 32)
    assert steering.dtype == torch.complex64

    steering_1d = build_steering_vector_1d(torch.tensor([0, 3]), azi_grid)
    assert steering_1d.shape == (2, 8)


def test_steering_vector_boresight_is_unit_phasor() -> None:
    azi_ele_id = torch.tensor([[5, 0], [-3, 6]])
    azi_grid = torch.tensor([0.0])
    ele_grid = torch.tensor([0.0])
    steering = build_steering_vector(azi_ele_id, azi_grid, ele_grid)
    assert torch.allclose(steering, torch.ones_like(steering))
