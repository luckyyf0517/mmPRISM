"""Array geometry and waveform parameters for complex echo simulation."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from mmprism.simulation.radar_config import (
    ACTIVE_MMWCAS_TDM_CONFIG_ID,
    RadarConfig,
    get_radar_config,
)

SIMULATION_LIGHT_SPEED = 2.99792458e8

DEFAULT_RADAR_POSITION = (0.0, 0.0, -1.0)

MEASURED_LINK_AMPLITUDE_CALIBRATION_V1 = 2.182e7


def get_index_full() -> tuple[list[list[int]], np.ndarray]:
    """Full virtual-array layout: 116 half-wavelength grid coordinates."""

    coordinates = [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [6, 0],
        [7, 0],
        [8, 0],
        [9, 0],
        [10, 0],
        [11, 0],
        [11, 1],
        [11, 4],
        [11, 6],
        [12, 0],
        [12, 1],
        [12, 4],
        [12, 6],
        [13, 0],
        [14, 0],
        [15, 0],
        [16, 0],
        [17, 0],
        [18, 0],
        [19, 0],
        [20, 0],
        [21, 0],
        [22, 0],
        [22, 1],
        [22, 4],
        [22, 6],
        [23, 0],
        [23, 1],
        [23, 4],
        [23, 6],
        [24, 0],
        [25, 0],
        [26, 0],
        [27, 0],
        [28, 0],
        [29, 0],
        [30, 0],
        [31, 0],
        [32, 0],
        [33, 0],
        [34, 0],
        [35, 0],
        [36, 0],
        [37, 0],
        [38, 0],
        [39, 0],
        [40, 0],
        [41, 0],
        [42, 0],
        [43, 0],
        [44, 0],
        [45, 0],
        [46, 0],
        [47, 0],
        [48, 0],
        [49, 0],
        [50, 0],
        [51, 0],
        [52, 0],
        [53, 0],
        [54, 0],
        [55, 0],
        [56, 0],
        [57, 0],
        [57, 1],
        [57, 4],
        [57, 6],
        [58, 0],
        [58, 1],
        [58, 4],
        [58, 6],
        [59, 0],
        [59, 1],
        [59, 4],
        [59, 6],
        [60, 0],
        [60, 1],
        [60, 4],
        [60, 6],
        [61, 0],
        [61, 1],
        [61, 4],
        [61, 6],
        [62, 0],
        [62, 1],
        [62, 4],
        [62, 6],
        [63, 0],
        [64, 0],
        [65, 0],
        [66, 0],
        [67, 0],
        [68, 0],
        [69, 0],
        [70, 0],
        [71, 0],
        [72, 0],
        [73, 0],
        [74, 0],
        [75, 0],
        [76, 0],
        [77, 0],
        [78, 0],
        [79, 0],
        [80, 0],
        [81, 0],
        [82, 0],
        [83, 0],
        [84, 0],
        [85, 0],
    ]
    return coordinates, np.arange(0, len(coordinates))


def get_index_large() -> tuple[list[list[int]], list[int]]:
    """44-element sub-array of the full layout."""

    coordinates = [
        [11, 0],
        [11, 1],
        [11, 4],
        [11, 6],
        [12, 0],
        [12, 1],
        [12, 4],
        [12, 6],
        [13, 0],
        [14, 0],
        [15, 0],
        [16, 0],
        [17, 0],
        [18, 0],
        [19, 0],
        [20, 0],
        [21, 0],
        [22, 0],
        [22, 1],
        [22, 4],
        [22, 6],
        [23, 0],
        [23, 1],
        [23, 4],
        [23, 6],
        [24, 0],
        [25, 0],
        [26, 0],
        [27, 0],
        [28, 0],
        [29, 0],
        [30, 0],
        [31, 0],
        [32, 0],
        [33, 0],
        [34, 0],
        [35, 0],
        [36, 0],
        [37, 0],
        [38, 0],
        [39, 0],
        [40, 0],
        [41, 0],
        [42, 0],
    ]
    full, _ = get_index_full()
    return coordinates, [full.index(item) for item in coordinates]


def get_index_middle() -> tuple[list[list[int]], list[int]]:
    """24-element sub-array of the full layout."""

    coordinates = [
        [11, 0],
        [11, 1],
        [11, 4],
        [12, 0],
        [12, 1],
        [12, 4],
        [13, 0],
        [14, 0],
        [15, 0],
        [16, 0],
        [17, 0],
        [18, 0],
        [19, 0],
        [20, 0],
        [21, 0],
        [22, 0],
        [22, 1],
        [22, 4],
        [23, 0],
        [23, 1],
        [23, 4],
        [24, 0],
        [25, 0],
        [26, 0],
    ]
    full, _ = get_index_full()
    return coordinates, [full.index(item) for item in coordinates]


def get_index_small() -> tuple[list[list[int]], list[int]]:
    """10-element sub-array of the full layout."""

    coordinates = [
        [11, 0],
        [11, 1],
        [12, 0],
        [12, 1],
        [13, 0],
        [14, 0],
        [15, 0],
        [16, 0],
        [17, 0],
        [18, 0],
    ]
    full, _ = get_index_full()
    return coordinates, [full.index(item) for item in coordinates]


class RadarArrayGeometry(nn.Module):
    """Declared TX/RX geometry consumed by direct mesh-surface paths."""

    tx_position: Tensor
    rx_positions: Tensor

    def __init__(
        self,
        radar_config: RadarConfig | None = None,
        *,
        radar_position: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        config = (
            radar_config
            if radar_config is not None
            else get_radar_config(ACTIVE_MMWCAS_TDM_CONFIG_ID)
        )
        position = DEFAULT_RADAR_POSITION if radar_position is None else radar_position
        if len(position) != 3:
            raise ValueError("radar_position must contain exactly three coordinates")
        try:
            position_array = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("radar_position must contain numeric coordinates") from error
        if not bool(np.isfinite(position_array).all()):
            raise ValueError("radar_position must contain finite coordinates")
        array_grid = np.asarray(get_index_full()[0], dtype=np.float64) - np.asarray([43, 0])
        spacing = (SIMULATION_LIGHT_SPEED / config.start_freq) / 2.0
        receiver_offsets = (
            np.column_stack((array_grid[:, 0], array_grid[:, 1], np.zeros(len(array_grid))))
            * spacing
        )
        # Keep float64 layout buffers for the same declared array geometry;
        # the surface backend converts them to its active vertex dtype.
        self.register_buffer("tx_position", torch.tensor(position_array), persistent=False)
        self.register_buffer(
            "rx_positions", torch.tensor(position_array[None] + receiver_offsets), persistent=False
        )
