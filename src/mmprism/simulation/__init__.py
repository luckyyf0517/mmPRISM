"""Point-reflector FMCW radar simulation (port of legacy ``src/fmcw/``)."""

from mmprism.simulation.point_cloud import (
    densify_body_hand_frames,
    densify_dual_hand_pose,
    temporal_smooth_decimate,
)
from mmprism.simulation.processor import (
    Processor,
    build_steering_vector,
    build_steering_vector_1d,
)
from mmprism.simulation.radar_config import (
    IWR1843_SIM_V1,
    RADAR_CONFIG_REGISTRY,
    RadarConfig,
    RadarConfigError,
    get_radar_config,
)
from mmprism.simulation.simulator import (
    RADAR_POSITION,
    SIMULATION_LIGHT_SPEED,
    PointReflectorSimulator,
    Simulation,
    get_index_full,
    get_index_large,
    get_index_middle,
    get_index_small,
)

__all__ = [
    "IWR1843_SIM_V1",
    "RADAR_CONFIG_REGISTRY",
    "RADAR_POSITION",
    "SIMULATION_LIGHT_SPEED",
    "PointReflectorSimulator",
    "Processor",
    "RadarConfig",
    "RadarConfigError",
    "Simulation",
    "build_steering_vector",
    "build_steering_vector_1d",
    "densify_body_hand_frames",
    "densify_dual_hand_pose",
    "get_index_full",
    "get_index_large",
    "get_index_middle",
    "get_index_small",
    "get_radar_config",
    "temporal_smooth_decimate",
]
