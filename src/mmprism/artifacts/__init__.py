"""Versioned run artifacts and paper-facing evidence exports."""

from mmprism.artifacts.predictions import (
    PREDICTION_AGGREGATION_SCHEMA_VERSION,
    PREDICTION_INDEX_NAME,
    PREDICTION_NAME,
    PREDICTION_SHARD_SCHEMA_VERSION,
    PredictionAggregation,
    PredictionShard,
    aggregate_prediction_shards,
    write_prediction_shard,
    write_single_rank_predictions,
)
from mmprism.artifacts.prepare import (
    PREPARE_REPORT_SCHEMA_VERSION,
    PrepareError,
    build_prepare_report,
    validate_split_bindings,
)
from mmprism.artifacts.run import (
    METRICS_SCHEMA_VERSION,
    RUN_INPUTS_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    ArtifactError,
    RunArtifactWriter,
    RunInput,
    RunInputKind,
    sha256_file,
)

__all__ = [
    "METRICS_SCHEMA_VERSION",
    "PREDICTION_AGGREGATION_SCHEMA_VERSION",
    "PREDICTION_INDEX_NAME",
    "PREDICTION_NAME",
    "PREDICTION_SHARD_SCHEMA_VERSION",
    "PREPARE_REPORT_SCHEMA_VERSION",
    "RUN_INPUTS_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "ArtifactError",
    "PredictionAggregation",
    "PredictionShard",
    "PrepareError",
    "RunArtifactWriter",
    "RunInput",
    "RunInputKind",
    "build_prepare_report",
    "aggregate_prediction_shards",
    "sha256_file",
    "validate_split_bindings",
    "write_prediction_shard",
    "write_single_rank_predictions",
]
