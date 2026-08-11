"""Versioned acquisition and verification of external research assets."""

from mmprism.assets.models import (
    MODEL_ASSET_COLLECTION_SCHEMA,
    MODEL_ASSET_CONFIG_SCHEMA,
    MODEL_ASSET_MANIFEST_SCHEMA,
    MODEL_ASSET_SMOKE_SCHEMA,
    ModelAssetError,
    ModelAssetSetConfig,
    ModelAssetSpec,
    ResolvedModelAsset,
    download_model_assets,
    load_model_asset_config,
    plan_model_assets,
    resolve_model_asset,
    run_model_asset_smoke,
    verify_model_assets,
    write_model_asset_smoke,
)

__all__ = [
    "MODEL_ASSET_COLLECTION_SCHEMA",
    "MODEL_ASSET_CONFIG_SCHEMA",
    "MODEL_ASSET_MANIFEST_SCHEMA",
    "MODEL_ASSET_SMOKE_SCHEMA",
    "ModelAssetError",
    "ModelAssetSetConfig",
    "ModelAssetSpec",
    "ResolvedModelAsset",
    "download_model_assets",
    "load_model_asset_config",
    "plan_model_assets",
    "resolve_model_asset",
    "run_model_asset_smoke",
    "verify_model_assets",
    "write_model_asset_smoke",
]
