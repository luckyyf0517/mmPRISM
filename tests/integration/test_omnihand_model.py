from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from mmprism.models import (  # noqa: E402
    CubeNetSpatialEncoder,
    OmniHandCubeNet,
    PathAggregationFPN3D,
    TemporalTransformerAggregator,
)


def _spatial_encoder(
    *,
    attention: tuple[bool, bool, bool] = (True, True, True),
    use_pafpn: bool = True,
) -> CubeNetSpatialEncoder:
    channel, spatial, se = attention
    return CubeNetSpatialEncoder(
        in_channels=4,
        stem_channels=8,
        stage_channels=(8, 16, 32),
        stage_depths=(1, 1, 1),
        channel_attention=channel,
        spatial_attention=spatial,
        se_attention=se,
        use_pafpn=use_pafpn,
        fpn_channels=32,
    )


def _model() -> OmniHandCubeNet:
    spatial = _spatial_encoder()
    temporal = TemporalTransformerAggregator(
        spatial.feature_dim,
        max_frames=4,
        layers=2,
        heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    return OmniHandCubeNet(spatial, temporal)


def test_omnihand_runs_temporal_and_single_frame_forward_backward() -> None:
    torch.manual_seed(31)
    model = _model()
    radar_cube = torch.rand(2, 3, 4, 9, 7, 5)
    frame_mask = torch.tensor([[True, True, True], [True, True, False]])

    output = model(radar_cube, frame_mask)
    assert output.joints.shape == (2, 2, 24, 3)
    assert output.frame_features.shape == (2, 3, 32)
    assert output.sequence_features.shape == (2, 32)
    assert torch.all(torch.isfinite(output.joints))

    output.joints.square().mean().backward()
    assert model.spatial_encoder.stem.conv.weight.grad is not None
    assert model.temporal_aggregator.encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.pose_head[1].weight.grad is not None

    single = model.forward_single_frame(radar_cube[:, 0])
    assert single.joints.shape == (2, 2, 24, 3)
    assert single.frame_features.shape == (2, 1, 32)


def test_temporal_padding_values_cannot_change_valid_prediction() -> None:
    torch.manual_seed(37)
    model = _model().eval()
    radar_cube = torch.rand(2, 3, 4, 9, 7, 5)
    frame_mask = torch.tensor([[True, True, True], [True, True, False]])
    changed = radar_cube.clone()
    changed[1, 2] = 1000

    with torch.inference_mode():
        baseline = model(radar_cube, frame_mask).joints
        counterfactual = model(changed, frame_mask).joints

    torch.testing.assert_close(baseline, counterfactual, rtol=0, atol=0)


@pytest.mark.parametrize(
    "disabled",
    ["channel", "spatial", "se"],
)
def test_attention_components_are_independently_ablatable(disabled: str) -> None:
    enabled = {"channel": True, "spatial": True, "se": True}
    enabled[disabled] = False
    encoder = _spatial_encoder(attention=(enabled["channel"], enabled["spatial"], enabled["se"]))
    first_block = encoder.stages[0][0]

    assert encoder.attention_configuration[disabled] is False
    assert isinstance(getattr(first_block, f"{disabled}_attention"), nn.Identity)
    output = encoder(torch.rand(2, 4, 9, 7, 5))
    assert output.shape == (2, 32)


def test_pafpn_preserves_each_level_shape_with_odd_spatial_sizes() -> None:
    neck = PathAggregationFPN3D((4, 8, 16), out_channels=6)
    levels = (
        torch.rand(2, 4, 9, 7, 5),
        torch.rand(2, 8, 5, 4, 3),
        torch.rand(2, 16, 3, 2, 2),
    )

    outputs = neck(levels)

    assert [tuple(level.shape) for level in outputs] == [
        (2, 6, 9, 7, 5),
        (2, 6, 5, 4, 3),
        (2, 6, 3, 2, 2),
    ]
