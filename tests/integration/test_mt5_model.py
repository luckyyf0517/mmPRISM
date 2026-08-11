from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mmprism.models import GeometryGuidedMT5  # noqa: E402


def _model() -> GeometryGuidedMT5:
    config = transformers.MT5Config(
        vocab_size=64,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    language_model = transformers.MT5ForConditionalGeneration(config)
    return GeometryGuidedMT5(
        language_model,
        hidden_size=32,
        radar_feature_dim=16,
        joint_count=24,
        coordinate_dim=3,
        pose_channels=(8, 16),
        temporal_kernel_size=3,
        dropout=0.0,
        label_smoothing=0.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return {
        "pose": torch.randn(2, 3, 2, 24, 3, generator=generator),
        "pose_confidence": torch.rand(2, 3, 2, 24, generator=generator),
        "radar_features": torch.randn(2, 3, 16, generator=generator),
        "frame_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "prompt_input_ids": torch.tensor([[2, 3, 1], [2, 3, 1]]),
        "prompt_attention_mask": torch.ones(2, 3, dtype=torch.long),
    }


def test_mt5_model_runs_forward_backward_and_generation() -> None:
    torch.manual_seed(23)
    model = _model()
    batch = _batch()
    labels = torch.tensor([[4, 5, 1, -100], [6, 7, 8, 1]])

    output = model(**batch, labels=labels)

    assert output.logits.shape == (2, 4, 64)
    assert output.encoder_attention_mask.shape == (2, 6)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.radar_projector.projection[0].weight.grad is not None
    assert model.fusion.confidence_gate[0].weight.grad is not None

    model.eval()
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=3, num_beams=2)
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 4


def test_confidence_gate_has_explicit_low_confidence_fallback() -> None:
    torch.manual_seed(29)
    model = _model().eval()
    batch = _batch()
    with torch.inference_mode():
        low = model.encode_modalities(
            batch["pose"],
            torch.zeros_like(batch["pose_confidence"]),
            batch["radar_features"],
            batch["frame_attention_mask"],
        )
        high = model.encode_modalities(
            batch["pose"],
            torch.ones_like(batch["pose_confidence"]),
            batch["radar_features"],
            batch["frame_attention_mask"],
        )

    assert torch.count_nonzero(low.pose_gate) == 0
    assert torch.all(high.pose_gate[:, :2] > 0)
    assert not torch.equal(low.embeddings, high.embeddings)
