"""Dual-decoder pose objective for personal reconstruction adaptation.

The direct-coordinate decoder and fixed-body decoder each contribute half of
the pose loss, using the same reference poses and validity mask.
"""

from mmprism.training.reconstruction.dense_token_direct_xyz import masked_direct_xyz_coordinate_l1


def pose_only_terms(*, prediction, fk_full24, target_global, valid_mask):
    """Reuse the producer coordinate loss for direct and fixed-body FK heads."""
    shape = valid_mask.shape
    direct = masked_direct_xyz_coordinate_l1(
        predicted_coarse18_m=prediction.direct.coarse18_m.reshape(*shape, 18),
        predicted_hand120_wrist_relative_m=prediction.direct.hand120_wrist_relative_m.reshape(
            *shape, 120
        ),
        target_global_m=target_global,
        valid_mask=valid_mask,
        capacity_frames=32,
    )
    fk = masked_direct_xyz_coordinate_l1(
        predicted_coarse18_m=fk_full24[..., :3, :].reshape(*shape, 18),
        predicted_hand120_wrist_relative_m=(fk_full24[..., 4:, :] - fk_full24[..., 2:3, :]).reshape(
            *shape, 120
        ),
        target_global_m=target_global,
        valid_mask=valid_mask,
        capacity_frames=32,
    )
    return {
        "total": 0.5 * direct.total + 0.5 * fk.total,
        "direct_xyz": direct.total,
        "fk_xyz": fk.total,
    }
