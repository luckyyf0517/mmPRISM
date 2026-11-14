"""Differentiable Momentum Human Rig decoding and full-pose coordinate transforms.

The external surface rig and joint constants supply the body geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mmprism.data.sam3d_full24_yoke import _SIDE_SLOTS, POSE_SHAPE

HAND_POSE_DIMS = 108

UPPER_POSE_DIMS = 48

PARAM_DIMS = HAND_POSE_DIMS + UPPER_POSE_DIMS

NUM_JOINTS = 127

AXIS_FLIP = (1.0, -1.0, -1.0)

DEGENERATE_SPAN_M = 1e-6

LEFT_WRIST_SKELETON_JOINT = 42

RIGHT_WRIST_SKELETON_JOINT = 78


class MhrParamFKError(RuntimeError):
    """Raised when the portable MHR FK path cannot run."""


def _unitquat_to_rotmat(quat: Tensor) -> Tensor:
    """roma.unitquat_to_rotmat verbatim (XYZW, real part last)."""

    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return torch.stack(
        (
            x2 - y2 - z2 + w2,
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            -x2 + y2 - z2 + w2,
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            -x2 - y2 + z2 + w2,
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _batch_xyz_from_6d(poses: Tensor) -> Tensor:
    """mhr_utils.batchXYZfrom6D verbatim (6D rotation-continuity -> Euler)."""

    import torch.nn.functional as F

    x_raw, y_raw = poses[..., :3], poses[..., 3:]
    x = F.normalize(x_raw, dim=-1)
    z = F.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack([x, y, z], dim=-1)
    sy = torch.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
    singular = (sy < 1e-6).to(matrix.dtype)
    euler_x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    euler_y = torch.atan2(-matrix[..., 2, 0], sy)
    euler_z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0
    out = torch.zeros_like(matrix[..., 0])
    out = torch.stack(
        (
            euler_x * (1 - singular) + xs * singular,
            euler_y * (1 - singular) + ys * singular,
            euler_z * (1 - singular) + zs * singular,
        ),
        dim=-1,
    )
    return out


def _centered_moving_average(values: Tensor, window: int = 5) -> Tensor:
    """Centered moving average along axis 0 with replicated boundaries."""

    if window < 1 or window % 2 != 1:
        raise MhrParamFKError("smoothing window must be a positive odd integer")
    if values.ndim < 1 or values.shape[0] == 0:
        raise MhrParamFKError("smoothing input must contain at least one frame")
    radius = window // 2
    indices = torch.arange(-radius, radius + 1, device=values.device)
    frames = torch.arange(values.shape[0], device=values.device)[:, None]
    frames = (frames + indices[None, :]).clamp(0, values.shape[0] - 1)
    return values[frames].mean(dim=1)


def _rotmat_to_unitquat(matrix: Tensor) -> Tensor:
    """Convert proper rotation matrices to XYZW unit quaternions."""

    # The four candidates avoid a trace-only singularity near pi rotations.
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    candidates = (
        torch.stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            dim=-1,
        )
        .clamp_min(0)
        .sqrt()
        * 0.5
    )
    choice = candidates.argmax(dim=-1)
    eps = torch.finfo(matrix.dtype).eps
    w = candidates[..., 0]
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * w.clamp_min(eps))
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * w.clamp_min(eps))
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * w.clamp_min(eps))
    q0 = torch.stack((x, y, z, w), dim=-1)
    x = candidates[..., 1]
    q1 = torch.stack(
        (
            x,
            (matrix[..., 0, 1] + matrix[..., 1, 0]) / (4 * x.clamp_min(eps)),
            (matrix[..., 0, 2] + matrix[..., 2, 0]) / (4 * x.clamp_min(eps)),
            (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * x.clamp_min(eps)),
        ),
        dim=-1,
    )
    y = candidates[..., 2]
    q2 = torch.stack(
        (
            (matrix[..., 0, 1] + matrix[..., 1, 0]) / (4 * y.clamp_min(eps)),
            y,
            (matrix[..., 1, 2] + matrix[..., 2, 1]) / (4 * y.clamp_min(eps)),
            (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * y.clamp_min(eps)),
        ),
        dim=-1,
    )
    z = candidates[..., 3]
    q3 = torch.stack(
        (
            (matrix[..., 0, 2] + matrix[..., 2, 0]) / (4 * z.clamp_min(eps)),
            (matrix[..., 1, 2] + matrix[..., 2, 1]) / (4 * z.clamp_min(eps)),
            z,
            (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * z.clamp_min(eps)),
        ),
        dim=-1,
    )
    quat = torch.where((choice == 0)[..., None], q0, q1)
    quat = torch.where((choice == 2)[..., None], q2, quat)
    quat = torch.where((choice == 3)[..., None], q3, quat)
    return torch.nn.functional.normalize(quat, dim=-1)


def _smooth_unitquats(quat: Tensor, window: int = 5) -> Tensor:
    """Sign-align each centered window, average, and renormalize."""

    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, device=quat.device)
    centers = torch.arange(quat.shape[0], device=quat.device)
    indices = (centers[:, None] + offsets[None, :]).clamp(0, quat.shape[0] - 1)
    neighborhood = quat[indices]
    reference = quat[:, None]
    signs = torch.where(
        (neighborhood * reference).sum(dim=-1, keepdim=True) < 0,
        -torch.ones((), dtype=quat.dtype, device=quat.device),
        torch.ones((), dtype=quat.dtype, device=quat.device),
    )
    return torch.nn.functional.normalize((neighborhood * signs).mean(dim=1), dim=-1)


def map_keypoints_70_to_full24_torch(keypoints: Tensor) -> Tensor:
    """Torch port of ``map_keypoints_70_to_full24`` (pure gather)."""

    if keypoints.ndim < 2 or tuple(keypoints.shape[-2:]) != (70, 3):
        raise MhrParamFKError(f"expected [..., 70, 3] keypoints, got {tuple(keypoints.shape)}")
    used = torch.tensor(
        [slot for slots in _SIDE_SLOTS for slot in slots], dtype=torch.long, device=keypoints.device
    )
    return keypoints[..., used, :].reshape(keypoints.shape[:-2] + POSE_SHAPE)


def yoke_normalize_torch(full24: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Torch port of ``yoke_normalize``; differentiable a.e. in the points.

    Returns ``(pose_yoke, shoulder_midpoint_m, applied_yaw_rad)`` with the
    same semantics as the numpy original, including the degenerate-span
    identity-rotation guard.
    """

    if full24.ndim < 3 or tuple(full24.shape[-3:]) != POSE_SHAPE:
        raise MhrParamFKError(f"expected [..., {POSE_SHAPE}] full24, got {tuple(full24.shape)}")
    left = full24[..., 0, 0, :]
    right = full24[..., 1, 0, :]
    midpoint = (left + right) * 0.5
    delta = (right - left).to(torch.float64)
    span = torch.hypot(delta[..., 0], delta[..., 2])
    yaw = torch.where(
        span >= DEGENERATE_SPAN_M,
        torch.arctan2(delta[..., 2], delta[..., 0]),
        torch.zeros_like(span),
    )
    cos_psi = torch.cos(yaw).to(full24.dtype)
    sin_psi = torch.sin(yaw).to(full24.dtype)
    centered = full24 - midpoint[..., None, None, :]
    x = centered[..., 0]
    z = centered[..., 2]
    pose = torch.empty_like(centered)
    pose = torch.stack(
        (
            x * cos_psi[..., None, None] + z * sin_psi[..., None, None],
            centered[..., 1],
            -x * sin_psi[..., None, None] + z * cos_psi[..., None, None],
        ),
        dim=-1,
    )
    return pose, midpoint, yaw.to(torch.float32)


class MhrParamFK:
    """Portable differentiable MHR FK (joints-only fast path).

    Loads the SAM TorchScript rig (``mhr_model.pt``) directly — no
    provider-env subprocess — plus the exported constants npz, and
    decodes ``(hand[B,108], upper[B,48])`` parameters to full24 Yoke
    points ``[B, 2, 24, 3]`` in metres with gradients to the inputs.
    """

    def __init__(
        self,
        mhr_model_path: Path | str,
        constants_path: Path | str,
        device: torch.device | str = "cpu",
    ) -> None:
        mhr_path = Path(mhr_model_path)
        if not mhr_path.is_file():
            raise MhrParamFKError(f"mhr model archive is missing: {mhr_path}")
        const_path = Path(constants_path)
        if not const_path.is_file():
            raise MhrParamFKError(f"fk constants are missing: {const_path}")
        self.device = torch.device(device)
        try:
            self.mhr = torch.jit.load(str(mhr_path), map_location=self.device)
        except Exception as error:
            raise MhrParamFKError(f"unable to load MHR rig: {error}") from error
        self.mhr.eval()
        for param in self.mhr.parameters():
            param.requires_grad_(False)
        self.ct = self.mhr.character_torch
        self.pcm = self.mhr.pose_correctives_model
        self.sparse_in = getattr(self.pcm.pose_dirs_predictor, "0")

        data = np.load(const_path)

        def f32(key: str) -> Tensor:
            return torch.from_numpy(np.asarray(data[key], dtype=np.float32)).to(self.device)

        def i64(key: str) -> Tensor:
            return torch.from_numpy(np.asarray(data[key], dtype=np.int64)).to(self.device)

        self.hand_pose_mean = f32("hand_pose_mean")
        self.hand_pose_comps = f32("hand_pose_comps")
        self.idx_cont_3dof = i64("idx_cont_3dof")
        self.idx_cont_1dof = i64("idx_cont_1dof")
        self.idx_model_3dof = i64("idx_model_3dof")
        self.idx_model_1dof = i64("idx_model_1dof")
        self.hand_joint_idxs_left = i64("hand_joint_idxs_left")
        self.hand_joint_idxs_right = i64("hand_joint_idxs_right")
        self.upper_body_dims = i64("upper_body_dims")
        self.scale_mean = f32("scale_mean")
        # Older P1 constants intentionally omitted this matrix because their
        # target fixed morphology at the mean template.  Complete camera-global
        # MHR decoding additionally needs the delivered 28-D scale coordinates.
        self.scale_comps = f32("scale_comps") if "scale_comps" in data.files else None
        self.wj = f32("keypoint_wj")
        self.wv = f32("keypoint_wv")
        self.base_sub = f32("base_sub")
        self.sv_sub = f32("sv_sub")
        self.corr_w = f32("corr_w")
        self.corr_b = f32("corr_b")
        self.lbs_si = i64("lbs_si")
        self.lbs_sw = f32("lbs_sw")
        self.lbs_vlocal = i64("lbs_vlocal")
        self.lbs_rb_k = f32("lbs_rb_k")
        self.lbs_tb_k = f32("lbs_tb_k")
        self.num_sub = int(self.wv.shape[1])
        self.flip = torch.tensor(AXIS_FLIP, dtype=torch.float32, device=self.device)
        self.wrist_skeleton_joints = torch.tensor(
            (LEFT_WRIST_SKELETON_JOINT, RIGHT_WRIST_SKELETON_JOINT),
            dtype=torch.long,
            device=self.device,
        )
        parents = self.ct.skeleton.joint_parents.to(device="cpu", dtype=torch.long)
        expected_parents = torch.tensor((41, 77), dtype=torch.long)
        expected_children = ((43, 48, 52, 56, 60), (79, 84, 88, 92, 96))
        if (
            parents.shape != (NUM_JOINTS,)
            or not torch.equal(parents[self.wrist_skeleton_joints.cpu()], expected_parents)
            or any(
                tuple(torch.nonzero(parents == joint, as_tuple=False).reshape(-1).tolist())
                != children
                for joint, children in zip(
                    (LEFT_WRIST_SKELETON_JOINT, RIGHT_WRIST_SKELETON_JOINT),
                    expected_children,
                    strict=True,
                )
            )
        ):
            raise MhrParamFKError("MHR rig wrist joint topology differs")
        self._upper_idx = None  # built lazily per batch size

    # -- hand cont(54) -> model(27), mhr_utils verbatim -------------------
    def _hand_cont_to_model(self, hand_cont: Tensor) -> Tensor:
        three = hand_cont[..., self.idx_cont_3dof].unflatten(-1, (-1, 6))
        three_euler = _batch_xyz_from_6d(three).flatten(-2, -1)
        one = hand_cont[..., self.idx_cont_1dof].unflatten(-1, (-1, 2))
        one_euler = torch.atan2(one[..., 0], one[..., 1])
        model = torch.zeros(
            hand_cont.shape[:-1] + (27,), dtype=hand_cont.dtype, device=hand_cont.device
        )
        model = model.index_copy(-1, self.idx_model_3dof, three_euler)
        model = model.index_copy(-1, self.idx_model_1dof, one_euler)
        return model

    def decode_points70(self, hand: Tensor, upper: Tensor) -> Tensor:
        """(hand[B,108], upper[B,48]) -> root-space 70 points [B,70,3] m.

        Axis-flipped to the label-chain convention; differentiable in
        both inputs.
        """

        if hand.ndim != 2 or hand.shape[1] != HAND_POSE_DIMS:
            raise MhrParamFKError(
                f"hand params must be [B, {HAND_POSE_DIMS}], got {tuple(hand.shape)}"
            )
        if upper.shape != hand.shape[:1] + (UPPER_POSE_DIMS,):
            raise MhrParamFKError(
                f"upper params must be [B, {UPPER_POSE_DIMS}], got {tuple(upper.shape)}"
            )
        batch = hand.shape[0]
        device = hand.device

        body = torch.zeros(batch, 133, dtype=torch.float32, device=device)
        upper = upper.to(torch.float32)
        body = body.index_copy(1, self.upper_body_dims, upper)

        # scale_params = 0 -> scales = scale_mean row
        scales = self.scale_mean[None, :].expand(batch, -1)

        zeros6 = torch.zeros(batch, 6, dtype=torch.float32, device=device)
        full_pose = torch.cat([zeros6, body[..., :130]], dim=1)  # [B, 136] (trans*10 and rot are 0)
        hand = hand.to(torch.float32)
        both = self.hand_pose_mean[None, :] + torch.einsum(
            "da,ab->db",
            torch.cat((hand[:, :54], hand[:, 54:]), dim=0),
            self.hand_pose_comps,
        )
        model_hands = self._hand_cont_to_model(both)  # [2B, 27]
        full_pose = full_pose.index_copy(1, self.hand_joint_idxs_left, model_hands[:batch])
        full_pose = full_pose.index_copy(1, self.hand_joint_idxs_right, model_hands[batch:])

        model_params = torch.cat([full_pose, scales], dim=1)  # [B, 204]
        jp = self.ct.model_parameters_to_joint_parameters(
            torch.cat(
                [model_params, torch.zeros(batch, 45, dtype=torch.float32, device=device)], dim=1
            )
        )
        skel = self.ct.joint_parameters_to_skeleton_state(jp)  # [B, 127, 8]
        coords_cm = skel[..., :3]
        coords = coords_cm / 100
        joint_rots = _unitquat_to_rotmat(skel[..., 3:7])

        # subset vertices: template (+ identity blend = 0) + pose correctives
        rest = self.base_sub[None, :].expand(batch, -1).reshape(batch, self.num_sub, 3)
        hidden = torch.relu(self.sparse_in(self.pcm._pose_features_from_joint_params(jp)))
        corr = (hidden @ self.corr_w.T + self.corr_b).reshape(batch, self.num_sub, 3)
        posed_template = rest + corr
        local = (
            torch.einsum("eij,bej->bei", self.lbs_rb_k, posed_template[:, self.lbs_vlocal])
            + self.lbs_tb_k
        )
        skin = (
            torch.einsum("beij,bej->bei", joint_rots[:, self.lbs_si], local)
            * skel[..., 7:8][:, self.lbs_si]
            + coords_cm[:, self.lbs_si]
        ) * self.lbs_sw[:, None]
        verts_sub = torch.zeros(batch, self.num_sub, 3, dtype=torch.float32, device=device)
        verts_sub = verts_sub.index_add(1, self.lbs_vlocal, skin)
        keypoints = torch.matmul(self.wv, verts_sub / 100) + torch.matmul(
            self.wj, coords
        )  # [B, 70, 3]
        return keypoints * self.flip

    def _complete_skeleton_state(
        self,
        hand: Tensor,
        upper: Tensor,
        global_rotation: Tensor,
        scale_params: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build complete MHR skeleton state and joint parameters for decode."""

        if hand.ndim != 2 or hand.shape[1] != HAND_POSE_DIMS:
            raise MhrParamFKError(
                f"hand params must be [B, {HAND_POSE_DIMS}], got {tuple(hand.shape)}"
            )
        batch = hand.shape[0]
        if upper.shape != (batch, UPPER_POSE_DIMS):
            raise MhrParamFKError(
                f"upper params must be [B, {UPPER_POSE_DIMS}], got {tuple(upper.shape)}"
            )
        if global_rotation.shape != (batch, 3):
            raise MhrParamFKError("global rotation must be [B, 3]")
        if scale_params.shape != (batch, 28):
            raise MhrParamFKError("scale params must be [B, 28]")
        if self.scale_comps is None or self.scale_comps.shape != (28, self.scale_mean.numel()):
            raise MhrParamFKError("complete MHR decoding requires [28, 68] scale_comps constants")
        if not (
            bool(torch.isfinite(hand).all())
            and bool(torch.isfinite(upper).all())
            and bool(torch.isfinite(global_rotation).all())
            and bool(torch.isfinite(scale_params).all())
        ):
            raise MhrParamFKError("complete MHR state is non-finite")

        device = hand.device
        body = torch.zeros(batch, 133, dtype=torch.float32, device=device)
        body = body.index_copy(1, self.upper_body_dims, upper.to(torch.float32))
        global_rotation = global_rotation.to(device=device, dtype=torch.float32)
        scales = (
            self.scale_mean[None, :]
            + scale_params.to(device=device, dtype=torch.float32) @ self.scale_comps
        )

        # The MHR model parameter prefix is global translation followed by
        # global rotation.  The camera-global chain keeps global translation
        # zero here and adds its fitted camera translation after decoding.
        global_translation = torch.zeros(batch, 3, dtype=torch.float32, device=device)
        full_pose = torch.cat((global_translation, global_rotation, body[..., :130]), dim=1)
        hand = hand.to(device=device, dtype=torch.float32)
        both = self.hand_pose_mean[None, :] + torch.einsum(
            "da,ab->db",
            torch.cat((hand[:, :54], hand[:, 54:]), dim=0),
            self.hand_pose_comps,
        )
        model_hands = self._hand_cont_to_model(both)
        full_pose = full_pose.index_copy(1, self.hand_joint_idxs_left, model_hands[:batch])
        full_pose = full_pose.index_copy(1, self.hand_joint_idxs_right, model_hands[batch:])

        model_params = torch.cat((full_pose, scales), dim=1)
        jp = self.ct.model_parameters_to_joint_parameters(
            torch.cat(
                (model_params, torch.zeros(batch, 45, dtype=torch.float32, device=device)),
                dim=1,
            )
        )
        return self.ct.joint_parameters_to_skeleton_state(jp), jp

    def decode_wrist_rotations_complete(
        self,
        hand: Tensor,
        upper: Tensor,
        global_rotation: Tensor,
        scale_params: Tensor,
    ) -> Tensor:
        """Return left/right wrist bases to the final axis-flipped camera frame.

        The rows are the rig's complete ancestor-composed wrist rotations.  The
        axis flip matches :meth:`decode_points70_complete`, so a row-vector
        wrist-relative point enters its native local basis by right-multiplying
        this result.
        """

        skel, _joint_parameters = self._complete_skeleton_state(
            hand, upper, global_rotation, scale_params
        )
        joint_rots = _unitquat_to_rotmat(skel[..., 3:7])
        camera_axis_flip = torch.diag(self.flip).view(1, 1, 3, 3)
        result = camera_axis_flip @ joint_rots[:, self.wrist_skeleton_joints]
        if result.shape != (hand.shape[0], 2, 3, 3) or not bool(torch.isfinite(result).all()):
            raise MhrParamFKError("complete wrist rotation decode differs")
        return result

    def decode_points70_complete(
        self,
        hand: Tensor,
        upper: Tensor,
        global_rotation: Tensor,
        scale_params: Tensor,
    ) -> Tensor:
        """Decode the complete MHR state to root-space 70-point geometry.

        Unlike :meth:`decode_points70`, this variant consumes the delivered
        global rotation and frozen per-person morphology vector.  Translation
        and standard-person scale stay outside this method because the
        camera-global label chain applies them after the root-space MHR
        decode.  The method is intentionally evaluation-only for the dense
        deterministic MHR program; it has no training-side loss consumer.
        """

        skel, joint_parameters = self._complete_skeleton_state(
            hand, upper, global_rotation, scale_params
        )
        batch = hand.shape[0]
        device = hand.device
        coords_cm = skel[..., :3]
        joint_rots = _unitquat_to_rotmat(skel[..., 3:7])
        rest = self.base_sub[None, :].expand(batch, -1).reshape(batch, self.num_sub, 3)
        hidden = torch.relu(
            self.sparse_in(self.pcm._pose_features_from_joint_params(joint_parameters))
        )
        corr = (hidden @ self.corr_w.T + self.corr_b).reshape(batch, self.num_sub, 3)
        posed_template = rest + corr
        local = (
            torch.einsum("eij,bej->bei", self.lbs_rb_k, posed_template[:, self.lbs_vlocal])
            + self.lbs_tb_k
        )
        skin = (
            torch.einsum("beij,bej->bei", joint_rots[:, self.lbs_si], local)
            * skel[..., 7:8][:, self.lbs_si]
            + coords_cm[:, self.lbs_si]
        ) * self.lbs_sw[:, None]
        verts_sub = torch.zeros(batch, self.num_sub, 3, dtype=torch.float32, device=device)
        verts_sub = verts_sub.index_add(1, self.lbs_vlocal, skin)
        keypoints = torch.matmul(self.wv, verts_sub / 100) + torch.matmul(self.wj, coords_cm / 100)
        return keypoints * self.flip

    def decode_points70_s2(self, hand: Tensor, upper: Tensor, window: int = 5) -> Tensor:
        """Decode one episode after quaternion-aware temporal smoothing.

        The time axis is the batch axis. Hand 6D groups and skeleton rotations
        are averaged as sign-aligned quaternions; hand 2D groups, joint
        parameters, and translations are averaged linearly with replicated
        boundaries.
        """

        if hand.ndim != 2 or hand.shape[1] != HAND_POSE_DIMS:
            raise MhrParamFKError(
                f"hand params must be [T, {HAND_POSE_DIMS}], got {tuple(hand.shape)}"
            )
        if upper.shape != hand.shape[:1] + (UPPER_POSE_DIMS,):
            raise MhrParamFKError(
                f"upper params must be [T, {UPPER_POSE_DIMS}], got {tuple(upper.shape)}"
            )
        frames = hand.shape[0]
        body = torch.zeros(frames, 133, dtype=torch.float32, device=hand.device)
        body = body.index_copy(1, self.upper_body_dims, upper.to(torch.float32))
        scales = self.scale_mean[None, :].expand(frames, -1)

        both_flat = self.hand_pose_mean[None, :] + torch.einsum(
            "da,ab->db",
            torch.cat((hand[:, :54], hand[:, 54:]), dim=0).to(torch.float32),
            self.hand_pose_comps,
        )
        both = both_flat.reshape(2, frames, 54).permute(1, 0, 2)
        three = both[..., self.idx_cont_3dof].unflatten(-1, (-1, 6))
        x = torch.nn.functional.normalize(three[..., :3], dim=-1)
        z = torch.nn.functional.normalize(torch.cross(x, three[..., 3:], dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        rotations = torch.stack((x, y, z), dim=-1)
        smooth_rotations = _unitquat_to_rotmat(
            _smooth_unitquats(_rotmat_to_unitquat(rotations), window)
        )
        smooth_three = torch.cat((smooth_rotations[..., :, 0], smooth_rotations[..., :, 1]), dim=-1)
        smooth_both = both.clone()
        smooth_both[..., self.idx_cont_3dof] = smooth_three.flatten(-2, -1)
        smooth_both[..., self.idx_cont_1dof] = _centered_moving_average(
            both[..., self.idx_cont_1dof], window
        )
        model_hands = self._hand_cont_to_model(smooth_both.permute(1, 0, 2).reshape(2 * frames, 54))

        full_pose = torch.cat(
            (torch.zeros(frames, 6, dtype=torch.float32, device=hand.device), body[..., :130]),
            dim=1,
        )
        full_pose = full_pose.index_copy(1, self.hand_joint_idxs_left, model_hands[:frames])
        full_pose = full_pose.index_copy(1, self.hand_joint_idxs_right, model_hands[frames:])
        model_params = torch.cat((full_pose, scales), dim=1)
        jp = self.ct.model_parameters_to_joint_parameters(
            torch.cat(
                (model_params, torch.zeros(frames, 45, dtype=torch.float32, device=hand.device)),
                dim=1,
            )
        )
        jp = _centered_moving_average(jp, window)
        skel = self.ct.joint_parameters_to_skeleton_state(jp)
        coords_cm = skel[..., :3]
        smooth_quat = _smooth_unitquats(skel[..., 3:7], window)
        joint_rots = _unitquat_to_rotmat(smooth_quat)

        rest = self.base_sub[None, :].expand(frames, -1).reshape(frames, self.num_sub, 3)
        hidden = torch.relu(self.sparse_in(self.pcm._pose_features_from_joint_params(jp)))
        corr = (hidden @ self.corr_w.T + self.corr_b).reshape(frames, self.num_sub, 3)
        posed_template = rest + corr
        local = (
            torch.einsum("eij,bej->bei", self.lbs_rb_k, posed_template[:, self.lbs_vlocal])
            + self.lbs_tb_k
        )
        skin = (
            torch.einsum("beij,bej->bei", joint_rots[:, self.lbs_si], local)
            * skel[..., 7:8][:, self.lbs_si]
            + coords_cm[:, self.lbs_si]
        ) * self.lbs_sw[:, None]
        verts_sub = torch.zeros(frames, self.num_sub, 3, dtype=torch.float32, device=hand.device)
        verts_sub = verts_sub.index_add(1, self.lbs_vlocal, skin)
        keypoints = torch.matmul(self.wv, verts_sub / 100) + torch.matmul(self.wj, coords_cm / 100)
        return keypoints * self.flip

    def decode_full24_yoke(self, hand: Tensor, upper: Tensor) -> Tensor:
        """(hand, upper) -> full24 Yoke points [B, 2, 24, 3] in metres."""

        points70 = self.decode_points70(hand, upper)
        full24 = map_keypoints_70_to_full24_torch(points70)
        pose_yoke, _, _ = yoke_normalize_torch(full24)
        return pose_yoke

    def identity_check(
        self,
        params: np.ndarray,
        pose_3d_targets: np.ndarray,
        frames: int = 200,
        tolerance_mm: float = 1.0,
    ) -> dict[str, float]:
        """Gate: decode(label params) must equal the archived label points.

        ``params`` is ``[N, 156]`` (hand108 + upper48), ``pose_3d_targets``
        is the matching ``[N, 144]`` flat Yoke labels. Returns the max
        hand21 deviation in mm; raises when it exceeds ``tolerance_mm``.
        """

        params = np.asarray(params, dtype=np.float32)
        targets = np.asarray(pose_3d_targets, dtype=np.float32).reshape(-1, 2, 24, 3)
        if params.ndim != 2 or params.shape[1] != PARAM_DIMS:
            raise MhrParamFKError(f"params must be [N, {PARAM_DIMS}], got {params.shape}")
        if targets.shape[0] != params.shape[0]:
            raise MhrParamFKError("params and targets row counts differ")
        count = min(frames, params.shape[0])
        worst = 0.0
        with torch.no_grad():
            for start in range(0, count, 64):
                chunk = params[start : start + 64]
                hand = torch.from_numpy(chunk[:, :HAND_POSE_DIMS]).to(self.device)
                upper = torch.from_numpy(chunk[:, HAND_POSE_DIMS:]).to(self.device)
                pose = self.decode_full24_yoke(hand, upper).cpu().numpy()
                target = targets[start : start + 64]
                err = np.linalg.norm(pose - target, axis=-1)[..., 3:24]
                worst = max(worst, float(err.max()) * 1000)
        if worst > tolerance_mm:
            raise MhrParamFKError(
                f"identity check failed: max hand21 deviation {worst:.4f} mm "
                f"exceeds {tolerance_mm} mm — the portable FK disagrees with "
                "the label chain"
            )
        return {"identity_max_hand21_mm": worst, "frames_checked": count}

    def gradient_check(
        self,
        batch: int = 4,
        seed: int = 20260905,
    ) -> dict[str, float]:
        """Finite-gradient sanity: grads from a Yoke L1 reach both blocks."""

        generator = torch.Generator(device="cpu").manual_seed(seed)
        hand = torch.randn(batch, HAND_POSE_DIMS, generator=generator).to(self.device) * 0.3
        upper = torch.randn(batch, UPPER_POSE_DIMS, generator=generator).to(self.device) * 0.3
        hand = hand.clone().requires_grad_(True)
        upper = upper.clone().requires_grad_(True)
        pose = self.decode_full24_yoke(hand, upper)
        pose.abs().mean().backward()
        for name, tensor in (("hand", hand), ("upper", upper)):
            if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
                raise MhrParamFKError(f"FK gradient to {name} params is not finite")
        return {
            "hand_grad_norm": float(hand.grad.norm().item()),
            "upper_grad_norm": float(upper.grad.norm().item()),
        }
