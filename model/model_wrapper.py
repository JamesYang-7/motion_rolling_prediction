# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np

import torch as th
import torch.nn as nn
from utils.utils_transform import bgs
from utils.constants import (
    DataTypeGT,
    ModelOutputType,
    PredictionInputType,
    PredictionTargetType,
    sparse_point_slices,
)


def canonicalize_sparse_xy(sparse: th.Tensor) -> th.Tensor:
    """Subtract the window's last (= current) frame's head xy from the xy of
    every present position block (head, both hands); rotations and velocities
    untouched. Port of VR_Pose_Pred model_wrapper._canonicalize_xy. A zeroed
    (confidence-masked) hand block must stay zero, hence the presence test."""
    head_pos = sparse_point_slices(0)[2]
    ref = sparse[:, -1:, head_pos.start : head_pos.start + 2]  # (B, 1, 2)
    sparse = sparse.clone()
    for k in range(3):
        sl = sparse_point_slices(k)[2]
        present = (sparse[..., sl].abs().sum(-1, keepdim=True) > 0).to(sparse.dtype)
        sparse[..., sl.start : sl.start + 2] = sparse[..., sl.start : sl.start + 2] - ref * present
    return sparse


def _yaw_matrix(angle: th.Tensor) -> th.Tensor:
    """(B,) angles -> (B, 3, 3) rotations about the vertical (z) axis."""
    c, s = th.cos(angle), th.sin(angle)
    zero, one = th.zeros_like(angle), th.ones_like(angle)
    return th.stack([
        th.stack([c, -s, zero], -1),
        th.stack([s, c, zero], -1),
        th.stack([zero, zero, one], -1),
    ], -2)


def _rotate_vec3(x: th.Tensor, rz: th.Tensor) -> th.Tensor:
    """Apply a per-sample rotation rz (B, 3, 3) to (B, T, 3k) blocks of
    3-vectors. Linear, so all-zero (masked) blocks stay zero. A 6D rotation is
    the first two columns of its matrix (utils_transform.matrot2sixd), so
    left-multiplying the matrix by rz is exactly this map on its two 3-vectors."""
    B, T, D = x.shape
    return th.einsum("bij,btkj->btki", rz, x.reshape(B, T, D // 3, 3)).reshape(B, T, D)


ROOT_ROT = slice(0, 6)   # pelvis (global orientation) 6D in the 132-d motion vector

ALL_FUNCTIONS = {
    PredictionTargetType.PCAF_COSINE: lambda num_timesteps: [
        1 - math.cos(t / num_timesteps * math.pi / 2)
        for t in range(1, num_timesteps + 1)
    ],
    PredictionTargetType.PCAF_COSINE_SQ: lambda num_timesteps: [
        1 - (math.cos(t / num_timesteps * math.pi / 2)) ** 2
        for t in range(1, num_timesteps + 1)
    ],
    PredictionTargetType.PCAF_LINEAR: lambda num_timesteps: [
        t / num_timesteps for t in range(1, num_timesteps + 1)
    ],
}


class ModelWrapper(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        target_type: PredictionTargetType,
        prediction_input_type: PredictionInputType,
        prediction_length: int,
        sparse_canon_xy: bool = False,
        sparse_canon_heading: str = "none",
        heading_ref_noise_deg: float = 0.0,
    ):
        super().__init__()
        self.sparse_canon_xy = sparse_canon_xy
        # Heading (yaw) canonicalisation -- port of VR_Pose_Pred
        # model_wrapper (canonicalize_heading: pelvis, research/progress.md
        # section 6 item 14b). Every world-frame quantity the network sees is
        # rotated about the vertical axis so the root yaw of the last committed
        # pose (the last motion-context frame) faces +x: the sparse rotations,
        # positions and position velocities of all three points (rotation
        # velocities are body-frame R_{t-1}^T R_t, yaw-invariant), and the
        # pelvis orientation of the motion context; the network's raw pelvis
        # output is rotated back to world before RPM's own (world-frame) PCAF
        # blend, so the anchor window and its zero new-frame slot are untouched. Pelvis rotations are normalised (gorp_mean/std), so they are
        # de- and re-normalised around the rotation (set_motion_normalization).
        # The reference is GT in teacher-forced frames and the model's own at
        # rollout, hence Gaussian yaw noise on it in training
        # (heading_ref_noise_deg, std; ours: 9 deg = measured rollout
        # pelvis-yaw error). Rotates about the canon-xy origin, so it requires
        # sparse_canon_xy.
        assert sparse_canon_heading in ("none", "pelvis"), sparse_canon_heading
        assert sparse_canon_heading == "none" or sparse_canon_xy, \
            "sparse_canon_heading rotates about the canon-xy origin; set --sparse_canon_xy"
        self.sparse_canon_heading = sparse_canon_heading
        self.heading_ref_noise_deg = float(heading_ref_noise_deg)
        self.root_mean = None   # set_motion_normalization
        self.root_std = None
        self._norm_set = False
        self.target_type = target_type
        self.prediction_input_type = prediction_input_type
        self.model = model
        self.prediction_length = prediction_length
        self.DEFAULT_NUM_BETAS = 16

    def set_motion_normalization(self, mean, std):
        """Pelvis-rotation slice of the dataset's motion normalisation
        (dataloader transform: (x - mean) / (std + 1e-8)); required by heading
        canonicalisation. Pass None, None under --no_normalization."""
        self._norm_set = True
        if mean is None:
            self.root_mean = self.root_std = None
            return
        self.root_mean = th.as_tensor(mean).reshape(-1)[ROOT_ROT].float().clone()
        self.root_std = th.as_tensor(std).reshape(-1)[ROOT_ROT].float().clone() + 1e-8

    def _rotate_root(self, x, rz, keep_zero_frames=False):
        """Rotate the (normalised) pelvis orientation of a (B, T, 132) motion
        tensor by rz. keep_zero_frames: frames whose pelvis block is exactly
        zero stay zero -- RPM's 'new frame' placeholder at the end of the PCAF
        anchor window (generators.py, rolling_model.freerunning_step)."""
        root = x[..., ROOT_ROT]
        if self.root_mean is not None:
            m, sd = self.root_mean.to(x), self.root_std.to(x)
            out = (_rotate_vec3(root * sd + m, rz) - m) / sd
        else:
            out = _rotate_vec3(root, rz)
        if keep_zero_frames:
            out = th.where((root == 0).all(-1, keepdim=True), root, out)
        x = x.clone()
        x[..., ROOT_ROT] = out
        return x

    def _reference_yaw(self, motion_ctx):
        """(B,) world yaw of the pelvis in the last motion-context frame."""
        root = motion_ctx[:, -1, ROOT_ROT].float()
        if self.root_mean is not None:
            root = root * self.root_std.to(root) + self.root_mean.to(root)
        facing = bgs(root)[:, :, 2]                      # pelvis local +z = facing
        yaw = th.atan2(facing[:, 1], facing[:, 0])
        if self.training and self.heading_ref_noise_deg > 0:
            yaw = yaw + th.randn_like(yaw) * math.radians(self.heading_ref_noise_deg)
        return yaw

    def _rotate_sparse(self, sparse, rz):
        """World -> canonical on the 54-d sparse window: global rotation (6D),
        position and position velocity of the three points."""
        sparse = sparse.clone()
        for k in range(3):
            rot, _rotvel, pos, posvel = sparse_point_slices(k)
            for sl in (rot, pos, posvel):
                sparse[..., sl] = _rotate_vec3(sparse[..., sl], rz)
        return sparse

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def convert_to_fp16(self):
        self.model.convert_to_fp16()

    def transform_model_output(self, model_output, prev_pred=None):
        """
        The model_output is the clean motion. This method implements several transformations in the output of the
        model, depending on the target_type.
        """
        if self.target_type == PredictionTargetType.POSITIONS:
            return model_output
        elif (
            self.target_type == PredictionTargetType.PCAF_COSINE
            or self.target_type == PredictionTargetType.PCAF_COSINE_SQ
            or self.target_type == PredictionTargetType.PCAF_LINEAR
        ):
            assert prev_pred is not None, "prev_pred is required for PCAF reparameterization"
            uncertainty = th.tensor(ALL_FUNCTIONS[self.target_type](self.prediction_length))
            uncertainty = uncertainty.to(device=model_output.device)
            return prev_pred + uncertainty[None, :, None] * th.tanh(model_output - prev_pred)
        else:
            raise NotImplementedError

    def forward(self, prev_pred, cond, **kwargs):
        """
        prev_pred: previous prediction
        t: timestep
        cond: conditioning
        """
        if self.prediction_input_type == PredictionInputType.NONE:
            model_input = th.zeros_like(prev_pred)
        elif self.prediction_input_type == PredictionInputType.CLEAN:
            # this was for ablation in the supp. mat.
            model_input = prev_pred
        elif self.prediction_input_type == PredictionInputType.NOISY:
            # this was for ablation in the supp. mat. It wasn't adapted during code refactor.
            # you could implement it by implementing a pseudo-q-sample function from diffusion models.
            raise NotImplementedError 
        else:
            raise NotImplementedError

        if self.sparse_canon_xy:
            cond = dict(cond)
            cond[DataTypeGT.SPARSE] = canonicalize_sparse_xy(cond[DataTypeGT.SPARSE])
        rz = None
        if self.sparse_canon_heading != "none":
            if not self._norm_set:
                raise RuntimeError("heading canonicalisation needs set_motion_normalization(mean, std)")
            rz = _yaw_matrix(-self._reference_yaw(cond[DataTypeGT.MOTION_CTX]))   # world -> canonical
            cond = dict(cond)
            cond[DataTypeGT.SPARSE] = self._rotate_sparse(cond[DataTypeGT.SPARSE].float(), rz)
            cond[DataTypeGT.MOTION_CTX] = self._rotate_root(cond[DataTypeGT.MOTION_CTX].float(), rz)
            if self.prediction_input_type == PredictionInputType.CLEAN:
                model_input = self._rotate_root(model_input.float(), rz, keep_zero_frames=True)
            # The PCAF anchor (prev_pred) stays in world: RPM seeds each new
            # frame at 0 (normalised) and moves it by at most tanh(.) per
            # step, and in the canonical frame the pelvis sits ~3 std from the
            # world-normalisation mean in every frame -- blending there pins
            # the pelvis against the tanh bound (2026-09-28 run: pelvis
            # jitter 2.4 vs 0.75 deg/frame). So only the raw network output is
            # rotated back, before the (upstream, world-frame) blend.
        model_output = self.model(model_input.float(), cond, **kwargs)
        assert isinstance(model_output, dict), "model output must be a dict"
        assert (
            ModelOutputType.RELATIVE_ROTS in model_output.keys()
        ), "RELATIVE_ROTS must be in model output"
        relative_rots = model_output[ModelOutputType.RELATIVE_ROTS]
        if rz is not None:
            # raw pelvis output canonical -> world, then the upstream blend
            relative_rots = self._rotate_root(relative_rots, rz.transpose(1, 2))
        model_output[ModelOutputType.RELATIVE_ROTS] = self.transform_model_output(
            relative_rots, prev_pred
        )
        return model_output
