# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np

import torch as th
import torch.nn as nn
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
    ):
        super().__init__()
        self.sparse_canon_xy = sparse_canon_xy
        self.target_type = target_type
        self.prediction_input_type = prediction_input_type
        self.model = model
        self.prediction_length = prediction_length
        self.DEFAULT_NUM_BETAS = 16

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
        model_output = self.model(model_input.float(), cond, **kwargs)
        assert isinstance(model_output, dict), "model output must be a dict"
        assert (
            ModelOutputType.RELATIVE_ROTS in model_output.keys()
        ), "RELATIVE_ROTS must be in model output"
        relative_rots = model_output[ModelOutputType.RELATIVE_ROTS]
        model_output[ModelOutputType.RELATIVE_ROTS] = self.transform_model_output(
            relative_rots, prev_pred
        )
        return model_output
