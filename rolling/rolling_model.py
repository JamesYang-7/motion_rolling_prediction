# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch as th
import utils.constants as constants
from data_loaders.dataloader import TrainDataset
from evaluation.utils import BodyModelsWrapper
from utils.constants import (
    DataTypeGT,
    LossDistType,
    ModelOutputType,
    MotionLossType,
    PredictionTargetType,
)
from utils.utils_transform import bgs, sixd2aa
from collections.abc import Callable


def loss_distance(
    a: th.Tensor,
    b: th.Tensor,
    dist_type: LossDistType = LossDistType.L2,
    joint_dim: int = 6,
    joint_weights: Optional[th.Tensor] = None,
):
    if dist_type == LossDistType.L1:
        p = 1
    elif dist_type == LossDistType.L2:
        p = 2
    else:
        raise ValueError(f"Unknown loss type {dist_type}")

    bs, n, c = a.shape
    loss = (
        th.norm(
            (a - b).reshape(-1, joint_dim),
            p,
            1,
        )
        .reshape(bs, n, c // joint_dim)
    )
    if joint_weights is not None:
        loss = loss * joint_weights.to(loss)[None, None, :]
    return loss.mean(dim=-1)


def loss_velocity(
    a: th.Tensor,
    b: th.Tensor,
    dist_type: LossDistType = LossDistType.L2,
    joint_dim: int = 6,
    joint_weights: Optional[th.Tensor] = None,
):
    bs, n, c = a.shape
    loss = th.zeros((bs, n), device=a.device)
    a_vel = a[:, 1:] - a[:, :-1]
    b_vel = b[:, 1:] - b[:, :-1]
    loss[:, :-1] = loss_distance(a_vel, b_vel, dist_type, joint_dim, joint_weights)
    return loss


class RollingPredictionModel():
    def __init__(
        self,
        mask_cond_fn:Callable,
        rolling_prediction_window:int=10,
        rolling_motion_ctx:int=10,
        rolling_sparse_ctx:int=10,
        rolling_fr_frames:int=0,
        rolling_latency:int=0,
        target_type:PredictionTargetType=PredictionTargetType.POSITIONS,
        loss_dist_type:LossDistType=LossDistType.L2,
        loss_velocity:float=0.0,
        loss_fk:float=0.0,
        loss_fk_vel:float=0.0,
        support_dir:Optional[str]=None,
        loss_hand_align:float=0.0,
        loss_lower_weight:float=1.0,
    ):
        self.rolling_prediction_window = rolling_prediction_window
        self.motion_cxt_len = rolling_motion_ctx
        self.sparse_cxt_len = rolling_sparse_ctx
        self.lat = rolling_latency
        self.max_freerunning_steps = rolling_fr_frames
        self.mask_cond_fn = mask_cond_fn
        self.target_type = target_type

        # losses
        self.loss_dist_type = loss_dist_type
        self.loss_velocity = loss_velocity
        self.loss_fk = loss_fk
        self.loss_fk_vel = loss_fk_vel
        # portable tricks (VR_Pose_Pred research/progress.md §6 item 10)
        self.loss_hand_align = loss_hand_align   # soft hand alignment, FK wrists vs lagged controllers
        self.loss_lower_weight = loss_lower_weight   # lower-body weight on the FK position terms
        self.fk_joint_weights = None
        if loss_lower_weight != 1.0:
            self.fk_joint_weights = th.ones(constants.NUM_JOINTS_SMPL)
            self.fk_joint_weights[list(constants.LOWER_BODY_JOINTS)] = loss_lower_weight
        if (
            self.loss_fk > 0
            or self.loss_fk_vel > 0
            or self.loss_hand_align > 0
        ):
            assert support_dir is not None, "Support dir is required for FK loss"
            self.body_model = BodyModelsWrapper(support_dir)

    def process_prediction_through_fk(
        self,
        output: dict,
        gt_data: dict,
        body_model: BodyModelsWrapper,
        dataset: TrainDataset,
    ):
        gender_fk = gt_data[DataTypeGT.SMPL_GENDER][0]
        model_type = gt_data[DataTypeGT.SMPL_MODEL_TYPE][0]
        local_rots = dataset.inv_transform(output[ModelOutputType.RELATIVE_ROTS])
        bs, seq_len = local_rots.shape[:2]
        local_rots_aa = sixd2aa(local_rots.reshape(-1, 6)).float().reshape(-1, 66)
        body_model = body_model.to(local_rots_aa.device)

        body_params = {
            "root_orient": local_rots_aa[:, :3],
            "pose_body": local_rots_aa[:, 3:66],
        }
        if (
            ModelOutputType.SHAPE_PARAMS in output
            and output[ModelOutputType.SHAPE_PARAMS] is not None
        ):
            # if shape params are predicted, use them
            body_params["betas"] = (
                output[ModelOutputType.SHAPE_PARAMS]
                .repeat(1, seq_len, 1)
                .reshape(-1, 16)
            )  # [bs * seq_len, 16]
        elif DataTypeGT.SHAPE_PARAMS in gt_data:
            # if shape params are not predicted, use the GT
            body_params["betas"] = gt_data[DataTypeGT.SHAPE_PARAMS].reshape(
                -1, 16
            )  # [bs * seq_len, 16]
        # if shape params are not in GT and not predicted, use default shape
        body_pose = body_model.grad_fk(
            body_params,
            model_type,
            gender_fk,
        )
        pred_joints = body_pose.Jtr[:, : constants.NUM_JOINTS_SMPL].reshape(
            bs, seq_len, constants.NUM_JOINTS_SMPL, 3
        )
        gt_joints = gt_data[DataTypeGT.WORLD_JOINTS]
        assert (
            pred_joints.shape == gt_joints.shape
        ), f"{pred_joints.shape} != {gt_joints.shape}"
        # make them relative to the head as in HMD-Poser + add GT head translation
        pred_joints = (
            pred_joints
            - pred_joints[:, :, [constants.HEAD_JOINT_IDX]]
            + gt_joints[:, :, [constants.HEAD_JOINT_IDX]]
        )
        output[ModelOutputType.WORLD_JOINTS] = pred_joints
        return output

    def freerunning_step(
        self, model, i, x_start, cond, model_kwargs, update_context=True
    ):
        nframes = self.rolling_prediction_window
        x_start_ = x_start[:, i : i + nframes]
        x_start_[:, -1] = th.zeros_like(
            x_start_[:, -1]
        )  # we set the new frame of the long-term future to 0. This is done to let the network know which is the initial prediction, and be able to reduce uncertainty.
        cond_ = {
            DataTypeGT.SPARSE: cond[DataTypeGT.SPARSE][
                :, i : i + self.sparse_cxt_len + 1 + self.lat
            ],
            DataTypeGT.MOTION_CTX: cond[DataTypeGT.MOTION_CTX][
                :, i : i + self.motion_cxt_len
            ],
        }

        model_output = model(x_start_, cond_, **model_kwargs)

        if update_context:
            x_start[:, i : i + nframes] = model_output[ModelOutputType.RELATIVE_ROTS]
            cond[DataTypeGT.MOTION_CTX][:, i + self.motion_cxt_len] = (
                model_output[ModelOutputType.RELATIVE_ROTS][:, 0]
            )
        return model_output

    def run_freerunning(self, model, gt_data, cond, model_kwargs=None):
        nframes = self.rolling_prediction_window
        x_start = gt_data[
            DataTypeGT.RELATIVE_ROTS
        ].clone()  # we initialize noisy prediction with GT for the first iteration.
        x_start[:, nframes:] = 0.0  # we set the future to 0
        cond[DataTypeGT.MOTION_CTX] = cond[DataTypeGT.MOTION_CTX].clone()

        fr = th.randint(0, self.max_freerunning_steps + 1, (1,))[0].item()

        # slice gt_data so that it matches the motion segment predicted after the FreeRunning frames
        s0 = fr # start timeframe where we start to compute loss
        s1 = fr + nframes  # end timeframe where we stop computing loss
        gt_data = {
            k: (
                gt_data[k][:, s0:s1]
                if isinstance(gt_data[k], th.Tensor) and len(gt_data[k].shape) >= 3
                else gt_data[k]
            )
            for k in gt_data.keys()
        }

        # NO GRAD stage - FreeRunning stage
        model.eval()
        with th.no_grad():
            for i in range(int(fr)):
                self.freerunning_step(
                    model, i, x_start, cond, model_kwargs, update_context=True
                )
        model.train()
        # GRAD stage
        model_output = self.freerunning_step(
            model,
            fr,
            x_start,
            cond,
            model_kwargs,
            update_context=False,
        )

        last_ctx_frame = int(fr + self.motion_cxt_len)
        prev_pred = x_start[:, fr : fr + nframes]
        return model_output, gt_data, prev_pred, last_ctx_frame

    def compute_losses(
        self,
        output: dict,
        gt_data: dict,
        dataset: Optional[TrainDataset],
    ):
        """
        Computes the losses given:
        - output: the output dictionary of the model
        - gt_data: the ground truth dictionary
        """
        terms = {}
        terms[MotionLossType.LOSS] = 0
        hand_offset = gt_data.get("_hand_offset", None)
        terms[MotionLossType.ROT_MSE] = loss_distance(
            gt_data[DataTypeGT.RELATIVE_ROTS],
            output[ModelOutputType.RELATIVE_ROTS],
            dist_type=self.loss_dist_type,
        )
        terms[MotionLossType.LOSS] += terms[MotionLossType.ROT_MSE]

        if self.loss_velocity != 0:
            terms[MotionLossType.VEL_MSE] = loss_velocity(
                gt_data[DataTypeGT.RELATIVE_ROTS],
                output[ModelOutputType.RELATIVE_ROTS],
                dist_type=self.loss_dist_type,
            )
            terms[MotionLossType.LOSS] += (
                terms[MotionLossType.VEL_MSE] * self.loss_velocity
            )
        if self.loss_fk != 0 or self.loss_fk_vel != 0 or self.loss_hand_align > 0:
            assert (
                dataset is not None
            ), "Dataset is required for FK loss (inv transform)"
            output = self.process_prediction_through_fk(
                output, gt_data, self.body_model, dataset=dataset
            )
            bs, seq_len = output[ModelOutputType.WORLD_JOINTS].shape[:2]
            pred_joints = output[ModelOutputType.WORLD_JOINTS].reshape(
                (bs, seq_len, -1)
            )
            gt_joints = gt_data[DataTypeGT.WORLD_JOINTS].reshape((bs, seq_len, -1))
            if self.loss_fk != 0:
                terms[MotionLossType.JOINTS_MSE] = loss_distance(
                    pred_joints,
                    gt_joints,
                    dist_type=self.loss_dist_type,
                    joint_dim=3,
                    joint_weights=self.fk_joint_weights,
                )
                terms[MotionLossType.LOSS] += (
                    terms[MotionLossType.JOINTS_MSE] * self.loss_fk
                )
            if self.loss_fk_vel != 0:
                terms[MotionLossType.JOINTS_VEL_MSE] = loss_velocity(
                    pred_joints,
                    gt_joints,
                    dist_type=self.loss_dist_type,
                    joint_dim=3,
                    joint_weights=self.fk_joint_weights,
                )
                terms[MotionLossType.LOSS] += (
                    terms[MotionLossType.JOINTS_VEL_MSE]
                    * self.loss_fk_vel
                )
            if self.loss_hand_align > 0:
                terms[MotionLossType.HAND_ALIGN] = self.hand_alignment_loss(
                    output[ModelOutputType.WORLD_JOINTS], gt_data, hand_offset
                )
                terms[MotionLossType.LOSS] += terms[MotionLossType.HAND_ALIGN] * self.loss_hand_align

        return terms

    def hand_alignment_loss(self, pred_joints, gt_data, hand_offset):
        """Soft hand alignment (AvatarJLM-style; VR_Pose_Pred utils/loss.py
        lambda_hand): distance between the head-anchored FK wrists and the
        lagged controller positions (+ R_controller @ learned offset), masked
        where the hand was unobserved (confidence) or held by the train
        masker. Returned as (bs, n) whose mean equals
        sum(err) / (3 * #valid hand-frames), the reduction our model uses."""
        tgt = gt_data[DataTypeGT.HAND_TARGET]                      # (bs, n, 2, 3)
        valid = gt_data[DataTypeGT.HAND_VALID].to(tgt.dtype)      # (bs, n, 2)
        if DataTypeGT.HAND_HELD in gt_data and gt_data[DataTypeGT.HAND_HELD] is not None:
            valid = valid * (~gt_data[DataTypeGT.HAND_HELD]).to(tgt.dtype)
        if hand_offset is not None:
            bs, n = tgt.shape[:2]
            R_c = bgs(gt_data[DataTypeGT.HAND_ROT].reshape(-1, 6)).reshape(bs, n, 2, 3, 3)
            tgt = tgt + th.einsum("bnkij,kj->bnki", R_c, hand_offset)
        wrists = pred_joints[:, :, [constants.LHAND_JOINT_IDX, constants.RHAND_JOINT_IDX]]  # (bs, n, 2, 3)
        err = wrists - tgt
        err = err.abs() if self.loss_dist_type == LossDistType.L1 else err ** 2
        err = err * valid.unsqueeze(-1)
        bs, n = err.shape[:2]
        return err.sum(dim=(-1, -2)) * (bs * n) / (3.0 * valid.sum()).clamp_min(1.0)

    def training_losses(
        self, model, gt_data, cond, model_kwargs=None, dataset=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        if self.loss_hand_align > 0:
            # frames the train masker masked must not be pulled to a hand the
            # model did not see: map the sparse-window mask onto the gt window
            # (gt frame j <-> sparse frame sparse_ctx + j; the gt window runs
            # past the sparse window into the future, which is never masked)
            cond[DataTypeGT.SPARSE], held = self.mask_cond_fn(
                cond[DataTypeGT.SPARSE], True, return_mask=True
            )
            n_gt = gt_data[DataTypeGT.HAND_VALID].shape[1]
            held_gt = th.zeros_like(gt_data[DataTypeGT.HAND_VALID], dtype=th.bool)
            m = max(0, min(held.shape[1] - self.sparse_cxt_len, n_gt))
            held_gt[:, :m] = held[:, self.sparse_cxt_len : self.sparse_cxt_len + m]
            gt_data[DataTypeGT.HAND_HELD] = held_gt
            inner = model.module if hasattr(model, "module") else model
            gt_data["_hand_offset"] = getattr(inner.model, "hand_offset", None)
        else:
            cond[DataTypeGT.SPARSE] = self.mask_cond_fn(
                cond[DataTypeGT.SPARSE], True
            )
        if self.max_freerunning_steps > 0:
            model_output, gt_data, _, _ = self.run_freerunning(
                model, gt_data, cond, model_kwargs
            )
        else:
            # no freerunning
            x_start = gt_data[DataTypeGT.RELATIVE_ROTS]
            model_output = model(x_start, cond, **model_kwargs)

        # it returns dictionary of losses outputs with shape (bs, sl)
        return self.compute_losses(
            model_output,
            gt_data,
            dataset,
        )
