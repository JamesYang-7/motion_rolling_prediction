# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import os
import sys

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import utils.constants as constants
from loguru import logger

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.constants import (
    ConditionMasker,
    DatasetType,
    DataTypeGT,
    SMPLGenderParam,
    SMPLModelType,
)
from pathlib import Path
from utils.utils_transform import matrot2sixd, sixd2matrot


# ---------------------------------------------------------------------------
# Portable tricks (VR_Pose_Pred research/progress.md §6 item 10): hold masking
# of confidence-masked hands and the lagged controller targets of the
# hand-alignment loss. Off unless --mask_mode hold / --loss_hand_align > 0.
# ---------------------------------------------------------------------------
def hold_hand_block(sp: torch.Tensor, valid: torch.Tensor, k: int) -> None:
    """Hold masking of hand k (1 left, 2 right) in place; port of
    VR_Pose_Pred dataloader/dataloader.py:_hold_mask_block.

    `sp` (T, 54), `valid` (T,) bool. Frames with valid=False that have an
    earlier valid frame get rot / pos copied from the last valid frame,
    rotvel = identity, posvel = 0. The first valid frame after a held run
    gets rotvel = inv(R_held) @ R_now and posvel = pos_now - pos_held (the
    true jump from the last seen hand). Frames before the first valid frame
    are left as they are (the caller zeroes them)."""
    rot_s, rotvel_s, pos_s, posvel_s = constants.sparse_point_slices(k)
    T = sp.shape[0]
    idx = torch.arange(T)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).cummax(dim=0).values
    held = (~valid) & (last >= 0)
    if held.any():
        src = last[held]
        sp[held, rot_s] = sp[src, rot_s]
        sp[held, pos_s] = sp[src, pos_s]
        sp[held, rotvel_s] = torch.tensor(constants.IDENTITY_6D, dtype=sp.dtype)
        sp[held, posvel_s] = 0.0
    regained = valid.clone()
    regained[0] = False
    regained[1:] &= held[:-1]
    if regained.any():
        prev = torch.nonzero(regained).squeeze(1) - 1
        R_prev = sixd2matrot(sp[prev, rot_s])
        R_now = sixd2matrot(sp[regained, rot_s])
        sp[regained, rotvel_s] = matrot2sixd(torch.matmul(R_prev.transpose(1, 2), R_now)).to(sp.dtype)
        sp[regained, posvel_s] = sp[regained, pos_s] - sp[prev, pos_s]


def hold_segment(sp: torch.Tensor, t0: int, t1: int, k: int, reappear_static: int = 0) -> None:
    """Hold-masking form of a train-time masker segment [t0, t1) of hand k
    on one window `sp` (n, 54): same rule as `hold_hand_block` with frame
    t0-1 as the last valid frame; a segment with no seen frame before it
    (starting at frame 0, or right after a hand that has not been tracked
    yet, i.e. an all-zero block) is zeroed instead (a leading gap). With
    `reappear_static` > 0 the frames after the segment then follow the
    reappearance rule (`static_reappear_frames`). An empty segment (the
    masker draws lengths from 0) loses no frame and changes nothing."""
    n = sp.shape[0]
    if t1 <= t0:
        return
    rot_s, rotvel_s, pos_s, posvel_s = constants.sparse_point_slices(k)
    block = constants.ENTITIES_IDCES[DatasetType.GORP][k]
    if t0 <= 0 or not bool((sp[t0 - 1, block] != 0).any()):
        for s in (rot_s, rotvel_s, pos_s, posvel_s):
            sp[t0:t1, s] = 0.0
    else:
        sp[t0:t1, rot_s] = sp[t0 - 1, rot_s]
        sp[t0:t1, pos_s] = sp[t0 - 1, pos_s]
        sp[t0:t1, rotvel_s] = torch.tensor(constants.IDENTITY_6D, dtype=sp.dtype, device=sp.device)
        sp[t0:t1, posvel_s] = 0.0
        if t1 < n:
            R_prev = sixd2matrot(sp[t0 - 1 : t0, rot_s])
            R_now = sixd2matrot(sp[t1 : t1 + 1, rot_s])
            sp[t1, rotvel_s] = matrot2sixd(torch.matmul(R_prev.transpose(1, 2), R_now))[0].to(sp.dtype)
            sp[t1, posvel_s] = sp[t1, pos_s] - sp[t0 - 1, pos_s]
    if reappear_static > 0 and t1 < n:
        static_reappear_frames(sp, list(range(t1, min(t1 + reappear_static, n))), k)


def static_reappear_frames(sp: torch.Tensor, frames, k: int) -> None:
    """Reappearance rule on the given frames of hand k, in place: rotation
    velocity -> identity, position velocity -> 0, pose untouched. Frames
    whose hand block is all zero (a leading gap: the hand was never seen)
    are skipped, as the rule only touches tracked frames."""
    _, rotvel_s, _, posvel_s = constants.sparse_point_slices(k)
    block = constants.ENTITIES_IDCES[DatasetType.GORP][k]
    for t in frames:
        if bool((sp[t, block] != 0).any()):
            sp[t, rotvel_s] = torch.tensor(constants.IDENTITY_6D, dtype=sp.dtype, device=sp.device)
            sp[t, posvel_s] = 0.0


def static_reappear_block(sp: torch.Tensor, valid: torch.Tensor, k: int, n_frames: int) -> None:
    """Reappearance rule (REAPPEAR_STATIC_FRAMES) of hand k over a whole
    sequence, in place; port of VR_Pose_Pred dataloader/dataloader.py:
    _static_reappear_block. On the first `n_frames` valid frames after an
    invalid run (a leading gap included), rotvel -> identity and posvel -> 0;
    rotation and position untouched. Removes the one-frame velocity spike a
    returning hand carries (under zero masking the stored jump from the
    tracker's placeholder, under hold the whole gap's displacement)."""
    _, rotvel_s, _, posvel_s = constants.sparse_point_slices(k)
    T = sp.shape[0]
    regained = valid.clone()
    regained[0] = False
    regained[1:] &= ~valid[:-1]
    hit = torch.zeros(T, dtype=torch.bool)
    for j in range(n_frames):
        hit[j:] |= regained[: T - j]
    hit &= valid
    if hit.any():
        sp[hit, rotvel_s] = torch.tensor(constants.IDENTITY_6D, dtype=sp.dtype)
        sp[hit, posvel_s] = 0.0


def lag_sequence(x: torch.Tensor, lag: float) -> torch.Tensor:
    """x read `lag` frames in the past along dim 0 (linear interpolation for
    fractional lags, frame 0 repeated at the start); port of VR_Pose_Pred
    train.py:lag_sparse. GORP's real tracking runs ~0.75 frame ahead of the
    mocap, so the hand-alignment target is read slightly in the past."""
    if lag <= 0.0:
        return x
    k = int(lag)
    f = lag - k

    def back(n):
        if n == 0:
            return x
        return torch.cat([x[:1].expand(n, *x.shape[1:]), x[:-n]], dim=0)

    out = back(k)
    if f > 0.0:
        out = (1.0 - f) * out + f * back(k + 1)
    return out


def _hand_confidence_masks(data_dict, input_conf_threshold: float, T: int):
    """(left, right) bool (T,) confidence-pass masks of the real input."""
    out = []
    for key in ("left_hand_confidence", "right_hand_confidence"):
        conf = torch.as_tensor(data_dict[key]).float()[:T]
        out.append(conf >= input_conf_threshold)
    return out


def hand_alignment_gt(data_dict, use_real_input: bool, input_conf_threshold: float, lag: float):
    """Per-frame targets of the hand-alignment loss for a whole sequence:
    HAND_TARGET (T, 2, 3) controller positions, HAND_ROT (T, 2, 6) controller
    rotations (for the learned rigid offset), HAND_VALID (T, 2) 1 where the
    confidence passed, all read `lag` frames in the past. A fractional lag
    mixes two frames, so validity requires both to be observed. Built from
    the raw (unmasked) signal, so independent of --mask_mode."""
    field = "hmd_position_global_full_real_list" if use_real_input else "hmd_position_global_full_gt_list"
    raw = data_dict[field].float()
    T = raw.shape[0]
    pos = torch.stack([raw[:, constants.sparse_point_slices(k)[2]] for k in (1, 2)], dim=1)   # (T, 2, 3)
    rot = torch.stack([raw[:, constants.sparse_point_slices(k)[0]] for k in (1, 2)], dim=1)   # (T, 2, 6)
    if use_real_input and input_conf_threshold > 0.0:
        valid = torch.stack(_hand_confidence_masks(data_dict, input_conf_threshold, T), dim=1).float()
    else:
        valid = torch.ones(T, 2)
    return {
        DataTypeGT.HAND_TARGET: lag_sequence(pos, lag),
        DataTypeGT.HAND_ROT: lag_sequence(rot, lag),
        DataTypeGT.HAND_VALID: (lag_sequence(valid, lag) > 0.999).float(),
    }


@dataclass
class DatasetDataStruct:
    filename_list: List[str]
    data: List[dict]
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class TrackingSignalGapsInfo:
    gaps: List[
        List[Tuple[int, int]]
    ]  # list with as many sublists of tuples (t0, t) as there are entities
    entities_idces: list  # list of idces of the entities in the sparse signal
    entities_smpl_idces: list  # list of idces of the entities in the SMPL model


def parse_data_struct(
    data_dict,
    keys_to_parse: Set[DataTypeGT],
    use_real_input=False,
    input_conf_threshold=0.0,
    mask_mode="zero",
    reappear_static=0,
):
    """
    Parse the data structure to get the gt_dict and cond_dict used
    in training and inference.
    """
    gt_dict = {}
    cond_dict = {}
    if DataTypeGT.RELATIVE_ROTS in keys_to_parse:
        gt_dict[DataTypeGT.RELATIVE_ROTS] = data_dict["rotation_local_full_gt_list"]
    if DataTypeGT.SPARSE in keys_to_parse and not use_real_input:
        cond_dict[DataTypeGT.SPARSE] = data_dict["hmd_position_global_full_gt_list"]
    elif DataTypeGT.SPARSE in keys_to_parse:
        # use real input
        assert "hmd_position_global_full_real_list" in data_dict, "No real input found"
        assert "left_hand_confidence" in data_dict, "No left-hand-conf found"
        assert "right_hand_confidence" in data_dict, "No right-hand-conf found"
        # clone: the stored sequence stays raw (the hand-alignment targets and
        # the hold rule read it), instead of upstream's in-place zeroing
        sparse = data_dict["hmd_position_global_full_real_list"].clone()  # [seq_len, 54]
        lhand_idces = constants.ENTITIES_IDCES[DatasetType.GORP][1]  # [seq_len]
        rhand_idces = constants.ENTITIES_IDCES[DatasetType.GORP][2]  # [seq_len]
        left_mask, right_mask = _hand_confidence_masks(data_dict, input_conf_threshold, sparse.shape[0])
        # below-threshold hand block -> zeros (RPM's rule) ...
        sparse[:, lhand_idces] = sparse[:, lhand_idces] * left_mask[:, None].to(sparse.dtype)
        sparse[:, rhand_idces] = sparse[:, rhand_idces] * right_mask[:, None].to(sparse.dtype)
        if mask_mode == "hold":
            # ... or the last observed hand frozen in place (leading gaps stay zero)
            hold_hand_block(sparse, left_mask, 1)
            hold_hand_block(sparse, right_mask, 2)
        elif mask_mode != "zero":
            raise ValueError(f"mask_mode must be zero|hold, got {mask_mode!r}")
        if reappear_static > 0:
            # the first frame(s) a hand is tracked again carry no velocity
            static_reappear_block(sparse, left_mask, 1, reappear_static)
            static_reappear_block(sparse, right_mask, 2, reappear_static)
        cond_dict[DataTypeGT.SPARSE] = sparse

    if DataTypeGT.GLOBAL_ROTS and "rotation_global_full_gt_list" in data_dict:
        gt_dict[DataTypeGT.GLOBAL_ROTS] = data_dict["rotation_global_full_gt_list"]
    if (
        DataTypeGT.WORLD_JOINTS in keys_to_parse
        and "position_global_full_gt_world" in data_dict
    ):
        gt_dict[DataTypeGT.WORLD_JOINTS] = data_dict["position_global_full_gt_world"]
    if DataTypeGT.BODY_PARAMS in keys_to_parse and "body_parms_list" in data_dict:
        gt_dict[DataTypeGT.BODY_PARAMS] = data_dict["body_parms_list"]
    if (
        DataTypeGT.HEAD_MOTION in keys_to_parse
        and "head_global_trans_list" in data_dict
    ):
        gt_dict[DataTypeGT.HEAD_MOTION] = data_dict["head_global_trans_list"]
    if (
        DataTypeGT.SHAPE_PARAMS in keys_to_parse
        and "body_parms_list" in data_dict
        and "betas" in data_dict["body_parms_list"]
    ):
        gt_dict[DataTypeGT.SHAPE_PARAMS] = data_dict["body_parms_list"]["betas"][1:]
    if DataTypeGT.SMPL_GENDER in keys_to_parse and "gender" in data_dict:
        gt_dict[DataTypeGT.SMPL_GENDER] = SMPLGenderParam[
            str(data_dict["gender"]).upper()
        ]
    if (
        DataTypeGT.SMPL_MODEL_TYPE in keys_to_parse
        and "surface_model_type" in data_dict
    ):
        model_type = data_dict["surface_model_type"]
        if not isinstance(model_type, str):
            model_type = model_type.item()
        gt_dict[DataTypeGT.SMPL_MODEL_TYPE] = SMPLModelType.parse(model_type)
    elif DataTypeGT.SMPL_MODEL_TYPE in keys_to_parse:
        gt_dict[DataTypeGT.SMPL_MODEL_TYPE] = SMPLModelType.SMPLH  # default

    if DataTypeGT.SHAPE_PARAMS not in gt_dict:
        # if shape params are not available, use the default shape --> retrocompatibility
        gt_dict[DataTypeGT.SMPL_GENDER] = SMPLGenderParam.MALE
    return gt_dict, cond_dict


class TrainDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        dataset_data: DatasetDataStruct,
        input_motion_length: int = 196,
        train_dataset_repeat_times: int = 1,
        no_normalization: bool = False,
        use_real_input: bool = False,
        input_conf_threshold: float = 0.0,
        mask_mode: str = "zero",
        reappear_static: int = 0,
        **kwargs,
    ):
        self.dataset = dataset
        self.data = dataset_data.data
        self.mean = dataset_data.mean
        self.std = dataset_data.std
        self.std[abs(self.std) < 1e-6] = (
            1.0  # if std is 0 --> set it to 1 (value will be set to 0 after normalization)
        )
        _dev = "cuda:0" if torch.cuda.is_available() else "cpu"  # --cpu smoke tests
        self.mean_gpu = dataset_data.mean.to(_dev)
        self.std_gpu = dataset_data.std.to(_dev)
        self.filename_list = dataset_data.filename_list
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.use_real_input = use_real_input
        self.input_conf_threshold = input_conf_threshold
        self.mask_mode = mask_mode
        self.reappear_static = reappear_static

    def __len__(self):
        return len(self.data) * self.train_dataset_repeat_times

    def transform(self, data):
        if self.no_normalization:
            return data
        return (data - self.mean) / (self.std + 1e-8)

    def inv_transform(self, data):
        if self.no_normalization:
            return data
        if data.is_cuda:
            return data * self.std_gpu + self.mean_gpu
        return data * self.std + self.mean

    def __getitem__(self, idx):
        data_dict = self.data[idx % len(self.data)]
        keys = {
            DataTypeGT.RELATIVE_ROTS,
            DataTypeGT.SPARSE,
            DataTypeGT.GLOBAL_ROTS,
            DataTypeGT.WORLD_JOINTS,
            DataTypeGT.SHAPE_PARAMS,
            DataTypeGT.SMPL_MODEL_TYPE,
            DataTypeGT.SMPL_GENDER,
        }
        gt_dict, cond_dict = parse_data_struct(
            data_dict, keys, self.use_real_input, self.input_conf_threshold, self.mask_mode,
            self.reappear_static,
        )

        seqlen = cond_dict[DataTypeGT.SPARSE].shape[0]
        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]

        # slice the data
        for k in gt_dict:
            if isinstance(gt_dict[k], torch.Tensor):
                gt_dict[k] = gt_dict[k][idx : idx + self.input_motion_length].float()
        for k in cond_dict:
            if isinstance(cond_dict[k], torch.Tensor):
                cond_dict[k] = cond_dict[k][
                    idx : idx + self.input_motion_length
                ].float()

        # Normalization
        if not self.no_normalization:
            gt_dict[DataTypeGT.RELATIVE_ROTS] = self.transform(
                gt_dict[DataTypeGT.RELATIVE_ROTS]
            )

        return gt_dict, cond_dict


class OnlineTrainDataset(TrainDataset):
    def __init__(
        self,
        dataset: str,
        dataset_data: DatasetDataStruct,
        input_motion_length: int = 196,
        train_dataset_repeat_times: int = 1,
        no_normalization: bool = False,
        sparse_context: int = 0,
        motion_context: int = 0,
        freerunning_frames: int = 0,
        latency: int = 0,  # access tu future 'latency' frames of sparse info
        use_real_input: bool = False,
        input_conf_threshold: float = 0.0,
        mask_mode: str = "zero",
        reappear_static: int = 0,
        loss_hand_align: float = 0.0,
        hand_align_lag: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            dataset,
            dataset_data,
            input_motion_length,
            train_dataset_repeat_times,
            no_normalization,
            use_real_input,
            input_conf_threshold,
            mask_mode,
            reappear_static,
        )
        self.hand_align = loss_hand_align > 0.0
        self.hand_align_lag = hand_align_lag
        self.latency = latency
        self.freerunning_frames = freerunning_frames
        self.sparse_ctx_len = sparse_context
        self.motion_ctx_len = motion_context
        self.pred_len = input_motion_length
        self.total_len = (
            input_motion_length
            + max(sparse_context, motion_context)
            + freerunning_frames
        )

    def __getitem__(self, idx):
        data_dict = self.data[idx % len(self.data)]
        keys = {
            DataTypeGT.RELATIVE_ROTS,
            DataTypeGT.SPARSE,
            DataTypeGT.GLOBAL_ROTS,
            DataTypeGT.WORLD_JOINTS,
            DataTypeGT.SHAPE_PARAMS,
            DataTypeGT.SMPL_MODEL_TYPE,
            DataTypeGT.SMPL_GENDER,
        }
        gt_dict, cond_dict = parse_data_struct(
            data_dict, keys, self.use_real_input, self.input_conf_threshold, self.mask_mode,
            self.reappear_static,
        )
        if self.hand_align:
            # sliced below like every other (T, ...) gt tensor
            gt_dict.update(hand_alignment_gt(data_dict, self.use_real_input, self.input_conf_threshold, self.hand_align_lag))

        seqlen = cond_dict[DataTypeGT.SPARSE].shape[0]
        if seqlen <= self.total_len:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.total_len), (1,))[0].item()

        # slice the data
        pred_idx0 = idx + max(self.sparse_ctx_len, self.motion_ctx_len)
        cond_dict[DataTypeGT.MOTION_CTX] = gt_dict[DataTypeGT.RELATIVE_ROTS][
            pred_idx0 - self.motion_ctx_len : pred_idx0 + self.freerunning_frames
        ].float()
        cond_dict[DataTypeGT.SPARSE] = cond_dict[DataTypeGT.SPARSE][
            pred_idx0
            - self.sparse_ctx_len : pred_idx0
            + 1
            + self.latency
            + self.freerunning_frames
        ].float()  # online --> self.context in the past and current frame

        for k in gt_dict:
            if isinstance(gt_dict[k], torch.Tensor):
                gt_dict[k] = gt_dict[k][
                    pred_idx0 : pred_idx0 + self.pred_len + self.freerunning_frames
                ]

        # Normalization
        if not self.no_normalization:
            gt_dict[DataTypeGT.RELATIVE_ROTS] = self.transform(
                gt_dict[DataTypeGT.RELATIVE_ROTS]
            )
            cond_dict[DataTypeGT.MOTION_CTX] = self.transform(
                cond_dict[DataTypeGT.MOTION_CTX]
            )

        return gt_dict, cond_dict


class TestDataset(Dataset):
    def __init__(
        self,
        name: DatasetType,
        dataset_path: str,
        no_normalization: bool,
        max_samples: int = -1,
        normalize_sparse: str = "none",
        min_frames: int = 0,
        max_frames: int = sys.maxsize,
        eval_gap_config: Optional[str] = None,
        num_features: Optional[int] = None,
        use_real_input: bool = False,
        input_conf_threshold: float = 0.0,
        test_split: str = "test",
        mask_mode: str = "zero",
        reappear_static: int = 0,
        **kwargs,
    ):
        dataset_data = load_data(
            name,
            Path(dataset_path),
            test_split,
            max_samples=max_samples,
        )
        self.name = name
        self.mean = dataset_data.mean
        self.std = dataset_data.std
        self.normalize_sparse = normalize_sparse
        self.no_normalization = no_normalization
        self.use_real_input = use_real_input
        self.input_conf_threshold = input_conf_threshold
        self.mask_mode = mask_mode
        self.reappear_static = reappear_static

        self.filename_list = []
        self.data = []
        filtered = 0
        for filename, info in zip(dataset_data.filename_list, dataset_data.data):
            hmd = info["hmd_position_global_full_gt_list"]
            if hmd.shape[0] < min_frames or hmd.shape[0] > max_frames:
                filtered += 1
                continue
            if info["rotation_local_full_gt_list"] is None:
                assert (
                    num_features is not None
                ), "num_features must be provided when GT is not available"
                info["rotation_local_full_gt_list"] = torch.zeros(
                    (hmd.shape[0], num_features)
                )
            self.data.append(info)
            self.filename_list.append(filename)
        logger.info(f"Filtered {filtered}/{len(dataset_data.data)} sequences")

        self.eval_gap_config = eval_gap_config
        if eval_gap_config is not None:
            self.tracking_gaps = self.inject_eval_gaps(name, dataset_path)
            logger.info("Tracking gaps applied to sparse tracking signal!")
        else:
            self.tracking_gaps = None

    def inject_eval_gaps(
        self, dataset_name: DatasetType, dataset_path: Path
    ) -> List[TrackingSignalGapsInfo]:
        assert self.eval_gap_config is not None, "No eval gap config provided!"
        dataset_path = dataset_path / dataset_name.value
        gaps_json = load_gaps(
            dataset_path, self.eval_gap_config
        )
        # sparse shape --> [seq_len, feats]
        masker_type = ConditionMasker.parse(gaps_json["metadata"]["masker"])
        entities_idces = ConditionMasker.get_entities_idces(masker_type, dataset_name)
        entities_smpl_idces = ConditionMasker.get_entities_smpl_idces(
            masker_type, dataset_name
        )
        all_gaps = []
        for i, filename in enumerate(self.filename_list):
            assert filename in gaps_json["gaps"], f"{filename} not in {gaps_json}"
            entities_gaps = gaps_json["gaps"][filename]
            assert len(entities_gaps) == len(entities_idces)
            for entity_idx, gaps in enumerate(entities_gaps):
                for t0, t in gaps:
                    self.data[i]["hmd_position_global_full_gt_list"][
                        t0:t, entities_idces[entity_idx]
                    ] = 0
            all_gaps.append(
                TrackingSignalGapsInfo(
                    entities_gaps, entities_idces, entities_smpl_idces
                )
            )
        return all_gaps

    def __len__(self):
        return len(self.data)

    def transform(self, data):
        if self.no_normalization:
            return data
        return (data - self.mean) / (self.std + 1e-8)

    def inv_transform(self, data):
        if self.no_normalization:
            return data
        return data * self.std + self.mean

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        keys = {
            DataTypeGT.RELATIVE_ROTS,
            DataTypeGT.SPARSE,
            DataTypeGT.GLOBAL_ROTS,
            DataTypeGT.WORLD_JOINTS,
            DataTypeGT.SHAPE_PARAMS,
            DataTypeGT.BODY_PARAMS,
            DataTypeGT.HEAD_MOTION,
            DataTypeGT.SMPL_MODEL_TYPE,
            DataTypeGT.SMPL_GENDER,
        }
        gt_dict, cond_dict = parse_data_struct(
            data_dict, keys, self.use_real_input, self.input_conf_threshold, self.mask_mode,
            self.reappear_static,
        )
        gt_dict[DataTypeGT.TRACKING_GAP] = (
            self.tracking_gaps[idx] if self.tracking_gaps else None
        )
        gt_dict[DataTypeGT.FILENAME] = self.filename_list[idx]
        gt_dict[DataTypeGT.NUM_FRAMES] = data_dict[
            "hmd_position_global_full_gt_list"
        ].shape[0]
        return gt_dict, cond_dict


def get_mean_std_path(dataset):
    return dataset + "_mean.pt", dataset + "_std.pt"


def get_motion(motion_list):
    # rotation_local_full_gt_list : 6d rotation parameters
    # hmd_position_global_full_gt_list : 3 joints(head, hands) 6d rotation/6d rotation velocity/global translation/global translation velocity
    motions = [i["rotation_local_full_gt_list"] for i in motion_list]
    sparses = [i["hmd_position_global_full_gt_list"] for i in motion_list]
    return motions, sparses


def get_path(dataset_path:Path, split:str):
    data_list_path = []
    parent_data_path = list(dataset_path.glob("*"))
    for path in parent_data_path:
        if path.is_dir():
            files = list(path.glob(f"{split}/*pt"))
            data_list_path.extend(files)
    return data_list_path


def load_data(
    dataset: DatasetType,
    dataset_path: Path,
    split: str,
    total_length:int=196,
    max_samples:int=-1,
) -> DatasetDataStruct:
    """
    Collect the data for the given split

    Args:
        - For test:
            dataset : the name of the testing dataset
            split : test or train
        - For train:
            dataset : the name of the training dataset
            split : train or test
            input_motion_length : the input motion length

    Outout:
        - For test:
            filename_list : List of all filenames in the dataset
            motion_list : List contains N dictoinaries, with
                        "hmd_position_global_full_gt_list" - sparse features of the 3 joints
                        "local_joint_parameters_gt_list" - body parameters Nx7[tx,ty,tz,rx,ry,rz] as the input of the human kinematic model
                        "head_global_trans_list" - Tx4x4 matrix which contains the global rotation and global translation of the head movement
            mean : mean of train dataset
            std : std of train dataset
        - For train:
            new_motions : motions indicates the sequences of rotation representation of each joint
            new_sparses : sparses indicates the sequences of sparse features of the 3 joints
            mean : mean of train dataset
            std : std of train dataset
    """
    dataset_path = dataset_path / dataset.value / "new_format_data"
    motion_list = get_path(dataset_path, split)
    assert len(motion_list) > 0, f"No data found in {dataset_path / split}"
    if max_samples != -1:
        motion_list = motion_list[:max_samples]
    mean_path, std_path = get_mean_std_path(dataset)
    logger.info(f"Loading '{split}' data: {dataset_path}")
    data = [torch.load(i) for i in tqdm(motion_list)]

    filename_list = [
        "-".join([i.parts[-3], i.parts[-1]]).split(".")[0]
        for i in motion_list
    ]
    if "test" in split:
        mean = torch.load(
            dataset_path / mean_path,
            weights_only=True,
        )
        std = torch.load(
            dataset_path / std_path,
            weights_only=True,
        )
        return DatasetDataStruct(filename_list, data, mean, std)

    assert split == "train"

    print("Filtering training sequences")
    filtered_data = []
    filtered_filenames = []
    for idx, data_dict in enumerate(data):
        nframes = data_dict["hmd_position_global_full_gt_list"].shape[0]
        if nframes < total_length:
            continue
        filtered_data.append(data_dict)
        filtered_filenames.append(filename_list[idx])
    print(f"Before filtering: {len(data)}\n  After filtering: {len(filtered_data)}")

    if (dataset_path / mean_path).exists() and (dataset_path / std_path).exists():
        mean = torch.load(os.path.join(dataset_path, mean_path))
        std = torch.load(os.path.join(dataset_path, std_path))
        print("Loading mean and std")
    else:
        print(
            "Mean and std not available! Computing mean and std from training data..."
        )
        all_motions = [d["rotation_local_full_gt_list"] for d in filtered_data]
        tmp_data_list = torch.cat(all_motions, dim=0)
        mean = tmp_data_list.mean(dim=0).float()
        std = tmp_data_list.std(dim=0).float()
        with open((dataset_path / mean_path), "wb") as f:
            torch.save(mean, f)
        with open((dataset_path / std_path), "wb") as f:
            torch.save(std, f)
        print("Mean and std saved to: ", dataset_path)

    return DatasetDataStruct(filtered_filenames, filtered_data, mean, std)


def load_gaps(
    dataset_path: Path,
    eval_name: str,
):
    local_path = os.path.join(dataset_path, "eval_gap_configs", eval_name + ".json")
    assert os.path.exists(local_path), f"File {local_path} does not exist"
    with open(local_path, "r") as f:
        data = json.load(f)
    return data


def get_dataloader(dataset, split, batch_size, num_workers=32, persistent_workers=True):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    return loader
