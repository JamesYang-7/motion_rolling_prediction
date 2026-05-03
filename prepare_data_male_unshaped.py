# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os

import numpy as np
import torch

from evaluation.utils import BodyModelsWrapper
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from loguru import logger
from tqdm import tqdm
from utils import utils_transform
from utils.constants import SMPLGenderParam, SMPLModelType
import sys


# AMASS Protocol 2: whole-dataset train/test split (HMD-Poser / VR_Pose_Pred convention)
AMASS_P2_TRAIN = [
    "ACCAD", "BioMotionLab_NTroje", "BMLmovi", "CMU", "EKUT",
    "Eyes_Japan_Dataset", "KIT", "MPI_HDM05", "MPI_Limits",
    "MPI_mosh", "SFU", "TotalCapture",
]
AMASS_P2_TEST = ["HumanEva", "Transitions_mocap"]


def replace_slashes(path: str) -> str:
    """
    Replaces forward slashes with backslashes in a path if the system is Windows.
    Args:
        path (str): The path to modify.
    Returns:
        str: The modified path.
    """
    if sys.platform == 'win32':  # Check if the system is Windows
        return path.replace('/', '\\')
    else:
        return path


def from_smpl_to_input_features(
    smpl_pose_vec: torch.Tensor, pose_joints_world: torch.Tensor, kintree
) -> dict:
    """
    smpl_pose_vec: [num_frames, 66] -> pose of the body in SMPL format
    pose_joints: [num_frames, 22, 3] -> position of the joints in the world coordinate system
    """
    gt_rotations_aa = torch.Tensor(smpl_pose_vec[:, :66]).reshape(-1, 3)
    gt_rotations_6d = utils_transform.aa2sixd(gt_rotations_aa).reshape(
        smpl_pose_vec.shape[0], -1
    )

    rotation_local_matrot = aa2matrot(
        torch.tensor(smpl_pose_vec).reshape(-1, 3)
    ).reshape(smpl_pose_vec.shape[0], -1, 9)
    rotation_global_matrot = local2global_pose(
        rotation_local_matrot, kintree[0].long()
    )  # rotation of joints relative to the origin
    head_rotation_global_matrot = rotation_global_matrot[1:, 15, :, :]

    rotation_global_6d = utils_transform.matrot2sixd(
        rotation_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_global_matrot.shape[0], -1, 6)
    input_rotation_global_6d = rotation_global_6d[1:, [15, 20, 21], :]

    rotation_velocity_global_matrot = torch.matmul(
        torch.inverse(rotation_global_matrot[:-1]),
        rotation_global_matrot[1:],
    )
    rotation_velocity_global_6d = utils_transform.matrot2sixd(
        rotation_velocity_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)
    input_rotation_velocity_global_6d = rotation_velocity_global_6d[:, [
        15, 20, 21], :]

    num_frames = pose_joints_world.shape[0] - 1
    hmd_cond = torch.cat(
        [
            input_rotation_global_6d.reshape(num_frames, -1),
            input_rotation_velocity_global_6d.reshape(num_frames, -1),
            pose_joints_world[1:, [15, 20, 21], :].reshape(num_frames, -1),
            pose_joints_world[1:, [15, 20, 21], :].reshape(num_frames, -1)
            - pose_joints_world[:-1, [15, 20, 21], :].reshape(num_frames, -1),
        ],
        dim=-1,
    )

    # world position of head
    position_head_world = pose_joints_world[1:, 15, :]
    head_global_trans = torch.eye(4).repeat(num_frames, 1, 1)
    head_global_trans[:, :3, :3] = head_rotation_global_matrot
    head_global_trans[:, :3, 3] = position_head_world

    data = {
        "rotation_local_full_gt_list": gt_rotations_6d[1:],
        "rotation_global_full_gt_list": rotation_global_6d[1:, :22]
        .reshape(num_frames, -1)
        .cpu()
        .float(),
        "hmd_position_global_full_gt_list": hmd_cond,
        "head_global_trans_list": head_global_trans,
        "position_global_full_gt_world": pose_joints_world[1:].cpu().float(),
    }
    return data


def _process_one_file(
    sample_path: str,
    rel_filepath: str,
    save_path: str,
    args,
    state: dict,
    device: str,
) -> None:
    """Load one AMASS .npz, downsample, run FK, save processed .pt.

    `state` holds the lazily-initialized body model and its type; the assertion
    keeps a run from mixing SMPL-H and SMPL-X sources.
    """
    if os.path.exists(save_path):
        return
    if not os.path.exists(sample_path):
        logger.warning("File {} does not exist, skipping...".format(sample_path))
        return

    bdata = np.load(sample_path, allow_pickle=True)
    if "mocap_framerate" in bdata:
        fps = bdata["mocap_framerate"]
    elif "mocap_frame_rate" in bdata:
        fps = bdata["mocap_frame_rate"]
    else:
        logger.info("No mocap_framerate found in {}".format(rel_filepath))
        return

    if "surface_model_type" in bdata:
        new_body_model_type = SMPLModelType.parse(bdata["surface_model_type"].item())
        assert state["type"] == "" or state["type"] == new_body_model_type, (
            "Can't mix different body models: {} vs {}".format(
                state["type"], new_body_model_type
            )
        )
        state["type"] = new_body_model_type
    else:
        state["type"] = SMPLModelType.SMPLH

    if state["model"] is None:
        logger.info("Initializing body model: {}".format(state["type"]))
        state["model"] = BodyModelsWrapper(args.support_dir)
    body_model = state["model"]
    body_model_type = state["type"]

    # Stride-based downsampling, matching VR_Pose_Pred / HMD-Poser exactly.
    # np.linspace with int-cast skips frames non-uniformly and shifts gt_jitter
    # by a few percent vs the reference implementation; plain stride slicing
    # reproduces their numbers bit-for-bit.
    target_fps = args.out_fps
    if fps == 2 * target_fps:
        stride = 2
    elif fps == target_fps:
        stride = 1
    elif args.protocol == "p1":
        raise AssertionError(
            "P1 expects framerate {} or {} (got {} in {})".format(
                target_fps, 2 * target_fps, fps, rel_filepath
            )
        )
    else:
        stride = round(fps / float(target_fps))
        if stride < 1:
            raise AssertionError(
                "Cannot supersample (fps={}, out_fps={})".format(fps, target_fps)
            )

    bdata_poses = bdata["poses"][::stride]
    bdata_trans = bdata["trans"][::stride]
    num_frames = bdata_poses.shape[0]
    fps = target_fps

    if num_frames < 10:
        logger.info("Too few frames in {}".format(rel_filepath))
        return
    smpl_gender = "male"

    body_parms = {
        "root_orient": torch.Tensor(bdata_poses[:, :3]),
        "pose_body": torch.Tensor(bdata_poses[:, 3:66]),
        "trans": torch.Tensor(bdata_trans),
        # uniform setting: zero betas, broadcast over all frames so
        # the dataloader's [1:] slice keeps shape (num_frames-1, 16)
        "betas": torch.zeros(bdata_poses.shape[0], 16),
    }
    body_pose_world = body_model(
        {k: v.to(device) for k, v in body_parms.items()},
        body_model_type,
        SMPLGenderParam[smpl_gender.upper()],
    )
    gt_joints_world_space = body_pose_world.Jtr[:, :22, :].cpu()
    kintree = body_model.get_kin_tree(
        body_model_type, SMPLGenderParam[smpl_gender.upper()]
    )
    data = from_smpl_to_input_features(bdata_poses, gt_joints_world_space, kintree)
    data["body_parms_list"] = body_parms
    data["framerate"] = fps
    data["gender"] = smpl_gender
    data["filepath"] = rel_filepath
    data["surface_model_type"] = body_model_type

    torch.save(data, save_path)


def process_amass_p1(args, device):
    """Per-file split via <splits_dir>/<dataset>/{train,test}_split.txt files."""
    state = {"model": None, "type": ""}
    out_root = os.path.join(args.save_dir, "new_format_data")
    all_datasets = sorted(os.listdir(args.splits_dir))
    for dataroot_subset in all_datasets:
        for phase in ["train", "test"]:
            logger.info(f"Processing {dataroot_subset} {phase}...")
            split_file = os.path.join(
                args.splits_dir, dataroot_subset, phase + "_split.txt"
            )
            if not os.path.exists(split_file):
                logger.info(f"{split_file} does not exist, skipping...")
                continue

            savedir = os.path.join(out_root, dataroot_subset, phase)
            os.makedirs(savedir, exist_ok=True)

            with open(split_file, "r") as f:
                filepaths = [replace_slashes(line.strip()) for line in f]

            for idx, filepath in enumerate(tqdm(filepaths), start=1):
                save_path = os.path.join(savedir, "{}.pt".format(idx))
                sample_path = os.path.join(args.root_dir, filepath)
                _process_one_file(
                    sample_path, filepath, save_path, args, state, device
                )


def process_amass_p2(args, device):
    """Whole-dataset split: each dataset goes entirely into train or test, files via glob."""
    state = {"model": None, "type": ""}
    out_root = os.path.join(args.save_dir, "new_format_data")
    pairs = (
        [(d, "train") for d in AMASS_P2_TRAIN]
        + [(d, "test") for d in AMASS_P2_TEST]
    )
    for dataroot_subset, phase in pairs:
        logger.info(f"Processing {dataroot_subset} {phase}...")
        savedir = os.path.join(out_root, dataroot_subset, phase)
        os.makedirs(savedir, exist_ok=True)

        sample_paths = sorted(glob.glob(
            os.path.join(args.root_dir, dataroot_subset, "*", "*_poses.npz")
        ))
        if not sample_paths:
            logger.warning(f"No files found under {args.root_dir}/{dataroot_subset}")
            continue

        for idx, sample_path in enumerate(tqdm(sample_paths), start=1):
            save_path = os.path.join(savedir, "{}.pt".format(idx))
            rel_filepath = os.path.relpath(sample_path, args.root_dir)
            _process_one_file(
                sample_path, rel_filepath, save_path, args, state, device
            )


def main(args, device="cuda:0"):
    if args.protocol == "p1":
        process_amass_p1(args, device)
    elif args.protocol == "p2":
        process_amass_p2(args, device)
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")


def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["p1", "p2"],
        default="p1",
        help="AMASS protocol: p1 (per-file split files) or p2 (whole-dataset split via glob)",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="prepare_data/data_split",
        help="=dir where the data splits are defined (p1 only; ignored for p2)",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="./SMPL/",
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="=dir where you put your AMASS data"
    )
    parser.add_argument(
        "--out_fps",
        type=int,
        default=60,
        help="Output framerate of the generated data",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu",
    )
    args = parser.parse_args()

    main(args, device="cpu" if args.cpu else "cuda:0")


if __name__ == "__main__":
    run()
