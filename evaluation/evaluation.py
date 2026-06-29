# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch
from evaluation.utils import (
    BodyModelsWrapper,
    get_body_poses_with_params,
    get_dataset_fps,
    get_GT_body_poses,
)

from loguru import logger

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from tqdm import tqdm

from utils.constants import DataTypeGT, ModelOutputType, SMPLGenderParam, SMPLModelType
from utils.metrics import (
    AccumulateArray,
    AverageValue,
    from_gaps_to_masks,
    get_all_metrics,
    get_all_metrics_trackingloss,
    get_metric_function,
    get_plots_configs,
    is_array_based_metric,
    keep_logging_metrics,
    MetricsInputData,
    PER_FRAME_METRIC_FUNCS_DICT,
    remove_frames_to_gaps,
)

from pathlib import Path
from typing import Optional

def padding_collate(batch):
    """
    Receives a list with #BATCH_SIZE tuples (gt_data, cond_data)
    It pads the sequences to the same length (only data structures specified in keys_to_pad)
    - All 'key_to_pad' are returned as tensors of shape (BATCH_SIZE, MAX_SEQ_LEN, ...)
    - All others are returned as lists of length BATCH_SIZE
    """
    keys_to_pad = {
        DataTypeGT.RELATIVE_ROTS,
        DataTypeGT.SPARSE,
        DataTypeGT.HEAD_MOTION,
        DataTypeGT.BOOTSTRAP
    }
    assert len(batch) > 0, "Batch is empty"
    gt_data, cond_data = {}, {}
    for i, data_dict in enumerate((gt_data, cond_data)):
        # pad the sequences to the same length
        for k in batch[0][i].keys():
            if k in keys_to_pad:
                data_dict[k] = pad_sequence([x[i][k] for x in batch], batch_first=True)
            else:
                data_dict[k] = [x[i][k] for x in batch]
    return gt_data, cond_data


class SortedSampler(Sampler):
    """Samples elements sequentially sorted by the sequence length, and filtered by min seq length."""

    def __init__(self, data_source, min_frames):
        self.data_source = data_source
        self.min_frames = min_frames

        # Filter and sort in one step
        self.filtered_and_sorted_indices = sorted(
            (
                idx
                for idx in range(len(self.data_source))
                if self.data_source[idx][0][DataTypeGT.NUM_FRAMES] >= self.min_frames
            ),
            key=lambda x: self.data_source[x][0][DataTypeGT.NUM_FRAMES],
        )

        num_filtered = len(self.data_source) - len(self.filtered_and_sorted_indices)
        logger.info(f"SortedSampler: {num_filtered} filtered, {len(self)} remaining")

    def __iter__(self):
        return iter(self.filtered_and_sorted_indices)

    def __len__(self):
        return len(self.filtered_and_sorted_indices)


class EvaluatorWrapper:
    def __init__(
        self,
        args,
        generator,
        dataset,
        body_model: BodyModelsWrapper,
        device,
        batch_size=1,
    ):
        self.args = args
        self.generator = generator
        self.dataset = dataset
        self.dataset_name = args.dataset
        self.device = device
        self.body_model = body_model
        self.fps = get_dataset_fps(self.dataset_name)
        # eval_skip_frames is a list of leading-frame counts to drop before
        # computing metrics. Default [0, 79] mirrors VR_Pose_Pred's two save
        # variants. The model FK runs once; only the metric slice changes.
        eval_skip = getattr(args, "eval_skip_frames", None)
        if eval_skip is None:
            eval_skip = [0, 79]
        elif isinstance(eval_skip, int):
            eval_skip = [eval_skip]
        self.eval_skip_frames = list(eval_skip)
        self.batch_size = batch_size
        # Per-frame body-parm + metric dump dir. None disables the dump (default).
        # Set via `set_save_frames_dir(...)` from the test driver after the
        # output dir has been resolved.
        self.save_frames_dir: Optional[Path] = None

        # initialize data loader. Sorting the dataset by sequence length to
        # speed up the evaluation, as sequences with similar length will be
        # processed together, thus minimizing the padding (i.e., useless computation).
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=padding_collate,
            # sampler=SortedSampler(self.dataset, min_frames=self.MIN_FRAMES_TO_EVAL + 3),
        )

    def evaluate_from_prediction(
        self,
        output,
        body_param_GT,
        head_motion,
        gaps,
        pred_gender: SMPLGenderParam,
        gt_gender: SMPLGenderParam,
        model_type: SMPLModelType,
        betas=None,
        save_frame_idx: Optional[int] = None,
        save_frame_filename: Optional[str] = None,
    ):
        """Run FK once, then compute metrics at every skip in self.eval_skip_frames.

        Returns dict[skip → eval_log]. Skips that exceed the sequence length
        (after the jitter T-3 guard) are silently dropped from the output.

        When `self.save_frames_dir` is set and `save_frame_idx` is given, also
        writes <save_frames_dir>/<save_frame_idx>.npz with predicted +
        GT body parameters and per-frame metrics. The dump uses the full
        sequence (no skip) so downstream callers can apply their own slice.
        """
        pr_body, pr_params = get_body_poses_with_params(
            output,
            self.body_model,
            head_motion,
            device=self.device,
            gender=pred_gender,
            model_type=model_type,
            betas=betas,
        )

        gt_body = get_GT_body_poses(
            self.body_model,
            body_param_GT,
            output.shape[0],
            self.device,
            gt_gender,
            model_type,
        )

        pr_pos_full = pr_body.Jtr[:, :22, :]
        pr_angle_full = pr_body.full_pose.reshape(pr_body.Jtr.shape)[:, :22]
        gt_pos_full = gt_body.Jtr[:, :22, :]
        gt_angle_full = gt_body.full_pose.reshape(pr_body.Jtr.shape)[:, :22]
        total_frames = pr_pos_full.shape[0]

        eval_logs_per_skip = {}
        for skip in self.eval_skip_frames:
            if skip >= total_frames - 3:
                # not enough frames left for jitter (needs T-3); drop this skip
                continue

            metrics_sets = [get_all_metrics()]
            masks, processed_gaps = None, None
            if gaps is not None:
                metrics_sets.append(get_all_metrics_trackingloss())
                masks = from_gaps_to_masks(gaps, total_frames, skip)
                processed_gaps = remove_frames_to_gaps(gaps, skip)

            metrics_input_data = MetricsInputData(
                pred_positions=pr_pos_full[skip:],
                pred_angles=pr_angle_full[skip:],
                pred_mesh=pr_body.v[skip:],
                gt_positions=gt_pos_full[skip:],
                gt_angles=gt_angle_full[skip:],
                gt_mesh=gt_body.v[skip:],
                fps=self.fps,
                trackingloss_masks=masks,
                gaps=processed_gaps,
            )
            eval_log = {}
            for metrics in metrics_sets:
                for metric in metrics:
                    eval_log[metric] = (
                        get_metric_function(metric)(metrics_input_data).cpu().numpy()
                    )
            eval_logs_per_skip[skip] = eval_log

        if self.save_frames_dir is not None and save_frame_idx is not None:
            self._dump_per_sequence_frames(
                seq_idx=save_frame_idx,
                filename=save_frame_filename or "",
                pr_pos_full=pr_pos_full,
                pr_angle_full=pr_angle_full,
                pr_params=pr_params,
                gt_pos_full=gt_pos_full,
                gt_angle_full=gt_angle_full,
                gt_params=body_param_GT,
            )
        return eval_logs_per_skip

    def _dump_per_sequence_frames(
        self,
        seq_idx: int,
        filename: str,
        pr_pos_full: torch.Tensor,
        pr_angle_full: torch.Tensor,
        pr_params: dict,
        gt_pos_full: torch.Tensor,
        gt_angle_full: torch.Tensor,
        gt_params: dict,
    ):
        """Write a per-sequence .npz with predicted + GT body params and
        per-frame metrics. File layout: <save_frames_dir>/<seq_idx>.npz with
        flat `__`-separated keys (np.savez can't nest dicts):

            pred__{root_orient,pose_body,trans,betas?}  (T, ...)
            gt__{root_orient,pose_body,trans,betas?}    (T, ...)
            frame_metrics__<metric>                     (T,) float32, NaN
                                                        where undefined
            filepath, seq_idx, T, fps                   scalars

        Coefficients from `metrics_coeffs` are already applied so the metric
        units match `avg_<dataset>_skip{N}f.csv` (deg / cm / cm·s⁻³).
        `np.nanmean` over `frame_metrics__<m>` recovers the corresponding
        scalar metric.
        """
        assert self.save_frames_dir is not None
        self.save_frames_dir.mkdir(parents=True, exist_ok=True)

        # Build a MetricsInputData with full-sequence (no skip) FK outputs and
        # run each registered per-frame metric in reduce_time=False mode.
        metrics_input_data = MetricsInputData(
            pred_positions=pr_pos_full,
            pred_angles=pr_angle_full,
            pred_mesh=None,  # mesh not needed by per-frame metrics
            gt_positions=gt_pos_full,
            gt_angles=gt_angle_full,
            gt_mesh=None,
            fps=self.fps,
            trackingloss_masks=None,
            gaps=None,
        )
        flat: dict = {
            "filepath": np.array(filename),
            "seq_idx": np.array(int(seq_idx)),
            "T": np.array(int(pr_pos_full.shape[0])),
            "fps": np.array(float(self.fps)),
        }
        for name, fn in PER_FRAME_METRIC_FUNCS_DICT.items():
            v = fn(metrics_input_data, reduce_time=False)
            flat[f"frame_metrics__{name}"] = v.detach().cpu().numpy().astype(np.float32)

        for k, v in pr_params.items():
            flat[f"pred__{k}"] = v.detach().cpu().numpy().astype(np.float32)
        for k in ("root_orient", "pose_body", "trans", "betas"):
            if k in gt_params:
                v = gt_params[k]
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                flat[f"gt__{k}"] = np.asarray(v).astype(np.float32)

        np.savez(self.save_frames_dir / f"{seq_idx}.npz", **flat)

    def evaluate_single(self, sample_idx):
        gt_dict, cond_dict = self.dataset[sample_idx]
        gt_data = gt_dict[DataTypeGT.RELATIVE_ROTS]
        sparse = cond_dict[DataTypeGT.SPARSE]
        body_param_GT = gt_dict[DataTypeGT.BODY_PARAMS]
        head_motion = gt_dict[DataTypeGT.HEAD_MOTION]
        gaps = gt_dict[DataTypeGT.TRACKING_GAP]
        gt_gender = gt_dict[DataTypeGT.SMPL_GENDER]
        model_type = gt_dict[DataTypeGT.SMPL_MODEL_TYPE]

        output, _ = self.generator(
            gt_data.unsqueeze(0),
            sparse.unsqueeze(0),
            body_model=self.body_model.get_body_model(
                SMPLModelType.SMPLH, SMPLGenderParam.MALE
            ),
            filenames=gt_dict[DataTypeGT.FILENAME],
        )
        local_rot = output[ModelOutputType.RELATIVE_ROTS][0]
        betas = None
        gt_contains_shape_params = DataTypeGT.SHAPE_PARAMS in gt_dict
        model_predicts_shape_params = (
            ModelOutputType.SHAPE_PARAMS in output
            and output[ModelOutputType.SHAPE_PARAMS] is not None
        )
        if gt_contains_shape_params and model_predicts_shape_params:
            # MODE 1: model predicts SHAPE params --› gender is always NEUTRAL for prediction, and betas are the ones predicted by the model
            betas = output[ModelOutputType.SHAPE_PARAMS][0]
            pred_gender = SMPLGenderParam.NEUTRAL
        elif gt_contains_shape_params:
            # MODE 2: used when shape is not predicted, but GT contains shape params --› use GT shape params as these are ASSUMED to be available at runtime
            betas = gt_dict[DataTypeGT.SHAPE_PARAMS]
            pred_gender = gt_gender
        else:
            # MODE 3: retrocompatibility with AGRoL/AvatarPoser benchmark where the shape params are not in GT --› default to 0's, and the gender is always assumed to be MALE
            pred_gender = SMPLGenderParam.MALE

        return self.evaluate_from_prediction(
            local_rot,
            body_param_GT,
            head_motion,
            gaps,
            pred_gender,
            gt_gender,
            model_type,
            betas=betas,
        )

    def evaluate_all(self):
        """Run inference + metrics for every skip in self.eval_skip_frames.

        Returns dict[skip → (summary_log, fine_grained_df, arr_based_metrics)].
        Per-sample short sequences (skip >= T-3) are dropped from that skip's
        aggregation but still contribute to lower skips.
        """
        logs_per_skip = {skip: {} for skip in self.eval_skip_frames}
        fine_grained_per_skip = {skip: [] for skip in self.eval_skip_frames}
        flag = False
        # Global sample counter so the per-frame dump (if enabled) writes one
        # `<global_idx>.npz` per sequence regardless of batch boundaries.
        global_sample_idx = 0
        with torch.no_grad():
            for i, (gt_dict, cond_dict) in tqdm(enumerate(self.data_loader)):
                gt_data = gt_dict[DataTypeGT.RELATIVE_ROTS]
                sparse = cond_dict[DataTypeGT.SPARSE]
                body_param = gt_dict[DataTypeGT.BODY_PARAMS]
                head_motion = gt_dict[DataTypeGT.HEAD_MOTION]
                gaps = gt_dict[DataTypeGT.TRACKING_GAP]
                num_frames = gt_dict[DataTypeGT.NUM_FRAMES]
                filenames = gt_dict[DataTypeGT.FILENAME]
                # `filepaths` are the original AMASS .npz paths (e.g.
                # "BioMotionLab_NTroje/rub098/0004_motorcycle_poses.npz")
                # — what callers want stored in the per-frame dump so they
                # can cross-reference with prepare_data/<dataset>/<group>/test_split.txt.
                # Pre-fix dataset .pt files may not carry this key, so default
                # to the slug to keep the dump non-empty.
                filepaths = gt_dict.get(DataTypeGT.FILEPATH, filenames)
                gt_genders = gt_dict[DataTypeGT.SMPL_GENDER]
                model_types = gt_dict[DataTypeGT.SMPL_MODEL_TYPE]
                bootstrap_data = gt_dict[DataTypeGT.BOOTSTRAP]
                # print(f"### len(bootstrap_data): {len(bootstrap_data)}, len gt_data: {len(gt_data)} ###")
                # for i in range(len(bootstrap_data)):
                #     print(f"### bootstrap_data[{i}] shape: {bootstrap_data[i].shape}, gt_data[{i}] shape: {gt_data[i].shape} ###")

                if not self.dataset.gt_init:
                    gt_data = bootstrap_data

                batch_size = len(num_frames)

                # inference
                output, _ = self.generator(
                    gt_data,
                    sparse,
                    body_model=self.body_model.get_body_model(
                        SMPLModelType.SMPLH, SMPLGenderParam.MALE
                    ),
                    filenames=filenames,
                )

                local_rot = output[ModelOutputType.RELATIVE_ROTS]
                betas = None
                version_info = ""
                if (
                    DataTypeGT.SHAPE_PARAMS in gt_dict
                    and ModelOutputType.SHAPE_PARAMS in output
                    and output[ModelOutputType.SHAPE_PARAMS] is not None
                ):
                    # new version of the dataset that has used shape params --> use NEUTRAL
                    betas = output[ModelOutputType.SHAPE_PARAMS]
                    pred_genders = [
                        SMPLGenderParam.NEUTRAL,
                    ] * batch_size
                    version_info = " (using 'neutral' gender and predicted shape params)"
                elif DataTypeGT.SHAPE_PARAMS in gt_dict:
                    # new version + no params predicted -- use GT
                    betas = gt_dict[DataTypeGT.SHAPE_PARAMS]
                    pred_genders = gt_genders
                    version_info = " (using GT gender and GT shape params)"
                else:
                    # old version of dataset where MALE is default, and shape is not predictable
                    pred_genders = [
                        SMPLGenderParam.MALE,
                    ] * batch_size
                    version_info = " (using male and no shape params)"
                if not flag:
                    logger.info(
                        f"EvaluatorWrapper: Dataset version info:{version_info}, GT init: {self.dataset.gt_init}"
                    )
                    flag = True

                # sequentially evaluate (FK not batcherized)
                for i in range(batch_size):
                    if betas is not None:
                        betas_ = betas[i][: num_frames[i]]
                    else:
                        betas_ = None

                    # we slice the output to the number of frames in the sequence, to remove the padding introduced in the padding_collate function
                    instance_logs = self.evaluate_from_prediction(
                        local_rot[i][: num_frames[i]],
                        body_param[i],
                        head_motion[i][: num_frames[i]],
                        gaps[i],
                        pred_genders[i],
                        gt_genders[i],
                        model_types[i],
                        betas=betas_,
                        save_frame_idx=global_sample_idx,
                        save_frame_filename=filepaths[i],
                    )
                    global_sample_idx += 1

                    for skip, instance_log in instance_logs.items():
                        log = logs_per_skip[skip]
                        for key in instance_log:
                            if key not in log:
                                log[key] = (
                                    AverageValue()
                                    if not is_array_based_metric(key)
                                    else AccumulateArray()
                                )
                            log[key].add_value(instance_log[key])

                        metrics_list_to_csv = [
                            instance_log[k]
                            for k in log.keys()
                            if not is_array_based_metric(k)
                        ]
                        fine_grained_per_skip[skip].append(
                            [filenames[i], num_frames[i]] + metrics_list_to_csv
                        )

        results_per_skip = {}
        for skip in self.eval_skip_frames:
            log = logs_per_skip[skip]
            df_titles = ["filename", "num_frames"] + [
                k for k in log.keys() if not is_array_based_metric(k)
            ]
            fine_grained_df = pd.DataFrame(
                fine_grained_per_skip[skip], columns=df_titles
            )
            arr_based_metrics = {}
            summary_log = {}
            for metric in log.keys():
                if is_array_based_metric(metric):
                    arr_based_metrics[metric] = log[metric].get_array()
                else:
                    summary_log[metric] = log[metric].get_average()
            results_per_skip[skip] = (summary_log, fine_grained_df, arr_based_metrics)

        return results_per_skip

    def store_all_results(self, df: pd.DataFrame, csv_path: Path):
        df.to_csv(csv_path, index=False)
        logger.info(f"Results successfully stored in a csv file: {csv_path=}")

    def store_avg_results(self, log: dict, csv_path: Path):
        """VR_Pose_Pred-style averaged-scalar CSV: two columns `metric, value`."""
        rows = [[metric, f"{float(value):.2f}"] for metric, value in log.items()]
        pd.DataFrame(rows, columns=["metric", "value"]).to_csv(csv_path, index=False)
        logger.info(f"Avg results stored in: {csv_path=}")

    def store_plots(self, metrics: dict, plot_dir: Path, suffix: str = "") -> dict:
        if len(metrics.keys()) == 0:
            return {}

        additional_metrics = {}
        for plot_cfg in get_plots_configs():
            # plot_cfg is PlotConfig data class from utils.metrics
            title = plot_cfg.title
            all_metrics_needed = plot_cfg.curve_styles.keys()
            if not all(m in metrics for m in all_metrics_needed):
                logger.warning(
                    f"Skipping plot '{title}' because not all metrics are available: {all_metrics_needed}"
                )
                continue
            plt.clf()
            plt.title(title)
            abs_max = -float("inf")
            saved_count = 0
            gt_key, pred_key = None, None
            for metric_name, style in plot_cfg.curve_styles.items():
                if metric_name not in metrics:
                    continue
                elif metrics[metric_name].shape[0] == 0:
                    continue
                y_values = list(
                    metrics[metric_name].mean(axis=0)
                )  # from [samples, T] --> [T]
                plt.plot(y_values, label=style.label)
                abs_max = max(max(y_values), abs_max)
                plt.ylim([0, 1.1 * abs_max])
                # store array to npz
                filename = "arr_" + metric_name + suffix + ".npz"
                np.savez(plot_dir / filename, values=metrics[metric_name])
                saved_count += 1
                if "gt" in metric_name:
                    gt_key = metric_name
                else:
                    pred_key = metric_name
            if saved_count == 0:
                logger.warning(f"No tracking input gaps found, so skipping plot/metric: {title}")
                continue
            plt.legend()
            plt.xlabel("Frame")
            plt.ylabel("Value")

            tgt_path = plot_dir / (plot_cfg.filename + suffix + ".png")
            plt.savefig(tgt_path)
            plt.close()
            logger.info(f"'{title}' plot saved in {tgt_path}")

            pred_arr_mean = metrics[pred_key].mean(axis=0) * 0.01 # jitter multiplied by 0.01
            gt_arr_mean = metrics[gt_key].mean(axis=0) * 0.01
            PJ = pred_arr_mean.max()
            AUJ = sum(abs(pred_arr_mean - gt_arr_mean))
            additional_metrics[f"PJ ({title})"] = PJ
            additional_metrics[f"AUJ ({title})"] = AUJ
        return additional_metrics



    def print_results(self, log):
        # print the value for all the metrics
        logger.info("Metrics:")
        for metric in log.keys():
            logger.info(f"{metric}: {round(log[metric], 2)}")

    def push_to_tb(self, log, tb_writer, iteration, suffix=""):
        logging_metrics = keep_logging_metrics(log)
        for metric in logging_metrics:
            tb_writer.add_scalar(
                f"eval{suffix}/{metric}",
                log[metric],
                iteration,
            )
