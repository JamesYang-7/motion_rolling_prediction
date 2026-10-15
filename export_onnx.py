"""Export RPM checkpoints (default: every released one) as ONNX step graphs,
one rolling step per graph, following VR_Pose_Pred's
scripts/export_app_bundle.py: CPU export with the TorchScript exporter,
opset 15, float32, batch 1, no Einsum node, eager and onnxruntime replays
checked against the eval's own rollout, and a manifest.json (provenance, IO
contract, normalisation, checks) next to the graph.

    checkpoints/<folder>/<exp>/model_latest.pt  (+ args.json)
 -> checkpoints/<folder>/<exp>/export/rpm_<folder>_<exp>.onnx + manifest.json

The step graph is ModelWrapper (network + PCAF blend) plus what the rolling
generator (evaluation/generators.py) and the eval (evaluation/utils.py
get_body_poses) do around it for the committed frame. With L =
input_motion_length (reactive 5 / smooth 10; AMASS-P1 10 / 20):

    inputs   sparse_ctx      (1, 11, 54)    input frames i-10 .. i, as the loader builds them
             motion_ctx      (1, 10, 132)   committed rows i-10 .. i-1, normalised
             prev_pred       (1, L, 132)    PCAF anchor window for frames i .. i+L-1, normalised
             joints_rest     (1, 22, 3)     SMPL-X rest joints with the subject's betas
    outputs  motion_out      (1, 198)       frame i: 22 local rot 6D, de-normalised (what the
                                            eval scores) + 22 world joint positions
             joint_rotmat    (1, 22, 3, 3)  those rotations as the eval's FK applies them
             next_motion_ctx (1, 10, 132)   motion_ctx shifted by one, frame i appended
             next_prev_pred  (1, L, 132)    the new window shifted by one, new slot 0

motion_ctx / prev_pred are model state: the app feeds next_* back unchanged.
They stay normalised because the foot rotations (dims 60:72) have std 0 in
the GORP and AMASS-P1 stats: de-normalising would replace the values the
network feeds itself by the mean (the generator keeps them normalised too).
The positions are the eval's 6D chain (sixd2aa -> Rodrigues,
`sixd_to_rotmat_eval`), FK on joints_rest, and the head joint moved onto the
tracked head of frame i. The eval anchors on the mocap head instead: the same
point for synthetic input, ~1.8 cm away for GORP real input.

The checks replay GORP test_controllers captures through RPM's own
RollingGenerationWrapper (on CPU) and through the eager and the onnxruntime
graph with the eval's seed, and score the replays with EvaluatorWrapper's
metric code. AMASS checkpoints are replayed on the same GORP captures with
their own normalisation: a numeric check only, no metrics.

Usage (from the repo root, RPM's own .venv):
    TQDM_DISABLE=1 .venv/Scripts/python.exe export_onnx.py
    TQDM_DISABLE=1 .venv/Scripts/python.exe export_onnx.py \
        --model_paths ./checkpoints/gorp_real_inputs/reactive/model_latest.pt
"""

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from loguru import logger

from data_loaders.dataloader import TestDataset, get_mean_std_path
from evaluation.evaluation import EvaluatorWrapper
from evaluation.generators import create_generator
from evaluation.utils import BodyModelsWrapper, get_dataset_fps
from human_body_prior.body_model.lbs import batch_rodrigues
from model.model_wrapper import ALL_FUNCTIONS
from utils.constants import (
    DataTypeGT,
    DatasetType,
    HEAD_JOINT_IDX,
    ModelOutputType,
    NUM_JOINTS_SMPL,
    SMPLModelType,
    sparse_point_slices,
)
from utils.model_util import load_rpm_model
from utils.parser_util import sample_args
from utils.utils_transform import sixd2aa

REPO = Path(__file__).resolve().parent
HEAD_POS = sparse_point_slices(0)[2]
INPUT_NAMES = ["sparse_ctx", "motion_ctx", "prev_pred", "joints_rest"]
OUTPUT_NAMES = ["motion_out", "joint_rotmat", "next_motion_ctx", "next_prev_pred"]
# GORP test_controllers captures replayed for the checks (the VR_Pose_Pred app
# bundle's): the only MC test capture with controller dropouts (exercises the
# real-input zero masking), the S11 capture recorded ~2.4 m from the training
# spot, and a capture close to the split's median error.
DEFAULT_SEQUENCES = (
    "GORP/S06/s06_fruitninja-q3-controllers_000.npz",
    "GORP/S11/s11_beatsaber-q3-controllers_900.npz",
    "GORP/S26/s26_fruitninja-q3-controllers_000.npz",
)


# ---- 6D chain + FK: verbatim from VR_Pose_Pred scripts/export_app_bundle.py ----
def _cross(a, b):
    """Row-wise cross product from slicing (no aten::linalg_cross in the graph)."""
    return torch.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], dim=1)


def sixd_to_rotmat_eval(d6):
    """(N, 6) raw 6D -> (N, 3, 3) the rotation the eval FK applies.

    The eval runs `sixd2aa` (columns a, b, a x b, NOT orthonormalised ->
    torchgeometry's rotation_matrix_to_quaternion -> angle-axis) and the body
    model turns the angle-axis back into a matrix (Rodrigues). The predicted 6D
    is not exactly orthonormal, where that chain and Gram-Schmidt differ, so the
    graph replicates the chain: the same Shepperd branch logic on the
    transposed matrix (float masks instead of bool products, which ONNX Mul
    does not take), then the rotation of the normalised quaternion -- what
    angle-axis + Rodrigues reproduce."""
    a, b = d6[:, :3], d6[:, 3:6]
    c = _cross(a, b)
    m = torch.stack([a, b, c], dim=1)          # rows = columns of M, i.e. M^T (tgm's rmat_t)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    d2 = m22 < 1e-6
    d0_d1 = m00 > m11
    d0_nd1 = m00 < -m11
    t0 = 1 + m00 - m11 - m22
    t1 = 1 - m00 + m11 - m22
    t2 = 1 - m00 - m11 + m22
    t3 = 1 + m00 + m11 + m22
    q0 = torch.stack([m12 - m21, t0, m01 + m10, m20 + m02], -1)
    q1 = torch.stack([m20 - m02, m01 + m10, t1, m12 + m21], -1)
    q2 = torch.stack([m01 - m10, m20 + m02, m12 + m21, t2], -1)
    q3 = torch.stack([t3, m12 - m21, m20 - m02, m01 - m10], -1)
    c0 = (d2 & d0_d1).to(d6.dtype).unsqueeze(1)
    c1 = (d2 & ~d0_d1).to(d6.dtype).unsqueeze(1)
    c2 = (~d2 & d0_nd1).to(d6.dtype).unsqueeze(1)
    c3 = (~d2 & ~d0_nd1).to(d6.dtype).unsqueeze(1)
    q = q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3
    q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True))   # tgm's 0.5/sqrt(t) scale drops out
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=1)


def forward_kinematics(rot, joints_rest, parents):
    """(B, J, 3, 3) local rotations + (B, J, 3) rest joints -> (B, J, 3) posed
    joints at zero translation (human_body_prior's batch_rigid_transform)."""
    g_rot, g_pos = [rot[:, 0]], [joints_rest[:, 0]]
    for i in range(1, len(parents)):
        p = parents[i]
        offset = (joints_rest[:, i] - joints_rest[:, p]).unsqueeze(-1)
        g_pos.append(g_pos[p] + torch.matmul(g_rot[p], offset).squeeze(-1))
        g_rot.append(torch.matmul(g_rot[p], rot[:, i]))
    return torch.stack(g_pos, dim=1)


class RPMStepGraph(nn.Module):
    """One rolling step as RollingGenerationWrapper runs it, plus the eval's
    rotation chain, FK and head anchoring of the committed frame (module
    docstring)."""

    def __init__(self, model, mean, std, parents):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.float().clone())   # dataset.inv_transform: x * std + mean
        self.register_buffer("std", std.float().clone())
        self.parents = [int(p) for p in parents]

    def forward(self, sparse_ctx, motion_ctx, prev_pred, joints_rest):
        B = sparse_ctx.shape[0]
        cond = {DataTypeGT.SPARSE: sparse_ctx, DataTypeGT.MOTION_CTX: motion_ctx}
        window = self.model(prev_pred, cond)[ModelOutputType.RELATIVE_ROTS]   # (B, L, 132), normalised
        row = window[:, 0]                                                    # the committed frame
        rot6 = row * self.std + self.mean
        rotmat = sixd_to_rotmat_eval(rot6.reshape(-1, 6)).reshape(B, NUM_JOINTS_SMPL, 3, 3)
        pos = forward_kinematics(rotmat, joints_rest, self.parents)
        head = sparse_ctx[:, -1, HEAD_POS].reshape(B, 1, 3)
        pos = pos - pos[:, HEAD_JOINT_IDX:HEAD_JOINT_IDX + 1] + head
        next_motion_ctx = torch.cat([motion_ctx[:, 1:], row.unsqueeze(1)], dim=1)
        # generators.py: shift the window, "add zeros to last frame, as it is done during training"
        next_prev_pred = torch.cat([window[:, 1:], torch.zeros_like(window[:, :1])], dim=1)
        return torch.cat([rot6, pos.reshape(B, -1)], dim=-1), rotmat, next_motion_ctx, next_prev_pred


def load_args(model_path):
    """test.py's arguments for `model_path` (sample_args: CLI defaults, then
    the model / losses groups and the paths from the checkpoint's args.json),
    on CPU; plus the raw args.json, which holds the training input flags."""
    argv = sys.argv
    sys.argv = [argv[0], "--model_path", str(model_path), "--cpu"]
    try:
        args = sample_args()
    finally:
        sys.argv = argv
    with open(model_path.parent / "args.json") as f:
        return args, json.load(f)


def motion_stats(args):
    """The checkpoint dataset's motion mean / std, from where load_data reads them."""
    root = REPO / args.dataset_path / args.dataset.value / "new_format_data"
    mean_name, std_name = get_mean_std_path(args.dataset)
    return (torch.load(root / mean_name, weights_only=True).float(),
            torch.load(root / std_name, weights_only=True).float())


_GORP_TEST = {}


def gorp_test(use_real_input, conf, mask_mode):
    key = (use_real_input, conf, mask_mode)
    if key not in _GORP_TEST:
        _GORP_TEST[key] = TestDataset(DatasetType.GORP, REPO / "datasets_processed", False,
                                      use_real_input=use_real_input, input_conf_threshold=conf,
                                      test_split="test_controllers", mask_mode=mask_mode)
    return _GORP_TEST[key]


def replay(step, gt_norm, sparse, joints_rest, m_ctx, s_ctx, L):
    """RollingGenerationWrapper's loop with the eval's seed (rows before frame
    `margin` = GT, first anchor window = GT) driven by a step function with the
    graph's signature, numpy in / out. Returns the (T - margin, 198)
    motion_out rows of frames margin .. T-1."""
    margin = max(m_ctx, s_ctx)
    motion_ctx = gt_norm[None, margin - m_ctx:margin]
    prev_pred = gt_norm[None, margin:margin + L]
    jr = joints_rest[None]
    rows = []
    for i in range(margin, gt_norm.shape[0]):
        motion_out, _, motion_ctx, prev_pred = step(sparse[None, i - s_ctx:i + 1], motion_ctx, prev_pred, jr)
        rows.append(motion_out[0])
    return np.stack(rows)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return ""


def input_description(args, train_args, fps):
    if args.dataset == DatasetType.GORP and train_args.get("use_real_input", False):
        return (f"GORP real device input at {fps} fps (RPM prepare_data_gorp.py + "
                f"transform_controllers_from_gorp_to_smpl / transform_hand_tracking_from_gorp_to_smpl); "
                f"a hand's 18 numbers are zeroed on frames with confidence < {train_args['input_conf_threshold']} "
                f"(RPM's rule, no hold)")
    if args.dataset == DatasetType.GORP:
        return (f"synthetic GORP input at {fps} fps: head / wrist features of the mocap body "
                f"(hmd_position_global_full_gt_list); the paper's sim->real cells feed it real device input")
    return f"synthetic {args.dataset.value} input at {fps} fps: head / wrist features of the AMASS mocap body"


def export_one(model_path, cli):
    folder, exp = model_path.parent.parent.name, model_path.parent.name
    name = f"rpm_{folder}_{exp}"
    out_dir = model_path.parent / "export"
    args, train_args = load_args(model_path)
    assert args.rolling_latency == 0, "the step graph is written for rolling_latency 0"
    m_ctx, s_ctx, L = args.rolling_motion_ctx, args.rolling_sparse_ctx, args.input_motion_length
    margin = max(m_ctx, s_ctx)
    fps = get_dataset_fps(args.dataset)
    logger.info(f"{name}: dataset {args.dataset.value}, L={L}, ctx {m_ctx}/{s_ctx}, {fps} fps -> {out_dir}")

    model, _ = load_rpm_model(args, device="cpu")
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = motion_stats(args)
    use_real = bool(train_args.get("use_real_input", False)) and args.dataset == DatasetType.GORP
    conf = float(train_args.get("input_conf_threshold", 0.0)) if use_real else 0.0
    ds = gorp_test(use_real, conf, args.mask_mode)
    ds.mean, ds.std = mean, std            # dataset.transform / inv_transform with this checkpoint's stats
    generator = create_generator(args, model, ds, "cpu")
    body_model = BodyModelsWrapper(str(REPO / args.support_dir))
    by_path = {ds.data[i]["filepath"]: i for i in range(len(ds))}
    missing = [s for s in cli.sequences if s not in by_path]
    assert not missing, f"not in GORP test_controllers: {missing}"

    # ---- the step graph (CPU) and its export ----
    kintree = body_model.get_kin_tree(SMPLModelType.SMPLX, ds[by_path[cli.sequences[0]]][0][DataTypeGT.SMPL_GENDER])
    parents = kintree[0, :NUM_JOINTS_SMPL].cpu().tolist()
    parents[0] = -1
    graph = RPMStepGraph(model, mean, std, parents).eval()

    seqs = []
    for seq in cli.sequences:
        idx = by_path[seq]
        gt_dict, cond_dict = ds[idx]
        gt = gt_dict[DataTypeGT.RELATIVE_ROTS].float()
        gender = gt_dict[DataTypeGT.SMPL_GENDER]
        betas = gt_dict[DataTypeGT.SHAPE_PARAMS].float()
        with torch.no_grad():
            joints_rest = body_model({"root_orient": torch.zeros(1, 3), "pose_body": torch.zeros(1, 63),
                                      "betas": betas[:1]}, SMPLModelType.SMPLX, gender).Jtr[0, :NUM_JOINTS_SMPL]
        seqs.append(dict(source=seq, idx=idx, gt_dict=gt_dict, gt=gt, sparse=cond_dict[DataTypeGT.SPARSE].float(),
                         gt_norm=ds.transform(gt).float(), betas=betas, gender=gender, joints_rest=joints_rest.float()))

    onnx_path = out_dir / f"{name}.onnx"
    s0 = seqs[0]
    example = (s0["sparse"][None, margin - s_ctx:margin + 1], s0["gt_norm"][None, margin - m_ctx:margin],
               s0["gt_norm"][None, margin:margin + L], s0["joints_rest"][None])
    with torch.no_grad():
        torch.onnx.export(graph, example, str(onnx_path), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                          opset_version=cli.opset, do_constant_folding=True)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    op_counts = collections.Counter(n.op_type for n in onnx_model.graph.node)
    assert op_counts.get("Einsum", 0) == 0, \
        f"{op_counts['Einsum']} Einsum node(s) in the export; Unity Inference Engine mis-evaluates them"
    del onnx_model
    size_mib = onnx_path.stat().st_size / 2 ** 20
    logger.info(f"wrote {onnx_path} ({size_mib:.1f} MiB, opset {cli.opset}, {sum(op_counts.values())} nodes, no Einsum)")

    so = ort.SessionOptions()
    so.intra_op_num_threads = cli.threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def eager_step(*ins):
        with torch.no_grad():
            return [t.numpy() for t in graph(*(torch.from_numpy(np.ascontiguousarray(x)) for x in ins))]

    def ort_step(*ins):
        return sess.run(OUTPUT_NAMES, {k: np.ascontiguousarray(x) for k, x in zip(INPUT_NAMES, ins)})

    # ---- replays vs the eval rollout ----
    evaluator = None
    if args.dataset == DatasetType.GORP:
        evaluator = EvaluatorWrapper(args, generator, ds, body_model, "cpu")
    released_csv = None
    if args.dataset == DatasetType.GORP:
        tag = "real" if use_real else "sim"
        sub = f"results_test_controllers_real_input_C{conf}" if use_real else "results_test_controllers"
        p = REPO / "results" / f"gorp_{tag}_released" / sub / exp / ("latest_rolling" + ("_real_input" if use_real else "")) / "results_gorp.csv"
        if p.exists():
            import pandas as pd
            released_csv = pd.read_csv(p).set_index("filename")
    checks = collections.defaultdict(float)
    seq_records = []
    for s in seqs:
        with torch.no_grad():
            ref, _ = generator(s["gt"][None], s["sparse"][None])
        ref_rot = ref[ModelOutputType.RELATIVE_ROTS][0].float()         # (T, 132), de-normalised
        args_np = (s["gt_norm"].numpy(), s["sparse"].numpy(), s["joints_rest"].numpy(), m_ctx, s_ctx, L)
        eager = replay(eager_step, *args_np)
        onnx_rows = replay(ort_step, *args_np)
        ref_np = ref_rot[margin:].numpy()
        checks["eager_replay_vs_rollout_rot6d_max_abs"] = max(checks["eager_replay_vs_rollout_rot6d_max_abs"],
                                                             float(np.abs(eager[:, :132] - ref_np).max()))
        checks["onnxruntime_replay_vs_rollout_rot6d_max_abs"] = max(checks["onnxruntime_replay_vs_rollout_rot6d_max_abs"],
                                                                   float(np.abs(onnx_rows[:, :132] - ref_np).max()))
        checks["onnxruntime_replay_vs_eager_replay_pos_max_abs_m"] = max(checks["onnxruntime_replay_vs_eager_replay_pos_max_abs_m"],
                                                                        float(np.abs(onnx_rows[:, 132:] - eager[:, 132:]).max()))
        # rotation chain and FK of the graph vs the eval's body model, head-relative
        rot6 = torch.from_numpy(eager[:, :132]).reshape(-1, 6)
        chain_err = float((sixd_to_rotmat_eval(rot6) - batch_rodrigues(sixd2aa(rot6))).abs().max())
        checks["rot_chain_vs_eval_max_abs"] = max(checks["rot_chain_vs_eval_max_abs"], chain_err)
        aa = sixd2aa(rot6).reshape(len(eager), -1)
        with torch.no_grad():
            jtr = body_model({"root_orient": aa[:, :3], "pose_body": aa[:, 3:66], "betas": s["betas"][margin:]},
                             SMPLModelType.SMPLX, s["gender"]).Jtr[:, :NUM_JOINTS_SMPL]
        pos = torch.from_numpy(eager[:, 132:]).reshape(-1, NUM_JOINTS_SMPL, 3)
        fk_err = float(((pos - pos[:, HEAD_JOINT_IDX:HEAD_JOINT_IDX + 1])
                        - (jtr - jtr[:, HEAD_JOINT_IDX:HEAD_JOINT_IDX + 1])).abs().max())
        checks["fk_vs_eval_body_model_head_relative_max_abs_m"] = max(checks["fk_vs_eval_body_model_head_relative_max_abs_m"], fk_err)
        head_err = float(np.abs(eager[:, 132 + 3 * HEAD_JOINT_IDX:135 + 3 * HEAD_JOINT_IDX]
                                - s["sparse"][margin:, HEAD_POS].numpy()).max())
        checks["head_joint_vs_tracked_head_max_abs_m"] = max(checks["head_joint_vs_tracked_head_max_abs_m"], head_err)
        rec = {"source": s["source"], "test_index": s["idx"], "frames": int(len(ref_rot)), "replayed_steps": int(len(eager))}
        if evaluator is not None:
            gd = s["gt_dict"]
            for key, rot in (("rollout", ref_rot), ("onnxruntime_replay", torch.cat([ref_rot[:margin], torch.from_numpy(onnx_rows[:, :132])]))):
                log = evaluator.evaluate_from_prediction(
                    rot, {k: v.clone() for k, v in gd[DataTypeGT.BODY_PARAMS].items()}, gd[DataTypeGT.HEAD_MOTION],
                    gd[DataTypeGT.TRACKING_GAP], s["gender"], s["gender"], gd[DataTypeGT.SMPL_MODEL_TYPE], betas=s["betas"])
                rec[f"{key}_mpjpe_cm"] = round(float(log["mpjpe"]), 4)
                rec[f"{key}_mpjre_deg"] = round(float(log["mpjre"]), 4)
            fname = ds.filename_list[s["idx"]]
            if released_csv is not None and fname in released_csv.index:
                rec["released_eval_csv_mpjpe_cm"] = round(float(released_csv.loc[fname, "mpjpe"]), 4)
        seq_records.append(rec)
        logger.info(f"  {s['source']}: " + ", ".join(f"{k} {v}" for k, v in rec.items() if k.endswith(("_cm", "_deg"))))
    logger.info("  checks: " + ", ".join(f"{k} {v:.2e}" for k, v in checks.items()))

    pcaf = ALL_FUNCTIONS[args.target_type](L)
    manifest = {
        "checkpoint": model_path.relative_to(REPO).as_posix(),
        "dataset": args.dataset.value, "exp_name": exp, "fps": fps,
        "input": input_description(args, train_args, fps),
        "use_real_input": use_real, "input_conf_threshold": conf,
        "target_type": str(args.target_type.value if hasattr(args.target_type, "value") else args.target_type),
        "pcaf_weights": pcaf,
        "geometry": {"sparse_ctx_frames": s_ctx + 1, "motion_ctx_frames": m_ctx, "prediction_window": L,
                     "rolling_latency": int(args.rolling_latency), "eval_seed_frames": margin},
        "io": {
            "inputs": {
                "sparse_ctx": [[1, s_ctx + 1, 54], f"input frames i-{s_ctx} .. i, oldest first (54-d sparse rows, unnormalised)"],
                "motion_ctx": [[1, m_ctx, 132], f"state: committed rows i-{m_ctx} .. i-1, normalised"],
                "prev_pred": [[1, L, 132], f"state: PCAF anchor window for frames i .. i+{L - 1}, normalised"],
                "joints_rest": [[1, NUM_JOINTS_SMPL, 3], "SMPL-X rest joints (zero pose) with the subject's 16 betas"],
            },
            "outputs": {
                "motion_out": [[1, 198], "frame i: [0:132] 22 local rot 6D (root global), de-normalised, not orthonormal; "
                                         "[132:198] 22 world joint positions, head joint on sparse_ctx[0, -1, 36:39]"],
                "joint_rotmat": [[1, NUM_JOINTS_SMPL, 3, 3], "the same rotations as the eval's FK applies them (sixd2aa -> Rodrigues)"],
                "next_motion_ctx": [[1, m_ctx, 132], "motion_ctx for frame i+1"],
                "next_prev_pred": [[1, L, 132], "prev_pred for frame i+1 (shifted window, last slot 0)"],
            },
        },
        "loop": [
            f"keep the last {s_ctx + 1} input frames S and the state (motion_ctx, prev_pred)",
            "every frame i: motion_out, joint_rotmat, motion_ctx, prev_pred = step(S, motion_ctx, prev_pred, joints_rest)",
            f"eval seed (the checks): motion_ctx = normalise(GT rows 0 .. {m_ctx - 1}), "
            f"prev_pred = normalise(GT rows {margin} .. {margin + L - 1}), first step at frame {margin}, rows before it = GT",
            "live seed: one 132-d standing-pose row r with identity rotations on joints 10 and 11 (std 0 there): "
            f"motion_ctx = {m_ctx} x normalise(r), prev_pred = {L} x normalise(r)",
            "normalise(x) = (x - motion_mean) / (motion_std + 1e-8) (TestDataset.transform); "
            "the graph de-normalises with x * motion_std + motion_mean (inv_transform)",
        ],
        "normalisation": {"motion_mean": [float(v) for v in mean.numpy()], "motion_std": [float(v) for v in std.numpy()]},
        "smplx_parents": parents,
        "onnx": {"file": onnx_path.name, "opset": cli.opset, "size_mib": round(size_mib, 2),
                 "op_types": dict(sorted(op_counts.items()))},
        "checks": dict(checks),
        "check_sequences": seq_records,
        "git_commit": git_commit(),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    with open(out_dir / "manifest.json", "w", newline="\n") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"wrote {out_dir / 'manifest.json'}")


def main():
    p = argparse.ArgumentParser(description="Export RPM checkpoints as ONNX step graphs")
    p.add_argument("--model_paths", nargs="+", type=Path, default=None,
                   help="model .pt files with an args.json beside them (default: every checkpoints/*/*/model_latest.pt)")
    p.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES),
                   help="GORP test_controllers filepaths replayed for the checks")
    p.add_argument("--opset", type=int, default=15, help="ONNX opset (15 keeps Unity Sentis' supported range)")
    p.add_argument("--threads", type=int, default=4, help="torch / onnxruntime CPU threads")
    cli = p.parse_args()
    torch.set_num_threads(cli.threads)
    paths = cli.model_paths
    if paths is None:
        paths = sorted(p for p in (REPO / "checkpoints").glob("*/*/model_latest.pt") if (p.parent / "args.json").exists())
        # the released zip's amass_p1/smooth/model_latest.pt is 0 bytes (only its opt_latest.pt has content)
        for p in [p for p in paths if p.stat().st_size == 0]:
            logger.warning(f"skipping {p.relative_to(REPO)}: empty file")
            paths.remove(p)
    for path in paths:
        export_one(path.resolve(), cli)


if __name__ == "__main__":
    main()
