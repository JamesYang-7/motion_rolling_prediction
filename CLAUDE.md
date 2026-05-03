# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rolling Prediction Models (RPM) is a research project for real-time full-body motion generation from sparse XR tracking signals (headset + controllers or hand tracking). The model generates smooth motion that transitions seamlessly between tracking mode (when input is available) and synthesis mode (when input is missing).

## Common Commands

### Data Preparation

This branch uses `prepare_data_male_unshaped.py` for AMASS — see "Local Conventions" below
for why (uniform SMPL-H/male/zero-betas + dataset-level P2 split). The original
`prepare_data.py` is left in place for reference but is not the active entry point.

**AMASS P1 (per-file splits, 60 fps):**
```bash
python prepare_data_male_unshaped.py --protocol p1 --save_dir ./datasets_processed/amass_p1 --root_dir ./dataset --splits_dir prepare_data/amass_p1 --out_fps 60
```

**AMASS P2 (whole-dataset split, also 60 fps):**
```bash
python prepare_data_male_unshaped.py --protocol p2 --save_dir ./datasets_processed/amass_p2 --root_dir ./dataset --out_fps 60
```
P2 ignores `--splits_dir` and globs `<root_dir>/<dataset>/*/*_poses.npz` for every
dataset in `AMASS_P2_TRAIN` / `AMASS_P2_TEST` (defined at the top of
`prepare_data_male_unshaped.py`). Each dataset goes entirely into one phase.

**GORP:**
```bash
python prepare_data_gorp.py --root_dir PATH_TO_GORP
```

### Training

```bash
# AMASS-P1 Reactive model
python train.py --results_dir ./results/amass_p1 --dataset amass_p1 --train_dataset_repeat_times 100 --batch_size 128 --input_motion_length 10 --exp_name reactive --rolling_fr_frames 60 --rolling_motion_ctx 10 --rolling_sparse_ctx 10 --loss_velocity 1 --loss_fk 1 --loss_fk_vel 1

# For Smooth model, use --input_motion_length 20
```

### Evaluation

```bash
# Basic evaluation (writes both skip-0 and skip-79f variants by default)
python test.py --model_path ./checkpoints/<dataset>/<model>/model_latest.pt --eval --eval_batch_size 4

# Override skip values (comma-separated; pass a single value to disable the
# second variant)
python test.py ... --eval --eval_skip_frames 0,79

# Hand tracking setup (with input gaps) — also writes per-skip artifacts
python test.py --model_path ./checkpoints/<dataset>/<model>/model_latest.pt --eval --eval_batch_size 4 --eval_gap_config hand_tracking

# With real input signals (GORP only)
python test.py --model_path ... --eval --use_real_input --input_conf_threshold 0.8
```

Per-skip artifacts written to the run's output dir:
- `results_<dataset>_skip{N}f.csv` — per-sequence rows (filename, num_frames, all metric columns).
- `avg_<dataset>_skip{N}f.csv` — averaged scalars in `metric, value` format (matches VR_Pose_Pred).

### Visualization

```bash
python test.py --model_path ./checkpoints/<dataset>/<model>/model_latest.pt --vis --vis_overwrite
```

## Architecture

### Core Components

- **`model/mdm_model.py`**: `RollingMDM` - Transformer-based architecture with cross-attention between motion and sparse tracking signals
- **`model/model_wrapper.py`**: `ModelWrapper` - Wraps the model with PCAF (Progressive Confidence-Adaptive Fusion) output transformation
- **`rolling/rolling_model.py`**: `RollingPredictionModel` - Orchestrates training with rolling windows, free-running simulation, and multi-component loss computation

### Data Flow

1. **Input**: Sparse tracking signal (54-dim: 3 joints x (6D rotation + 6D velocity + 3D position + 3D velocity))
2. **Context**: Motion context (past predicted poses) + sparse context (past tracking signals)
3. **Model**: Cross-attention between motion sequence and sparse embeddings, followed by transformer encoder
4. **Output**: 22-joint body pose in 6D rotation format (132-dim)

### Key Data Types (from `utils/constants.py`)

- `DataTypeGT.RELATIVE_ROTS`: Local joint rotations (6D representation, 132-dim)
- `DataTypeGT.SPARSE`: Sparse tracking signal from headset + wrists
- `DataTypeGT.MOTION_CTX`: Past motion frames for context
- `DataTypeGT.WORLD_JOINTS`: Joint positions in world coordinates

### Training Loop

The `RollingPredictionModel.training_losses()` method:
1. Applies condition masking (for training with missing inputs)
2. Runs free-running simulation (random number of auto-regressive steps without gradients)
3. Computes losses: rotation MSE + velocity + FK joints + FK velocity

### Dataset Structure

Processed data is stored in `datasets_processed/<dataset>/new_format_data/` with:
- Per-sequence `.pt` files containing motion features
- `<dataset>_mean.pt` and `<dataset>_std.pt` for normalization

## Dependencies

- Python 3.10 + PyTorch 2.5.1
- SMPL-H body model required at `./SMPL/smplh/{male,female,neutral}/model.npz` and
  DMPLs at `./SMPL/dmpls/{male,female,neutral}/model.npz`. SMPL-X folder structure
  (`./SMPL/smplx/...`) is unused on this branch — see "Local Conventions" below.
- External libraries: `human_body_prior`, `body_visualizer` (included in repo)

## Key Parameters

- `--input_motion_length`: Prediction window size (10 for reactive, 20 for smooth)
- `--rolling_fr_frames`: Max free-running steps during training
- `--rolling_motion_ctx` / `--rolling_sparse_ctx`: Context lengths
- `--target_type`: Output parameterization (default: PCAF_COSINE)
- `--eval_skip_frames`: comma-separated list of leading-frame counts to drop
  before computing metrics (default `0,79`).

## Local Conventions

This branch is configured as an RPM baseline that's directly comparable to
`/home/yy/Code/VR_Pose_Pred`. Four deviations from upstream RPM:

### 1. Uniform body model: SMPL-H + male + zero betas

Forced everywhere — training FK loss, eval FK, visualization. Implementation:
- `data_loaders/dataloader.py:90-101` (`parse_data_struct`) hardcodes
  `SMPL_GENDER = MALE` and `SMPL_MODEL_TYPE = SMPLH` regardless of dataset
  metadata; reads betas from `body_parms_list["betas"][1:]`.
- `prepare_data_male_unshaped.py` writes `body_parms_list["betas"] = zeros(num_frames, 16)`,
  `gender = "male"`, `surface_model_type = SMPLH`.
- `evaluation/evaluation.py:211-213, :276-278` and `evaluation/visualization.py:230-232`
  pin the generator's body model to `SMPLH/MALE` (was `SMPLX/NEUTRAL` upstream).

Implication: predictions and GT FK both run on SMPL-H male with zero betas.
Metrics are not directly comparable to published RPM/AGRoL P1 numbers (which use
real betas and GT gender), but they *are* comparable to VR_Pose_Pred.

### 2. AMASS P1 + P2 both at 60 fps

`evaluation/utils.py:18-22` sets `DATASETS_FPS[AMASS_P2] = 60` (was 30 upstream).
The `out_fps=60` invocation must be matched by this map — otherwise
velocity/jerk metrics are off by a factor of 2 (jerk by 8) and `MIN_FRAMES_TO_EVAL`
miscounts. P2 numbers from this branch are *not* directly comparable to published
P2 numbers (reported at 30 fps) — only to internal P1/P2 cross-comparison and to
VR_Pose_Pred at the same 60 fps.

### 3. AMASS P2 split is whole-dataset, not per-file

P2 follows the HMD-Poser / VR_Pose_Pred convention rather than the upstream
per-file split files. Hardcoded in `prepare_data_male_unshaped.py`:
```
AMASS_P2_TRAIN = [ACCAD, BioMotionLab_NTroje, BMLmovi, CMU, EKUT,
                  Eyes_Japan_Dataset, KIT, MPI_HDM05, MPI_Limits,
                  MPI_mosh, SFU, TotalCapture]
AMASS_P2_TEST  = [HumanEva, Transitions_mocap]
```
File enumeration: `glob(<root>/<dataset>/*/*_poses.npz)`. P1 still uses per-file
split files.

### 4. Eval comparable to VR_Pose_Pred

- `utils/metrics.py:METRIC_FUNCS_DICT` mirrors VR_Pose_Pred's `scalar_metric_funcs`:
  `mpjre` (body-only, joints 1-21), `rootre`, `upperre`, `lowerre`, `mpjpe`,
  `handpe`, `upperpe`, `lowerpe`, `rootpe`, `mpjve`, `upperve`, `lowerve`,
  `pred_jitter`, `gt_jitter`. Note `mpjre` semantics changed: upstream RPM
  averaged over all 22 joints; this branch averages over 21 body joints (root
  is reported separately as `rootre`).
- `MIN_FRAMES_TO_EVAL`-based implicit slice is gone. Skip is now controlled by
  `--eval_skip_frames` (default `[0, 79]`). FK runs once per sample; only the
  metric slice changes.
- `evaluate_all` returns `dict[skip → (summary_log, fine_grained_df, arr_metrics)]`.
- Each `--eval` run writes one CSV pair per skip:
  `results_<dataset>_skip{N}f.csv` (per-sequence rows) and
  `avg_<dataset>_skip{N}f.csv` (averaged scalars in `metric, value` form).
