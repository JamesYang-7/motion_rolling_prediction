# Model Details — Rolling Prediction Model (RPM)

This document records, end-to-end, what the network is, how it is trained, and
how it is evaluated. References use `file:line` to point at the source of truth.

---

## 1. Model Architecture

The network is a transformer-based motion generator (`RollingMDM`) wrapped
by a thin re-parameterization layer (`ModelWrapper`). At each rolling step
it consumes a short window of motion + sparse tracking signals and predicts
the next window of body poses.

### 1.1 I/O signature

- **Motion / output features (`nfeats = 132`)** — 22 SMPL body joints, each in
  the 6D rotation representation (22 × 6 = 132). Defined in `utils/parser_util.py:189`.
- **Sparse tracking signal (`sparse_dim = 54`)** — 3 entities (HMD head, left
  hand, right hand). For each entity: 6D rotation + 6D rotation-velocity + 3D
  global translation + 3D translation-velocity = 18, ×3 = 54. The exact channel
  layout per entity is in `utils/constants.py:86-148` (`ENTITIES_IDCES`).
- **Number of joints reported in metrics**: 22 (`utils/constants.py:12`,
  `evaluation/evaluation.py:171-174`).

### 1.2 `RollingMDM` — `model/mdm_model.py:14-110`

Hyper-parameters (CLI defaults from `utils/parser_util.py:194-209`; many runs
override these from `args.json`):

| Param | Default | Notes |
|---|---|---|
| `latent_dim` | 512 | transformer width |
| `ff_size` | 1024 | feed-forward width |
| `layers` | 4 | number of `TransformerEncoderLayer` |
| `num_heads` | 4 | heads in self-/cross-attention |
| `dropout` | 0.1 | |
| `activation` | `gelu` | |
| `input_motion_length` | 10 (reactive) / 20 (smooth) | length of the predicted window |
| `rolling_motion_ctx` | 10 (typical) | past-prediction frames fed in |
| `rolling_sparse_ctx` | 10 (typical) | past sparse frames fed in |
| `rolling_latency` | 0 | lookahead frames into future sparse |

Module graph (`model/mdm_model.py:32-105`):

1. **`InputProcess` for motion** — `nn.Linear(132 → latent_dim)`, applied to the
   concatenation `[motion_ctx ; x_start]` (length `rolling_motion_ctx +
   input_motion_length`). After permute → `[seq, B, D]`.
2. **`InputProcess` for sparse** — `nn.Linear(54 → latent_dim)` applied to the
   sparse window of length `rolling_sparse_ctx + 1 + rolling_latency`.
3. **Sinusoidal positional encoding** (`PositionalEncoding`,
   `model/mdm_model.py:114-132`) added to both motion and sparse embeddings.
4. **Cross-attention block** — `nn.MultiheadAttention(latent_dim, num_heads)`
   with motion as query and sparse as key/value, then residual + LayerNorm:
   `xseq = LayerNorm(x_attn(motion, sparse, sparse) + motion_emb)`.
   (`model/mdm_model.py:52-101`)
5. **Transformer encoder stack** — `nn.TransformerEncoder` of
   `seqTransEncoderLayer × layers`, self-attention + FFN per layer
   (`model/mdm_model.py:62-72`).
6. **`OutputProcess`** — `nn.Linear(latent_dim → 132)`, then crop to drop the
   `motion_ctx` prefix. The cropped tensor is the prediction for the
   `input_motion_length` future frames (`model/mdm_model.py:105-110`).

The forward returns
`{ModelOutputType.RELATIVE_ROTS: [B, input_motion_length, 132], ModelOutputType.SHAPE_PARAMS: None}`
(shape-parameter prediction is wired but unused here).

### 1.3 `ModelWrapper` and PCAF re-parameterization — `model/model_wrapper.py`

The backbone output is *not* used directly. `ModelWrapper.transform_model_output`
re-parameterizes it relative to the previous prediction (`prev_pred`) using
**Progressive Confidence-Adaptive Fusion (PCAF)**:

```
out[t] = prev_pred[t] + u[t] * tanh(raw[t] - prev_pred[t])
```

with `u[t]` an uncertainty schedule over the prediction window
(`model/model_wrapper.py:18-30`):

- `PCAF_COSINE` (default, `parser_util.py:154`): `u[t] = 1 - cos(t/T · π/2)`
- `PCAF_COSINE_SQ`: `1 - cos²(...)`
- `PCAF_LINEAR`: `t/T`
- `POSITIONS`: identity (no PCAF)

Effect: near-future frames (`t` small ⇒ `u` small) stay close to `prev_pred`,
far-future frames are allowed to diverge — frames close to "now" are confident,
far-out frames are uncertain.

`prediction_input_type` (`parser_util.py:158-163`): `NONE` (default — model
input is zeros, predictions come purely from cross-attention to sparse + motion
context), `CLEAN` (input = previous prediction), `NOISY` (not implemented).

### 1.4 Parameter count

`Total params: %.2fM` is logged at startup (`train.py:40-49`). Numbers depend on
`latent_dim`/`layers`; with the 512/4 defaults the model is on the order of a
few million parameters.

### 1.5 Temporal layout — what predicts frame `t`

Let `t = current_idx` be the frame the rolling loop is about to commit.
Defaults: `motion_ctx = sparse_ctx = 10`, `input_motion_length = 10`,
`latency = 0`.

```
time:         ... t-10 t-9 ... t-2 t-1   t   t+1 ... t+9 ...
motion_ctx:       [-------- 10 frames ------)
x_start:                                  [-------- 10 frames -------)
sparse:           [-------- 10 frames ------ ][1] (+ latency frames)
target (loss):                            [-------- 10 frames -------)   (training)
committed:                                 ^ only this one frame at inference
```

Window definitions (`data_loaders/dataloader.py:248-258` for training,
`evaluation/generators.py:130-141` for inference):

- **`MOTION_CTX`** = frames `[t - motion_ctx, t)` — strictly past, no overlap
  with the predicted window.
- **`x_start` / target window** = `[t, t + input_motion_length)` — `t` itself
  is the first frame of this window.
- **`SPARSE`** = `[t - sparse_ctx, t + 1 + latency)` — past **plus the present
  frame `t`** plus `latency` future sparse frames.

**Does the model see sparse for frame `t`?** Yes. The "+1" in
`sparse[... : current_idx + 1 + latency]` (`generators.py:138-140`) is exactly
the present frame's tracking signal. With the default `latency = 0` the model
is causal in sparse (past + present, no future). A non-zero `--rolling_latency`
adds that many future sparse frames (a peek-ahead the headset stack would have
to buffer).

**What flows into the prediction of frame `t`.** The transformer sequence is
built as `concat([motion_ctx ; x_start])` along time (`mdm_model.py:88`) — one
stream of length `motion_ctx + input_motion_length = 20`. Inside the network:

1. The concatenated motion stream cross-attends to the sparse stream
   (motion = query, sparse = K/V, `mdm_model.py:96-100`).
2. Then it self-attends within itself through the encoder stack
   (`mdm_model.py:103`). No causal mask — every position attends to every
   other; but input at the `x_start` positions is **zeros** (because
   `prediction_input_type = NONE`, `model_wrapper.py:89-90`), so the only real
   information at those positions is the positional encoding. Information
   leakage from "true future" is impossible because the future is never fed in.
3. `OutputProcess` keeps only the `x_start` slice as the raw prediction
   (`mdm_model.py:107`).
4. `ModelWrapper` re-parameterizes with PCAF:
   `out[i] = prev_pred[i] + u[i]·tanh(raw[i] - prev_pred[i])`. For frame
   `i = 0` (the one being committed) `u[0] ≈ 0.012`, so the committed value is
   anchored very close to the previous step's `prev_pred[0]` — i.e. small
   refinements per tick.

So, concretely, frame `t` is predicted from:

- **Motion context** (10 own past predictions, frames `t-10..t-1`).
- **Sparse** (frames `t-10..t`, optionally plus `latency` future frames).
- **Positional encoding** at the slot for frame `t` (the model input there is
  zeros, not the prior prediction).
- **`prev_pred[t]`** through the PCAF residual path, which makes the actual
  committed frame mostly "previous prediction + tiny correction."

**Rest of the `x_start` window (`t+1 .. t+9`).** Predicted in the same forward
pass but **only frame `t` is written to `output`** (`generators.py:151`). Next
step the window shifts left by one, the new far-end slot is zeroed
(`generators.py:161-163`), and `current_idx` advances. The far-end frames serve
as a soft horizon — they let the encoder reason about a 10-frame future
trajectory while the PCAF schedule lets confidence build up as a frame
approaches the commit point.

**Where the motion context comes from.**

- *Inference* (`generators.py:131-133`): the model's own past predictions
  (`output[current_idx - motion_ctx : current_idx]`). The first
  `ctx_margin = max(motion_ctx, sparse_ctx)` output frames are seeded with the
  GT prefix to bootstrap the rollout (`generators.py:110`).
- *Training* (`OnlineTrainDataset`, `dataloader.py:249-251`): GT relative rots
  are loaded as the motion context. During free-running
  (`rolling_model.py:179-217`) the no-grad rollout overwrites that buffer with
  own predictions, so by the grad step the context is already self-generated;
  the gradient step itself sees a context that mixes "older GT" + "recently
  rolled-out predictions," matching the inference distribution.

**With free-running (`rolling_fr_frames > 0`).** Each loaded sample is
`total_len = input_motion_length + max(motion_ctx, sparse_ctx) + rolling_fr_frames`
long, so the same sample supports a random `fr ∈ [0, rolling_fr_frames]` of
no-grad rolling steps before the grad step lands at offset `fr`
(`rolling_model.py:187-217`). The relationship above still holds at each step:
`motion_ctx` is strictly the previous `motion_ctx` frames, `sparse` is
past+present(+latency), and only one new frame "exists" per step.

---

## 2. Training Strategy

Entry point: `train.py`. Inner loop: `runner/training_loop.py:TrainLoop`.
Loss/free-running orchestration: `rolling/rolling_model.py:RollingPredictionModel`.

### 2.1 Data pipeline

- **Raw**: AMASS / GORP processed by `prepare_data_male_unshaped.py` (this
  branch — see `CLAUDE.md`) into per-sequence `.pt` files containing
  `rotation_local_full_gt_list` (132-d), `hmd_position_global_full_gt_list`
  (54-d), `position_global_full_gt_world` (22 × 3), head transform, etc.
- **Train dataset**: `OnlineTrainDataset` (`data_loaders/dataloader.py:190-275`).
  - Crop length: `total_len = input_motion_length + max(motion_ctx, sparse_ctx) + rolling_fr_frames`.
  - For each `__getitem__` it randomly samples a window of length `total_len`,
    splits it into:
    - `cond[MOTION_CTX]` = GT relative rots of length
      `motion_ctx + freerunning_frames` (taken from the GT, not from past
      predictions — these will be fed in turn during free-running).
    - `cond[SPARSE]` = sparse window of length
      `sparse_ctx + 1 + latency + freerunning_frames`.
    - `gt[RELATIVE_ROTS]` = GT for the `input_motion_length + freerunning_frames`
      frames being predicted.
  - Normalizes motion + motion-context with per-channel `(mean, std)` computed
    once on the train split (`dataloader.py:140-148, 521-552`).
- **Repeating**: `train_dataset_repeat_times` (default 1000) virtually inflates
  epoch length so checkpoint/eval cadence is reasonable.

### 2.2 Condition (sparse) masking — simulating tracking loss

`model/maskers.py` + `parser_util.py:135-151`. Default masker is
`SEG_HANDS_IDP` (segment-wise, hands, independent per hand):

- The HMD/head channel is **never** masked (always tracked).
- For each hand, with probability `cond_mask_prob = 0.1`, a contiguous segment
  of length sampled in `[masker_minf, masker_maxf]` is zeroed out within the
  current crop. Independence means left and right hand are masked by separate
  random draws.
- Other modes: `SEG_ALL` (mask all 3 entities together), `SEG_HANDS` (both
  hands jointly), `SEQ_*` (mask the entire sequence rather than a segment).

This teaches the model to switch between "tracking" and "synthesis" smoothly.

### 2.3 Free-running rollout — `rolling/rolling_model.py:179-221`

For each batch:

1. Initialize `x_start = gt[RELATIVE_ROTS].clone()`, then zero out frames after
   the first prediction window (`x_start[:, nframes:] = 0`).
2. Sample `fr ∈ [0, rolling_fr_frames]` uniformly (`rolling_model.py:187`).
3. **No-grad stage**: `model.eval()`, run `fr` rolling steps with
   `update_context=True`. Each step:
   - Predict frames `[i, i+nframes)` using context `[i, i+motion_ctx)`,
     sparse `[i, i+sparse_ctx+1+latency)`.
   - Write the prediction back into `x_start[i:i+nframes]` and append the first
     predicted frame as the next motion-context frame.
   - This makes the model accustomed to its own roll-out errors.
4. **Grad stage**: `model.train()`, run one more step at offset `fr` with
   `update_context=False` (we only back-prop through this final step).
5. Slice GT to the same `[fr, fr+nframes)` window for loss computation.

Crucially, the last frame of the `x_start` window is reset to zeros each step
(`rolling_model.py:158-160`, `generators.py:163`) — the model uses the zero
slot as a signal that this is the fresh / most-uncertain frame.

### 2.4 Losses — `rolling/rolling_model.py:223-286`

Per-frame distance (`loss_distance`) is the per-joint norm (default L1, see
`parser_util.py:107`) over the 6D / 3D groups, averaged across joints. Velocity
loss (`loss_velocity`) applies the same to first-difference signals.

Total loss weighted-sum:

- `ROT_MSE` — local 6D rotation loss vs GT (always on, weight 1).
- `VEL_MSE` — first-diff of rotations, weighted by `--loss_velocity`.
- `JOINTS_MSE` — 22 joint world positions after differentiable FK
  (`process_prediction_through_fk`), weighted by `--loss_fk`. Predicted joints
  are head-aligned to the GT head translation
  (`rolling_model.py:144-149`) so the loss doesn't penalize global drift.
- `JOINTS_VEL_MSE` — first-diff of FK joints, weighted by `--loss_fk_vel`.

FK uses `BodyModelsWrapper.grad_fk` (`evaluation/utils.py:81-83, 134-139`) with
SMPL-H + male + zero betas (forced on this branch — see `CLAUDE.md` "Local
Conventions"). Typical training run uses
`--loss_velocity 1 --loss_fk 1 --loss_fk_vel 1` (`CLAUDE.md`).

### 2.5 Optimization & schedule — `runner/training_loop.py`

- Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)` (`parser_util.py:333-335`).
- Mixed precision wrapper present (`MixedPrecisionTrainer`) but `use_fp16 = False`.
- LR schedule: a single one-shot decay — once `step > lr_anneal_steps`
  (default 50 000), `lr ← lr / 30` and the flag flips off
  (`training_loop.py:211-221`). No warmup, no per-step annealing.
- `num_steps` default 100 000 (`parser_util.py:363-367`), iterated as
  `num_epochs = num_steps // len(data) + 1`.
- Each epoch: full pass over the (repeated) dataset, then optional eval +
  checkpointing every `save_interval` epochs (default 10).
- Checkpoints: `model{step:09d}.pt` plus `model_latest.pt` and matching
  `opt*.pt`, into `<results>/checkpoints/<exp_name>/` (`training_loop.py:230-245`).
- Logging: TensorBoard scalar per loss type plus a `quartiles` Multiline view
  (`training_loop.py:34-39, 169-175`).
- Seeding: `random/np/torch.manual_seed(args.seed)` (default 10) in
  `train.py:67-69`. `cudnn.benchmark = False`.
- Multi-GPU: `nn.DataParallel` if `torch.cuda.device_count() > 1`
  (`train.py:35-42`).

### 2.6 In-training evaluation (optional)

If `--eval_during_training`, after each save tick the loop runs
`EvaluatorWrapper.evaluate_all()` on the test split (no gap config), writes
per-skip CSVs, and pushes the smallest-skip metrics to TensorBoard
(`training_loop.py:88-121, 247-267`).

---

## 3. Evaluation Process

Entry point: `test.py`. Generation logic: `evaluation/generators.py`. Metric
aggregation + I/O: `evaluation/evaluation.py`. Metric definitions:
`utils/metrics.py`.

### 3.1 Test dataset — `data_loaders/dataloader.py:278-437`

- Loads the test split with the train split's `(mean, std)` for normalization.
- Filters by `[min_frames, max_frames]`.
- Optional: `eval_gap_config` (a JSON in
  `datasets_processed/<dataset>/eval_gap_configs/`) injects gaps in the sparse
  signal — for each (entity, [t0, t)) range it zeros those channels. Stores
  the gap layout for tracking-loss metrics
  (`dataloader.py:369-398`).
- Optional `--use_real_input` (GORP): use real tracker stream + per-hand
  confidence threshold to mask low-confidence frames
  (`dataloader.py:60-74`).
- Optional `--no_gt_init` + `--bootstrap_data_path`: replace the GT-prefix
  initialization with predictions from another model (bootstrap chaining,
  `dataloader.py:309-334`).

### 3.2 Rolling generation — `evaluation/generators.py:RollingGenerationWrapper`

Inputs: full-sequence `gt_data [B, T, 132]` and `sparse [B, T, 54]`.

1. `ctx_margin = max(motion_ctx, sparse_ctx)`. The first `ctx_margin` output
   frames are filled with the (normalized) GT prefix — this is the warm-start
   the model needs to begin rolling.
2. Initialize `x_start = gt_data[:, ctx_margin : ctx_margin + input_motion_length]`
   (the first window the model will refine).
3. Loop `current_idx` from `ctx_margin` until end of sequence:
   - Build `MOTION_CTX = output[current_idx-motion_ctx : current_idx]`
     (the model's own past predictions).
   - Build `SPARSE = sparse[current_idx-sparse_ctx : current_idx+1+latency]`
     (past + present + lookahead).
   - Forward pass `model(x_start, cond)` returns the refined window.
   - Commit only the *first* frame of the refined window to `output[current_idx]`.
   - Shift `x_start` left by 1 (`x_start[:-1] = x_start[1:].clone()`) and zero
     the last (newly entering) frame.
   - `current_idx += 1`.
4. After the loop, inverse-transform (un-normalize) the full output buffer.

The result is a per-frame causal stream where each frame was produced with
`motion_ctx` past predictions and `sparse_ctx + 1 + latency` sparse frames.

### 3.3 Forward kinematics — building joint positions for metrics

`evaluation/evaluation.py:152-174` runs FK once per sequence with
`SMPLH/MALE` (`evaluation/evaluation.py:223-225, 293-295`):

- **Predicted body**: `get_body_poses` (`evaluation/utils.py:160-206`) takes
  the predicted local rotations, performs FK with `betas` (zero on this branch
  unless dataset provides them), then re-anchors the root so the predicted head
  matches the GT head global translation. This isolates joint-articulation
  error from global head-tracking noise.
- **GT body**: `get_GT_body_poses` calls FK on the stored GT body params.
- Outputs: `Jtr[:, :22]` (22 joint positions), `full_pose[:, :22]` (per-joint
  axis-angle reshaped), and the mesh `v` for each. All downstream metrics use
  these tensors.

### 3.4 Metrics — `utils/metrics.py`

The metric set mirrors VR_Pose_Pred (`CLAUDE.md` "Local Conventions"):

| Metric | Units | Joints | Notes |
|---|---|---|---|
| `mpjre` | deg | body (1–21) | mean per-joint rotation error, body-only — *not* the upstream RPM mpjre over all 22 joints |
| `rootre` | deg | root (0) | rotation error of the root joint |
| `upperre` / `lowerre` | deg | upper / lower body | |
| `mpjpe` | cm | all 22 | mean per-joint position error |
| `handpe`, `upperpe`, `lowerpe`, `rootpe` | cm | subgroups | |
| `mpjve`, `upperve`, `lowerve` | cm/s | all / subgroups | first-diff position error × fps |
| `pred_jitter`, `gt_jitter` | m/s³ | all 22 | third-difference × fps³ |

When an `eval_gap_config` is set, the tracking-loss metric set adds
**transition jerk** curves (`S_to_T_jerk`, `T_to_S_jerk` and GT counterparts):
0.5 s windows at the boundaries of synthesis ↔ tracking transitions
(`utils/metrics.py:175-231`). These are saved as `[N, T]` arrays for plotting
(`evaluation.py:399-455`); two derived scalars are reported:

- **PJ** (peak jerk): `max` of mean predicted jerk in the transition window.
- **AUJ** (area under jerk): `Σ |pred − gt|` over the same window.

### 3.5 Skip-frame averaging — `evaluation/evaluation.py:114-208`

`--eval_skip_frames` (default `[0, 79]`, parsed from `0,79`,
`parser_util.py:285-293`). For each skip:

- Drop the first `skip` frames of the FK output **before** computing metrics
  (`pr_pos_full[skip:]` etc., line 192-200). FK itself is run only once.
- Sequences with `skip ≥ T - 3` are dropped from that skip's aggregation
  (need 3 frames for jerk, line 179).
- `from_gaps_to_masks` and `remove_frames_to_gaps` re-anchor gap masks to the
  shifted timeline.

Per-skip aggregation (`evaluation.py:331-387`):

- Scalar metrics → `AverageValue` (NaN-safe sum/count → mean).
- Array metrics (transition jerk) → `AccumulateArray` (concatenate across
  sequences).
- Per-sequence rows go into a fine-grained DataFrame.

### 3.6 Output artifacts — `test.py:94-118`

For each skip `N`, written to
`results/<results_folder>/<exp_name>/<checkpoint_name>_rolling[<gap_cfg>]/`:

- `results_<dataset>_skip{N}f.csv` — one row per sequence: filename,
  num_frames, all scalar metric values.
- `avg_<dataset>_skip{N}f.csv` — averaged scalars in `metric, value` form (two
  columns), matching VR_Pose_Pred's averaged-results format.
- `arr_<metric><suffix>.npz` — raw `[N, T]` arrays for jerk plots.
- `<S_to_T|T_to_S>_transition<suffix>.png` — averaged jerk vs GT plot per
  transition direction.
- `summary log` printed to stdout (`evaluator.print_results`).

### 3.7 Visualization — `evaluation/visualization.py`

`--vis` / `--vis_gt` activates `VisualizerWrapper.visualize_subset` instead of /
in addition to evaluation. Body model is again pinned to SMPL-H / male
(`CLAUDE.md` Local Conventions).

---

## 4. Key Cross-References

- Local conventions deviating from upstream RPM (uniform SMPL-H/male/zero-betas,
  60 fps for both AMASS protocols, dataset-level P2 split, VR_Pose_Pred-style
  `mpjre`): `CLAUDE.md`.
- Architecture: `model/mdm_model.py`, `model/model_wrapper.py`.
- Training orchestration: `train.py`, `runner/training_loop.py`,
  `rolling/rolling_model.py`, `data_loaders/dataloader.py:OnlineTrainDataset`.
- Evaluation: `test.py`, `evaluation/generators.py`, `evaluation/evaluation.py`,
  `evaluation/utils.py`, `utils/metrics.py`,
  `data_loaders/dataloader.py:TestDataset`.
- All flag definitions: `utils/parser_util.py`.
