# RPM (Rolling Prediction Model)

Online, real-time full-body pose generator for XR. Input: sparse 6-DoF tracking from headset + 2 hands (either controllers or unreliable hand-tracking) at 30/60 FPS. Output: 22-joint SMPL/SMPL-X body represented as 6D local rotations (132-d) plus optional FK joint positions. "Rolling" prediction means at each timestep the model does not emit a single pose; it re-refines a length-`W` window of future poses, committing one frame per step and carrying the rest forward as a strong prior for the next step. PCAF (Prediction Consistency Anchor Function) bounds how much the next iterate can deviate from the previous one, which is the mechanism that makes transitions (tracking-to-synthesis and back) smooth instead of snapping.

## Methodology

**Architecture.** Small MDM-style transformer encoder (default 4 layers, `latent_dim=512`, `ff_size=1024`, 4 heads, GELU, dropout 0.1). Two linear input heads embed (a) past motion context `X_t` of length `M` and (b) past + present tracking `C_t` of length `I+1` (+ `latency` future frames if non-zero); their tokens are concatenated via cross-attention, then passed through the encoder. The output head predicts the whole `W`-frame window `P_t = {x_t, x_{t+1}, ..., x_{t+W}}`. Source: `model/mdm_model.py` (built by `utils/model_util.py:create_model_and_rpm`). Default `W = input_motion_length = 10`, `M = rolling_motion_ctx = 10`, `I = rolling_sparse_ctx = 10`, `latency = 0`.

**Rolling prediction.** Paper Eq. 1: each step outputs `W` future poses given `M` past poses + `I+1` past tracking inputs. After a step, shift the window by one: commit `x_t`, drop it from the front of `P`, and append a zero "blank" pose at the end. See the inference loop in `evaluation/generators.py:130-165` (line 163 zeros out the last frame).

**PCAF.** Eq. 2: `P_t = P_{t-1} + U * tanh(f_theta(X_t, C_t) - P_{t-1})`, with `U = (u_1, ..., u_W) in [0,1]^W`. Three hand-crafted schedules and one "cosine-based" default matching DDPM noise schedules:
- `pcaf_cosine` (default): `u_t = 1 - cos(t/W * pi/2)`
- `pcaf_cosine_sq`: `u_t = 1 - cos^2(...)`
- `pcaf_linear`: `u_t = t/W`
See `model/model_wrapper.py:18-30`. The effect: near-present frames (small `t`) are nearly locked to the previous window (low `u`), far-future frames (large `t`) can move freely (high `u`). This forces long-term prediction to absorb new information gradually.

**Inference.** Pure autoregressive rollout starting from `ctx_margin` GT frames. One forward per output frame, ~207 FPS on A100 (paper Sec. 4). No diffusion/denoising chain. Gaps in tracking input are just zeros in `C_t` — the masker that injects them at training time teaches the model to synthesize through them. Concretely: `ctx_margin = max(rolling_motion_ctx, rolling_sparse_ctx)` (= 10 at defaults, `evaluation/generators.py:103`); the output buffer's first `ctx_margin` frames are seeded from `gt_data` at `generators.py:110`; the rolling loop from `current_idx = ctx_margin` (`:113`) reads `output[current_idx - M : current_idx]` as motion context (`:131-133`) and writes exactly one self-predicted frame per step back into `output[current_idx]` (`:151`).

## Training details

- **Datasets & FPS.** Synthetic: AMASS P1 at 60 FPS (intra-dataset), AMASS P2 at 30 FPS (cross-dataset). Real: GORP at 30 FPS (~14h, 28 subjects, Meta Quest 3 + OptiTrack).
- **Sequence window.** `total_len = input_motion_length + max(sparse_ctx, motion_ctx) + freerunning_frames` (`dataloader.py:232`). For default training: `10 + 10 + 60 = 80` frames per sample. Sequences shorter than `total_len` are filtered out (`dataloader.py:533`).
- **Preprocessing.** 6D rotations (Zhou et al.) for global body orientation + 21 body joints (132 dim). Non-gendered SMPL-X meshes with each subject's true shape coefficients (paper Sec. 4). Normalization is `(x-mean)/std` computed from train split unless `--no_normalization`.
- **Loss.** `L = lambda_ori*L_ori + lambda_rot*L_rot + lambda_pos*L_pos + lambda_ori_vel*L_ori_vel + lambda_rot_vel*L_rot_vel + lambda_pos_vel*L_pos_vel` (Eq. 3 of paper). In code (`rolling_model.py:236-284`) this is four terms:
  1. **Relative-rotation loss** on `RELATIVE_ROTS` (the 132-d 6D rotation output): **always added, weight 1, no scaling factor exposed** (`rolling_model.py:236-241`). Covers both L_ori and L_rot since "relative rots" includes global body orientation plus 21 body joints.
  2. **Relative-rotation velocity** — weighted by `loss_velocity` (gated on `!= 0`).
  3. **Global joint position** (FK'd world joints) — weighted by `loss_fk`. This is the L_pos (global position) factor, not an orientation factor.
  4. **Global joint-position velocity** — weighted by `loss_fk_vel`.
  - `loss_dist_type = L1` (default; paper says "L1 distance of absolute values between last predicted window and GT")
  - `--loss_velocity 1 --loss_fk 1 --loss_fk_vel 1` (from `train_gorp.sh` and the commented AMASS-P1 command in `prepare_data_and_train_test.sh`)
- **Condition masking.** `masker=seg_hands_idp`, `cond_mask_prob=0.1`, both hands masked independently per *segment* of random length within the window (`model/maskers.py:67-99`). Head is never masked.
- **Optimizer.** AdamW, `lr=3e-4`, `weight_decay=1e-4`, `lr_anneal_steps=50000` (at which point lr is divided by 30, `training_loop.py:211-221`).
- **Duration / batch.** `num_steps=100000`, `batch_size=128` (shell scripts override default 32), `train_dataset_repeat_times=100` (epochs are small since the dataset is re-iterated many times per "epoch"). Paper: 12h on a single NVIDIA A100.
- **Free-running.** `rolling_fr_frames = 60` for AMASS P1 / P2, `30` for GORP (1s of real time). See `train_gorp.sh` and `prepare_data_and_train_test.sh`.

## Data split

- **AMASS P1 (intra-dataset, 60 FPS).** Subjects drawn from **BioMotionLab_NTroje, CMU, MPI_HDM05**. Per-dataset `train_split.txt` / `test_split.txt` lists. Totals: train 4725 sequences (2754+1778+193), test 526 sequences (307+197+22). Lists live in `prepare_data/amass_p1/<dataset>/`.
- **AMASS P2 (cross-dataset, 30 FPS).** Train on 14 AMASS subsets (ACCAD, BMLmovi, BioMotionLab_NTroje, CMU, EKUT, Eyes_Japan_Dataset, KIT, MPI_HDM05, MPI_Limits, MPI_mosh, SFU, TotalCapture, Transitions_mocap, etc.); only `train_split.txt` files are provided — the test set follows the standard cross-dataset AvatarPoser protocol (HumanEva + subject-held-out evaluation). Train totals ~11781 sequences.
- **GORP.** `train_split.txt` (1274), `test_controllers_split.txt` (147), `test_tracking_split.txt` (162) in `prepare_data/gorp/GORP/`. Separate test sets for the two sensing modalities. No explicit val split.
- **No val split / no held-out subjects listed in code.** Splits are pre-computed file lists, not identity-based partitions derivable from code alone.

## Checkpoint selection for test

- The training loop does **not** track a "best" checkpoint. It writes `model_latest.pt` every `save_interval=10` epochs (`training_loop.py:241-245`), plus timestamped `modelXXXXXXXXX.pt`. Optional `eval_during_training` runs the full rolling evaluator during training but only dumps csvs and TB logs — it never selects a checkpoint.
- The released shell script simply evaluates `model_latest.pt` (`prepare_data_and_train_test.sh:5`). **Reported test numbers use the last-epoch checkpoint after `num_steps=100000`, no early stopping, no EMA.**

## Evaluation

- **Metrics.** MPJRE (mean per-joint rotation error, deg), MPJPE (position, cm), MPJVE (velocity, cm/s), Jitter (10^2 m/s^3 — third derivative), Peak Jerk (PJ) and Area-Under-Jerk (AUJ) computed over a 1s window around synthesis<->tracking transitions (`PJ_{T-S}`, `AUJ_{T-S}`, `PJ_{S-T}`, `AUJ_{S-T}`).
- **Initial 1s is skipped** in every sequence (warm-up, paper Sec. 4). Defined as `MIN_FRAMES_TO_EVAL = fps * 1` in `evaluation/utils.py:25` and applied by slicing `pr_pos[MIN_FRAMES_TO_EVAL - 1 :]` (and matching GT/mesh) in `evaluation/evaluation.py:158-185`. This is what nullifies the GT seed from `ctx_margin`: at the first scored frame (index 59 at 60 fps, ctx_margin=10), the rolling loop has already overwritten `output[49:59]` with 49 AR self-predictions, so the reported metrics never see a GT-seeded motion context.
- **Paper framing of skip-1s (Sec. 4 "Evaluation" verbatim):** *"Our evaluation skips an initial 1-second padding in all sequences to allow the methods to warm up."* That is the **entire** justification in the paper — no discussion of GT-bootstrap fairness, no discussion of state-carrying vs stateless. Skip-1s is presented as uniform warm-up across all methods. The `ctx_margin` GT seed in the code is never mentioned as a variable; **GT-init is an implementation detail, not a reported knob**. Implication for our port: skip-1s serves two mechanical purposes — (a) finite-difference boundary relief for velocity/jitter metrics, (b) diffusing away the GT-seeded motion context for state-carrying methods. Stateless baselines get it gratis as a fairness hedge.
- **Baseline composition (Sec. 4 p.1854).** Paper distinguishes two camps explicitly:
  - *State-less:* AvatarPoser, SAGE, EgoPoser, AvatarJLM — "match the new hand-tracking signal as soon as it reappears, leading to abrupt transitions (i.e., very high AUJ_{S-T})."
  - *State-carrying:* HMD-Poser (RNN), AGRoL (diffusion), RPM (motion_ctx + FR). Paper: *"Even HMD-Poser, which leverages an RNN that carries on the temporal information, shows very rough transitions after tracking input losses."*
  - **HMD-Poser is not motion-context-free** — it has an RNN hidden state. Common mischaracterization to avoid: "baselines are all motion-context-free" is wrong; the split is stateless (4) vs state-carrying (3, including RPM itself). Skip-1s is the community leveling mechanism between these camps.
- **Gap protocol.** Tracking-signal losses of 0.5–2s simulated on AMASS to build HT variants; real gaps on GORP HT. Configured via `eval_gap_config` JSON loaded in `dataloader.py:561`.
- **Two operating points reported.** *RPM-Reactive* (`W=10` on A-P1, `W=5` on A-P2/GORP) trades smoothness for responsiveness; *RPM-Smooth* (`W=20` on A-P1, `W=10` on A-P2/GORP) is the default smooth setting.

## Ablation (Table 3, paper Sec. 4; A-P1, HT setup)

| RPM | FR | PCAF | MPJRE | MPJPE | MPJVE | Jitter | AUJ_T-S | AUJ_S-T |
|-----|----|------|-------|-------|-------|--------|---------|---------|
| ✓   | —  | —    | 6.69  | **10.59** | 31.84 | 5.84 | 101.35 | 210.63 |
| ✓   | ✓  | —    | 3.79  | 5.04  | 23.89 | 8.32 | 428.32 | 799.03 |
| ✓   | —  | ✓    | 5.53  | 9.07  | 27.85 | 3.82 | 138.17 | 116.68 |
| ✓   | ✓  | ✓    | 3.82  | 5.18  | 22.83 | 4.35 | 60.51  | 69.02  |

- **FR is a degeneracy-fix, not a drift-robustness improvement.** Paper Sec. 4.1: *"Motion generated with RPM tends to degenerate unless combined with free-running (FR)."* Without FR, MPJPE doubles (5.04 → 10.59). This is RPM's rolling architecture specifically breaking down, not a generic exposure-bias gain.
- **PCAF is the smoothness lever, not the accuracy lever.** Turning PCAF on with FR moves MPJPE 5.04 ↔ 5.18 (no real accuracy change) but slashes transition jerk (AUJ_T-S 428 → 61, AUJ_S-T 799 → 69) and the raw jitter (8.32 → 4.35).
- **Cross-reference with our runs.** Our `per_sensor_embedders` (no FR) is MPJPE **2.58**; `freerunning_fr10` (FR=10) is **2.59**. Our PatchTransformer arch does **not** degenerate without FR, so the RPM ablation's main FR benefit doesn't have an analogue here. The architectural difference matters: RPM commits the **first** window frame per step (requires ctx consistency for forward rolling); our PatchTransformer commits the **last** predicted frame (consumes less of its own history per step). The exposure pattern differs, and "need FR" transfers to architecture-dependent conditions that our setup doesn't trigger.

## Cross-validation notes

- **Free-running step count.** Paper Alg. 1 line 5: `fr ~ U(0, FR)`. Code at `rolling/rolling_model.py:187` confirms `th.randint(0, self.max_freerunning_steps + 1, (1,))` per **batch** (not per sample). The `fr` no-grad rollout is done on `model.eval()` mode, then a single `model.train()` forward carries gradients (`rolling_model.py:201-217`). `fr=60` for AMASS (~1s at 60 FPS), `fr=30` for GORP (~1s at 30 FPS). Gated on `max_freerunning_steps > 0` at `rolling_model.py:297-300`; when 0, training reduces to a single teacher-forced forward on `x_start = gt_data[RELATIVE_ROTS]`.
- **Free-running overwrites the motion context in-place.** Inside `freerunning_step` (`rolling_model.py:153-177`), each no-grad iteration both (a) writes its W-frame prediction back into `x_start[i:i+nframes]` (line 173) and (b) overwrites **one** frame of the motion context: `cond[MOTION_CTX][:, i + motion_cxt_len] = prediction[:, 0]` (lines 174-176). After `fr` iterations the motion context for the grad-bearing step is entirely self-predicted once `fr >= motion_cxt_len`. The grad-stage target is shifted by `fr`: `gt_data = gt_data[:, fr : fr+nframes]` (`rolling_model.py:190-199`) — loss is computed on the AR-offset window, not the original window.
- **Training sample geometry.** `total_len = input_motion_length + max(sparse_ctx, motion_ctx) + freerunning_frames` in `data_loaders/dataloader.py:232-236` (default AMASS: `10 + 10 + 60 = 80` frames). Sequences shorter than `total_len` are filtered; the extra `fr_max` frames of headroom are what the no-grad FR rollout consumes.
- **Loss weights.** Paper Eq. 3 lists six terms; code collapses to four: a mandatory relative-rotation L1 (always weight 1, no scaling) plus three weighted terms (`loss_velocity`, `loss_fk`, `loss_fk_vel`). `loss_fk` is the factor on the **global joint-position** loss (FK'd world joints), not a rotation term. The relative-rotation term already covers global body orientation (L_ori) and the 21 body-joint rotations (L_rot) jointly since `RELATIVE_ROTS` is the 132-d stack. All three scalable weights are set to `1` in the shell scripts.
- **Distance norm.** Paper Sec. 3.3 says "L1 distance of absolute values". Default `loss_dist_type=LossDistType.L1` (`parser_util.py:108`).
- **Masker default.** Paper is silent on the exact masking scheme; code default is `seg_hands_idp` with `cond_mask_prob=0.1` and unbounded max segment length (`masker_maxf=sys.maxsize`, `parser_util.py:148`).
- **Training uses GT body shape.** `BodyModelsWrapper.grad_fk` uses GT `betas` for FK loss (`rolling_model.py:126-131`), i.e. shape is assumed known (paper Sec. 4 mentions a "prior calibration step").
- **Final checkpoint is last, not best.** No validation-based selection in `training_loop.run_loop`.
- **GORP split has no val.** Only train / test_controllers / test_tracking. Hyperparameters were tuned on AMASS P1 and transferred.

## Parameters relevant to our PatchTransformer port

| Knob | RPM value | Our repo (`config/patch_transformer_ap1.yaml`) | Note |
|---|---|---|---|
| `max_freerunning_steps` | 60 (AMASS) / 30 (GORP) | **0** | RPM's core smoothness contribution. Enabling it means also setting `input_motion_length = W + M + fr`. |
| Masker | `seg_hands_idp`, `cond_mask_prob=0.1` | **none** | RPM's hand-dropout is what lets it handle missing HT. Worth enabling for any HT-realistic eval. |
| Prediction target | `pcaf_cosine` | `pcaf_cosine` (match) | PCAF schedule is identical; both use `1 - cos(t/W * pi/2)`. |
| Window geometry | `W=10, M=10, I=10` | `target_window_length=10, motion_context_window_length=20, prediction_shift=10` | Our `M=20` is larger; check whether any of the extra context is actually fed to the model vs just buffered. |
| Loss weights | relative-rot L1 **mandatory (coef 1, not exposed)** + `loss_velocity=1` (rel-rot vel) + `loss_fk=1` (global pos) + `loss_fk_vel=1` (global pos vel) | Check `model.loss_*` in yaml | `loss_fk` is the **global-position** factor, not a rotation factor. Relative-rot loss can't be disabled/rescaled in RPM. |
| Checkpoint for test | `model_latest.pt` (last epoch) | `best_ckpt.pth.tar` for val, `ckpt.pth.tar` (last) for final autoreg | We *do* pick best-val for eval during training; RPM does not. |
