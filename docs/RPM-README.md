# RPM: full-body pose from a VR headset and two controllers

RPM (Rolling Prediction Model; Barquero et al., *From Sparse Signal to Smooth Motion: Real-Time Motion Generation with Rolling Prediction Models*, CVPR 2025) predicts a 22-joint SMPL-X body pose (local joint rotations plus world joint positions) from the head and two hand poses of a VR device. It runs causally, one output frame per input frame, with no lookahead. This export holds the authors' released checkpoints, each as one ONNX "step" graph with its reference output, plus three reference captures from GORP's motion-controller test split shared by all of them. The weights are the released ones, unchanged.

| Path | What it is |
|---|---|
| `RPM-README.md` | This file. Shared by all model versions. |
| `data/gorp/<capture>.npz` | Three GORP test captures: raw device stream, mocap ground truth and the subject's skeleton ([section 6](#6-checking-a-port)). The same for every model. Float32 arrays only. |
| `models/<name>/rpm_<name>.onnx` | One checkpoint's rolling step (contract in [section 3](#3-the-step-graph)). Opset 15, float32, batch 1, 37 MiB (9.6 M weights). |
| `models/<name>/manifest.json` | That checkpoint's provenance, IO contract, loop, the normalisation statistics needed to seed the state ([section 5](#5-the-inference-loop)), per-capture metrics and the numeric checks made at export. |
| `models/<name>/output_ref/<capture>.npz` | That checkpoint's input and reference output for each capture in `data/gorp/` ([section 6](#6-checking-a-port)). |

## Model versions

`<name>` is `<release folder>_<variant>`. Reactive and Smooth differ only in the length L of the rolling prediction window; every version reads 11 input frames and a 10-row motion state.

| `<name>` | Trained on | L | Input rate | GORP MC, real device input: MPJPE / MPJVE / jitter (RPM paper Table 4) |
|---|---|---|---|---|
| `gorp_real_inputs_reactive` | GORP, real device input (controller and hand-tracking captures) | 5 | 30 fps | 6.83 cm / 10.93 cm/s / 0.57 |
| `gorp_real_inputs_smooth` | same | 10 | 30 fps | 7.36 cm / 12.09 cm/s / 0.41 |
| `gorp_reactive` | GORP, synthetic input computed from the mocap body | 5 | 30 fps | 7.36 cm / 12.93 cm/s / 0.77 |
| `gorp_smooth` | same | 10 | 30 fps | 7.58 cm / 12.92 cm/s / 0.47 |
| `amass_p1_reactive` | AMASS, protocol 1, synthetic input | 10 | 60 fps | — |
| `amass_p2_reactive` | AMASS, protocol 2, synthetic input | 5 | 30 fps | — |
| `amass_p2_smooth` | same | 10 | 30 fps | — |

The release's `amass_p1/smooth` checkpoint is an empty file in the authors' `rpm_checkpoints.zip`, so it has no graph.

This README describes `gorp_real_inputs_reactive` in detail, the version trained on what a headset delivers. The accuracy, sizes, timings and measurements quoted are its own. The others share the contract and the loop; their manifests state their window, frame rate and normalisation statistics.

Accuracy of `gorp_real_inputs_reactive` on GORP with real device input, under RPM's own evaluation (first second skipped; the first 10 frames and the first prediction window seeded from mocap). These are the RPM paper's Table 4 and Table 5 values, reproduced with the released checkpoint:

| Test split | MPJPE | MPJRE | MPJVE | Jitter |
|---|---|---|---|---|
| motion controllers (147 captures) | 6.83 cm | 5.36° | 10.93 cm/s | 0.57 |
| hand tracking (162 captures) | 7.95 cm | 6.59° | 5.64 cm/s | 0.37 |

## 1. Conventions

- **World frame.** SMPL world: **Z up**, metres, floor at z = 0. GORP's raw device poses are Y-up with the floor at y = 0 and are converted as described in [section 4](#4-input-raw-device-poses--model-frames). The app's tracking origin must be floor level.
- **Frame rate.** 30 fps for the GORP and AMASS-P2 versions, 60 fps for `amass_p1_reactive` (manifest `fps`). All velocities are per frame, not per second. Resample the device stream to that rate before processing. Quest runs at 72 to 120 Hz.
- **Quaternions** are `(w, x, y, z)`. **Rotation matrices** act on column vectors and map a joint's or device's local axes to world.
- **6D rotation** = the first two **columns** of the matrix, `[R00, R10, R20, R01, R11, R21]`.
- **Joints** are the 22 SMPL-X body joints, in this order: 0 pelvis, 1 L hip, 2 R hip, 3 spine1, 4 L knee, 5 R knee, 6 spine2, 7 L ankle, 8 R ankle, 9 spine3, 10 L foot, 11 R foot, 12 neck, 13 L collar, 14 R collar, 15 head, 16 L shoulder, 17 R shoulder, 18 L elbow, 19 R elbow, 20 L wrist, 21 R wrist.
- **Parents:** `[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]` (also `smplx_parents` in each manifest). Joint 0's rotation is the global root orientation in the Z-up world. The others are local, relative to the parent.
- **Skeleton.** `joints_rest` (22 × 3) gives the rest-pose joint locations of a neutral SMPL-X body with the subject's shape. The graph needs it for forward kinematics.

## 2. The row types

**Model input frame, 54 floats.** It covers three tracked points: head, left wrist, right wrist.

| Slice | Content |
|---|---|
| `[0:6]` `[6:12]` `[12:18]` | global rotation 6D of head, L, R |
| `[18:24]` `[24:30]` `[30:36]` | rotation velocity 6D `R(t-1)^-1 · R(t)` of head, L, R |
| `[36:39]` `[39:42]` `[42:45]` | world position of head, L, R |
| `[45:48]` `[48:51]` `[51:54]` | position velocity `p(t) - p(t-1)` of head, L, R |

A hand's 18 numbers are its slices `[6:12]`, `[24:30]`, `[39:42]`, `[48:51]` (left) or `[12:18]`, `[30:36]`, `[42:45]`, `[51:54]` (right).

**Model state row, 132 floats.** 22 × 6D joint rotations (joint `j` at `[6j : 6j+6]`), local, root global, **normalised** with the version's dataset statistics ([section 5](#5-the-inference-loop)). The model's state holds rotations only: there is no root translation, and the body is placed by head alignment.

**Output motion row, 198 floats.** This is one body pose, in physical units.

| Slice | Content |
|---|---|
| `[0:132]` | 22 × 6D joint rotations, local, root global, de-normalised |
| `[132:198]` | 22 × 3 world joint positions (joint `j` at `[132+3j : 135+3j]`) |

## 3. The step graph

`models/<name>/rpm_<name>.onnx` runs one step of the rolling loop for the newest frame `i`. L is the version's window (manifest `geometry.prediction_window`); the shapes below are for `gorp_real_inputs_reactive`, L = 5.

| Input | Shape | Content |
|---|---|---|
| `sparse_ctx` | (1, 11, 54) | input frames `i-10 … i`, oldest first, as section 4 builds them (not normalised) |
| `motion_ctx` | (1, 10, 132) | state: the committed rows of frames `i-10 … i-1`, normalised |
| `prev_pred` | (1, L, 132) | state: the rolling prediction window for frames `i … i+L-1`, normalised |
| `joints_rest` | (1, 22, 3) | the user's rest skeleton |

| Output | Shape | Content |
|---|---|---|
| `motion_out` | (1, 198) | the output motion row for frame `i` |
| `joint_rotmat` | (1, 22, 3, 3) | the same pose as rotation matrices (local; root global), for driving the avatar |
| `next_motion_ctx` | (1, 10, 132) | `motion_ctx` for frame `i+1` |
| `next_prev_pred` | (1, L, 132) | `prev_pred` for frame `i+1` |

**What the graph does internally:**
- **No canonicalisation.** The network sees the world positions exactly as fed. It was trained that way, so its output depends on where the user stands ([section 7](#7-things-to-know-before-shipping)).
- **The network and RPM's rolling blend.** The network refines the L-frame window `prev_pred` into a new window (per-slot blend weights `pcaf_weights` in the manifest). Its first slot is frame `i`'s pose, which is committed.
- **De-normalisation** of the committed row, then **6D to rotation** with the same conversion the evaluation used.
- **Forward kinematics** on `joints_rest`.
- **Head alignment**, which translates the body so its head joint sits on the tracked head position of frame `i` (`sparse_ctx[0, -1, 36:39]`). RPM's own evaluation anchors the body on the mocap head instead. On GORP real input the two points are about 1.8 cm apart. MPJPE is unaffected, since it re-anchors at the head.
- **The state update.** `next_motion_ctx` is `motion_ctx` shifted by one with the committed row appended. `next_prev_pred` is the new window shifted by one, with a new last slot of zeros (in normalised units, i.e. the dataset-mean pose), as in RPM's training.

**Feed the state back unchanged.** `next_motion_ctx` and `next_prev_pred` go into the next step exactly as returned. Never de-normalise them, edit them, or rebuild them from `motion_out`. In the GORP and AMASS-P1 statistics the two foot joints (10, 11; state dims `[60:72]`) have standard deviation 0, so a round trip through physical units does not reproduce the state. `motion_out` is output only, and nothing the app does to it reaches the model.

**Take avatar rotations from `joint_rotmat`.** The 6D in `motion_out[0:132]` is not exactly orthonormal (it is the network's raw output). `joint_rotmat` holds the rotations the evaluation's forward kinematics applies, and `motion_out[132:198]` is that forward kinematics.

**No Einsum in the graph.** Unity Inference Engine 2.6 returns `Einsum("bij,btkj->btki")` with its output axes in the wrong order. The exporter refuses to write a graph that contains an Einsum node. Each manifest lists its graph's op types.

## 4. Input: raw device poses → model frames

The GORP versions were trained on GORP as preprocessed by the RPM codebase: `prepare_data_gorp.py` together with `utils/gorp_util.py: transform_controllers_from_gorp_to_smpl`. All three stages below have to be reproduced in the app, in this order. Stages 4.1 to 4.3 are the same as for PMT (the PMT export's input processing). Only the confidence handling (4.4) differs.

The synthetic-input versions (`gorp_*`, `amass_*`) were trained on the same 54-number frames computed from the mocap body instead of the devices, so they never saw a device offset residual or a tracking dropout. At run time they take the device input of this section like the real-input versions (the RPM paper's "simulated → real" setting). The AMASS versions were never trained on GORP.

### 4.1 What the dataset processing does (RPM `prepare_data_gorp.py`, motion-controller captures)

**Raw device stream.** A GORP capture (`GORP/<subject>/<capture>.npz`) stores the device stream in `tracking_data`:
- `hmdTransforms`, `lConTransforms`, `rConTransforms`: (N, 7) float32, each row `[px, py, pz, qw, qx, qy, qz]` in GORP's Y-up frame, metres, floor at y = 0.
- `lConValid`, `rConValid`: (N,) int, 0/1 per frame.

`sparse_raw` in `data/gorp/` holds exactly these fields, stacked per frame as (N, 3, 8): `[pose(7), valid]` for head, left, right, with the head's valid set to 1.

**The steps:**
1. GORP is recorded at 30 fps, and RPM resamples to 30 fps, so no frame is dropped or interpolated.
2. Every raw frame goes through the device conversion in 4.2.
3. Features are formed from consecutive frames (4.3), so raw frame 0 is used only as the "previous" frame and the processed sequence has N-1 frames.
4. The per-frame hand confidence is the valid flag of the *current* raw frame: `left_hand_confidence[t] = lConValid[t+1]`.

**Ground truth.** The mocap side is the capture's SMPL-X `poses` / `trans` / `betas`, which are already in the Z-up world:
- local joint rotations of the first 22 joints → 6D
- world joint positions from a neutral SMPL-X with the subject's 16 betas

Both drop frame 0 to stay aligned with the input. The raw device stream is the only thing converted from Y-up.

### 4.2 Device pose → SMPL head / wrist joint (RPM `transform_controllers_from_gorp_to_smpl`)

**Steps.** Per raw frame, for each device `d` (head, left controller, right controller), with position `p` and quaternion `q = (w, x, y, z)` in GORP's frame:
1. **Position axes:** `p ← (p.x, -p.z, p.y)`.
2. **Quaternion conjugate:** `q ← (w, -x, -y, -z)`.
3. **World fix:** `q ← q_x(90°) ⊗ q`.
4. **Device axis fix**, right-multiplied:
   - head: `q ← q ⊗ q_y(180°)`
   - left: `q ← q ⊗ q_z(90°) ⊗ q_y(90°)`
   - right: `q ← q ⊗ q_z(-90°) ⊗ q_y(-90°)`
5. **Matrix:** `R = matrix(q)`.
6. **Fixed device-to-joint offset:** `p ← p - R · offset_d`. This moves the tracked device point onto the SMPL joint: the head joint for the headset, the wrist joint for each controller.

**Quaternion conventions.** `⊗` is the Hamilton product, so `matrix(a ⊗ b) = matrix(a) · matrix(b)`. The constants, as `(w, x, y, z)`:
- `q_x(90°) = (0.7071068, 0.7071068, 0, 0)`
- `q_y(180°) = (0, 0, 1, 0)`
- `q_z(±90°) = (0.7071068, 0, 0, ±0.7071068)`
- `q_y(±90°) = (0.7071068, 0, ±0.7071068, 0)`

`matrix(q)` is the standard quaternion matrix with `s = 2 / |q|²`: row 0 `(1 - s(y²+z²), s(xy - zw), s(xz + yw))`, row 1 `(s(xy + zw), 1 - s(x²+z²), s(yz - xw))`, row 2 `(s(xz - yw), s(yz + xw), 1 - s(x²+y²))`.

**The fixed offsets**, in metres, in the device frame *after* the axis fix:

| Device | `offset_d` | Length |
|---|---|---|
| head | `(-0.01568759, 0.02414007, 0.09251691)` | 9.7 cm |
| left controller | `(0.12183987, -0.05452635, 0.06948892)` | 15.0 cm |
| right controller | `(-0.11596493, -0.04973321, 0.06753435)` | 14.3 cm |

They are RPM's hand-measured calibration constants, the same for every subject and capture. Because they rotate with the device, a pure head turn leaves the computed head joint in place. On GORP the processed head still sits about 1.8 cm from the mocap head joint, with about 1.5 cm spread between subjects. That residual is part of the input the real-input versions were trained on.

**The same conversion as matrices.** This form is equivalent to the steps above to 1e-15:

```
p_d = Rx · p_gorp − R_d · offset_d
R_d = Rx · R_gorpᵀ · F_d          (R_gorp = matrix of the stored quaternion)

Rx      = [[1,0,0],[0,0,-1],[0,1,0]]      (+90° about x: Y-up → Z-up)
F_head  = [[-1,0,0],[0,1,0],[0,0,-1]]
F_left  = [[0,-1,0],[0,0,1],[-1,0,0]]
F_right = [[0,1,0],[0,0,1],[1,0,0]]
```

The transpose on `R_gorp` comes from step 2's conjugate. It is GORP's stored-rotation convention, and has to be reproduced as is.

**Hand-tracking captures** use RPM's `transform_hand_tracking_from_gorp_to_smpl` instead. Its steps 1–3 are the same for all three points. Steps 4 and 6 apply to the head only: the tracked hands get no axis fix and no offset. Their confidence is continuous rather than 0/1.

### 4.3 Features

Model frame `t` comes from converted raw frames `t+1` (current) and `t` (previous), laid out as in [section 2](#2-the-row-types):
- **rotation:** `6D(R_cur)`, the first two columns of the matrix
- **rotation velocity:** `6D(inverse(R_prev) · R_cur)`
- **position:** `p_cur`
- **position velocity:** `p_cur − p_prev`, per frame

The first raw frame only seeds the velocities, so the app's first model frame appears on its second raw frame. The velocities are always taken between consecutive raw frames, also when the previous frame was untracked (then against whatever pose the device reported on it).

### 4.4 Controller confidence (RPM's rule: threshold 0.8, zeroing)

Per hand, per frame:
- **Hand tracked** (confidence ≥ 0.8; GORP controllers report a 0/1 valid flag): use the features as computed.
- **Hand not tracked:** set all 18 numbers of that hand to 0.

There is no hold of the last tracked pose and no special rule on reappearance: the first tracked frame after a gap uses its features from 4.3 as they are. The head is never masked. (This is where PMT differs: it holds the last tracked hand and resets the velocities on reappearance.)

### 4.5 Worked example (unit test for a port)

`data/gorp/s26_fruitninja-q3-controllers_000.npz`, `sparse_raw` rows 0 and 1, gives model frame 0 (values rounded to 6 digits). Both hands are tracked, so 4.4 leaves the frame unchanged. It is `sparse_ctx[0]` of `models/gorp_real_inputs_reactive/output_ref/s26_fruitninja-q3-controllers_000.npz`.

```
raw row 0   [px py pz | qw qx qy qz | valid]
  head   -0.980979  1.574224  0.981465 | 0.807055  0.051097 -0.587253 -0.034437 | 1
  left   -1.147543  1.409308  1.085072 | 0.681421 -0.349691 -0.625936  0.146921 | 1
  right  -1.056766  1.430890  0.822214 | 0.752012 -0.200173 -0.507583  0.369821 | 1
raw row 1
  head   -0.981341  1.575140  0.981141 | 0.807310  0.048710 -0.587223 -0.032369 | 1
  left   -1.155779  1.376902  1.081186 | 0.703041 -0.304321 -0.630434  0.125197 | 1
  right  -1.071952  1.395979  0.821313 | 0.774523 -0.165899 -0.518704  0.321774 | 1

4.2 on row 1, head:  p = (-0.896106, -1.025318, 1.562036)
                     R = [[-0.308243, -0.109471, -0.944988],
                          [-0.951295,  0.040632,  0.305593],
                          [ 0.004943,  0.993159, -0.116664]]

model frame 0              head                                   left                                    right
  rot 6D   [0:18]   -0.308243 -0.951295  0.004943 -0.109471  0.040632  0.993159 | -0.810242  0.019883  0.585758 -0.173757 -0.962643 -0.207671 |  0.696732 -0.406848 -0.590795  0.254817  0.910259 -0.326338
  rotvel   [18:36]   1.000000  0.000556  0.000299 -0.000554  0.999980 -0.006304 |  0.994256 -0.017781 -0.105538  0.020310  0.999530  0.022942 |  0.993014 -0.015136  0.117018  0.009297  0.998692  0.050286
  pos      [36:45]  -0.896106 -1.025318  1.562036 | -1.105430 -1.117332  1.239770 | -1.023767 -0.828412  1.261406
  posvel   [45:54]  -0.000457  0.000309  0.000327 |  0.004477 -0.003911 -0.026616 |  0.002423  0.000412 -0.029829
```

## 5. The inference loop

Keep the last 11 input frames `S` and the state (`motion_ctx`, `prev_pred`).

**Normalisation.** `normalise(x) = (x − motion_mean) / (motion_std + 1e-8)`, with the 132-float `motion_mean` and `motion_std` from the version's `manifest.json → normalisation`. The app only needs it once, to build the seed. The graph de-normalises its own output.

**Startup.** The first step needs 11 input frames, so it runs on model frame 10, about 0.37 s after the first raw frame at 30 fps. Before that there is no model output.
- **Seed the state with a standing pose `r`** (132 floats): SMPL-X rest pose (all local rotations identity, 6D `(1,0,0,0,1,0)`), stood upright in the Z-up world and turned to the head's heading. The root rotation is `Rz(yaw_head + 90°) · Rx(90°)`, where `yaw_head = atan2(R_head[1,2], R_head[0,2])` is taken from the head rotation of model frame 10. Then `motion_ctx` = 10 copies of `normalise(r)` and `prev_pred` = L copies of `normalise(r)`.
- **Keep the feet exactly at identity.** In the GORP and AMASS-P1 statistics the foot joints 10 and 11 (`r[60:72]`) have standard deviation 0, and their `motion_mean` is exactly the identity 6D. Identity therefore normalises to exactly 0, while any other value is divided by 1e-8. Never write anything but the exact identity there.
- **The evaluation seeds differently:** `motion_ctx` = normalised mocap rows of frames 0 … 9 and `prev_pred` = normalised mocap rows of frames 10 … 10+L-1. That needs mocap of frames ahead of the first step, so it is possible offline only. The reference output in `output_ref/` uses it (section 6).
- **What the standing seed costs** (`gorp_real_inputs_reactive`, measured on the three captures). Over the first 20 frames after the first step, MPJPE is 7–10 cm with the standing seed, against 1.6–4.8 cm with the mocap seed (which has seen frames 10–14). From frame 30 on there is no systematic cost: MPJPE 8.25 / 11.50 / 4.57 cm with the standing seed vs 8.60 / 11.57 / 5.57 cm with the mocap seed (S06 / S11 / S26). The two runs never converge, though: after 10 s their per-frame MPJPE still differs by 0.04–1.5 cm on average. RPM's rollout keeps the imprint of its seed, so a port can only be compared value by value with the identical seed.

**Every frame `i ≥ 10`:**

```
S.append(frame_i)                                    # keep the last 11
motion_out, joint_rotmat, motion_ctx, prev_pred = step(sparse_ctx  = S[-11:],
                                                       motion_ctx  = motion_ctx,
                                                       prev_pred   = prev_pred,
                                                       joints_rest = joints_rest)
render(joint_rotmat, motion_out[132:198])
```

**Driving an SMPL-X avatar:** use `joint_rotmat` as the 22 body joint rotations. The root translation is `trans = pelvis_world - joints_rest[0]`, where `pelvis_world = motion_out[132:135]`.

**The avatar's head orientation comes from the model, not the HMD.** It is not snapped to the headset rotation. If the app needs the avatar to look exactly where the user looks, override the head joint's local rotation with `R_parent_globalᵀ · R_hmd` on the rendered pose. The model state is not affected, since it never reads the rendered pose.

## 6. Checking a port

Each capture from GORP `test_controllers` has one file in `data/gorp/`, the same for every version, and one in each version's `output_ref/`, under the same name. Every array is float32. `T` = 899 model frames (30 s).

`data/gorp/<capture>.npz`:

| Key | Shape | Content |
|---|---|---|
| `sparse_raw` | (T+1, 3, 8) | raw GORP device stream, rows = head, left controller, right controller, each `[px py pz qw qx qy qz valid]` in GORP's Y-up frame (head `valid` = 1). Input frame `t` is built from raw rows `t` and `t+1`. |
| `gt_motion` | (T, 198) | mocap ground truth: local rotation 6D + world joint positions |
| `joints_rest` | (22, 3) | the subject's SMPL-X rest skeleton |

`models/<name>/output_ref/<capture>.npz`:

| Key | Shape | Content |
|---|---|---|
| `sparse_ctx` | (T, 54) | the input frames the reference was run on. Real-input versions: section 4 applied to `sparse_raw`. Synthetic-input versions: the 54-number frames computed from the mocap body, as they were trained (manifest `reference_input`). |
| `ref_pred_motion` | (T, 198) | rows 0 … 9 = `gt_motion` (the evaluation's seed); rows 10 and later = the graph's `motion_out`, run as RPM's evaluation runs it (its rotations are RPM's own rolling generator, bit for bit). |

| Capture | Subject / game | Why it is here |
|---|---|---|
| `s06_fruitninja-q3-controllers_000.npz` | S06, Fruit Ninja | the only MC test capture with controller dropouts (66 left, 2 right frames): exercises the zeroing of 4.4 |
| `s11_beatsaber-q3-controllers_900.npz` | S11, Beat Saber | recorded about 2.4 m from where the training captures stand |
| `s26_fruitninja-q3-controllers_000.npz` | S26, Fruit Ninja | an ordinary capture |

**The check, for a version `<name>`:**
- **Input processing** (real-input versions). Feed `sparse_raw` through your section 4 code and compare with `sparse_ctx`. Both should agree to float32 rounding (about 4e-7).
- **The loop.** Seed as the evaluation does: `motion_ctx = normalise(gt_motion[0:10, 0:132])` and `prev_pred = normalise(gt_motion[10:10+L, 0:132])`, with the version's statistics. Run the steps for frames 10 … T-1 on `sparse_ctx` and compare `motion_out` with `ref_pred_motion[10:]`. On CPU this export's onnxruntime replay agrees to the manifest's `checks.onnxruntime_replay_vs_rollout_rot6d_max_abs` (1.5e-6 for `gorp_real_inputs_reactive`, at most 2.2e-5 for any version). A different backend (GPU, fp16) can drift further over the 889 autoregressive steps. In that case compare the MPJPE: re-anchor the predicted and the mocap joints at their head joint and average the joint error over frames 30 and later. For the GORP versions each manifest lists RPM's evaluator's values under `sequences[].rollout_mpjpe_cm` (`gorp_real_inputs_reactive`: 8.594 / 11.563 / 5.571 cm). Computed this way from `gt_motion`, the same rollout scores 0.004–0.011 cm higher (8.599 / 11.570 / 5.575 cm): RPM's evaluator rebuilds the mocap joints with its own body model.

The AMASS versions' references run the GORP captures at 30 fps through graphs trained on AMASS (`amass_p1_reactive` at 60 fps): a numeric reference for a port only, not an accuracy figure.

**What was checked at export** (each manifest's `checks` and `sequences`; 889 steps per capture):
- The exported network replayed in PyTorch is bit-identical to RPM's own rolling generator.
- The onnxruntime replay stays within 1.6e-6 of it in the 6D for `gorp_real_inputs_reactive` (at most 2.2e-5 over all versions).
- The 6D-to-rotation conversion matches the evaluation's to 6e-7. The forward kinematics matches the SMPL-X body model to 8.4e-7 m.
- For the GORP versions, the onnxruntime replay's per-capture MPJPE equals RPM's evaluation of the released checkpoint to 1e-4 cm.

## 7. Things to know before shipping

- **Accuracy depends on where the user stands.** The released networks were trained on absolute world positions, with no re-centring, and the graph feeds them the positions as they are. For `gorp_real_inputs_reactive`, the same capture moved horizontally by 0.5 to 2 m changes its MPJPE by −3.8 to +2.5 cm (S06, S11, S26; seven shifts each, tracked points only). It can get better or worse, depending on how the new spot relates to where GORP's captures stand. PMT's graph re-centres horizontally on the head, which makes it independent of the standing spot; that cannot be added here without retraining. The floor must be at z = 0 in both.
- **Compute.** `gorp_real_inputs_reactive` on onnxruntime 1.23.2, CPU (Intel Core i9-13900K, Windows 11), the session call only, over 2607 steps after warm-up: **1.3 ms** median and 1.9 ms p99 with onnxruntime's default threads; 3.9 ms median and 5.0 ms p99 on one thread. All versions have the same network size. GPU backends were not measured. Contention with the app's own VR rendering, Unity Sentis, and standalone headsets have not been measured either. The graph's size (37 MiB, 9.6 M weights) is small enough that a standalone headset is not ruled out.
- **Device conventions.** The conversion in section 4 assumes the app's poses mean the same thing as GORP's: the same headset point, the same controller anchor, and the same axes, from Quest 3 captures. That is not verified for any particular SDK. Quick test: record a few seconds, run it through section 4 and the loop. With the right conventions the wrists land on the controllers and the body faces where the headset faces. Wrong axes show up at once as a backwards body or swapped, mirrored arms.
- **Device offsets.** The device-to-joint offsets are GORP's single calibration for all users. On GORP the processed head still sits about 1.8 cm from the mocap head joint, with about 1.5 cm spread between subjects.
- **Hand tracking.** The GORP versions were trained on controller and hand-tracking captures; `gorp_real_inputs_reactive`'s hand-tracking accuracy is in the table at the top. Hand-tracking input uses a different raw conversion (section 4.2). This export's references use controller captures only.
- **Body shape.** `joints_rest` should match the user. Without a body scan, a neutral SMPL-X with zero shape, or a shape fitted to the user's height, is the practical choice. The accuracy cost of a mismatched skeleton has not been measured.

## Provenance

- **Checkpoints:** `checkpoints/<folder>/<variant>/model_latest.pt` + `args.json` from the authors' release, [`rpm_checkpoints.zip`](https://github.com/facebookresearch/motion_rolling_prediction/releases/download/v0/rpm_checkpoints.zip) (facebookresearch/motion_rolling_prediction, release `v0`). Each manifest's `checkpoint` names its file.
- **Exporter:** `export_onnx.py` in the RPM checkout (each manifest's `git_commit`; `-dirty` = uncommitted changes at export time) with torch 2.5.1, onnx 1.23.1 and onnxruntime 1.23.2. This README's source is `docs/RPM-README.md` there.
- **Normalisation statistics:** the authors' per-dataset `<dataset>_mean.pt` / `<dataset>_std.pt`, shipped with the release. They are baked into each graph and copied to its manifest.
