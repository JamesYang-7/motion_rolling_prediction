#!/bin/bash
# Fresh P1 train -> P1 eval -> P2 train -> P2 eval, sequential.
# Trains both reactive models from scratch on the stride-corrected data
# (gt_jitter targets P1=403.43, P2=297.50 already verified).
# Phase markers + tail-friendly output go to ./results/orchestration.log.
set -u

ORCH_LOG="./results/orchestration.log"
mkdir -p ./results ./results/amass_p1 ./results/amass_p2

phase() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$ORCH_LOG"
}

phase "ORCH_START"

# ---- P1 train ----
phase "STARTING_P1_TRAIN"
TQDM_MININTERVAL=30 PYTHONUNBUFFERED=1 stdbuf -oL -eL python train.py \
    --results_dir ./results/amass_p1 \
    --dataset amass_p1 \
    --train_dataset_repeat_times 100 \
    --batch_size 128 \
    --input_motion_length 10 \
    --exp_name reactive \
    --rolling_fr_frames 60 \
    --rolling_motion_ctx 10 \
    --rolling_sparse_ctx 10 \
    --loss_velocity 1 \
    --loss_fk 1 \
    --loss_fk_vel 1 \
    > ./results/amass_p1/train.log 2>&1
P1_TRAIN_RC=$?
phase "P1_TRAIN_FINISHED exit=$P1_TRAIN_RC"
[ $P1_TRAIN_RC -ne 0 ] && { phase "P1_TRAIN_ERROR"; exit 1; }

# ---- P1 eval ----
phase "STARTING_P1_EVAL"
TQDM_MININTERVAL=30 PYTHONUNBUFFERED=1 stdbuf -oL -eL python test.py \
    --model_path ./results/amass_p1/checkpoints/reactive/model_latest.pt \
    --eval --eval_batch_size 4 \
    > ./results/amass_p1/eval.log 2>&1
P1_EVAL_RC=$?
phase "P1_EVAL_FINISHED exit=$P1_EVAL_RC"
[ $P1_EVAL_RC -ne 0 ] && { phase "P1_EVAL_ERROR"; exit 1; }

# ---- P2 train ----
phase "STARTING_P2_TRAIN"
TQDM_MININTERVAL=30 PYTHONUNBUFFERED=1 stdbuf -oL -eL python train.py \
    --results_dir ./results/amass_p2 \
    --dataset amass_p2 \
    --train_dataset_repeat_times 100 \
    --batch_size 128 \
    --input_motion_length 10 \
    --exp_name reactive \
    --rolling_fr_frames 60 \
    --rolling_motion_ctx 10 \
    --rolling_sparse_ctx 10 \
    --loss_velocity 1 \
    --loss_fk 1 \
    --loss_fk_vel 1 \
    > ./results/amass_p2/train.log 2>&1
P2_TRAIN_RC=$?
phase "P2_TRAIN_FINISHED exit=$P2_TRAIN_RC"
[ $P2_TRAIN_RC -ne 0 ] && { phase "P2_TRAIN_ERROR"; exit 1; }

# ---- P2 eval ----
phase "STARTING_P2_EVAL"
TQDM_MININTERVAL=30 PYTHONUNBUFFERED=1 stdbuf -oL -eL python test.py \
    --model_path ./results/amass_p2/checkpoints/reactive/model_latest.pt \
    --eval --eval_batch_size 4 \
    > ./results/amass_p2/eval.log 2>&1
P2_EVAL_RC=$?
phase "P2_EVAL_FINISHED exit=$P2_EVAL_RC"
[ $P2_EVAL_RC -ne 0 ] && { phase "P2_EVAL_ERROR"; exit 1; }

phase "ALL_DONE"
