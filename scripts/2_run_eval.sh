#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Local helper for evaluating a single WinQ checkpoint.
#
# Step 1: Ensure CHECKPOINT_PATH, TRAIN_DATA_PATH, and EVAL_DATA_PATH
#         below are set to existing local files/directories.
# Step 2: Run `bash 2_run_eval.sh <w_bits>` matching the trained model.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <w_bits>" >&2
  exit 1
fi

W_BITS="$1"
if ! [[ "${W_BITS}" =~ ^[0-9]+$ ]]; then
  echo "w_bits must be an integer." >&2
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_ENTRY="${REPO_ROOT}/train.py"

OUTPUT_ROOT=${OUTPUT_ROOT:-"/tmp/llama"}
MODEL_NAME=${MODEL_NAME:-"winq_w${W_BITS}"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${OUTPUT_ROOT%/}/winq_w${W_BITS}/models/${MODEL_NAME}"}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/tmp/train.jsonl"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"/tmp/eval.jsonl"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"${OUTPUT_ROOT%/}/winq_w${W_BITS}_eval"}
A_BITS=${A_BITS:-16}
KV_BITS=${KV_BITS:-32}

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint directory not found at ${CHECKPOINT_PATH}." >&2
  exit 1
fi

if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
  echo "Training data not found at ${TRAIN_DATA_PATH}." >&2
  exit 1
fi

if [[ ! -f "${EVAL_DATA_PATH}" ]]; then
  echo "Evaluation data not found at ${EVAL_DATA_PATH}." >&2
  exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

torchrun --nnodes=1 --nproc_per_node=1 "${TRAIN_ENTRY}" \
  --local_dir "${EVAL_OUTPUT_DIR}" \
  --input_model_filename "${CHECKPOINT_PATH}" \
  --output_model_filename "${MODEL_NAME}-eval" \
  --train_data_local_path "${TRAIN_DATA_PATH}" \
  --eval_data_local_path "${EVAL_DATA_PATH}" \
  --do_train False \
  --do_eval True \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --logging_dir "${EVAL_OUTPUT_DIR}/logs" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --report_to "tensorboard" \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 False \
  --gradient_checkpointing False \
  --qat True \
  --share_embedding True \
  --contain_weight_clip_val True \
  --w_bits "${W_BITS}" \
  --a_bits "${A_BITS}" \
  --kv_bits "${KV_BITS}"
