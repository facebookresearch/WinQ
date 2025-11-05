#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Local helper for running a single WinQ fine-tuning job.
#
# Step 1: Update MODEL_ID and TRAIN_DATA_PATH below to point to your
#         full-precision checkpoint and local training dataset.
# Step 2: Run `bash 1_run_train.sh <w_bits>` (e.g. `bash 1_run_train.sh 2`).

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

MODEL_ID=${MODEL_ID:-"meta-llama/Llama-3.2-1B"}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/tmp/train.jsonl"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/tmp/llama"}
MODEL_NAME=${MODEL_NAME:-"winq_w${W_BITS}"}
A_BITS=${A_BITS:-16}
KV_BITS=${KV_BITS:-32}

LOCAL_DIR="${OUTPUT_ROOT%/}/winq_w${W_BITS}"
mkdir -p "${LOCAL_DIR}"

if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
  echo "Training data not found at ${TRAIN_DATA_PATH}." >&2
  exit 1
fi

torchrun --nnodes=1 --nproc_per_node=1 "${TRAIN_ENTRY}" \
  --local_dir "${LOCAL_DIR}" \
  --input_model_filename "${MODEL_ID}" \
  --output_model_filename "${MODEL_NAME}" \
  --train_data_local_path "${TRAIN_DATA_PATH}" \
  --do_train True \
  --do_eval False \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --logging_dir "${LOCAL_DIR}/logs" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
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
  --train_nso True \
  --nso_within_model True \
  --nso_pre_quantization_noise True \
  --nso_sigma_weights 0.001 \
  --nso_sigma_clipvals 0.001 \
  --reinitialize_weights True \
  --reinitialize_steps 200 \
  --reinitialize_alpha 0.2 \
  --share_embedding True \
  --w_bits "${W_BITS}" \
  --a_bits "${A_BITS}" \
  --kv_bits "${KV_BITS}"
