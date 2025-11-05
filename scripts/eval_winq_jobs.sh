#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if [[ -z "${RUN_TAG:-}" ]]; then
  echo "RUN_TAG must be set to match the training run (e.g. export RUN_TAG=20240101_123000)." >&2
  exit 1
fi

RUN_TAG=${RUN_TAG}
OUTDIR=${OUTDIR:-"eval_scripts_winq_${RUN_TAG}"}
mkdir -p "${OUTDIR}"

declare -a DEFAULT_W_BITS=(0 1 2)
declare -a DEFAULT_A_BITS=(8 16)
W_BITS=(${W_BITS_OVERRIDE:-${DEFAULT_W_BITS[@]}})
A_BITS=(${A_BITS_OVERRIDE:-${DEFAULT_A_BITS[@]}})
KV_BITS=${KV_BITS:-32}

BASE_TRAIN_PREFIX=${BASE_TRAIN_PREFIX:-"xx"}
BASE_EVAL_PREFIX=${BASE_EVAL_PREFIX:-"xx"}
BASE_TRAIN_PREFIX=${BASE_TRAIN_PREFIX%/}
BASE_EVAL_PREFIX=${BASE_EVAL_PREFIX%/}

DEFAULT_TORCHXCONFIG=${DEFAULT_TORCHXCONFIG:-'xx'}
TORCHXCONFIG_PATH=${TORCHXCONFIG:-${DEFAULT_TORCHXCONFIG}}
if [[ "${TORCHXCONFIG_PATH}" == ~* ]]; then
  TORCHXCONFIG_PATH="${HOME}${TORCHXCONFIG_PATH:1}"
fi

if [[ ! -f "${TORCHXCONFIG_PATH}" ]]; then
  echo "Expected TORCHXCONFIG at ${TORCHXCONFIG_PATH}, but it does not exist." >&2
  echo "Set TORCHXCONFIG to a valid config file and retry." >&2
  exit 1
fi

for w in "${W_BITS[@]}"; do
  for a in "${A_BITS[@]}"; do
    bits_folder="${w}_${a}_${KV_BITS}"

    max_steps=240000
    lr="2e-5"
    reinit_steps=60000

    if (( a == 8 )); then
      lr="4e-5"
    elif (( w >= 3 )); then
      max_steps=80000
      lr="1e-5"
      reinit_steps=40000
    fi

    max_k=$(( max_steps / 1000 ))
    case "${lr}" in
      1e-5) lr_label="1e5" ;;
      2e-5) lr_label="2e5" ;;
      4e-5) lr_label="4e5" ;;
      *) lr_label=${lr//./p} ;;
    esac

    tail="winq_lr_${lr_label}_reinit_${reinit_steps}_alpha_0.2_max_${max_k}k"

    checkpoint_path="${BASE_TRAIN_PREFIX}/${bits_folder}/${tail}/run_${RUN_TAG}/${max_steps}/"
    if [[ -n ${CHECKPOINT_PATH_OVERRIDE:-} ]]; then
      checkpoint_path="${CHECKPOINT_PATH_OVERRIDE}"
    fi

    if [[ ${VERIFY_CHECKPOINT:-1} -eq 1 ]]; then
      if [[ "${checkpoint_path}" == manifold://* ]]; then
        :
      elif [[ ! -d "${checkpoint_path}" ]]; then
        echo "Warning: checkpoint path ${checkpoint_path} does not exist locally." >&2
      fi
    fi

    eval_out_path="${BASE_EVAL_PREFIX}/${bits_folder}/${tail}/run_${RUN_TAG}"
    eval_syn_path="${eval_out_path}"

    default_local_dir="/tmp/llama/winq_eval_w${w}_a${a}_${RUN_TAG}"
    script="${OUTDIR}/eval_qat_w${w}_a${a}.sh"

    cat > "${script}" <<'SCRIPT_EOF'
#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${TORCHXCONFIG:-__DEFAULT_CONFIG__}"
if [[ "${CONFIG_PATH}" == ~* ]]; then
  CONFIG_PATH="${HOME}${CONFIG_PATH:1}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Expected TORCHXCONFIG at ${CONFIG_PATH}, but it does not exist." >&2
  echo "Set TORCHXCONFIG to a valid config file and retry." >&2
  exit 1
fi

LOCAL_DIR="${LOCAL_DIR:-__DEFAULT_LOCAL_DIR__}"
LOCAL_DIR="${LOCAL_DIR%/}"
CHECKPOINT_PATH="__CHECKPOINT_PATH__"
EVAL_OUT_PATH="__EVAL_OUT_PATH__"
EVAL_SYN_PATH="__EVAL_SYN_PATH__"
LEARNING_RATE="__LEARNING_RATE__"
KV_BITS="__KV_BITS__"
W_BITS="__W_BITS__"
A_BITS="__A_BITS__"
REINIT_STEPS="__REINIT_STEPS__"

mkdir -p "${LOCAL_DIR}" >/dev/null 2>&1

APP_ARGS=(
  --local_dir "${LOCAL_DIR}"
  --backend "xl"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --input_model_filename "xx"
  --output_model_filename "${EVAL_OUT_PATH}"
  --synthesis_path "${EVAL_SYN_PATH}"
  --eval_data_path "xx"
  --do_train False
  --do_eval True
  --model_max_length 2048
  --fp16 False
  --bf16 True
  --log_on_each_node False
  --logging_dir /tmp/output/runs/current
  --num_train_epochs 1
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 8
  --gradient_accumulation_steps 1
  --eval_strategy "no"
  --save_strategy "steps"
  --save_steps 2000
  --report_to "tensorboard"
  --save_total_limit 1
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0.
  --warmup_ratio 0.
  --lr_scheduler_type "cosine"
  --logging_steps 1
  --gradient_checkpointing False
  --zeroshot_tasks "arc_easy,arc_challenge,boolq,piqa,siqa,hellaswag,obqa,winogrande_1.1,wiki2"
  --wrap_layer_xl True
  --qat "experimental"
  --w_bits "${W_BITS}"
  --a_bits "${A_BITS}"
  --kv_bits "${KV_BITS}"
  --share_embedding True
  --add_bos_token False
  --add_eos_token False
  --llama_version "3.2"
  --train_nso True
  --nso_within_model True
  --nso_pre_quantization_noise True
  --nso_post_quantization_noise False
  --nso_sigma_weights 0.001
  --nso_sigma_clipvals 0.001
  --nso_num_perturbs 1
  --reinitialize_weights True
  --reinitialize_steps "${REINIT_STEPS}"
  --reinitialize_alpha 0.2
)

RUN_COMPONENT="${TORCHX_COMPONENT:-fb.dist.hpc}"
RUN_ARGS=(--scheduler mast "${RUN_COMPONENT}" -- "${APP_ARGS[@]}")

echo "Using TORCHXCONFIG: ${CONFIG_PATH}"
echo "Resolved local_dir: ${LOCAL_DIR}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Evaluation output path: ${EVAL_OUT_PATH}"
echo "Launching with torchx component: ${RUN_COMPONENT}"
TORCHXCONFIG="${CONFIG_PATH}" torchx run "${RUN_ARGS[@]}"
SCRIPT_EOF

    sed -i "s|__DEFAULT_CONFIG__|${TORCHXCONFIG_PATH}|g" "${script}"
    sed -i "s|__DEFAULT_LOCAL_DIR__|${default_local_dir}|g" "${script}"
    sed -i "s|__CHECKPOINT_PATH__|${checkpoint_path}|g" "${script}"
    sed -i "s|__EVAL_OUT_PATH__|${eval_out_path}|g" "${script}"
    sed -i "s|__EVAL_SYN_PATH__|${eval_syn_path}|g" "${script}"
    sed -i "s|__LEARNING_RATE__|${lr}|g" "${script}"
    sed -i "s|__KV_BITS__|${KV_BITS}|g" "${script}"
    sed -i "s|__W_BITS__|${w}|g" "${script}"
    sed -i "s|__A_BITS__|${a}|g" "${script}"
    sed -i "s|__REINIT_STEPS__|${reinit_steps}|g" "${script}"

    chmod +x "${script}"
    echo "Generated: ${script}"
  done
done

echo "All evaluation scripts written to: ${OUTDIR}/"

echo "To run all evaluations, execute:"
echo "  for f in ${OUTDIR}/*.sh; do bash \"${f}\"; done"

if [[ "${SKIP_AUTO_RUN:-0}" != "1" ]]; then
  for f in "${OUTDIR}"/*.sh; do
    echo "Running ${f}..."
    bash "${f}"
  done
else
  echo "SKIP_AUTO_RUN is set; not executing generated scripts."
fi
