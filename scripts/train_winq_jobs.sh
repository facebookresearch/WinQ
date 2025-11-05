#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUTDIR=${OUTDIR:-"winq_jobs_${RUN_TAG}"}
mkdir -p "${OUTDIR}"

declare -a DEFAULT_W_BITS=(0 1 2)
declare -a DEFAULT_A_BITS=(8 16)
W_BITS=(${W_BITS_OVERRIDE:-${DEFAULT_W_BITS[@]}})
A_BITS=(${A_BITS_OVERRIDE:-${DEFAULT_A_BITS[@]}})
KV_BITS=${KV_BITS:-32}

BASE_OUT_PREFIX=${BASE_OUT_PREFIX:-"xx"}
BASE_SYN_PREFIX=${BASE_SYN_PREFIX:-"${BASE_OUT_PREFIX}"}
BASE_OUT_PREFIX=${BASE_OUT_PREFIX%/}
BASE_SYN_PREFIX=${BASE_SYN_PREFIX%/}

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
    reinit_alpha="0.2"

    if (( a == 8 )); then
      lr="4e-5"
    elif (( w >= 3 )); then
      max_steps=80000
      lr="1e-5"
      reinit_steps=40000
    fi

    eval_steps=${max_steps}

    max_k=$(( max_steps / 1000 ))
    case "${lr}" in
      1e-5) lr_label="1e5" ;;
      2e-5) lr_label="2e5" ;;
      4e-5) lr_label="4e5" ;;
      *) lr_label=${lr//./p} ;;
    esac

    tail="winq_lr_${lr_label}_reinit_${reinit_steps}_alpha_${reinit_alpha}_max_${max_k}k"

    out_path="${BASE_OUT_PREFIX}/${bits_folder}/${tail}/run_${RUN_TAG}"
    synth_path="${BASE_SYN_PREFIX}/${bits_folder}/${tail}/run_${RUN_TAG}"

    default_local_dir="/tmp/llama/winq_w${w}_a${a}_${RUN_TAG}"
    script="${OUTDIR}/run_winq_w${w}_a${a}.sh"

    cat > "${script}" <<'EOF'
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
OUT_PATH="__OUT_PATH__"
SYNTH_PATH="__SYNTH_PATH__"
RUN_TAG="__RUN_TAG__"

mkdir -p "${LOCAL_DIR}" >/dev/null 2>&1

APP_ARGS=(
  --local_dir "${LOCAL_DIR}"
  --backend "xl"
  --input_model_filename "xx"
  --checkpoint_path "xx"
  --output_model_filename "${OUT_PATH}"
  --synthesis_path "${SYNTH_PATH}"
  --resume "${RESUME_FLAG:-True}"
  --data_path "xx"
  --eval_data_path "xx"
  --use_mdl True
  --namespace "oculus"
  --hive_table "finewebedu"
  --hive_json_data_column "text"
  --single_file True
  --pretrain False
  --max_parallel_files 2
  --do_train True
  --do_eval True
  --model_max_length 2048
  --fp16 False
  --bf16 True
  --log_on_each_node False
  --logging_dir "/tmp/output/runs/current"
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 8
  --gradient_accumulation_steps 1
  --save_steps 5000
  --eval_steps "__EVAL_STEPS__"
  --logging_steps 10
  --eval_strategy "steps"
  --save_strategy "steps"
  --report_to "tensorboard"
  --save_total_limit 1
  --learning_rate "__LEARNING_RATE__"
  --weight_decay 0.
  --lr_scheduler_type "cosine"
  --gradient_checkpointing False
  --max_steps "__MAX_STEPS__"
  --warmup_step 1000
  --zeroshot_tasks "arc_easy,arc_challenge,boolq,piqa,siqa,hellaswag,obqa,winogrande_1.1,wiki2"
  --wrap_layer_xl True
  --qat "experimental"
  --add_bos_token False
  --add_eos_token False
  --w_bits "__W_BITS__"
  --a_bits "__A_BITS__"
  --kv_bits "__KV_BITS__"
  --share_embedding True
  --train_nso True
  --nso_within_model True
  --nso_pre_quantization_noise True
  --nso_post_quantization_noise False
  --nso_sigma_weights 0.001
  --nso_sigma_clipvals 0.001
  --nso_num_perturbs 1
  --reinitialize_weights True
  --reinitialize_steps "__REINIT_STEPS__"
  --reinitialize_alpha "__REINIT_ALPHA__"
)

RUN_COMPONENT="${TORCHX_COMPONENT:-fb.dist.hpc}"
RUN_ARGS=(--scheduler mast "${RUN_COMPONENT}" -- "${APP_ARGS[@]}")

echo "Using TORCHXCONFIG: ${CONFIG_PATH}"
echo "Resolved local_dir: ${LOCAL_DIR}"
echo "Remote output path: ${OUT_PATH}"
echo "Launching with torchx component: ${RUN_COMPONENT}"
TORCHXCONFIG="${CONFIG_PATH}" torchx run "${RUN_ARGS[@]}"
EOF

    sed -i "s|__DEFAULT_CONFIG__|${TORCHXCONFIG_PATH}|g" "${script}"
    sed -i "s|__DEFAULT_LOCAL_DIR__|${default_local_dir}|g" "${script}"
    sed -i "s|__OUT_PATH__|${out_path}|g" "${script}"
    sed -i "s|__SYNTH_PATH__|${synth_path}|g" "${script}"
    sed -i "s|__RUN_TAG__|${RUN_TAG}|g" "${script}"
    sed -i "s|__EVAL_STEPS__|${eval_steps}|g" "${script}"
    sed -i "s|__LEARNING_RATE__|${lr}|g" "${script}"
    sed -i "s|__MAX_STEPS__|${max_steps}|g" "${script}"
    sed -i "s|__W_BITS__|${w}|g" "${script}"
    sed -i "s|__A_BITS__|${a}|g" "${script}"
    sed -i "s|__KV_BITS__|${KV_BITS}|g" "${script}"
    sed -i "s|__REINIT_STEPS__|${reinit_steps}|g" "${script}"
    sed -i "s|__REINIT_ALPHA__|${reinit_alpha}|g" "${script}"

    chmod +x "${script}"
    echo "Generated: ${script}"
  done
done

echo "All scripts written to: ${OUTDIR}/"

if [[ "${SKIP_AUTO_RUN:-0}" != "1" ]]; then
  echo "# (Optional) Run all generated scripts one after another"
  for f in "${OUTDIR}"/*.sh; do
    echo "Running ${f}..."
    bash "${f}"
  done
else
  echo "SKIP_AUTO_RUN is set; not executing generated scripts."
fi
