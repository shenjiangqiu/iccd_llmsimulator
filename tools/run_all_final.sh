#!/bin/bash
# Full experiment suite: all models × all configs × all PE variants + 4-bit
set -e

BIN="./build_release/run"
LOG_DIR="./log"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODELS=("llama3_8B" "glm4_9B" "opt_6_7B" "qwen3_8B")
MODEL_CTX=(16400 16400 2047 16400)

BASE_CONFIGS=(
  "config_exp1_fp16_gpu"
  "config_exp2_fp16_pim"
  "config_exp3_2bit_gpu"
  "config_exp4_2bit_hybrid"
  "config_exp5_2bit_allpim"
  "config_exp6_2bit_dequant_pim"
)

PE_CONFIGS=(8 16)
PE_INT2=(16 32)

cleanup() { echo "Interrupted"; exit 1; }
trap cleanup INT

cd "$DIR"
mkdir -p "$LOG_DIR"

total=0
total_runs=$(( ${#MODELS[@]} * (${#BASE_CONFIGS[@]} + 16 + 6) ))
echo "=== Total estimated runs: $total_runs ==="

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  ctx="${MODEL_CTX[$i]}"
  echo ""
  echo "========== Model: $model (ctx=$ctx) =========="

  # Base configs (exp1-exp6)
  for cfg in "${BASE_CONFIGS[@]}"; do
    exp="${cfg#config_}"
    echo -n "  $exp ... "
    sed "s/model_name: llama3_8B/model_name: $model/" "${cfg}.yaml" | \
      sed "s/input_len: [0-9]*/input_len: $ctx/" > "/tmp/${model}_${exp}.yaml"
    $BIN "/tmp/${model}_${exp}.yaml" 2>&1 | \
      grep -E "AttentionGen |LLM |decoder_0" > "$LOG_DIR/${model}_${exp}.txt"
    echo '{"total_time_us": 0}' > "$LOG_DIR/${model}_${exp}.json"
    ((total++))
    echo "OK ($total/$total_runs)"
  done

  # PE variants: exp4 (Hybrid) and exp5 (All-PIM)
  for f16 in "${PE_CONFIGS[@]}"; do
    for i2 in "${PE_INT2[@]}"; do
      # Hybrid
      cfg="config_exp4_2bit_hybrid_pe_f16_${f16}_i2_${i2}"
      if [ -f "${cfg}.yaml" ]; then
        echo -n "  hybrid f16=${f16}B i2=${i2}B ... "
        sed "s/model_name: llama3_8B/model_name: $model/" "${cfg}.yaml" | \
          sed "s/input_len: [0-9]*/input_len: $ctx/" > "/tmp/${model}_${cfg}.yaml"
        $BIN "/tmp/${model}_${cfg}.yaml" 2>&1 | \
          grep -E "AttentionGen " > "$LOG_DIR/${model}_exp4_pe_f16_${f16}_i2_${i2}.txt"
        echo '{"total_time_us": 0}' > "$LOG_DIR/${model}_exp4_pe_f16_${f16}_i2_${i2}.json"
        ((total++))
        echo "OK ($total/$total_runs)"
      fi

      # All-PIM
      cfg="config_exp5_2bit_allpim_pe_f16_${f16}_i2_${i2}"
      if [ -f "${cfg}.yaml" ]; then
        echo -n "  allpim f16=${f16}B i2=${i2}B ... "
        sed "s/model_name: llama3_8B/model_name: $model/" "${cfg}.yaml" | \
          sed "s/input_len: [0-9]*/input_len: $ctx/" > "/tmp/${model}_${cfg}.yaml"
        $BIN "/tmp/${model}_${cfg}.yaml" 2>&1 | \
          grep -E "AttentionGen " > "$LOG_DIR/${model}_exp5_pe_f16_${f16}_i2_${i2}.txt"
        echo '{"total_time_us": 0}' > "$LOG_DIR/${model}_exp5_pe_f16_${f16}_i2_${i2}.json"
        ((total++))
        echo "OK ($total/$total_runs)"
      fi
    done
  done

  # 4-bit experiments
  for bits4 in exp3_4bit_gpu exp4_4bit_hybrid exp5_4bit_allpim; do
    cfg="config_${bits4}"
    if [ -f "${cfg}.yaml" ]; then
      expl="${bits4}"
      echo -n "  ${bits4} ... "
      sed "s/model_name: llama3_8B/model_name: $model/" "${cfg}.yaml" | \
        sed "s/input_len: [0-9]*/input_len: $ctx/" > "/tmp/${model}_${cfg}.yaml"
      $BIN "/tmp/${model}_${cfg}.yaml" 2>&1 | \
        grep -E "AttentionGen " > "$LOG_DIR/${model}_${bits4}.txt"
      echo '{"total_time_us": 0}' > "$LOG_DIR/${model}_${bits4}.json"
      ((total++))
      echo "OK ($total/$total_runs)"
    fi
  done
done

echo ""
echo "========== All experiments complete ($total_runs runs) =========="
echo ""
echo "=== Generating report ==="
python3 tools/report_all.py "$LOG_DIR" --md
