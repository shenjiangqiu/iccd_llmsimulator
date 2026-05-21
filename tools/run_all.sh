#!/bin/bash
set -e
BIN="./build_release/run"
LOG_DIR="./log"

mkdir -p "$LOG_DIR"

echo "=== Running all experiments ==="
for cfg in config_exp1_fp16_gpu config_exp2_fp16_pim config_exp3_2bit_gpu \
           config_exp4_2bit_hybrid config_exp5_2bit_allpim config_exp6_2bit_dequant_pim; do
    echo "  $cfg ..."
    $BIN "${cfg}.yaml" 2>&1 | python3 tools/parse_layer.py -o "$LOG_DIR/${cfg}.json"
done

echo ""
echo "=== Generating report ==="
python3 tools/report.py "$LOG_DIR"/config_exp*.json
echo "Saved to docs/experiment_comparison.md"
