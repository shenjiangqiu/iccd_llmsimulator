#!/bin/bash
# Run all experiments and generate comparison report
# Usage: bash tools/run_all.sh

set -e
BIN="./build_release/run"
LOG_DIR="./log"
TOOLS="./tools"

mkdir -p "$LOG_DIR"

echo "=== Running all experiments ==="
for cfg in config_exp1_fp16_gpu config_exp2_fp16_pim config_exp3_2bit_gpu \
           config_exp4_2bit_hybrid config_exp5_2bit_allpim config_exp6_2bit_dequant_pim; do
    echo "  $cfg ..."
    $BIN "${cfg}.yaml" 2>&1 | python3 $TOOLS/parse_layer.py -o "$LOG_DIR/${cfg}.json"
done

echo ""
echo "=== Generating report ==="
python3 -c "
import json, os, glob, re, math

files = sorted(glob.glob('$LOG_DIR/config_exp*.json'))

def find_node(tree, name):
    if isinstance(tree, dict):
        if tree.get('name') == name: return tree
        for c in tree.get('children', []):
            r = find_node(c, name)
            if r: return r
    return None

rows = []
for f in files:
    exp = os.path.basename(f).replace('config_','').replace('.json','')
    d = json.load(open(f))
    ops = d.get('operations', {})
    gen = find_node(ops, 'AttentionGen')
    gen_dur = gen['duration_us'] if gen else 0
    gen_proc = gen.get('processor', '?') if gen else '?'
     txt = f.replace('.json', '.txt')
     qk = softmax = score_v = kv_quant = pim_rb = pim_pe = num_seq = 0.0
     qk_rb = qk_pe = sv_rb = sv_pe = 0.0
     if os.path.exists(txt):
        with open(txt) as tf:
            for line in tf:
                if 'AttentionGen' in line and 'qk=' in line:
                    m = re.search(r'tensor\((\d+)', line)
                    if m: num_seq = int(m.group(1))
                    m2 = re.search(r'qk=([\d.]+)us softmax=([\d.]+)us score_v=([\d.]+)us kv_quant=([\d.]+)us', line)
                    if m2:
                        qk = float(m2.group(1)); softmax = float(m2.group(2))
                        score_v = float(m2.group(3)); kv_quant = float(m2.group(4))
                    m3 = re.search(r'pim_rb=([\d.]+)us pim_pe=([\d.]+)us', line)
                    if m3:
                        pim_rb = float(m3.group(1)); pim_pe = float(m3.group(2))
                    m4 = re.search(r'qk_rb=([\d.]+)us qk_pe=([\d.]+)us sv_rb=([\d.]+)us sv_pe=([\d.]+)us', line)
                    if m4:
                        qk_rb = float(m4.group(1)); qk_pe = float(m4.group(2))
                        sv_rb = float(m4.group(3)); sv_pe = float(m4.group(4))
                    break
    per_seq = qk / num_seq if num_seq > 0 else 0
    rows.append({
        'exp': exp, 'total': d.get('total_time_us',0), 'layer': d.get('layer_duration_us',0),
        'gen': gen_dur, 'proc': gen_proc, 'qk': qk, 'per_seq': per_seq,
        'softmax': softmax, 'score_v': score_v, 'kv_quant': kv_quant,
        'pim_rb': pim_rb, 'pim_pe': pim_pe,
        'qk_rb': qk_rb, 'qk_pe': qk_pe, 'sv_rb': sv_rb, 'sv_pe': sv_pe,
    })

# Markdown table
lines = []
lines.append('# Experiment Comparison Report')
lines.append('')
lines.append('| Experiment | Total(us) | Layer(us) | Gen(us) | Proc | Q@K(us) | per_seq | S@V(us) | qk_rb | qk_pe | sv_rb | sv_pe | pim_rb | pim_pe |')
lines.append('|------------|-----------|-----------|---------|------|---------|---------|---------|-------|-------|-------|-------|--------|--------|')
for r in rows:
    lines.append(f'| {r[\"exp\"]:28s} | {r[\"total\"]:9.0f} | {r[\"layer\"]:9.0f} | {r[\"gen\"]:7.0f} | {r[\"proc\"]:4s} | {r[\"qk\"]:7.0f} | {r[\"per_seq\"]:7.1f} | {r[\"score_v\"]:7.0f} | {r[\"qk_rb\"]:5.0f} | {r[\"qk_pe\"]:5.0f} | {r[\"sv_rb\"]:5.0f} | {r[\"sv_pe\"]:5.0f} | {r[\"pim_rb\"]:6.0f} | {r[\"pim_pe\"]:6.0f} |')

# PIM cycle verification
for r in rows:
    if 'fp16_pim' in r['exp'] and r['pim_rb'] > 0:
        M, K_v, N = 1, 128, 16528
        elem = 2; banks = 2048; group = 4; kv_heads = 8
        rb_size = 2048; dram_bw = 32e9; pe_width = 16; pe_cycle = 0.5
        total_elem = M*K_v + K_v*N + M*N
        total_bytes = total_elem * elem
        per_bank = total_bytes / banks
        rb_fills = max(1, math.ceil(per_bank / rb_size))
        rb_fill_one = rb_size / dram_bw * 1e9
        pe_total = per_bank / pe_width * pe_cycle
        if per_bank <= rb_size:
            latency = max(per_bank/dram_bw*1e9, pe_total)
        else:
            latency = rb_fill_one + max((rb_fills-1)*rb_fill_one, pe_total)
        per_gemv_rb = rb_fills * rb_fill_one
        per_gemv_pe = pe_total
        per_step_rb = per_gemv_rb * group * kv_heads * 2 * num_seq
        per_step_pe = per_gemv_pe * group * kv_heads * 2 * num_seq

        lines.append('')
        lines.append('## PIM Cycle Verification (exp2 FP16 PIM)')
        lines.append('')
        lines.append(f'- GEMV: M={M} K={K_v} N={N} elem={elem}B, banks={banks}, rb_fills={rb_fills}')
        lines.append(f'- Analytical per-GEMV: rb={per_gemv_rb:.1f}ns pe={per_gemv_pe:.1f}ns')
        lines.append(f'- Analytical per-step (×{group}group ×{kv_heads}heads ×2dir ×{int(num_seq)}seqs): rb={per_step_rb/1000:.1f}us pe={per_step_pe/1000:.1f}us')
        lines.append(f'- Reported: pim_rb={r[\"pim_rb\"]:.1f}us pim_pe={r[\"pim_pe\"]:.1f}us')
        lines.append(f'- RB ratio: {r[\"pim_rb\"]/(per_step_rb/1000):.2f}x  PE ratio: {r[\"pim_pe\"]/(per_step_pe/1000):.2f}x')
        break

report = '\n'.join(lines)
print(report)

with open('docs/experiment_comparison.md', 'w') as f:
    f.write(report)
    f.write('\n')
print(f'\nSaved to docs/experiment_comparison.md')
"