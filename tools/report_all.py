#!/usr/bin/env python3
"""
Generate comprehensive comparison report across all models and experiments.
Usage:
    python3 tools/report_all.py [log_dir] [--md | --csv]
    python3 tools/report_all.py log/
    python3 tools/report_all.py log/ --md
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

# Model configurations
MODEL_PARAMS = {
    "llama3_8B": {"kv_heads": 8, "head_dim": 128, "num_heads": 32, "num_layers": 32, "hidden": 4096},
    "opt_6_7B": {"kv_heads": 32, "head_dim": 128, "num_heads": 32, "num_layers": 32, "hidden": 4096},
    "qwen3_8B": {"kv_heads": 8, "head_dim": 128, "num_heads": 32, "num_layers": 36, "hidden": 4096},
    "glm4_9B": {"kv_heads": 4, "head_dim": 128, "num_heads": 32, "num_layers": 40, "hidden": 4096},
}

EXP_LABELS = {
    "exp1": "FP16 GPU", "exp1_fp16_gpu": "FP16 GPU",
    "exp2": "FP16 PIM", "exp2_fp16_pim": "FP16 PIM",
    "exp3": "2-bit GPU", "exp3_2bit_gpu": "2-bit GPU",
    "exp4": "2-bit Hyb.", "exp4_2bit_hybrid": "2-bit Hyb.",
    "exp5": "2-bit All", "exp5_2bit_allpim": "2-bit All",
    "exp6": "2-bit Deq.", "exp6_2bit_dequant_pim": "2-bit Deq.",
    "exp4_pe_f16_8_i2_16": "Hyb (8B/16B)",
    "exp4_pe_f16_8_i2_32": "Hyb (8B/32B)",
    "exp4_pe_f16_16_i2_16": "Hyb (16B/16B)",
    "exp4_pe_f16_16_i2_32": "Hyb (16B/32B)",
    "exp5_pe_f16_8_i2_16": "All (8B/16B)",
    "exp5_pe_f16_8_i2_32": "All (8B/32B)",
    "exp5_pe_f16_16_i2_16": "All (16B/16B)",
    "exp5_pe_f16_16_i2_32": "All (16B/32B)",
    "exp3_4bit_gpu": "4-bit GPU",
    "exp4_4bit_hybrid": "4-bit Hyb.",
    "exp5_4bit_allpim": "4-bit All",
    # 2-bit PE variants
    "exp4_pe_f16_8_i2_16": "2-bit Hyb (FP16=8B/INT2=16B)",
    "exp4_pe_f16_8_i2_32": "2-bit Hyb (FP16=8B/INT2=32B)",
    "exp4_pe_f16_16_i2_16": "2-bit Hyb (FP16=16B/INT2=16B)",
    "exp4_pe_f16_16_i2_32": "2-bit Hyb (FP16=16B/INT2=32B)",
    "exp5_pe_f16_8_i2_16": "2-bit All (FP16=8B/INT2=16B)",
    "exp5_pe_f16_8_i2_32": "2-bit All (FP16=8B/INT2=32B)",
    "exp5_pe_f16_16_i2_16": "2-bit All (FP16=16B/INT2=16B)",
    "exp5_pe_f16_16_i2_32": "2-bit All (FP16=16B/INT2=32B)",
    # 4-bit PE variants
    "exp4_4bit_pe_f16_8_i2_16": "4-bit Hyb (FP16=8B/INT2=16B)",
    "exp4_4bit_pe_f16_8_i2_32": "4-bit Hyb (FP16=8B/INT2=32B)",
    "exp4_4bit_pe_f16_16_i2_16": "4-bit Hyb (FP16=16B/INT2=16B)",
    "exp4_4bit_pe_f16_16_i2_32": "4-bit Hyb (FP16=16B/INT2=32B)",
    "exp5_4bit_pe_f16_8_i2_16": "4-bit All (FP16=8B/INT2=16B)",
    "exp5_4bit_pe_f16_8_i2_32": "4-bit All (FP16=8B/INT2=32B)",
    "exp5_4bit_pe_f16_16_i2_16": "4-bit All (FP16=16B/INT2=16B)",
    "exp5_4bit_pe_f16_16_i2_32": "4-bit All (FP16=16B/INT2=32B)",
}

def normalize_exp(exp_key):
    """Map experiment keys like 'exp1_fp16_gpu' to short name like 'exp1'."""
    mapping = {
        "exp1_fp16_gpu": "exp1", "exp2_fp16_pim": "exp2",
        "exp3_2bit_gpu": "exp3", "exp4_2bit_hybrid": "exp4",
        "exp5_2bit_allpim": "exp5", "exp6_2bit_dequant_pim": "exp6",
    }
    return mapping.get(exp_key, exp_key)


def find_node(tree, name):
    if isinstance(tree, dict):
        if tree.get("name") == name:
            return tree
        for c in tree.get("children", []):
            r = find_node(c, name)
            if r:
                return r
    return None


def parse_raw_output(text):
    """Parse raw simulator output for per-stage timing."""
    d = {}
    for line in text.splitlines():
        if "LLM " in line and "|" in line:
            m = re.search(r"LLM\s*\|\s*([\d.]+)us", line)
            if m:
                d["total_time_us"] = float(m.group(1))
            m = re.search(r"qk=([\d.]+)us.*score_v=([\d.]+)us.*kv_quant=([\d.]+)us", line)
            if m:
                d["qk_total"] = float(m.group(1))
                d["sv_total"] = float(m.group(2))
                d["kv_quant_total"] = float(m.group(3))
        if "decoder_0 |" in line and "|" in line:
            m = re.search(r"decoder_0\s*\|\s*([\d.]+)us", line)
            if m:
                d["layer_total_us"] = float(m.group(1))
        if "AttentionGen |" in line and "qk=" in line:
            m = re.search(r"tensor\((\d+)", line)
            if m:
                d["num_seq"] = int(m.group(1))
            m = re.search(r"qk=([\d.]+)us\s+softmax=([\d.]+)us\s+score_v=([\d.]+)us\s+kv_quant=([\d.]+)us", line)
            if m:
                d["qk"] = float(m.group(1))
                d["softmax"] = float(m.group(2))
                d["score_v"] = float(m.group(3))
                d["kv_quant"] = float(m.group(4))
                break
            # Also try without softmax (GPU path)
            m = re.search(r"qk=([\d.]+)us\s+score_v=([\d.]+)us\s+kv_quant=([\d.]+)us", line)
            if m:
                d["qk"] = float(m.group(1))
                d["score_v"] = float(m.group(2))
                d["kv_quant"] = float(m.group(3))
                break
    return d


def parse_json(json_path):
    """Parse JSON output from parse_layer.py."""
    if not os.path.exists(json_path):
        return {}
    d = json.load(open(json_path))
    gen = find_node(d.get("operations", {}), "AttentionGen")
    result = {
        "total_time_us": d.get("total_time_us", 0),
        "layer_duration_us": d.get("layer_duration_us", 0),
    }
    if gen:
        result["gen_us"] = gen["duration_us"]
        result["processor"] = gen.get("processor", "?")
    return result


def compute_throughput(data, model_params, batch_size=64, decode_steps=128, input_len=16400):
    """Compute decode throughput (tokens/s)."""
    n_layers = model_params["num_layers"]
    per_layer = data.get("layer_total_us", 0)
    if per_layer == 0:
        # Fallback: compute from per-step gen time
        gen = data.get("qk", 0) + data.get("score_v", 0)
        if gen > 0:
            per_layer = gen * 1.5  # rough estimate: gen is ~1.5x of total layer

    if per_layer == 0:
        return 0

    # Decode throughput (tokens/s)
    decode_step_us = per_layer * n_layers
    decode_tps = batch_size / (decode_step_us * 1e-6) if decode_step_us > 0 else 0

    return decode_tps


def collect_data(log_dir):
    """Collect data from all JSON+TXT pairs in log_dir."""
    results = defaultdict(dict)
    default_model = "llama3_8B"

    def parse_filename(basename):
        """Parse model and experiment from filename like 'glm4_9B_exp1_fp16_gpu.txt' or 'config_exp1_fp16_gpu.json'."""
        parts = basename.replace(".txt", "").replace(".json", "")
        # Look for experiment pattern in the name
        exps = ["exp1_fp16_gpu", "exp2_fp16_pim", "exp3_2bit_gpu", "exp4_2bit_hybrid",
                "exp5_2bit_allpim", "exp6_2bit_dequant_pim",
                "exp3_4bit_gpu",                 "exp4_4bit_hybrid", "exp5_4bit_allpim",
                "exp4_4bit_pe_f16_8_i2_16", "exp4_4bit_pe_f16_8_i2_32",
                "exp4_4bit_pe_f16_16_i2_16", "exp4_4bit_pe_f16_16_i2_32",
                "exp5_4bit_pe_f16_8_i2_16", "exp5_4bit_pe_f16_8_i2_32",
                "exp5_4bit_pe_f16_16_i2_16", "exp5_4bit_pe_f16_16_i2_32",
                "exp4_pe_f16_8_i2_16", "exp4_pe_f16_8_i2_32",
                "exp4_pe_f16_16_i2_16", "exp4_pe_f16_16_i2_32",
                "exp5_pe_f16_8_i2_16", "exp5_pe_f16_8_i2_32",
                "exp5_pe_f16_16_i2_16", "exp5_pe_f16_16_i2_32"]
        for e in exps:
            if e in parts:
                idx = parts.index(e)
                prefix = parts[:idx].rstrip("_")
                model = prefix if prefix in MODEL_PARAMS else default_model
                return model, e
        # Fallback: try base_exp patterns
        for e in ["exp1", "exp2", "exp3_2bit", "exp4", "exp5", "exp6"]:
            if e in parts:
                return default_model, e
        return default_model, parts

    # Process all txt files in log_dir
    txt_files = sorted(glob.glob(os.path.join(log_dir, "*.txt")))
    for tf in txt_files:
        basename = os.path.basename(tf)
        model, exp = parse_filename(basename)
        d = parse_raw_output(open(tf).read())
        if d:
            # Also try JSON for layer_total info
            jf = tf.replace(".txt", ".json")
            if os.path.exists(jf):
                jd = parse_json(jf)
                d.update(jd)
            results[model][exp] = d

    return results


def format_num(v, fmt=".0f", default="-"):
    if v == 0 or v is None:
        return default
    return f"{v:{fmt}}"


def generate_markdown(results, output_path=None):
    """Generate markdown report."""
    lines = ["# Full Comparison Report", ""]

    for model in sorted(results.keys()):
        model_data = results[model]
        if not model_data:
            continue
        params = MODEL_PARAMS.get(model, {})
        kv_h = params.get("kv_heads", "?")

        lines.append(f"## Model: {model} (kv_heads={kv_h})")
        lines.append("")

        # === Per-Step Attention Table ===
        lines.append("### Per-Step Attention Latency (us, 16 seqs/DP)")
        lines.append("")
        headers = ["Config", "Score", "Aggregate", "Gen", "Softmax", "KV Quant",
                    "pim_rb", "pim_pe", "qk_rb", "qk_pe", "sv_rb", "sv_pe", "per_seq"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for exp in sorted(model_data.keys()):
            d = model_data.get(exp, {})
            if not d: continue
            decode_tps = compute_throughput(d, params)
            per_layer = d.get("layer_total_us", 0)
            row = [
                EXP_LABELS.get(exp, exp),
                format_num(per_layer),
                format_num(decode_tps, ",.0f"),
                format_num(per_layer * n_layers if per_layer > 0 else 0, ",.0f"),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("")

    # === Decode Speedup Summary (vs FP16 GPU) ===
    lines.append("## Decode Speedup vs. FP16 GPU (Gen latency)")
    lines.append("")

    # Build config list: base configs + PE variants
    base_configs = ["exp1_fp16_gpu", "exp2_fp16_pim", "exp3_2bit_gpu",
                    "exp4_2bit_hybrid", "exp5_2bit_allpim", "exp6_2bit_dequant_pim",
                    "exp3_4bit_gpu", "exp4_4bit_hybrid", "exp5_4bit_allpim"]
    pe_configs = [k for k in EXP_LABELS if "pe_f16" in k]
    all_configs = base_configs + sorted(pe_configs)

    models_sorted = sorted(results.keys())
    headers = ["Config"] + models_sorted
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for exp in all_configs:
        row = [EXP_LABELS.get(exp, exp)]
        for model in models_sorted:
            d = results[model].get(exp, {})
            gen = d.get("qk", 0) + d.get("score_v", 0)
            if gen > 0:
                fp16_key = "exp1_fp16_gpu"
                fp16 = results[model].get(fp16_key, {})
                fp16_gen = fp16.get("qk", 0) + fp16.get("score_v", 0)
                if fp16_gen > 0:
                    speedup = fp16_gen / gen
                    row.append(f"{speedup:.1f}×")
                else:
                    row.append(format_num(gen))
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Decode Throughput Speedup ===
    lines.append("## Decode Throughput Speedup vs. FP16 GPU")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for exp in all_configs:
        row = [EXP_LABELS.get(exp, exp)]
        for model in models_sorted:
            d = results[model].get(exp, {})
            params = MODEL_PARAMS.get(model, {})
            tps = compute_throughput(d, params)
            if tps > 0:
                fp16_tps = compute_throughput(results[model].get("exp1_fp16_gpu", {}), params)
                if fp16_tps > 0:
                    speedup = tps / fp16_tps
                    row.append(f"{speedup:.1f}×")
                else:
                    row.append(f"{tps:,.0f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved to {output_path}")
    return report


def generate_csv(results, output_path=None):
    """Generate CSV report."""
    lines = ["model,config,score_us,aggregate_us,gen_us,layer_us,decode_step_us,decode_tps,pim_rb,pim_pe,qk_rb,qk_pe,sv_rb,sv_pe,per_seq"]

    for model in sorted(results.keys()):
        params = MODEL_PARAMS.get(model, {})
        n_layers = params.get("num_layers", 32)
        for exp in sorted(EXP_LABELS.keys()):
            d = results[model].get(exp, {})
            decode_tps = compute_throughput(d, params)
            per_layer = d.get("layer_total_us", 0)
            ns = d.get("num_seq", 16)
            per_seq = d.get("qk", 0) / ns if ns > 0 else 0
            row = [
                model, exp,
                d.get("qk", 0), d.get("score_v", 0),
                d.get("qk", 0) + d.get("score_v", 0),
                per_layer,
                per_layer * n_layers if per_layer > 0 else 0,
                decode_tps,
                d.get("pim_rb", 0), d.get("pim_pe", 0),
                d.get("qk_rb", 0), d.get("qk_pe", 0),
                d.get("sv_rb", 0), d.get("sv_pe", 0),
                per_seq,
            ]
            lines.append(",".join(str(x) for x in row))

    csv = "\n".join(lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(csv + "\n")
        print(f"CSV saved to {output_path}")
    return csv


def main():
    fmt = "--md"
    log_dir = "log"

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--md", "--csv"):
            fmt = sys.argv[1]
        else:
            log_dir = sys.argv[1]
    if len(sys.argv) > 2:
        fmt = sys.argv[2]

    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        sys.exit(1)

    results = collect_data(log_dir)

    if fmt == "--csv":
        generate_csv(results, "docs/experiment_comparison.csv")
    else:
        generate_markdown(results, "docs/experiment_comparison.md")


if __name__ == "__main__":
    main()
