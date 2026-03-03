#!/usr/bin/env python3
"""
Extract all recipe configs from recipes/ folder into CSV format matching dynamo_infmax_v1.csv.
- Configs under recipes/trtllm/ -> sflow_config = disagg_trtllm
- All others -> sflow_config = disagg_sglang
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

# CSV header matching dynamo_infmax_v1.csv
CSV_HEADER = [
    "recipe_source",
    "enable_run",
    "model",
    "sflow_config",
    "machine",
    "precision",
    "isl",
    "osl",
    "num_nodes",
    "num_prefill_server",
    "num_decode_server",
    "enable_chunked_prefill",
    "prefill_gpu_mem_fraction",
    "prefill_tp",
    "prefill_moe_tp",
    "prefill_dep",
    "prefill_max_batch_size",
    "prefill_max_num_tokens",
    "cache_transceiver_max_num_tokens",
    "prefill_enable_block_reuse",
    "decode_gpu_mem_fraction",
    "decode_tp",
    "decode_moe_tp",
    "decode_dep",
    "decode_max_batch_size",
    "decode_max_num_tokens",
    "decode_enable_block_reuse",
    "decode_moe_backend",
    "mtp_size",
    "spec_decoding",
    "concurrency",
    "folder_path",
    "prefill_dp",
    "prefill_ep",
    "extra_prefill_args",
    "decode_dp",
    "decode_ep",
    "extra_decode_args",
    "raw_aic_cmd",
]

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"

# gpu_type -> machine name for CSV
MACHINE_MAP = {
    "gb200": "pre-tyche",
    "b200": "pre-nyx",
    "gb300": "theia",
    "b300": "bia",
    "h200": "cw",
    "h100": "eos",
}

# Keys to skip when building extra_prefill_args / extra_decode_args from sglang config
SKIP_CLI_KEYS = frozenset({
    "disaggregation-mode",
    "served-model-name",
    "model-path",
    "trust-remote-code",
    "mem-fraction-static",
    "max-running-requests",
    "max-prefill-tokens",
    "load-balance-method",
    "disaggregation-transfer-backend",
    "disaggregation-bootstrap-port",
    "cuda-graph-max-bs",
    "enable-dp-attention",
    "prefill-round-robin-balance",
    "skip-tokenizer-init",
    # All parallel-size args
    "tensor-parallel-size",
    "tp-size",
    "dp-size",
    "ep-size",
    "data-parallel-size",
    "expert-parallel-size",
    "moe-dense-tp-size",
    "pipeline-parallel-size",
})


def _get(d: dict | None, *keys: str, default: str = "hardcoded"):
    if d is None:
        return default
    for k in keys:
        if isinstance(d, dict) and k in d:
            v = d[k]
            return v if v is not None else default
    return default


def _yes_no(val) -> str:
    if val is None:
        return "no"
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, str):
        return "yes" if val.lower() in ("true", "yes", "1") else "no"
    return "yes" if val else "no"


def _config_to_cli_args(config: dict, skip_keys: frozenset[str] = SKIP_CLI_KEYS) -> list[str]:
    """Convert config dict to CLI arguments, skipping keys in skip_keys."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if flag_name in skip_keys:
            continue
        if flag_name.endswith("-parallel-size"):
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag_name}")
        elif isinstance(value, list):
            args.append(f"--{flag_name}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag_name}", str(value)])
    return args


def extract_row(recipe_path: Path, data: dict) -> list[str]:
    rel_path = recipe_path.relative_to(RECIPES_DIR)
    path_str = str(rel_path).replace("\\", "/")
    is_trtllm = "trtllm" in path_str
    sflow_config = "disagg_trtllm" if is_trtllm else "disagg_sglang"
    folder_path = str(rel_path.parent).replace("\\", "/")
    name = _get(data, "name", default="") or recipe_path.stem

    model_block = data.get("model") or {}
    resources = data.get("resources") or {}
    benchmark = data.get("benchmark") or {}
    backend = data.get("backend") or {}

    model_path = _get(model_block, "path", default=name)
    if model_path == "hardcoded":
        model_path = name
    gpu_type = _get(resources, "gpu_type", default="hardcoded")
    machine = MACHINE_MAP.get(str(gpu_type).lower(), gpu_type) if gpu_type != "hardcoded" else "hardcoded"
    precision = _get(model_block, "precision", default="hardcoded")
    isl = _get(benchmark, "isl", default="hardcoded")
    osl = _get(benchmark, "osl", default="hardcoded")

    pn = resources.get("prefill_nodes")
    dn = resources.get("decode_nodes")
    agg_n = resources.get("agg_nodes")
    agg_w = resources.get("agg_workers")
    if agg_n is not None and agg_w is not None:
        num_nodes = agg_n
        num_prefill = 0
        num_decode = agg_w
    elif pn is not None and dn is not None:
        num_nodes = pn + (dn if dn > 0 else 0)
        num_prefill = _get(resources, "prefill_workers", default="hardcoded")
        num_decode = _get(resources, "decode_workers", default="hardcoded")
    else:
        num_nodes = "hardcoded"
        num_prefill = _get(resources, "prefill_workers", default="hardcoded")
        num_decode = _get(resources, "decode_workers", default="hardcoded")

    # MTP from path or name
    mtp_size = 0
    if "mtp" in path_str.lower() or "mtp" in name.lower():
        for part in path_str.split("/") + name.split("_"):
            if "mtp" in part.lower() and part[-1].isdigit():
                try:
                    mtp_size = int(part[-1])
                    break
                except ValueError:
                    pass
    spec_decoding = "mtp" if mtp_size else "none"
    concurrency = "up_to_4096"

    prefill_gpu_mem = "hardcoded"
    prefill_tp = "hardcoded"
    prefill_moe_tp = 1
    prefill_dep = "hardcoded"
    prefill_max_batch = "hardcoded"
    prefill_max_tokens = "hardcoded"
    cache_transceiver_max = "hardcoded"
    prefill_block_reuse = "no"
    decode_gpu_mem = "hardcoded"
    decode_tp = "hardcoded"
    decode_moe_tp = 1
    decode_dep = "hardcoded"
    decode_max_batch = "hardcoded"
    decode_max_tokens = "hardcoded"
    decode_block_reuse = "no"
    enable_chunked_prefill = "no"
    extra_prefill_args = "empty"
    extra_decode_args = "empty"
    prefill_dp = "1"
    prefill_ep = "1"
    decode_dp = "1"
    decode_ep = "1"

    if is_trtllm:
        if agg_n is not None:
            sflow_config = "agg_trtllm"
        tc = backend.get("trtllm_config") or {}
        prefill = tc.get("prefill") or {}
        decode = tc.get("decode") or {}
        kv_prefill = prefill.get("kv_cache_config") or {}
        kv_decode = decode.get("kv_cache_config") or {}
        ctc_prefill = prefill.get("cache_transceiver_config") or {}
        ctc_decode = decode.get("cache_transceiver_config") or {}

        prefill_gpu_mem = kv_prefill.get("free_gpu_memory_fraction", "hardcoded")
        prefill_tp = prefill.get("tensor_parallel_size", "hardcoded")
        prefill_moe_tp = 1
        prefill_ep = prefill.get("moe_expert_parallel_size", 1)
        prefill_dep = _yes_no(prefill.get("enable_attention_dp"))
        prefill_max_batch = prefill.get("max_batch_size", "hardcoded")
        prefill_max_tokens = prefill.get("max_num_tokens", "hardcoded")
        cache_transceiver_max = ctc_prefill.get("max_tokens_in_buffer") or ctc_decode.get("max_tokens_in_buffer") or "hardcoded"
        prefill_block_reuse = _yes_no(kv_prefill.get("enable_block_reuse"))

        decode_gpu_mem = kv_decode.get("free_gpu_memory_fraction", "hardcoded")
        decode_tp = decode.get("tensor_parallel_size", "hardcoded")
        decode_moe_tp = 1
        decode_ep = decode.get("moe_expert_parallel_size", 1)
        decode_dep = _yes_no(decode.get("enable_attention_dp"))
        decode_max_batch = decode.get("max_batch_size", "hardcoded")
        decode_max_tokens = decode.get("max_num_tokens", "hardcoded")
        decode_block_reuse = _yes_no(kv_decode.get("enable_block_reuse"))
    else:
        sc = backend.get("sglang_config") or {}
        agg = sc.get("aggregated")
        if agg is not None:
            sflow_config = "agg_sglang"
            prefill = agg
            prefill_gpu_mem = prefill.get("mem-fraction-static", "hardcoded")
            prefill_tp = prefill.get("tensor-parallel-size") or prefill.get("tp-size", "hardcoded")
            prefill_dep = _yes_no(prefill.get("enable-dp-attention"))
            prefill_max_batch = prefill.get("max-running-requests") or prefill.get("cuda-graph-max-bs") or "hardcoded"
            prefill_max_tokens = prefill.get("max-total-tokens") or prefill.get("max-prefill-tokens") or 0
            cache_transceiver_max = prefill_max_tokens
            if prefill.get("chunked-prefill-size") or prefill.get("chunked-prefill"):
                enable_chunked_prefill = "yes"
            prefill_dp = prefill.get("data-parallel-size") or prefill.get("dp-size") or 1
            prefill_ep = prefill.get("expert-parallel-size") or prefill.get("ep-size") or 1
            prefill_cli = _config_to_cli_args(prefill)
            extra_prefill_args = " ".join(prefill_cli) if prefill_cli else "empty"
            decode_gpu_mem = "0.0"
            decode_tp = "0"
            decode_dep = "no"
            decode_max_batch = 0
            decode_max_tokens = 0
            decode_block_reuse = "no"
            extra_decode_args = "empty"
        else:
            prefill = sc.get("prefill") or {}
            decode = sc.get("decode") or {}
            prefill_gpu_mem = prefill.get("mem-fraction-static", "hardcoded")
            prefill_tp = prefill.get("tensor-parallel-size") or prefill.get("tp-size", "hardcoded")
            prefill_dep = _yes_no(prefill.get("enable-dp-attention"))
            decode_gpu_mem = decode.get("mem-fraction-static", "hardcoded")
            decode_tp = decode.get("tensor-parallel-size") or decode.get("tp-size", "hardcoded")
            decode_dep = _yes_no(decode.get("enable-dp-attention"))
            if prefill.get("chunked-prefill-size") or prefill.get("chunked-prefill"):
                enable_chunked_prefill = "yes"
            prefill_max_batch = prefill.get("max-running-requests") or prefill.get("cuda-graph-max-bs") or "hardcoded"
            prefill_max_tokens = prefill.get("max-total-tokens") or prefill.get("max-prefill-tokens") or 0
            cache_transceiver_max = prefill_max_tokens
            decode_max_batch = decode.get("max-running-requests") or decode.get("cuda-graph-max-bs") or "hardcoded"
            decode_max_tokens = decode.get("max-total-tokens") or decode.get("max-prefill-tokens") or prefill_max_tokens
            prefill_dp = prefill.get("data-parallel-size") or prefill.get("dp-size") or 1
            decode_dp = decode.get("data-parallel-size") or decode.get("dp-size") or 1
            prefill_ep = prefill.get("expert-parallel-size") or prefill.get("ep-size") or 1
            decode_ep = decode.get("expert-parallel-size") or decode.get("ep-size") or 1
            prefill_cli = _config_to_cli_args(prefill)
            decode_cli = _config_to_cli_args(decode)
            extra_prefill_args = " ".join(prefill_cli) if prefill_cli else "empty"
            extra_decode_args = " ".join(decode_cli) if decode_cli else "empty"

        # SGLang: if extra args contain speculative-num-draft-tokens, set spec_decoding and mtp_size
        for src in (extra_decode_args, extra_prefill_args):
            if "speculative-num-draft-tokens" in src:
                idx = src.find("speculative-num-draft-tokens") + len("speculative-num-draft-tokens")
                rest = src[idx:].strip()
                if rest:
                    parts = rest.split()
                    try:
                        mtp_size = int(parts[0])
                        spec_decoding = "mtp"
                    except (ValueError, IndexError):
                        pass
                break

    decode_moe_backend = "auto"

    def _str(v):
        if v is None:
            return "hardcoded"
        if isinstance(v, list):
            return "x".join(str(x) for x in v) if v else "none"
        return str(v)

    return [
        "infmax_v1.5",
        "yes",
        "deepseek_r1",
        sflow_config,
        _str(machine),
        _str(precision),
        _str(isl),
        _str(osl),
        _str(num_nodes),
        _str(num_prefill),
        _str(num_decode),
        enable_chunked_prefill,
        _str(prefill_gpu_mem),
        _str(prefill_tp),
        str(prefill_moe_tp),
        _str(prefill_dep),
        _str(prefill_max_batch),
        _str(prefill_max_tokens),
        _str(cache_transceiver_max),
        prefill_block_reuse,
        _str(decode_gpu_mem),
        _str(decode_tp),
        str(decode_moe_tp),
        _str(decode_dep),
        _str(decode_max_batch),
        _str(decode_max_tokens),
        decode_block_reuse,
        decode_moe_backend,
        str(mtp_size),
        spec_decoding,
        _str(concurrency),
        folder_path,
        prefill_dp,
        str(prefill_ep),
        extra_prefill_args,
        decode_dp,
        str(decode_ep),
        extra_decode_args,
        name,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract recipe configs to CSV")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: only one config per (model, machine, sflow_config) combo",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent
    out_path = out_dir / ("recipes_extracted_smoke.csv" if args.smoke else "recipes_extracted.csv")
    yaml_files = sorted(RECIPES_DIR.rglob("*.yaml"))
    rows = []

    for recipe_path in yaml_files:
        try:
            text = recipe_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if not data:
                continue
            benchmark = data.get("benchmark") or {}
            isl = benchmark.get("isl")
            osl = benchmark.get("osl")
            if isl != 8192 or osl != 1024:
                continue
            rows.append(extract_row(recipe_path, data))
        except Exception as e:
            print(f"Skip {recipe_path}: {e}", file=sys.stderr)

    if args.smoke:
        # One row per (model, machine, sflow_config, precision), keep smallest num_nodes
        model_idx = CSV_HEADER.index("model")
        machine_idx = CSV_HEADER.index("machine")
        sflow_idx = CSV_HEADER.index("sflow_config")
        precision_idx = CSV_HEADER.index("precision")
        num_nodes_idx = CSV_HEADER.index("num_nodes")

        def _num_nodes_sort_val(row: list[str]) -> int:
            raw = row[num_nodes_idx]
            try:
                return int(raw)
            except (ValueError, TypeError):
                return 999999

        best: dict[tuple[str, str, str, str], list[str]] = {}
        for row in rows:
            key = (row[model_idx], row[machine_idx], row[sflow_idx], row[precision_idx])
            n = _num_nodes_sort_val(row)
            if key not in best or _num_nodes_sort_val(best[key]) > n:
                best[key] = row
        rows = list(best.values())
        concurrency_idx = CSV_HEADER.index("concurrency")
        for row in rows:
            row[concurrency_idx] = "only_64"
        print(f"Smoke mode: {len(rows)} rows (one per model+machine+sflow_config+precision, smallest num_nodes)", file=sys.stderr)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
