#!/usr/bin/env python3
"""
Convert srtslurm TRTLLM recipes under recipes/trtllm/ to sflow format
following the structure of sflow_trtllm_disagg.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


class _LiteralBlockDumper(yaml.Dumper):
    """YAML dumper that uses literal block style (|) for multi-line strings."""

    pass


def _literal_str_representer(dumper: yaml.Dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralBlockDumper.add_representer(str, _literal_str_representer)


def _norm_container(container: str) -> str:
    """Convert nvcr.io#... to nvcr.io/..."""
    if not container:
        return "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0"
    return container.replace("#", "/", 1)


def _parse_concurrencies(concurrencies) -> list[int]:
    """Parse concurrencies from string (e.g. '10x15x25'), list, or single int."""
    if concurrencies is None:
        return [50]
    if isinstance(concurrencies, list):
        return [int(c) for c in concurrencies]
    s = str(concurrencies).strip()
    if not s:
        return [50]
    if "x" in s:
        return [int(p) for p in s.split("x") if p.strip()]
    return [int(s)]


def _concurrency_domain(concurrencies) -> list[int]:
    """Always return concurrency as a list (domain), even for a single value."""
    values = _parse_concurrencies(concurrencies)
    return values if values else [50]


def _yaml_to_literal_block(obj: dict) -> str:
    """Dump YAML dict to a string for artifact content (PyYAML will emit as block scalar)."""
    return yaml.dump(obj, default_flow_style=False, allow_unicode=True, sort_keys=False).rstrip()


def convert_recipe(recipe_path: Path, data: dict) -> dict:
    """Convert one srtslurm recipe to sflow workflow structure."""
    name = data.get("name", recipe_path.stem)
    model = data.get("model", {})
    resources = data.get("resources", {})
    backend = data.get("backend", {})
    benchmark = data.get("benchmark", {})
    infra = data.get("infra", {})

    trtllm_config = backend.get("trtllm_config") or {}
    prefill_cfg = trtllm_config.get("prefill") or {}
    decode_cfg = trtllm_config.get("decode") or {}

    prefill_env = backend.get("prefill_environment") or {}
    decode_env = backend.get("decode_environment") or {}

    gpus_per_node = int(resources.get("gpus_per_node", 8))
    prefill_workers = int(resources.get("prefill_workers", 1))
    prefill_nodes = int(resources.get("prefill_nodes", 1))
    decode_workers = int(resources.get("decode_workers", 1))
    decode_nodes = int(resources.get("decode_nodes", 1))

    ctx_tp = int(prefill_cfg.get("tensor_parallel_size", 1))
    gen_tp = int(decode_cfg.get("tensor_parallel_size", 1))

    # Total nodes: worker nodes + 1 head (frontend/etcd/nats); optional +1 for dedicated infra
    extra_node = 1 if infra.get("etcd_nats_dedicated_node") else 0
    slurm_nodes = prefill_nodes + decode_nodes + 1 + extra_node

    model_path = model.get("path", "model")
    served_model_name = model_path.replace("/", "-")
    container = _norm_container(model.get("container", ""))

    isl = int(benchmark.get("isl", 1024))
    osl = int(benchmark.get("osl", 1024))
    concurrencies = benchmark.get("concurrencies", "50")
    concurrency_domain = _concurrency_domain(concurrencies)

    gpu_type = str(resources.get("gpu_type", "")).lower()
    use_arm_aiperf = gpu_type.startswith("gb200") or gpu_type.startswith("gb300")
    aiperf_image = (
        "gitlab-master.nvidia.com/perflab-compute/unified-benchmarks/aiperf:0.3.0-arm"
        if use_arm_aiperf
        else "gitlab-master.nvidia.com/perflab-compute/unified-benchmarks/aiperf:0.3.0-x86"
    )

    # Build variables section (same shape as sample)
    variables = {
        "SLURM_ACCOUNT": {"description": "SLURM account", "value": "rogliu"},
        "SLURM_PARTITION": {"description": "SLURM partition", "value": "gamoraq"},
        "SLURM_TIMELIMIT": {"description": "SLURM time limit", "value": 120},
        "GPUS_PER_NODE": {"description": "GPUs per node", "value": gpus_per_node},
        "SLURM_NODES": {"description": "Number of nodes", "value": slurm_nodes},
        "SERVED_MODEL_NAME": {
            "description": "Served model name",
            "value": served_model_name,
        },
        "MODEL_PATH": {"description": "Model path (fs or name)", "value": model_path},
        "NUM_CTX_SERVERS": {
            "description": "Number of context/prefill servers",
            "value": prefill_workers,
        },
        "CTX_TP_SIZE": {"description": "Context tensor parallel size", "value": ctx_tp},
        "CTX_DP_SIZE": {"description": "Context data parallel size", "value": 1},
        "CTX_EP_SIZE": {"description": "Context expert parallel size", "value": 1},
        "CTX_MOE_TP_SIZE": {
            "description": "Context MOE tensor parallel size",
            "value": ctx_tp,
        },
        "CTX_PP_SIZE": {
            "description": "Context pipeline parallel size",
            "value": int(prefill_cfg.get("pipeline_parallel_size", 1)),
        },
        "CTX_REPLICAS_POLICY": {
            "description": "Context replicas policy",
            "value": "parallel",
        },
        "CTX_BATCH_SIZE": {
            "description": "Context batch size",
            "value": int(prefill_cfg.get("max_batch_size", 128)),
        },
        "CTX_MAX_NUM_TOKENS": {
            "description": "Context max number of tokens",
            "value": int(prefill_cfg.get("max_num_tokens", 4096)),
        },
        "CTX_MAX_SEQ_LEN": {
            "description": "Context max sequence length",
            "value": int(prefill_cfg.get("max_seq_len", 1280)),
        },
        "CTX_FREE_GPU_MEMORY_FRACTION": {
            "description": "Context free GPU memory fraction",
            "value": float(prefill_cfg.get("kv_cache_config", {}).get("free_gpu_memory_fraction", 0.9)),
        },
        "CTX_ENABLE_ATTENTION_DP": {
            "description": "Context enable attention DP",
            "value": prefill_cfg.get("enable_attention_dp", False),
        },
        "KV_CACHE_DTYPE": {
            "description": "KV cache dtype",
            "value": prefill_cfg.get("kv_cache_config", {}).get("dtype", "fp8"),
        },
        "NUM_GEN_SERVERS": {
            "description": "Number of generation/decode servers",
            "value": decode_workers,
        },
        "GEN_TP_SIZE": {
            "description": "Generation tensor parallel size",
            "value": gen_tp,
        },
        "GEN_DP_SIZE": {"description": "Generation data parallel size", "value": 1},
        "GEN_EP_SIZE": {"description": "Generation expert parallel size", "value": 1},
        "GEN_MOE_TP_SIZE": {
            "description": "Generation MOE tensor parallel size",
            "value": gen_tp,
        },
        "GEN_PP_SIZE": {
            "description": "Generation pipeline parallel size",
            "value": int(decode_cfg.get("pipeline_parallel_size", 1)),
        },
        "GEN_REPLICAS_POLICY": {
            "description": "Generation replicas policy",
            "value": "parallel",
        },
        "GEN_BATCH_SIZE": {
            "description": "Generation batch size",
            "value": int(decode_cfg.get("max_batch_size", 128)),
        },
        "GEN_MAX_NUM_TOKENS": {
            "description": "Generation max number of tokens",
            "value": int(decode_cfg.get("max_num_tokens", 4096)),
        },
        "GEN_MAX_SEQ_LEN": {
            "description": "Generation max sequence length",
            "value": int(decode_cfg.get("max_seq_len", 2304)),
        },
        "GEN_FREE_GPU_MEMORY_FRACTION": {
            "description": "Generation free GPU memory fraction",
            "value": float(decode_cfg.get("kv_cache_config", {}).get("free_gpu_memory_fraction", 0.9)),
        },
        "GEN_ENABLE_ATTENTION_DP": {
            "description": "Generation enable attention DP",
            "value": decode_cfg.get("enable_attention_dp", False),
        },
        "EXTRA_FRONTEND_ARGS": {"description": "Extra frontend arguments", "value": ""},
        "EXTRA_PREFILL_ARGS": {"description": "Extra prefill arguments", "value": ""},
        "EXTRA_DECODE_ARGS": {"description": "Extra decode arguments", "value": ""},
        "ISL": {"description": "Input sequence length", "value": isl},
        "OSL": {"description": "Output sequence length", "value": osl},
        "MULTI_ROUND": {"description": "Number of benchmark rounds", "value": 8},
        "CONCURRENCY": {
            "description": "Concurrency",
            "value": concurrency_domain[0],
            "domain": concurrency_domain,
        },
        "AIPERF_IMAGE": {
            "description": "AIPerf container image",
            "value": aiperf_image,
        },
        "DYNAMO_IMAGE": {
            "description": "Dynamo TRTLLM container image",
            "value": container,
        },
    }

    # Artifacts: LOCAL_MODEL_PATH, PREFILL_CONFIG, DECODE_CONFIG (literal from recipe to preserve all options)
    if "/" in model_path and not model_path.startswith(("fs://", "file://")):
        model_uri = f"fs://{model_path}"
    else:
        model_uri = "fs://${{ variables.MODEL_PATH }}"
    artifacts = [
        {"name": "LOCAL_MODEL_PATH", "uri": model_uri},
        {
            "name": "PREFILL_CONFIG",
            "uri": "file://prefill_config.yaml",
            "content": _yaml_to_literal_block(prefill_cfg) if prefill_cfg else "",
        },
        {
            "name": "DECODE_CONFIG",
            "uri": "file://decode_config.yaml",
            "content": _yaml_to_literal_block(decode_cfg) if decode_cfg else "",
        },
    ]

    # Build prefill_server script: only export envs from recipe prefill_environment
    prefill_script = [
        "set -x",
        "echo ${CUDA_VISIBLE_DEVICES}",
    ]
    for k, v in prefill_env.items():
        prefill_script.append(f'export {k}="{v}"')
    prefill_script.append(
        "trtllm-llmapi-launch python3 -m dynamo.trtllm "
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }} "
        "--served-model-name ${SERVED_MODEL_NAME} "
        "--disaggregation-mode prefill "
        "--extra-engine-args ${{ artifacts.PREFILL_CONFIG.path }} ${EXTRA_PREFILL_ARGS}",
    )

    # Build decode_server script: only export envs from recipe decode_environment
    decode_script = [
        "set -x",
        "echo ${CUDA_VISIBLE_DEVICES}",
    ]
    for k, v in decode_env.items():
        decode_script.append(f'export {k}="{v}"')
    decode_script.append(
        "trtllm-llmapi-launch python3 -m dynamo.trtllm "
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }} "
        "--served-model-name ${SERVED_MODEL_NAME} "
        "--disaggregation-mode decode "
        "--extra-engine-args ${{ artifacts.DECODE_CONFIG.path }} ${EXTRA_DECODE_ARGS}",
    )

    # Load full workflow template from sample and substitute
    sample_path = Path(__file__).resolve().parent.parent / "sflow_trtllm_disagg.yaml"
    with open(sample_path, encoding="utf-8") as f:
        template = yaml.safe_load(f)

    out = {
        "version": "0.1",
        "variables": variables,
        "artifacts": artifacts,
        "backends": template["backends"],
        "operators": template["operators"],
        "workflow": {
            "name": name.replace(" ", "_").replace('"', ""),
            "timeout": "115m",
            "variables": template["workflow"]["variables"],
            "tasks": [],
        },
    }

    # Copy tasks from template; substitute prefill/decode script and env
    for task in template["workflow"]["tasks"]:
        t = dict(task)
        if t["name"] == "prefill_server":
            t["script"] = prefill_script
        elif t["name"] == "decode_server":
            t["script"] = decode_script
        out["workflow"]["tasks"].append(t)

    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    recipes_dir = repo_root / "recipes" / "trtllm"
    if not recipes_dir.is_dir():
        print(f"Recipes dir not found: {recipes_dir}", file=sys.stderr)
        return 1

    yaml_files = list(recipes_dir.rglob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files under {recipes_dir}", file=sys.stderr)
        return 1

    for path in sorted(yaml_files):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue

        if not data or data.get("backend", {}).get("type") != "trtllm":
            print(f"Skip {path}: not a TRTLLM recipe", file=sys.stderr)
            continue

        try:
            sflow = convert_recipe(path, data)
        except Exception as e:
            print(f"Convert failed {path}: {e}", file=sys.stderr)
            continue

        rel = path.relative_to(repo_root / "recipes")
        out_path = repo_root / "sflow_recipes" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(
                sflow,
                f,
                Dumper=_LiteralBlockDumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        # print(f"Converted: {path} -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
