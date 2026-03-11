#!/usr/bin/env python3
"""
Convert srtslurm recipes under recipes/ to sflow format.

Supports both TRTLLM (recipes/trtllm/) and SGLang (all other recipe dirs) backends.
TRTLLM recipes follow the sflow_trtllm_disagg.yaml template.
SGLang recipes follow the sflow_sglang_disagg.yaml template and support both
disaggregated (prefill/decode) and aggregated modes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    values = _parse_concurrencies(concurrencies)
    return values if values else [50]


def _yaml_to_literal_block(obj: dict) -> str:
    return yaml.dump(obj, default_flow_style=False, allow_unicode=True, sort_keys=False).rstrip()


def _aiperf_image(gpu_type: str) -> str:
    use_arm = gpu_type.startswith("gb200") or gpu_type.startswith("gb300")
    return (
        "gitlab-master.nvidia.com/perflab-compute/unified-benchmarks/aiperf:0.3.0-arm"
        if use_arm
        else "gitlab-master.nvidia.com/perflab-compute/unified-benchmarks/aiperf:0.3.0-x86"
    )


def _sglang_config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert sglang_config dict to CLI argument strings.

    Keys like 'tp-size' become '--tp-size 4', booleans become '--flag' or are omitted,
    lists become '--key val1 val2 ...'.
    """
    args: list[str] = []
    for key, value in config.items():
        flag = f"--{key}" if key.startswith("-") else f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([flag, str(value)])
    return args


def _generate_nginx_config(num_frontends: int, slurm_nodes: int, frontend_port: int = 8180) -> str:
    """Generate nginx config content with sflow node IP expressions.

    Frontends run on worker nodes (node[1]..node[N]), nginx proxies to them.
    The number of frontend nodes is capped by available worker nodes.
    """
    actual_count = min(num_frontends, slurm_nodes - 1)
    lines = [
        "worker_processes auto;",
        "http {",
        "    access_log off;",
        "    upstream backend_servers {",
    ]
    for i in range(1, actual_count + 1):
        lines.append(f"        server ${{{{ backends.slurm_cluster.nodes[{i}].ip_address }}}}:{frontend_port};")
    lines.extend(
        [
            "    }",
            "    server {",
            "        listen 8000;",
            "        location / {",
            "            proxy_pass http://backend_servers;",
            "            proxy_buffering off;",
            "            proxy_read_timeout 24h;",
            "            proxy_send_timeout 24h;",
            "        }",
            "    }",
            "}",
            "events {",
            "    worker_connections 65535;",
            "    multi_accept on;",
            "    use epoll;",
            "}",
        ]
    )
    return "\n".join(lines)


def _base_variables(
    *,
    gpus_per_node: int,
    slurm_nodes: int,
    served_model_name: str,
    model_path: str,
    isl: int,
    osl: int,
    concurrency_domain: list[int],
    aiperf_image: str,
) -> dict[str, Any]:
    """Variables common to both TRTLLM and SGLang sflow recipes."""
    return {
        "SLURM_ACCOUNT": {"description": "SLURM account", "value": "rogliu"},
        "SLURM_PARTITION": {"description": "SLURM partition", "value": "gamoraq"},
        "SLURM_TIMELIMIT": {"description": "SLURM time limit", "value": 120},
        "GPUS_PER_NODE": {"description": "GPUs per node", "value": gpus_per_node},
        "SLURM_NODES": {"description": "Number of nodes", "value": slurm_nodes},
        "SERVED_MODEL_NAME": {"description": "Served model name", "value": served_model_name},
        "MODEL_PATH": {"description": "Model path (fs or name)", "value": model_path},
        "EXTRA_FRONTEND_ARGS": {"description": "Extra frontend arguments", "value": ""},
        "ISL": {"description": "Input sequence length", "value": isl},
        "OSL": {"description": "Output sequence length", "value": osl},
        "MULTI_ROUND": {"description": "Number of benchmark rounds", "value": 8},
        "CONCURRENCY": {
            "description": "Concurrency",
            "value": concurrency_domain[0],
            "domain": concurrency_domain,
        },
        "AIPERF_IMAGE": {"description": "AIPerf container image", "value": aiperf_image},
    }


# ---------------------------------------------------------------------------
# TRTLLM conversion
# ---------------------------------------------------------------------------


def _norm_container(container: str) -> str:
    if not container:
        return "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0"
    return container.replace("#", "/", 1)


def convert_trtllm_recipe(recipe_path: Path, data: dict) -> dict:
    """Convert one srtslurm TRTLLM recipe to sflow workflow structure."""
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

    extra_node = 1 if infra.get("etcd_nats_dedicated_node") else 0
    slurm_nodes = prefill_nodes + decode_nodes + 1 + extra_node

    model_path = model.get("path", "model")
    served_model_name = model_path.replace("/", "-")
    container = _norm_container(model.get("container", ""))

    isl = int(benchmark.get("isl", 1024))
    osl = int(benchmark.get("osl", 1024))
    concurrency_domain = _concurrency_domain(benchmark.get("concurrencies", "50"))
    gpu_type = str(resources.get("gpu_type", "")).lower()

    variables = _base_variables(
        gpus_per_node=gpus_per_node,
        slurm_nodes=slurm_nodes,
        served_model_name=served_model_name,
        model_path=model_path,
        isl=isl,
        osl=osl,
        concurrency_domain=concurrency_domain,
        aiperf_image=_aiperf_image(gpu_type),
    )
    variables.update(
        {
            "NUM_CTX_SERVERS": {"description": "Number of context/prefill servers", "value": prefill_workers},
            "CTX_TP_SIZE": {"description": "Context tensor parallel size", "value": ctx_tp},
            "CTX_DP_SIZE": {"description": "Context data parallel size", "value": 1},
            "CTX_EP_SIZE": {"description": "Context expert parallel size", "value": 1},
            "CTX_MOE_TP_SIZE": {"description": "Context MOE tensor parallel size", "value": ctx_tp},
            "CTX_PP_SIZE": {
                "description": "Context pipeline parallel size",
                "value": int(prefill_cfg.get("pipeline_parallel_size", 1)),
            },
            "CTX_REPLICAS_POLICY": {"description": "Context replicas policy", "value": "parallel"},
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
            "NUM_GEN_SERVERS": {"description": "Number of generation/decode servers", "value": decode_workers},
            "GEN_TP_SIZE": {"description": "Generation tensor parallel size", "value": gen_tp},
            "GEN_DP_SIZE": {"description": "Generation data parallel size", "value": 1},
            "GEN_EP_SIZE": {"description": "Generation expert parallel size", "value": 1},
            "GEN_MOE_TP_SIZE": {"description": "Generation MOE tensor parallel size", "value": gen_tp},
            "GEN_PP_SIZE": {
                "description": "Generation pipeline parallel size",
                "value": int(decode_cfg.get("pipeline_parallel_size", 1)),
            },
            "GEN_REPLICAS_POLICY": {"description": "Generation replicas policy", "value": "parallel"},
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
            "EXTRA_PREFILL_ARGS": {"description": "Extra prefill arguments", "value": ""},
            "EXTRA_DECODE_ARGS": {"description": "Extra decode arguments", "value": ""},
            "DYNAMO_IMAGE": {"description": "Dynamo TRTLLM container image", "value": container},
        }
    )

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

    prefill_script = ["set -x", "echo ${CUDA_VISIBLE_DEVICES}"]
    for k, v in prefill_env.items():
        prefill_script.append(f'export {k}="{v}"')
    prefill_script.append(
        "trtllm-llmapi-launch python3 -m dynamo.trtllm "
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }} "
        "--served-model-name ${SERVED_MODEL_NAME} "
        "--disaggregation-mode prefill "
        "--extra-engine-args ${{ artifacts.PREFILL_CONFIG.path }} ${EXTRA_PREFILL_ARGS}",
    )

    decode_script = ["set -x", "echo ${CUDA_VISIBLE_DEVICES}"]
    for k, v in decode_env.items():
        decode_script.append(f'export {k}="{v}"')
    decode_script.append(
        "trtllm-llmapi-launch python3 -m dynamo.trtllm "
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }} "
        "--served-model-name ${SERVED_MODEL_NAME} "
        "--disaggregation-mode decode "
        "--extra-engine-args ${{ artifacts.DECODE_CONFIG.path }} ${EXTRA_DECODE_ARGS}",
    )

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

    for task in template["workflow"]["tasks"]:
        t = dict(task)
        if t["name"] == "prefill_server":
            t["script"] = prefill_script
        elif t["name"] == "decode_server":
            t["script"] = decode_script
        out["workflow"]["tasks"].append(t)

    return out


# ---------------------------------------------------------------------------
# SGLang conversion
# ---------------------------------------------------------------------------


def _get_sglang_tp(config: dict[str, Any]) -> int:
    """Extract tensor-parallel size from sglang_config dict."""
    for key in ("tp-size", "tensor-parallel-size", "tp_size", "tensor_parallel_size"):
        if key in config:
            return int(config[key])
    return 1


def _is_aggregated_recipe(data: dict) -> bool:
    """Detect if a recipe uses aggregated mode (no prefill/decode split)."""
    resources = data.get("resources", {})
    if resources.get("agg_nodes") or resources.get("agg_workers"):
        return True
    sglang_config = data.get("backend", {}).get("sglang_config") or {}
    return bool(
        sglang_config.get("aggregated") and not sglang_config.get("prefill") and not sglang_config.get("decode")
    )


def _sglang_container(model: dict) -> str:
    """Resolve the SGLang container image from the model config."""
    container = model.get("container", "")
    if not container:
        return "lmsysorg/sglang:v0.5.8-runtime"
    if "/" in container or ":" in container:
        return container.replace("#", "/", 1)
    return container


def _get_frontend_config(data: dict) -> tuple[bool, int, str]:
    """Extract multi-frontend settings from recipe.

    Returns:
        (enable_multi, num_frontends, nginx_container)
    """
    fe = data.get("frontend", {})
    enable = fe.get("enable_multiple_frontends", False)
    num_additional = int(fe.get("num_additional_frontends", 9))
    nginx_container = fe.get("nginx_container", "nginx:1.27.4")
    # Total frontends = num_additional + 1 (the first one)
    return enable, num_additional + 1, nginx_container


def _apply_multi_frontend(
    out: dict,
    enable_multi: bool,
    num_frontends: int,
    nginx_container: str,
    slurm_nodes: int,
) -> None:
    """Apply multi-frontend configuration to the sflow output dict in-place.

    When multi-frontend is enabled:
    - Adds NUM_FRONTENDS, FRONTEND_PORT, NGINX_IMAGE variables
    - Generates NGINX_CONFIG artifact with upstream entries
    - Includes nginx_server task, adjusts frontend_server
    - Updates benchmark depends_on to include nginx_server

    When disabled:
    - Removes nginx_server task
    - Keeps single frontend on port 8000
    """
    if enable_multi:
        frontend_port = 8180
        actual_frontends = min(num_frontends, slurm_nodes - 1)
        out["variables"]["NUM_FRONTENDS"] = {
            "description": "Number of frontend instances",
            "value": actual_frontends,
        }
        out["variables"]["FRONTEND_PORT"] = {
            "description": "Frontend listening port (8180 behind nginx, 8000 direct)",
            "value": frontend_port,
        }
        out["variables"]["NGINX_IMAGE"] = {
            "description": "Nginx container image",
            "value": nginx_container,
        }

        nginx_cfg = _generate_nginx_config(actual_frontends, slurm_nodes, frontend_port)
        out["artifacts"].append(
            {
                "name": "NGINX_CONFIG",
                "uri": "file://nginx.conf",
                "content": nginx_cfg,
            }
        )

        # Update benchmark depends_on to include nginx_server
        for task in out["workflow"]["tasks"]:
            if task["name"] == "benchmark":
                deps = task.get("depends_on", [])
                if "nginx_server" not in deps:
                    deps.append("nginx_server")
                # Remove direct frontend_server dep — nginx handles it
                if "frontend_server" in deps:
                    deps.remove("frontend_server")
                task["depends_on"] = deps
    else:
        # Single frontend: remove nginx_server task, keep defaults
        out["variables"]["NUM_FRONTENDS"] = {
            "description": "Number of frontend instances",
            "value": 1,
        }
        out["variables"]["FRONTEND_PORT"] = {
            "description": "Frontend listening port",
            "value": 8000,
        }
        out["workflow"]["tasks"] = [t for t in out["workflow"]["tasks"] if t["name"] != "nginx_server"]


def _build_sglang_server_script(
    env_vars: dict[str, str],
    sglang_cli_args: list[str],
    mode: str | None = None,
    extra_args_var: str = "EXTRA_PREFILL_ARGS",
) -> list[str]:
    """Build the shell script lines for a sglang worker task."""
    script = ["set -x", "echo ${CUDA_VISIBLE_DEVICES}"]
    for k, v in env_vars.items():
        script.append(f'export {k}="{v}"')

    cmd_parts = [
        "python3 -m dynamo.sglang",
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}",
        "--served-model-name ${SERVED_MODEL_NAME}",
    ]
    if mode:
        cmd_parts.append(f"--disaggregation-mode {mode}")
    cmd_parts.extend(sglang_cli_args)
    cmd_parts.append(f"${{{extra_args_var}}}")

    script.append(" ".join(cmd_parts))
    return script


def convert_sglang_disagg_recipe(recipe_path: Path, data: dict) -> dict:
    """Convert a disaggregated SGLang recipe to sflow format."""
    name = data.get("name", recipe_path.stem)
    model = data.get("model", {})
    resources = data.get("resources", {})
    backend = data.get("backend", {})
    benchmark = data.get("benchmark", {})
    infra = data.get("infra", {})

    sglang_config = backend.get("sglang_config") or {}
    prefill_cfg = dict(sglang_config.get("prefill") or {})
    decode_cfg = dict(sglang_config.get("decode") or {})
    prefill_env = backend.get("prefill_environment") or {}
    decode_env = backend.get("decode_environment") or {}

    gpus_per_node = int(resources.get("gpus_per_node", 8))
    prefill_workers = int(resources.get("prefill_workers", 1))
    prefill_nodes = int(resources.get("prefill_nodes", 1))
    decode_workers = int(resources.get("decode_workers", 1))
    decode_nodes = int(resources.get("decode_nodes", 0))

    ctx_tp = _get_sglang_tp(prefill_cfg)
    gen_tp = _get_sglang_tp(decode_cfg)

    extra_node = 1 if infra.get("etcd_nats_dedicated_node") else 0
    slurm_nodes = prefill_nodes + decode_nodes + 1 + extra_node

    model_path = model.get("path", "model")
    container = _sglang_container(model)
    gpu_type = str(resources.get("gpu_type", "")).lower()

    isl = int(benchmark.get("isl", 1024))
    osl = int(benchmark.get("osl", 1024))
    concurrency_domain = _concurrency_domain(benchmark.get("concurrencies", "50"))

    served_model_name = (
        prefill_cfg.get("served-model-name") or prefill_cfg.get("served_model_name") or model_path.replace("/", "-")
    )

    enable_multi, num_frontends, nginx_container = _get_frontend_config(data)

    # Strip keys that are handled by sflow variables/command structure
    for cfg in (prefill_cfg, decode_cfg):
        for k in ("served-model-name", "served_model_name", "disaggregation-mode", "disaggregation_mode"):
            cfg.pop(k, None)

    prefill_cli = _sglang_config_to_cli_args(prefill_cfg)
    decode_cli = _sglang_config_to_cli_args(decode_cfg)

    variables = _base_variables(
        gpus_per_node=gpus_per_node,
        slurm_nodes=slurm_nodes,
        served_model_name=served_model_name,
        model_path=model_path,
        isl=isl,
        osl=osl,
        concurrency_domain=concurrency_domain,
        aiperf_image=_aiperf_image(gpu_type),
    )
    variables.update(
        {
            "NUM_CTX_SERVERS": {"description": "Number of context/prefill servers", "value": prefill_workers},
            "CTX_TP_SIZE": {"description": "Context tensor parallel size", "value": ctx_tp},
            "NUM_GEN_SERVERS": {"description": "Number of generation/decode servers", "value": decode_workers},
            "GEN_TP_SIZE": {"description": "Generation tensor parallel size", "value": gen_tp},
            "EXTRA_PREFILL_ARGS": {"description": "Extra prefill arguments", "value": ""},
            "EXTRA_DECODE_ARGS": {"description": "Extra decode arguments", "value": ""},
            "SGLANG_IMAGE": {"description": "SGLang container image", "value": container},
        }
    )

    if "/" in model_path and not model_path.startswith(("fs://", "file://")):
        model_uri = f"fs://{model_path}"
    else:
        model_uri = "fs://${{ variables.MODEL_PATH }}"
    artifacts = [{"name": "LOCAL_MODEL_PATH", "uri": model_uri}]

    prefill_script = _build_sglang_server_script(
        prefill_env, prefill_cli, mode="prefill", extra_args_var="EXTRA_PREFILL_ARGS"
    )
    decode_script = _build_sglang_server_script(
        decode_env, decode_cli, mode="decode", extra_args_var="EXTRA_DECODE_ARGS"
    )

    template_path = Path(__file__).resolve().parent.parent / "sflow_sglang_disagg.yaml"
    with open(template_path, encoding="utf-8") as f:
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

    for task in template["workflow"]["tasks"]:
        t = dict(task)
        if t["name"] == "prefill_server":
            t["script"] = prefill_script
        elif t["name"] == "decode_server":
            t["script"] = decode_script
        out["workflow"]["tasks"].append(t)

    _apply_multi_frontend(out, enable_multi, num_frontends, nginx_container, slurm_nodes)

    return out


def convert_sglang_agg_recipe(recipe_path: Path, data: dict) -> dict:
    """Convert an aggregated SGLang recipe to sflow format."""
    name = data.get("name", recipe_path.stem)
    model = data.get("model", {})
    resources = data.get("resources", {})
    backend = data.get("backend", {})
    benchmark = data.get("benchmark", {})
    infra = data.get("infra", {})

    sglang_config = backend.get("sglang_config") or {}
    agg_cfg = dict(sglang_config.get("aggregated") or {})
    agg_env = backend.get("aggregated_environment") or {}

    gpus_per_node = int(resources.get("gpus_per_node", 8))
    agg_workers = int(resources.get("agg_workers", 1))
    agg_nodes = int(resources.get("agg_nodes", 1))
    agg_tp = _get_sglang_tp(agg_cfg)

    extra_node = 1 if infra.get("etcd_nats_dedicated_node") else 0
    slurm_nodes = agg_nodes + 1 + extra_node

    model_path = model.get("path", "model")
    container = _sglang_container(model)
    gpu_type = str(resources.get("gpu_type", "")).lower()

    isl = int(benchmark.get("isl", 1024))
    osl = int(benchmark.get("osl", 1024))
    concurrency_domain = _concurrency_domain(benchmark.get("concurrencies", "50"))

    served_model_name = (
        agg_cfg.get("served-model-name") or agg_cfg.get("served_model_name") or model_path.replace("/", "-")
    )

    enable_multi, num_frontends, nginx_container = _get_frontend_config(data)

    for k in ("served-model-name", "served_model_name"):
        agg_cfg.pop(k, None)

    agg_cli = _sglang_config_to_cli_args(agg_cfg)

    variables = _base_variables(
        gpus_per_node=gpus_per_node,
        slurm_nodes=slurm_nodes,
        served_model_name=served_model_name,
        model_path=model_path,
        isl=isl,
        osl=osl,
        concurrency_domain=concurrency_domain,
        aiperf_image=_aiperf_image(gpu_type),
    )
    variables.update(
        {
            "NUM_AGG_SERVERS": {"description": "Number of aggregated servers", "value": agg_workers},
            "AGG_TP_SIZE": {"description": "Aggregated tensor parallel size", "value": agg_tp},
            "EXTRA_AGG_ARGS": {"description": "Extra aggregated server arguments", "value": ""},
            "SGLANG_IMAGE": {"description": "SGLang container image", "value": container},
        }
    )

    if "/" in model_path and not model_path.startswith(("fs://", "file://")):
        model_uri = f"fs://{model_path}"
    else:
        model_uri = "fs://${{ variables.MODEL_PATH }}"
    artifacts = [{"name": "LOCAL_MODEL_PATH", "uri": model_uri}]

    agg_script = _build_sglang_server_script(agg_env, agg_cli, mode=None, extra_args_var="EXTRA_AGG_ARGS")

    template_path = Path(__file__).resolve().parent.parent / "sflow_sglang_disagg.yaml"
    with open(template_path, encoding="utf-8") as f:
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

    for task in template["workflow"]["tasks"]:
        t = dict(task)
        if t["name"] == "prefill_server":
            t = dict(task)
            t["name"] = "agg_server"
            t["script"] = agg_script
            t["operator"] = {
                "name": "dynamo_sglang",
                "ntasks": "${{ variables.AGG_TP_SIZE }}",
                "ntasks_per_node": "${{ [ variables.AGG_TP_SIZE, variables.GPUS_PER_NODE ] | min }}",
            }
            t["replicas"] = {"count": "${{ variables.NUM_AGG_SERVERS }}", "policy": "parallel"}
            t["resources"] = {"gpus": {"count": "${{ variables.AGG_TP_SIZE }}"}}
            out["workflow"]["tasks"].append(t)
        elif t["name"] == "decode_server":
            continue
        elif t["name"] == "benchmark":
            t = dict(task)
            t["depends_on"] = ["agg_server", "frontend_server"]
            out["workflow"]["tasks"].append(t)
        else:
            out["workflow"]["tasks"].append(t)

    _apply_multi_frontend(out, enable_multi, num_frontends, nginx_container, slurm_nodes)

    return out


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_trtllm_recipe(data: dict, path: Path) -> bool:
    """Check if a recipe is TRTLLM-based."""
    if "trtllm" in str(path):
        return True
    backend_type = data.get("backend", {}).get("type", "")
    return backend_type == "trtllm"


def _is_sglang_recipe(data: dict, path: Path) -> bool:
    """Check if a recipe is SGLang-based."""
    backend = data.get("backend", {})
    if backend.get("type") == "sglang":
        return True
    return bool(backend.get("sglang_config"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    recipes_dir = repo_root / "recipes"
    if not recipes_dir.is_dir():
        print(f"Recipes dir not found: {recipes_dir}", file=sys.stderr)
        return 1

    yaml_files = list(recipes_dir.rglob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files under {recipes_dir}", file=sys.stderr)
        return 1

    converted = 0
    for path in sorted(yaml_files):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue

        if not data:
            print(f"Skip {path}: empty file", file=sys.stderr)
            continue

        try:
            if _is_trtllm_recipe(data, path):
                sflow = convert_trtllm_recipe(path, data)
            elif _is_sglang_recipe(data, path):
                if _is_aggregated_recipe(data):
                    sflow = convert_sglang_agg_recipe(path, data)
                else:
                    sflow = convert_sglang_disagg_recipe(path, data)
            else:
                print(f"Skip {path}: unrecognized backend type", file=sys.stderr)
                continue
        except Exception as e:
            print(f"Convert failed {path}: {e}", file=sys.stderr)
            continue

        rel = path.relative_to(recipes_dir)
        out_path = repo_root / "sflow_recipes" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(
                sflow,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        converted += 1

    print(f"Converted {converted} recipes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
