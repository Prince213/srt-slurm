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
    return yaml.dump(
        obj, default_flow_style=False, allow_unicode=True, sort_keys=False
    ).rstrip()


# ---------------------------------------------------------------------------
# Multi-frontend helpers (shared with sglang converter)
# ---------------------------------------------------------------------------


def _get_frontend_config(data: dict, slurm_nodes: int) -> tuple[bool, int, str]:
    """Extract multi-frontend settings from recipe.

    Matches srt-slurm FrontendConfig defaults (enable_multiple_frontends=True)
    and topology rules: single-node disables multi-frontend regardless of config.
    """
    fe = data.get("frontend", {})
    enable = fe.get("enable_multiple_frontends", False)
    num_additional = int(fe.get("num_additional_frontends", 0))
    nginx_container = (
        fe.get("nginx_container", "nginx:1.27.4")
        if fe.get("nginx_container", "nginx:1.27.4") != "nginx-sqsh"
        else "nginx:1.27.4"
    )

    if slurm_nodes <= 1:
        enable = False

    return enable, num_additional + 1, nginx_container


def _get_frontend_extra_args(data: dict) -> str:
    """Extract extra frontend CLI args from recipe frontend.args dict."""
    fe = data.get("frontend", {})
    args_dict = fe.get("args", {})
    if not args_dict:
        return ""
    parts: list[str] = []
    for key, value in args_dict.items():
        flag = f"--{key}" if key.startswith("-") else f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        elif value is not None:
            parts.extend([flag, str(value)])
    return " ".join(parts)


def _generate_nginx_config(
    num_frontends: int, slurm_nodes: int, frontend_port: int = 8180
) -> str:
    """Generate nginx config content with sflow node IP expressions."""
    actual_count = min(num_frontends, slurm_nodes)
    lines = [
        "worker_processes auto;",
        "http {",
        "    access_log off;",
        "    upstream backend_servers {",
    ]
    for i in range(0, actual_count):
        lines.append(
            f"        server ${{{{ backends.slurm_cluster.nodes[{i}].ip_address }}}}:{frontend_port};"
        )
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

    extra_node = 1 if infra.get("etcd_nats_dedicated_node") else 0
    slurm_nodes = prefill_nodes + decode_nodes + extra_node

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

    enable_multi, num_frontends, nginx_container = _get_frontend_config(
        data, slurm_nodes
    )
    frontend_extra = _get_frontend_extra_args(data)

    variables = {
        "SLURM_ACCOUNT": {"description": "SLURM account", "value": "rogliu"},
        "SLURM_PARTITION": {"description": "SLURM partition", "value": "gamoraq"},
        "SLURM_TIMELIMIT": {"description": "SLURM time limit", "value": 120},
        "GPUS_PER_NODE": {"description": "GPUs per node", "value": gpus_per_node},
        "SLURM_NODES": {
            "description": "Number of nodes",
            "type": "integer",
            "value": slurm_nodes,
        },
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
            "value": float(
                prefill_cfg.get("kv_cache_config", {}).get(
                    "free_gpu_memory_fraction", 0.9
                )
            ),
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
            "value": float(
                decode_cfg.get("kv_cache_config", {}).get(
                    "free_gpu_memory_fraction", 0.9
                )
            ),
        },
        "GEN_ENABLE_ATTENTION_DP": {
            "description": "Generation enable attention DP",
            "value": decode_cfg.get("enable_attention_dp", False),
        },
        "EXTRA_FRONTEND_ARGS": {
            "description": "Extra frontend arguments",
            "value": frontend_extra if frontend_extra else "",
        },
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

    # Multi-frontend variables
    if enable_multi:
        actual_frontends = min(num_frontends, slurm_nodes)
        variables["NUM_FRONTENDS"] = {
            "description": "Number of frontend instances",
            "value": actual_frontends,
        }
        variables["FRONTEND_PORT"] = {
            "description": "Frontend listening port (8180 behind nginx, 8000 direct)",
            "value": 8180,
        }
        variables["NGINX_IMAGE"] = {
            "description": "Nginx container image",
            "value": nginx_container,
        }
    else:
        variables["NUM_FRONTENDS"] = {
            "description": "Number of frontend instances",
            "value": 1,
        }
        variables["FRONTEND_PORT"] = {
            "description": "Frontend listening port",
            "value": 8000,
        }

    # Artifacts
    if "/" in model_path and not model_path.startswith(("fs://", "file://")):
        model_uri = f"fs://{model_path}"
    else:
        model_uri = "fs://${{ variables.MODEL_PATH }}"
    artifacts: list[dict] = [
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

    if enable_multi:
        actual_frontends = min(num_frontends, slurm_nodes)
        nginx_cfg = _generate_nginx_config(actual_frontends, slurm_nodes, 8180)
        artifacts.append(
            {"name": "NGINX_CONFIG", "uri": "file://nginx.conf", "content": nginx_cfg}
        )

    # Build prefill/decode scripts
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

    # Load template
    sample_path = Path(__file__).resolve().parent.parent / "sflow_trtllm_disagg.yaml"
    with open(sample_path, encoding="utf-8") as f:
        template = yaml.safe_load(f)

    # Build operators with optional nginx
    operators = list(template["operators"])
    if enable_multi:
        operators.append(
            {
                "name": "nginx",
                "type": "srun",
                "container_image": "${{ variables.NGINX_IMAGE }}",
                "container_writable": True,
            }
        )

    out = {
        "version": "0.1",
        "variables": variables,
        "artifacts": artifacts,
        "backends": template["backends"],
        "operators": operators,
        "workflow": {
            "name": name.replace(" ", "_").replace('"', ""),
            "timeout": "115m",
            "variables": template["workflow"]["variables"],
            "tasks": [],
        },
    }

    # Build tasks from template with multi-frontend modifications
    nginx_bin = (
        "/usr/sbin/nginx" if gpu_type.startswith(("gb200", "gb300")) else "nginx"
    )

    for task in template["workflow"]["tasks"]:
        t = dict(task)

        if t["name"] == "frontend_server":
            t["replicas"] = {
                "count": "${{ variables.NUM_FRONTENDS }}",
                "policy": "parallel",
            }
            t["script"] = [
                "python3 -m dynamo.frontend"
                " --http-port ${{ variables.FRONTEND_PORT }}"
                " ${{ variables.EXTRA_FRONTEND_ARGS }}"
            ]
            t["resources"] = {"nodes": {"count": 1}}
            t["probes"] = {
                "readiness": {
                    "tcp_port": {"port": "${{ variables.FRONTEND_PORT }}"},
                    "timeout": 120,
                    "interval": 5,
                }
            }
            if enable_multi:
                # Insert nginx_server task before frontend
                out["workflow"]["tasks"].append(
                    {
                        "name": "nginx_server",
                        "operator": "nginx",
                        "script": [
                            f"{nginx_bin} -c ${{{{ artifacts.NGINX_CONFIG.path }}}} -g 'daemon off;'"
                        ],
                        "resources": {"nodes": {"indices": [0]}},
                        "probes": {
                            "readiness": {
                                "tcp_port": {"port": 8000},
                                "timeout": 60,
                                "interval": 2,
                            }
                        },
                        "depends_on": ["frontend_server"],
                    }
                )
            out["workflow"]["tasks"].append(t)
        elif t["name"] == "prefill_server":
            t["script"] = prefill_script
            out["workflow"]["tasks"].append(t)
        elif t["name"] == "decode_server":
            t["script"] = decode_script
            out["workflow"]["tasks"].append(t)
        elif t["name"] == "benchmark":
            if enable_multi:
                deps = list(t.get("depends_on", []))
                if "frontend_server" in deps:
                    deps.remove("frontend_server")
                if "nginx_server" not in deps:
                    deps.append("nginx_server")
                t["depends_on"] = deps
            out["workflow"]["tasks"].append(t)
        else:
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

    converted = 0
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
        converted += 1

    print(f"Converted {converted} TRTLLM recipes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
