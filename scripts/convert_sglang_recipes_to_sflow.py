#!/usr/bin/env python3
"""
Convert srtslurm SGLang recipes under recipes/ (excluding trtllm/) to sflow format
following the structure of sflow_sglang_disagg.yaml.

Supports both disaggregated (prefill/decode) and aggregated modes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


class _LiteralBlockDumper(yaml.Dumper):
    """YAML dumper that uses literal block style (|) for multi-line strings."""

    pass


def _literal_str_representer(dumper: yaml.Dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralBlockDumper.add_representer(str, _literal_str_representer)

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
    """Variables common to all SGLang sflow recipes."""
    return {
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
        "EXTRA_FRONTEND_ARGS": {"description": "Extra frontend arguments", "value": ""},
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
    }


def _get_sglang_tp(config: dict[str, Any]) -> int:
    """Extract tensor-parallel size from sglang_config dict."""
    for key in ("tp-size", "tensor-parallel-size", "tp_size", "tensor_parallel_size"):
        if key in config:
            return int(config[key])
    return 1


def _get_sglang_dp(config: dict[str, Any]) -> int:
    """Extract data-parallel size from sglang_config dict."""
    for key in ("dp-size", "data-parallel-size", "dp_size", "data_parallel_size"):
        if key in config:
            return int(config[key])
    return 1


def _get_sglang_pp(config: dict[str, Any]) -> int:
    """Extract pipeline-parallel size from sglang_config dict."""
    for key in (
        "pp-size",
        "pipeline-parallel-size",
        "pp_size",
        "pipeline_parallel_size",
    ):
        if key in config:
            return int(config[key])
    return 1


def _has_dp_attention(config: dict[str, Any]) -> bool:
    """Check if dp-attention is enabled in sglang_config."""
    for key in ("enable-dp-attention", "enable_dp_attention"):
        if key in config:
            return bool(config[key])
    return False


def _is_aggregated_recipe(data: dict) -> bool:
    """Detect if a recipe uses aggregated mode (no prefill/decode split)."""
    resources = data.get("resources", {})
    if resources.get("agg_nodes") or resources.get("agg_workers"):
        return True
    sglang_config = data.get("backend", {}).get("sglang_config") or {}
    return bool(
        sglang_config.get("aggregated")
        and not sglang_config.get("prefill")
        and not sglang_config.get("decode")
    )


def _sglang_container(model: dict) -> str:
    """Resolve the SGLang container image from the model config."""
    return "lmsysorg/sglang:v0.5.8.post1-cu130"
    # container = model.get("container", "")
    # if not container:
    #     return "lmsysorg/sglang:v0.5.8.post1-cu130"
    # if "/" in container or ":" in container:
    #     return container.replace("#", "/", 1)
    # return container


def _get_frontend_config(data: dict, slurm_nodes: int) -> tuple[bool, int, str]:
    """Extract multi-frontend settings from recipe.

    Matches srt-slurm FrontendConfig defaults (enable_multiple_frontends=True)
    and topology rules: single-node disables multi-frontend regardless of config.

    Returns:
        (enable_multi, num_frontends, nginx_container)
    """
    fe = data.get("frontend", {})
    enable = fe.get("enable_multiple_frontends", False)
    num_additional = int(fe.get("num_additional_frontends", 0))
    nginx_container = fe.get("nginx_container", "nginx:1.27.4")

    # Match srt-slurm topology rule: single node → no multi-frontend
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


def _load_template_worker_script(task_name: str) -> list[str]:
    """Load a worker task's script from the sflow_sglang_disagg.yaml template."""
    return list(_get_template_task(task_name)["script"])


def _build_sglang_server_script(
    *,
    template_script: list[str],
    env_vars: dict[str, str],
    sglang_cli_args: list[str],
    mode: str | None = None,
    extra_args_var: str = "EXTRA_PREFILL_ARGS",
) -> list[str]:
    """Build a sglang worker script by merging template infrastructure with recipe args.

    - Preserves template infrastructure (NODES_PER_WORKER, MULTI_NODE_EXTRA_ARGS, etc.)
    - Inserts recipe env exports after "set -x"
    - Replaces the last script line (python command) with recipe CLI args,
      keeping --disaggregation-bootstrap-port and ${MULTI_NODE_EXTRA_ARGS} from template
    """
    script = list(template_script)

    env_lines = [f'export {k}="{v}"' for k, v in env_vars.items()]
    # Insert after "set -x" and "echo ${CUDA_VISIBLE_DEVICES}" (index 0, 1)
    for i, line in enumerate(env_lines):
        script.insert(2 + i, line)

    cmd_parts = [
        "python3 -m dynamo.sglang",
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}",
        "--served-model-name ${SERVED_MODEL_NAME}",
    ]
    if mode:
        cmd_parts.append(f"--disaggregation-mode {mode}")
    cmd_parts.extend(sglang_cli_args)
    cmd_parts.append(
        "--disaggregation-bootstrap-port"
        ' $(python3 -c "import socket; s=socket.socket();'
        " s.bind(('', 0)); print(s.getsockname()[1]); s.close()\")"
    )
    cmd_parts.append("${MULTI_NODE_EXTRA_ARGS}")
    cmd_parts.append("--host 0.0.0.0")
    cmd_parts.append(f"${{{extra_args_var}}}")

    script[-1] = " ".join(cmd_parts)
    return script


def _build_sglang_agg_server_script(
    env_vars: dict[str, str],
    sglang_cli_args: list[str],
    extra_args_var: str = "EXTRA_AGG_ARGS",
) -> list[str]:
    """Build an aggregated sglang worker script (no disaggregation-bootstrap-port)."""
    script: list[str] = ["set -x", "echo ${CUDA_VISIBLE_DEVICES}"]
    for k, v in env_vars.items():
        script.append(f'export {k}="{v}"')

    cmd_parts = [
        "python3 -m dynamo.sglang",
        "--model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}",
        "--served-model-name ${SERVED_MODEL_NAME}",
    ]
    cmd_parts.extend(sglang_cli_args)
    cmd_parts.append(f"${{{extra_args_var}}}")

    script.append(" ".join(cmd_parts))
    return script


# ---------------------------------------------------------------------------
# Sflow structure constants
# ---------------------------------------------------------------------------

_SFLOW_BACKENDS = [
    {
        "name": "slurm_cluster",
        "type": "slurm",
        "default": True,
        "time": "${{ variables.SLURM_TIMELIMIT }}",
        "nodes": "${{ variables.SLURM_NODES }}",
        "partition": "${{ variables.SLURM_PARTITION }}",
        "account": "${{ variables.SLURM_ACCOUNT }}",
        "gpus_per_node": "${{ variables.GPUS_PER_NODE }}",
    }
]

_SFLOW_WORKFLOW_VARIABLES = {
    "HEAD_NODE_IP": {
        "description": "Head node IP (resolved after allocation)",
        "value": "${{ backends.slurm_cluster.nodes[0].ip_address }}",
    },
    "ETCD_ENDPOINTS": {
        "description": "ETCD endpoints",
        "value": "${{ backends.slurm_cluster.nodes[0].ip_address }}:2379",
    },
    "NATS_SERVER": {
        "description": "NATS server URL",
        "value": "nats://${{ backends.slurm_cluster.nodes[0].ip_address }}:4222",
    },
}

_GPU_MONITOR_CMD = (
    "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,"
    "temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,"
    "memory.total,memory.used "
    "--format=csv,noheader,nounits -lms 2000 | "
    'while IFS= read -r input || [ -n "$input" ] ; '
    "do timestamp=$(date +%s%3N); "
    'printf "%s.%s,%s\\n" "${timestamp:0:10}" "${timestamp:10:3}" "${input}"; '
    "done "
    ">> ${SFLOW_TASK_OUTPUT_DIR}/gpu_monitor_node_${SLURM_NODEID}_${SLURMD_NODENAME}.log\n"
)

_ETCD_CMD = (
    'etcd --listen-client-urls "http://0.0.0.0:2379" '
    '--advertise-client-urls "http://0.0.0.0:2379" '
    '--listen-peer-urls "http://0.0.0.0:2380" '
    '--initial-advertise-peer-urls "http://${HEAD_NODE_IP}:2380" '
    '--initial-cluster "default=http://${HEAD_NODE_IP}:2380" '
    "--data-dir /tmp/etcd\n"
)

_BENCHMARK_CMD = (
    "aiperf profile "
    "--artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY} "
    "--model ${{ variables.SERVED_MODEL_NAME }} "
    "--tokenizer ${{ artifacts.LOCAL_MODEL_PATH.path }} "
    "--endpoint-type chat "
    "--endpoint /v1/chat/completions "
    "--streaming "
    "--url http://${{ variables.HEAD_NODE_IP }}:8000 "
    "--synthetic-input-tokens-mean ${{ variables.ISL }} "
    "--synthetic-input-tokens-stddev 0 "
    "--output-tokens-mean ${{ variables.OSL }} "
    "--output-tokens-stddev 0 "
    '--extra-inputs "max_tokens:${{ variables.OSL }}" '
    '--extra-inputs "min_tokens:${{ variables.OSL }}" '
    '--extra-inputs "ignore_eos:true" '
    '--extra-inputs "{\\"nvext\\":{\\"ignore_eos\\":true}}" '
    '--extra-inputs "repetition_penalty:1.0" '
    '--extra-inputs "temperature: 0.0" '
    "--concurrency ${CONCURRENCY} "
    "--request-count $((${{ variables.MULTI_ROUND }}*${CONCURRENCY})) "
    "--warmup-request-count ${CONCURRENCY} "
    "--num-dataset-entries $((${{ variables.MULTI_ROUND }}*${CONCURRENCY})) "
    "--random-seed 100 "
    "-H 'Authorization: Bearer NOT USED' "
    "-H 'Accept: text/event-stream' "
    "--record-processors 8 "
    "--ui simple\n"
)


# ---------------------------------------------------------------------------
# Sflow task builders
# ---------------------------------------------------------------------------


def _build_operators(*, enable_multi_frontend: bool) -> list[dict]:
    ops: list[dict] = [
        {
            "name": "dynamo_sglang",
            "type": "srun",
            "container_image": "${{ variables.DYNAMO_IMAGE }}",
            "container_writable": True,
            "mpi": "pmix",
        },
    ]
    if enable_multi_frontend:
        ops.append(
            {
                "name": "nginx",
                "type": "srun",
                "container_image": "${{ variables.NGINX_IMAGE }}",
                "container_writable": True,
            }
        )
    ops.append(
        {
            "name": "aiperf",
            "type": "srun",
            "container_image": "${{ variables.AIPERF_IMAGE }}",
            "container_writable": True,
            "mpi": "pmix",
        }
    )
    return ops


def _load_template() -> dict:
    """Load and cache the sflow_sglang_disagg.yaml template."""
    if not hasattr(_load_template, "_cache"):
        template_path = (
            Path(__file__).resolve().parent.parent / "sflow_sglang_disagg.yaml"
        )
        with open(template_path, encoding="utf-8") as f:
            _load_template._cache = yaml.safe_load(f)
    return _load_template._cache


def _get_template_task(task_name: str) -> dict:
    """Get a task dict from the template by name."""
    for task in _load_template()["workflow"]["tasks"]:
        if task["name"] == task_name:
            return task
    raise ValueError(f"Task {task_name!r} not found in template")


def _build_infra_tasks() -> list[dict]:
    """Build infrastructure tasks: load_image, install_aiperf, gpu_monitor, nats, etcd.

    nats_server and etcd_server scripts are loaded from the template to preserve
    install-check infrastructure.
    """
    nats_tmpl = _get_template_task("nats_server")
    etcd_tmpl = _get_template_task("etcd_server")

    return [
        {
            "name": "load_image",
            "operator": {
                "name": "dynamo_sglang",
                "ntasks": "${{ variables.SLURM_NODES }}",
                "ntasks_per_node": 1,
            },
            "script": ['echo "Image Loaded"', "sleep 3600"],
            "probes": {
                "readiness": {
                    "log_watch": {
                        "regex_pattern": "Image Loaded",
                        "match_count": "${{ variables.SLURM_NODES }}",
                    },
                    "timeout": 1200,
                    "interval": 2,
                }
            },
        },
        {
            "name": "install_aiperf",
            "operator": {"name": "aiperf", "ntasks_per_node": 1},
            "resources": {"nodes": {"indices": [0]}},
            "script": [
                "pip install aiperf==0.3.0",
                'echo "AIPerf installed"',
                "sleep 3600",
            ],
            "probes": {
                "readiness": {
                    "log_watch": {"regex_pattern": "AIPerf installed"},
                    "timeout": 1200,
                    "interval": 2,
                }
            },
        },
        {
            "name": "gpu_monitor",
            "operator": {"name": "dynamo_sglang", "ntasks_per_node": 1},
            "resources": {"nodes": {"count": "${{ variables.SLURM_NODES }}"}},
            "script": ['echo "Starting gpu monitor"', _GPU_MONITOR_CMD],
            "probes": {
                "readiness": {
                    "log_watch": {"regex_pattern": "Starting gpu monitor"},
                    "timeout": 60,
                    "interval": 2,
                }
            },
            "depends_on": ["load_image", "install_aiperf"],
        },
        {
            "name": "nats_server",
            "operator": "dynamo_sglang",
            "script": list(nats_tmpl["script"]),
            "resources": {"nodes": {"indices": [0]}},
            "probes": {
                "readiness": {
                    "tcp_port": {"port": 4222},
                    "timeout": 60,
                    "interval": 2,
                }
            },
            "depends_on": ["load_image", "install_aiperf"],
        },
        {
            "name": "etcd_server",
            "operator": "dynamo_sglang",
            "script": list(etcd_tmpl["script"]),
            "resources": {"nodes": {"indices": [0]}},
            "probes": {
                "readiness": {
                    "tcp_port": {"port": 2379},
                    "timeout": 60,
                    "interval": 2,
                }
            },
            "depends_on": ["load_image", "install_aiperf"],
        },
    ]


def _build_nginx_task(*, gpu_type: str) -> dict:
    nginx_bin = (
        "/usr/sbin/nginx" if gpu_type.startswith(("gb200", "gb300")) else "nginx"
    )
    return {
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


def _build_frontend_task() -> dict:
    """Build frontend_server task, preserving install-check script from template."""
    fe_tmpl = _get_template_task("frontend_server")
    # Keep install-check lines from template, replace the launch command
    tmpl_script = list(fe_tmpl["script"])
    tmpl_script[-1] = (
        "python3 -m dynamo.frontend --http-port ${{ variables.FRONTEND_PORT }} ${{ variables.EXTRA_FRONTEND_ARGS }}"
    )

    return {
        "name": "frontend_server",
        "operator": "dynamo_sglang",
        "replicas": {
            "count": "${{ variables.NUM_FRONTENDS }}",
            "policy": "parallel",
        },
        "script": tmpl_script,
        "resources": {"nodes": {"count": 1}},
        "probes": {
            "readiness": {
                "tcp_port": {"port": "${{ variables.FRONTEND_PORT }}"},
                "timeout": 120,
                "interval": 5,
            }
        },
        "depends_on": ["nats_server", "etcd_server"],
    }


def _build_worker_task(
    *,
    name: str,
    tp_var: str,
    num_servers_var: str,
    script: list[str],
) -> dict:
    return {
        "name": name,
        "operator": {
            "name": "dynamo_sglang",
            # "ntasks": f"${{{{ variables.{tp_var} }}}}",
            "ntasks_per_node": 1,
        },
        "replicas": {
            "count": f"${{{{ variables.{num_servers_var} }}}}",
            "policy": "parallel",
        },
        "script": script,
        "resources": {"gpus": {"count": f"${{{{ variables.{tp_var} }}}}"}},
        "depends_on": ["frontend_server"],
        "probes": {
            "readiness": {
                "log_watch": {"regex_pattern": "orker handler initialized"},
                "timeout": 900,
                "interval": 10,
            },
            "failure": {
                "log_watch": {"regex_pattern": "Traceback (most recent call last)"},
                "interval": 10,
            },
        },
        "retries": {"count": 3, "interval": 30, "backoff": 2},
    }


def _build_benchmark_task(*, depends_on: list[str]) -> dict:
    return {
        "name": "benchmark",
        "operator": {"name": "aiperf", "ntasks": 1},
        "script": [
            "set -x",
            "export COLUMNS=200",
            _BENCHMARK_CMD,
            'echo "Benchmarking finished"',
        ],
        "resources": {"nodes": {"indices": [0]}},
        "replicas": {"variables": ["CONCURRENCY"], "policy": "sequential"},
        "depends_on": depends_on,
    }


# ---------------------------------------------------------------------------
# SGLang conversion
# ---------------------------------------------------------------------------


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
    ctx_dp = _get_sglang_dp(prefill_cfg)
    ctx_pp = _get_sglang_pp(prefill_cfg)
    ctx_dp_attn = _has_dp_attention(prefill_cfg)
    gen_tp = _get_sglang_tp(decode_cfg)
    gen_dp = _get_sglang_dp(decode_cfg)
    gen_pp = _get_sglang_pp(decode_cfg)
    gen_dp_attn = _has_dp_attention(decode_cfg)

    extra_node = 1 if infra.get("etcd_nats_dedicated_node") else 0
    slurm_nodes = prefill_nodes + decode_nodes + extra_node

    model_path = model.get("path", "model")
    container = _sglang_container(model)
    gpu_type = str(resources.get("gpu_type", "")).lower()

    isl = int(benchmark.get("isl", 1024))
    osl = int(benchmark.get("osl", 1024))
    concurrency_domain = _concurrency_domain(benchmark.get("concurrencies", "50"))

    served_model_name = (
        prefill_cfg.get("served-model-name")
        or prefill_cfg.get("served_model_name")
        or model_path.replace("/", "-")
    )

    enable_multi, num_frontends, nginx_container = _get_frontend_config(
        data, slurm_nodes
    )
    frontend_extra = _get_frontend_extra_args(data)

    for cfg in (prefill_cfg, decode_cfg):
        for k in (
            "served-model-name",
            "served_model_name",
            "disaggregation-mode",
            "disaggregation_mode",
            "disaggregation-bootstrap-port",
            "disaggregation_bootstrap_port",
        ):
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
    if frontend_extra:
        variables["EXTRA_FRONTEND_ARGS"]["value"] = frontend_extra

    variables.update(
        {
            "NUM_CTX_SERVERS": {
                "description": "Number of context/prefill servers",
                "value": prefill_workers,
            },
            "CTX_TP_SIZE": {
                "description": "Context tensor parallel size",
                "type": "integer",
                "value": ctx_tp,
            },
            "CTX_DP_SIZE": {
                "description": "Context data parallel size",
                "type": "integer",
                "value": ctx_dp,
            },
            "CTX_PP_SIZE": {
                "description": "Context pipeline parallel size",
                "type": "integer",
                "value": ctx_pp,
            },
            "CTX_ENABLE_ATTENTION_DP": {
                "description": "Context enable attention DP",
                "value": "--enable-dp-attention" if ctx_dp_attn else "",
            },
            "NUM_GEN_SERVERS": {
                "description": "Number of generation/decode servers",
                "value": decode_workers,
            },
            "GEN_TP_SIZE": {
                "description": "Generation tensor parallel size",
                "type": "integer",
                "value": gen_tp,
            },
            "GEN_DP_SIZE": {
                "description": "Generation data parallel size",
                "type": "integer",
                "value": gen_dp,
            },
            "GEN_PP_SIZE": {
                "description": "Generation pipeline parallel size",
                "type": "integer",
                "value": gen_pp,
            },
            "GEN_ENABLE_ATTENTION_DP": {
                "description": "Generation enable attention DP",
                "value": "--enable-dp-attention" if gen_dp_attn else "",
            },
            "EXTRA_PREFILL_ARGS": {
                "description": "Extra prefill arguments",
                "value": "",
            },
            "EXTRA_DECODE_ARGS": {"description": "Extra decode arguments", "value": ""},
            "DYNAMO_IMAGE": {
                "description": "SGLang container image",
                "value": container,
            },
        }
    )

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

    if "/" in model_path and not model_path.startswith(("fs://", "file://")):
        model_uri = f"fs://{model_path}"
    else:
        model_uri = "fs://${{ variables.MODEL_PATH }}"
    artifacts: list[dict] = [{"name": "LOCAL_MODEL_PATH", "uri": model_uri}]

    if enable_multi:
        actual_frontends = min(num_frontends, slurm_nodes)
        nginx_cfg = _generate_nginx_config(actual_frontends, slurm_nodes, 8180)
        artifacts.append(
            {"name": "NGINX_CONFIG", "uri": "file://nginx.conf", "content": nginx_cfg}
        )

    prefill_tmpl_script = _load_template_worker_script("prefill_server")
    decode_tmpl_script = _load_template_worker_script("decode_server")

    prefill_script = _build_sglang_server_script(
        template_script=prefill_tmpl_script,
        env_vars=prefill_env,
        sglang_cli_args=prefill_cli,
        mode="prefill",
        extra_args_var="EXTRA_PREFILL_ARGS",
    )
    decode_script = _build_sglang_server_script(
        template_script=decode_tmpl_script,
        env_vars=decode_env,
        sglang_cli_args=decode_cli,
        mode="decode",
        extra_args_var="EXTRA_DECODE_ARGS",
    )

    tasks = _build_infra_tasks()
    if enable_multi:
        tasks.append(_build_nginx_task(gpu_type=gpu_type))
    tasks.append(_build_frontend_task())
    tasks.append(
        _build_worker_task(
            name="prefill_server",
            tp_var="CTX_TP_SIZE",
            num_servers_var="NUM_CTX_SERVERS",
            script=prefill_script,
        )
    )
    tasks.append(
        _build_worker_task(
            name="decode_server",
            tp_var="GEN_TP_SIZE",
            num_servers_var="NUM_GEN_SERVERS",
            script=decode_script,
        )
    )

    bench_deps = ["prefill_server", "decode_server"]
    if enable_multi:
        bench_deps.append("nginx_server")
    else:
        bench_deps.append("frontend_server")
    tasks.append(_build_benchmark_task(depends_on=bench_deps))

    return {
        "version": "0.1",
        "variables": variables,
        "artifacts": artifacts,
        "backends": _SFLOW_BACKENDS,
        "operators": _build_operators(enable_multi_frontend=enable_multi),
        "workflow": {
            "name": name.replace(" ", "_").replace('"', ""),
            "timeout": "115m",
            "variables": _SFLOW_WORKFLOW_VARIABLES,
            "tasks": tasks,
        },
    }


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
        agg_cfg.get("served-model-name")
        or agg_cfg.get("served_model_name")
        or model_path.replace("/", "-")
    )

    enable_multi, num_frontends, nginx_container = _get_frontend_config(
        data, slurm_nodes
    )
    frontend_extra = _get_frontend_extra_args(data)

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
    if frontend_extra:
        variables["EXTRA_FRONTEND_ARGS"]["value"] = frontend_extra

    variables.update(
        {
            "NUM_AGG_SERVERS": {
                "description": "Number of aggregated servers",
                "value": agg_workers,
            },
            "AGG_TP_SIZE": {
                "description": "Aggregated tensor parallel size",
                "value": agg_tp,
            },
            "EXTRA_AGG_ARGS": {
                "description": "Extra aggregated server arguments",
                "value": "",
            },
            "DYNAMO_IMAGE": {
                "description": "SGLang container image",
                "value": container,
            },
        }
    )

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

    if "/" in model_path and not model_path.startswith(("fs://", "file://")):
        model_uri = f"fs://{model_path}"
    else:
        model_uri = "fs://${{ variables.MODEL_PATH }}"
    artifacts: list[dict] = [{"name": "LOCAL_MODEL_PATH", "uri": model_uri}]

    if enable_multi:
        actual_frontends = min(num_frontends, slurm_nodes)
        nginx_cfg = _generate_nginx_config(actual_frontends, slurm_nodes, 8180)
        artifacts.append(
            {"name": "NGINX_CONFIG", "uri": "file://nginx.conf", "content": nginx_cfg}
        )

    agg_script = _build_sglang_agg_server_script(
        agg_env, agg_cli, extra_args_var="EXTRA_AGG_ARGS"
    )

    tasks = _build_infra_tasks()
    if enable_multi:
        tasks.append(_build_nginx_task(gpu_type=gpu_type))
    tasks.append(_build_frontend_task())
    tasks.append(
        _build_worker_task(
            name="agg_server",
            tp_var="AGG_TP_SIZE",
            num_servers_var="NUM_AGG_SERVERS",
            script=agg_script,
        )
    )

    bench_deps = ["agg_server"]
    if enable_multi:
        bench_deps.append("nginx_server")
    else:
        bench_deps.append("frontend_server")
    tasks.append(_build_benchmark_task(depends_on=bench_deps))

    return {
        "version": "0.1",
        "variables": variables,
        "artifacts": artifacts,
        "backends": _SFLOW_BACKENDS,
        "operators": _build_operators(enable_multi_frontend=enable_multi),
        "workflow": {
            "name": name.replace(" ", "_").replace('"', ""),
            "timeout": "115m",
            "variables": _SFLOW_WORKFLOW_VARIABLES,
            "tasks": tasks,
        },
    }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_sglang_recipe(data: dict) -> bool:
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

    trtllm_dir = recipes_dir / "trtllm"
    yaml_files = [
        p
        for p in sorted(recipes_dir.rglob("*.yaml"))
        if not p.is_relative_to(trtllm_dir)
    ]
    if not yaml_files:
        print(f"No YAML files under {recipes_dir} (excluding trtllm)", file=sys.stderr)
        return 1

    converted = 0
    for path in yaml_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue

        if not data:
            print(f"Skip {path}: empty file", file=sys.stderr)
            continue

        if not _is_sglang_recipe(data):
            print(f"Skip {path}: not an SGLang recipe", file=sys.stderr)
            continue

        try:
            if _is_aggregated_recipe(data):
                sflow = convert_sglang_agg_recipe(path, data)
            else:
                sflow = convert_sglang_disagg_recipe(path, data)
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
                Dumper=_LiteralBlockDumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        converted += 1

    print(f"Converted {converted} SGLang recipes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
