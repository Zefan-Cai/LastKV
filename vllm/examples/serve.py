"""Serve a released LastKV checkpoint through vLLM's OpenAI-compatible server."""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="Any other arguments are passed to `vllm serve`.",
    )
    parser.add_argument("--model", default="ZefanCai/LastKV-LLM-8B-Base", help="HF repository or local checkpoint directory")
    parser.add_argument("--revision", default="main", help="main (8K) or 256k-step-001209")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="default: 32, or 4 with --eager")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--eager", action="store_true",
                        help="reference path: no fast kernels, CUDA graphs or prefix caching")
    return parser.parse_known_args()


def configure_environment(eager):
    # The fast-path switches are read when the backend is imported.
    defaults = {
        "RMARIN_FAST": "0" if eager else "1",
        "RMARIN_CUDAGRAPH": "0" if eager else "1",
        "RMARIN_FUSED_ONREAD": "1",
        "RM_SM90_PLACEHOLDER_BYPASS": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_PLUGINS": "lastkv",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_BATCH_INVARIANT": "0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def launch(args, extra, scheduler_class_path):
    local = Path(args.model).is_dir()
    if local:
        config_path = Path(args.model) / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(args.model, "config.json", revision=args.revision)
    chunk = json.loads(Path(config_path).read_text())["chunk_size"]
    # The scheduler admits at most one chunk-aligned prefill span per step.
    factor = int(os.environ.get("RMARIN_PREFILL_FACTOR", "1"))
    # The reference path reserves per-sequence scratch outside vLLM's memory budget.
    max_num_seqs = args.max_num_seqs or (4 if args.eager else 32)

    sys.argv = [
        "vllm", "serve", args.model,
        "--served-model-name", args.served_model_name or args.model,
        "--host", args.host, "--port", args.port, "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(max_num_seqs), "--block-size", "16",
        "--max-num-batched-tokens", str(factor * chunk + max_num_seqs),
        "--long-prefill-token-threshold", str(chunk), "--enable-chunked-prefill",
        "--scheduler-cls", scheduler_class_path, "--attention-backend", "CUSTOM",
        "--model-impl", "vllm", "--no-enable-flashinfer-autotune",
        "--disable-custom-all-reduce", "--kv-cache-memory-bytes", str(2 << 30),
    ]
    if not local:
        sys.argv += ["--revision", args.revision, "--tokenizer-revision", args.revision]
    if args.eager:
        sys.argv += ["--enforce-eager", "--no-enable-prefix-caching"]
    else:
        sys.argv += ["--enable-prefix-caching", "--async-scheduling", "--compilation-config",
                     json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})]
    sys.argv += extra

    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    ARGS, EXTRA = parse_args()
    configure_environment(ARGS.eager)

# vLLM runs the engine in a spawned process that re-imports this file and
# inherits the environment above, so registration happens at import time.
from lastkv.vllm.backends.recurrent_vllm.runner import install_training_semantic_linear_mode  # noqa: E402
from lastkv.vllm.backends.recurrent_vllm.vllm_compat import installed_runtime_profile  # noqa: E402
from lastkv.vllm.backends.rmarin_flashinfer.batching import SCHEDULER_CLASS_PATH  # noqa: E402
from lastkv.vllm.backends.rmarin_flashinfer.vllm_plugin import vllm_general_plugin  # noqa: E402
from lastkv.vllm.sm90_bypass import install as install_sm90_bypass  # noqa: E402

vllm_general_plugin()
install_training_semantic_linear_mode(runtime_profile=installed_runtime_profile())
install_sm90_bypass()

if __name__ == "__main__":
    launch(ARGS, EXTRA, SCHEDULER_CLASS_PATH)
