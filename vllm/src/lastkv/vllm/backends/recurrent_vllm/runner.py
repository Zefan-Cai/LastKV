"""Bounded recurrent-vLLM parity runner.

P0 intentionally runs one active sequence per engine.  This makes vLLM's
existing ``max_num_batched_tokens=chunk_size`` chunked-prefill behavior an
exact scheduling boundary without copying its unstable Scheduler internals.
Scale out is data parallel: launch one process/GPU over disjoint requests.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .contract import (
    RecurrentProbeReport,
    probe_model_directory,
    sha256_file,
    validate_engine_contract,
)
from .registry import assert_recurrent_model_owner, register_recurrent_model
from .vllm_compat import (
    VLLM_0_11_0,
    VllmRuntimeProfile,
    configure_vllm_0_11_attention_environment,
    install_vllm_0_11_dummy_forward_guard,
    installed_runtime_profile,
    require_runtime_profile,
    versioned_engine_kwargs,
)


PARITY_TOP_LOGPROBS = 20
PARITY_ARTIFACT_SCHEMA_VERSION = 2
TRAINING_NUMERIC_MODE = "linear_only_batch_invariant_fa2_native_reductions_v1"
CANONICAL_HF_NUMERIC_CONTRACT = {
    "contract_id": "canonical_training_hf_bf16_v1",
    "role": "independent_training_hf_full_prefix_reference",
    "linear_primitive": "torch.nn.Linear",
    "attention_kernel": "external_flash_attention_2",
    "rmsnorm_reduction": "training_eager_float32_mean",
    "sampler_log_softmax": "torch_float32",
}
CANDIDATE_NUMERIC_ALIGNMENT_CONTRACT = {
    "contract_id": TRAINING_NUMERIC_MODE,
    "linear_primitive": "vllm.linear_batch_invariant",
    "attention_kernel": "vllm_vendored_flash_attention_2",
    "flash_attention_version": 2,
    "flash_attention_version_source": (
        "engine.llm_engine.vllm_config.attention_config.flash_attn_version"
    ),
    "rmsnorm_reduction": "torch_native_eager_float32_mean",
    "sampler_log_softmax": "torch_native",
    "global_vllm_batch_invariant": False,
}
VLLM_0_11_DIAGNOSTIC_NUMERIC_MODE = (
    "vllm_0_11_torch_2_8_native_linear_fa2_diagnostic_v1"
)
VLLM_0_11_CANDIDATE_NUMERIC_ALIGNMENT_CONTRACT = {
    "contract_id": VLLM_0_11_DIAGNOSTIC_NUMERIC_MODE,
    "linear_primitive": "torch.nn.functional.linear",
    "attention_kernel": "vllm_vendored_flash_attention_2",
    "flash_attention_version": 2,
    "flash_attention_version_source": (
        "vllm.attention.utils.fa_utils.get_flash_attn_version"
    ),
    "rmsnorm_reduction": "torch_native_eager_float32_mean",
    "sampler_log_softmax": "torch_native",
    "global_vllm_batch_invariant": False,
}
RECURRENT_ARTIFACT_CONTRACT_KEYS = frozenset(
    {"chunk_size", "v_gap", "deep_rescale_mode"}
)
SUPPORTED_DEEP_RESCALE_MODES = frozenset({"v", "vnorm", "vnormqk"})


@dataclass(frozen=True)
class PilotEngineConfig:
    attention_backend: str = "FLASH_ATTN"
    block_size: int = 16
    dtype: str = "bfloat16"
    enable_flashinfer_autotune: bool = False
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    seed: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_num_seqs: int = 1
    numeric_mode: str = TRAINING_NUMERIC_MODE

    def validate(self, *, chunk_size: int) -> None:
        if self.attention_backend != "FLASH_ATTN":
            raise ValueError("P0 recurrent cache mutation requires FLASH_ATTN")
        if self.enable_flashinfer_autotune:
            raise ValueError("P0 disables FlashInfer autotune on Blackwell")
        validate_engine_contract(
            chunk_size=chunk_size,
            block_size=self.block_size,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_model=None,
        )
        if self.dtype != "bfloat16":
            raise ValueError("P0 parity is restricted to bfloat16")
        if not 0.1 <= self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be in [0.1, 1.0)")
        if self.max_model_len < chunk_size:
            raise ValueError("max_model_len must contain at least one chunk")
        if self.max_num_seqs != 1:
            raise ValueError(
                "P0 requires max_num_seqs=1 until the boundary scheduler hook lands"
            )
        if self.numeric_mode != TRAINING_NUMERIC_MODE:
            raise ValueError(
                "recurrent P0 requires the training-semantic numeric mode "
                f"{TRAINING_NUMERIC_MODE!r}"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")


def force_in_process_vllm(
    *,
    numeric_mode: str = TRAINING_NUMERIC_MODE,
    runtime_profile: VllmRuntimeProfile | None = None,
) -> None:
    """Freeze process and sampler choices before importing vLLM.

    The Evaluation checkout is an immutable source snapshot, not an installed
    distribution.  A spawned EngineCore therefore cannot rediscover its
    ``vllm.general_plugins`` entry point.  TP=PP=1 does not need the extra
    process, so P0 refuses a conflicting environment value and forces vLLM's
    documented in-process path before vLLM is imported.
    """

    settings = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        # This package also ships the independent RMarin FlashInfer P0 model
        # under the same prepared-export architecture name.  vLLM loads
        # general plugins after EngineArgs construction and silently replaces
        # an existing ModelRegistry owner.  Pin the legacy runner to its own
        # entry point so the two implementations can never overwrite one
        # another implicitly.
        "VLLM_PLUGINS": "recurrent_marin",
        # vLLM 0.25's FlashInfer sampler AOTI warmup fails on the deployed
        # Blackwell runtime before the first request. Parity is greedy, so use
        # the native deterministic sampler and keep FlashAttention for KV.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Global batch-invariant mode also replaces mean/softmax/log_softmax.
        # That silently changes the eager training RMSNorm reduction selected
        # by model.py.  Recurrent P0 installs only the deterministic linear
        # primitive below and leaves the training-semantic reductions intact.
        "VLLM_BATCH_INVARIANT": "0",
    }
    if (
        runtime_profile is not None
        and runtime_profile.vllm_version == VLLM_0_11_0
    ):
        # The 0.11 cache contract depends on V1's per-layer metadata and
        # independent name-keyed KV bindings. V0 collapses attention modules
        # with the same numeric decoder-layer index onto one cache tensor.
        settings["VLLM_USE_V1"] = "1"
    if numeric_mode != TRAINING_NUMERIC_MODE:
        raise ValueError(f"unsupported recurrent numeric mode {numeric_mode!r}")
    for name, required in settings.items():
        configured = os.environ.get(name)
        if configured not in {None, required}:
            raise ValueError(
                f"recurrent P0 requires {name}={required}, observed={configured!r}"
            )
        os.environ[name] = required
    if runtime_profile is not None:
        configure_vllm_0_11_attention_environment(
            runtime_profile,
            os.environ,
        )


def _default_runtime_profile() -> VllmRuntimeProfile:
    """Keep dependency-free unit calls bound to the established 0.25 path."""

    return require_runtime_profile(
        vllm_version="0.25.0",
        torch_version="unconstrained",
    )


def install_training_semantic_linear_mode(
    *,
    runtime_profile: VllmRuntimeProfile | None = None,
) -> None:
    """Use deterministic GEMMs without replacing training reductions.

    vLLM's global ``VLLM_BATCH_INVARIANT=1`` mode is too broad for this
    backend: besides linear layers it overrides ``mean``, ``softmax``, and
    ``log_softmax`` process-wide.  The recurrent training implementation
    deliberately evaluates RMSNorm through an eager float32 ``mean``.  Keep
    that reduction (and sampler normalization) native while selecting the
    exact same deterministic linear primitive that global mode would use.

    The pilot is one model per process, so this process-local class patch is
    installed before the engine constructs any layers and is intentionally
    idempotent.
    """

    if os.environ.get("VLLM_BATCH_INVARIANT") != "0":
        raise ValueError(
            "linear-only recurrent numeric mode requires VLLM_BATCH_INVARIANT=0"
        )

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    profile = runtime_profile or _default_runtime_profile()
    if profile.vllm_version == VLLM_0_11_0:
        # vLLM 0.11 predates the batch-invariant module.  Keep its native
        # torch-2.8 linear primitive and bind that distinct diagnostic
        # contract into artifacts; it must not masquerade as the 0.25 mode.
        from torch.nn.functional import linear as linear_primitive
    else:
        from lastkv.vllm.backends.rmarin_flashinfer import fastmode

        if fastmode.LINEAR == "cublas":
            # SPEED: standard cuBLAS GEMM (Transformer-baseline numerics class).
            from torch.nn.functional import linear as linear_primitive
        else:
            from vllm.model_executor.layers.batch_invariant import (
                linear_batch_invariant as linear_primitive,
            )

    marker = "_recurrent_training_semantic_linear_mode"
    for method_class in (UnquantizedLinearMethod, UnquantizedEmbeddingMethod):
        installed_mode = getattr(method_class, marker, None)
        if installed_mode == profile.vllm_version:
            continue
        if installed_mode is not None:
            raise RuntimeError(
                "unquantized linear method already carries a different "
                "recurrent runtime patch"
            )

        def apply(self, layer, value, bias=None):
            del self
            return linear_primitive(value, layer.weight, bias)

        method_class.apply = apply
        setattr(method_class, marker, profile.vllm_version)


def numeric_alignment_metadata(
    *,
    flash_attention_version: int,
    runtime_profile: VllmRuntimeProfile | None = None,
) -> dict[str, Any]:
    """Return the exact runtime contract bound into every candidate artifact."""

    if (
        isinstance(flash_attention_version, bool)
        or not isinstance(flash_attention_version, int)
        or flash_attention_version != 2
    ):
        raise RuntimeError(
            "recurrent training-semantic numeric mode requires "
            f"FlashAttention 2, observed {flash_attention_version!r}"
        )
    if os.environ.get("VLLM_BATCH_INVARIANT") != "0":
        raise RuntimeError("global vLLM batch-invariant reductions are enabled")
    profile = runtime_profile or _default_runtime_profile()
    if profile.vllm_version == VLLM_0_11_0:
        return dict(VLLM_0_11_CANDIDATE_NUMERIC_ALIGNMENT_CONTRACT)
    return dict(CANDIDATE_NUMERIC_ALIGNMENT_CONTRACT)


def engine_numeric_alignment_metadata(
    engine: Any,
    *,
    runtime_profile: VllmRuntimeProfile | None = None,
    vllm_0_11_fa_version_reader: Any | None = None,
) -> dict[str, Any]:
    """Read back the constructed engine's FA version and bind it as evidence."""

    profile = runtime_profile or getattr(
        engine,
        "_recurrent_runtime_profile",
        _default_runtime_profile(),
    )
    if profile.vllm_version == VLLM_0_11_0:
        if vllm_0_11_fa_version_reader is None:
            from vllm.attention.utils.fa_utils import get_flash_attn_version

            vllm_0_11_fa_version_reader = get_flash_attn_version
        flash_attention_version = vllm_0_11_fa_version_reader()
    else:
        try:
            flash_attention_version = (
                engine.llm_engine.vllm_config.attention_config.flash_attn_version
            )
        except AttributeError as error:
            raise RuntimeError(
                "constructed recurrent engine has no FlashAttention version readback"
            ) from error
    return numeric_alignment_metadata(
        flash_attention_version=flash_attention_version,
        runtime_profile=profile,
    )


def validate_recurrent_artifact_contract(value: Any) -> dict[str, Any]:
    """Validate the compact recurrent contract embedded in parity artifacts."""

    if not isinstance(value, dict) or set(value) != RECURRENT_ARTIFACT_CONTRACT_KEYS:
        raise ValueError(
            "recurrent_contract must contain exactly "
            "chunk_size, v_gap, and deep_rescale_mode"
        )
    chunk_size = value["chunk_size"]
    v_gap = value["v_gap"]
    deep_rescale_mode = value["deep_rescale_mode"]
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("recurrent_contract.chunk_size must be a positive integer")
    if isinstance(v_gap, bool) or not isinstance(v_gap, int):
        raise ValueError("recurrent_contract.v_gap must be an integer")
    if (
        not isinstance(deep_rescale_mode, str)
        or deep_rescale_mode not in SUPPORTED_DEEP_RESCALE_MODES
    ):
        raise ValueError(
            "recurrent_contract.deep_rescale_mode must be one of "
            f"{sorted(SUPPORTED_DEEP_RESCALE_MODES)}"
        )
    return {
        "chunk_size": chunk_size,
        "v_gap": v_gap,
        "deep_rescale_mode": deep_rescale_mode,
    }


def ensure_fresh_report(
    report: RecurrentProbeReport,
    model_dir: str | Path,
) -> tuple[Path, Any]:
    directory = Path(model_dir).expanduser().resolve()
    if Path(report.model_dir).resolve() != directory:
        raise ValueError("compatibility report belongs to a different model directory")
    config_path = directory / "config.json"
    if report.config_sha256 != sha256_file(config_path):
        raise ValueError("compatibility report is stale for config.json")
    current = probe_model_directory(directory, check_runtime=True)
    if not current.ready_for_pilot or current.contract is None:
        evidence = [
            item.evidence for item in current.findings if item.severity == "error"
        ]
        raise ValueError(f"model export does not satisfy recurrent pilot: {evidence}")
    if report.contract != current.contract:
        raise ValueError("compatibility report recurrent contract is stale")
    return directory, current.contract


def verify_export_receipt(
    receipt_path: str | Path,
    *,
    model_dir: Path,
    config_sha256: str,
) -> dict[str, Any]:
    path = Path(receipt_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    prepared = payload.get("prepared_model", {})
    if payload.get("schema_version") != 1 or payload.get("status") != "success":
        raise ValueError("recurrent export receipt is not successful schema v1")
    legacy_receipt = "artifact_kind" not in payload
    artifact_kind = (
        payload.get("artifact_kind") or "saved_recurrent_training_checkpoint"
    )
    if artifact_kind not in {
        "saved_recurrent_training_checkpoint",
        "logical_recurrent_initialization_snapshot",
    }:
        raise ValueError("recurrent export receipt has an unknown artifact kind")
    if artifact_kind == "logical_recurrent_initialization_snapshot":
        checkpoint = payload.get("checkpoint") or {}
        initialized = prepared.get("initialized_model_keys")
        if (
            payload.get("step") != 0
            or payload.get("variant") != "model"
            or checkpoint.get("role") != "initial_load_source"
            or not isinstance(initialized, list)
            or not initialized
            or prepared.get("initialized_model_key_count") != len(initialized)
        ):
            raise ValueError("logical recurrent initialization receipt is incomplete")
    if Path(str(prepared.get("path", ""))).resolve() != model_dir:
        raise ValueError("recurrent export receipt belongs to a different model")
    if prepared.get("config_sha256") != config_sha256:
        raise ValueError("recurrent export receipt config digest differs")
    if prepared.get("missing_model_keys") or prepared.get("unexpected_model_keys"):
        raise ValueError("recurrent export receipt has model key errors")
    training_source = payload.get("training_source")
    if not legacy_receipt or training_source is not None:
        if not isinstance(training_source, dict):
            raise ValueError("recurrent export receipt lost training source")
        for path_key, digest_key in (
            ("config_path", "config_sha256"),
            ("modeling_path", "modeling_sha256"),
        ):
            source_path = Path(str(training_source.get(path_key, "")))
            if not source_path.is_file() or sha256_file(
                source_path
            ) != training_source.get(digest_key):
                raise ValueError(f"recurrent training source changed: {path_key}")
    files = prepared.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("recurrent export receipt has no file manifest")
    for item in files:
        file_path = model_dir / str(item.get("path", ""))
        if not file_path.is_file() or file_path.stat().st_size != item.get("bytes"):
            raise ValueError(f"prepared export file changed: {file_path.name}")
        if sha256_file(file_path) != item.get("sha256"):
            raise ValueError(f"prepared export digest changed: {file_path.name}")
    return payload


def build_engine(
    model_dir: str | Path,
    report: RecurrentProbeReport,
    export_receipt: str | Path,
    config: PilotEngineConfig,
):
    directory, contract = ensure_fresh_report(report, model_dir)
    receipt = verify_export_receipt(
        export_receipt,
        model_dir=directory,
        config_sha256=report.config_sha256,
    )
    config.validate(chunk_size=contract.chunk_size)
    runtime_profile = installed_runtime_profile()
    force_in_process_vllm(
        numeric_mode=config.numeric_mode,
        runtime_profile=runtime_profile,
    )
    register_recurrent_model()

    from vllm import LLM

    install_vllm_0_11_dummy_forward_guard(runtime_profile)
    install_training_semantic_linear_mode(runtime_profile=runtime_profile)
    engine_kwargs = {
        "model": str(directory),
        "tokenizer": str(directory),
        "trust_remote_code": False,
        "model_impl": "vllm",
        "dtype": config.dtype,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "block_size": config.block_size,
        "kv_cache_dtype": "auto",
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": contract.chunk_size,
        "long_prefill_token_threshold": contract.chunk_size,
        "max_num_seqs": config.max_num_seqs,
        "enforce_eager": True,
        "seed": config.seed,
        "disable_log_stats": False,
        **versioned_engine_kwargs(
            runtime_profile,
            attention_backend=config.attention_backend,
            enable_flashinfer_autotune=config.enable_flashinfer_autotune,
        ),
    }
    engine = LLM(
        **engine_kwargs,
    )
    # EngineArgs loads general plugins during construction. Verify again after
    # that load point so a future plugin-loader change cannot silently replace
    # the model selected above.
    assert_recurrent_model_owner()
    engine._recurrent_runtime_profile = runtime_profile
    # Read the immutable engine config after construction; this fails closed
    # if a future runtime ignores or rewrites the explicit FA2 pin.
    engine_numeric_alignment_metadata(engine)
    return engine, receipt


def _request_digest(requests: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(
        list(requests), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_requests(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("request file must contain a non-empty JSON list")
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"request {index} must be an object")
        identifier = item.get("id")
        prompt = item.get("prompt")
        prompt_token_ids = item.get("prompt_token_ids")
        max_tokens = item.get("max_tokens", 16)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"request {index} id must be a non-empty string")
        if identifier in identifiers:
            raise ValueError(f"duplicate request id {identifier!r}")
        if (prompt is None) == (prompt_token_ids is None):
            raise ValueError(
                f"request {identifier!r} must have exactly one of prompt or "
                "prompt_token_ids"
            )
        if prompt is not None and (not isinstance(prompt, str) or not prompt):
            raise ValueError(f"request {identifier!r} prompt must be non-empty")
        if prompt_token_ids is not None:
            if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
                raise ValueError(
                    f"request {identifier!r} prompt_token_ids must be non-empty"
                )
            if any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in prompt_token_ids
            ):
                raise ValueError(
                    f"request {identifier!r} prompt_token_ids must be "
                    "non-negative integers"
                )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError(f"request {identifier!r} max_tokens must be positive")
        try:
            temperature = float(item.get("temperature", 0.0))
            top_p = float(item.get("top_p", 1.0))
            seed = int(item.get("seed", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"request {identifier!r} sampling fields are invalid"
            ) from error
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"request {identifier!r} temperature must be finite and non-negative"
            )
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError(f"request {identifier!r} top_p must be within (0, 1]")
        if seed < 0:
            raise ValueError(f"request {identifier!r} seed must be non-negative")
        identifiers.add(identifier)
        result.append(
            {
                "id": identifier,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }
        )
    return result


def load_requests(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _validate_requests(json.load(handle))


def run_requests(
    *,
    model_dir: str | Path,
    report: RecurrentProbeReport,
    export_receipt: str | Path,
    requests: Sequence[dict[str, Any]],
    engine_config: PilotEngineConfig,
) -> dict[str, Any]:
    requests = _validate_requests(list(requests))
    engine, receipt = build_engine(model_dir, report, export_receipt, engine_config)
    numeric_alignment = engine_numeric_alignment_metadata(engine)

    from vllm import SamplingParams, TokensPrompt

    prompts = [
        item["prompt"]
        if item["prompt"] is not None
        else TokensPrompt(prompt_token_ids=item["prompt_token_ids"])
        for item in requests
    ]
    parameters = [
        SamplingParams(
            temperature=item["temperature"],
            top_p=item["top_p"],
            max_tokens=item["max_tokens"],
            seed=item["seed"],
            logprobs=PARITY_TOP_LOGPROBS,
        )
        for item in requests
    ]
    started = time.monotonic()
    outputs = engine.generate(prompts, parameters, use_tqdm=False)
    elapsed = time.monotonic() - started
    if len(outputs) != len(requests):
        raise RuntimeError("vLLM output count differs from request count")

    rows = []
    total_prompt = 0
    total_generated = 0
    for request, output in zip(requests, outputs, strict=True):
        if len(output.outputs) != 1:
            raise RuntimeError("P0 requires exactly one completion per request")
        completion = output.outputs[0]
        prompt_ids = list(output.prompt_token_ids)
        token_ids = list(completion.token_ids)
        if completion.logprobs is None or len(completion.logprobs) != len(token_ids):
            raise RuntimeError("vLLM selected-token logprobs are missing")
        selected_logprobs = []
        top_token_logprobs = []
        for token_id, token_logprobs in zip(
            token_ids, completion.logprobs, strict=True
        ):
            selected = token_logprobs.get(token_id)
            if selected is None:
                raise RuntimeError(f"vLLM logprobs omit selected token {token_id}")
            selected_logprobs.append(float(selected.logprob))
            ranked = sorted(
                (
                    {
                        "token_id": int(candidate_id),
                        "logprob": float(candidate.logprob),
                        "rank": (
                            int(candidate.rank) if candidate.rank is not None else None
                        ),
                    }
                    for candidate_id, candidate in token_logprobs.items()
                ),
                key=lambda item: (-item["logprob"], item["token_id"]),
            )[:PARITY_TOP_LOGPROBS]
            if not ranked or ranked[0]["token_id"] != token_id:
                raise RuntimeError("greedy selected token is not vLLM top rank")
            top_token_logprobs.append(ranked)
        total_prompt += len(prompt_ids)
        total_generated += len(token_ids)
        rows.append(
            {
                "id": request["id"],
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": token_ids,
                "selected_token_logprobs": selected_logprobs,
                "top_token_logprobs": top_token_logprobs,
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "cumulative_logprob": completion.cumulative_logprob,
            }
        )

    return {
        "schema_version": PARITY_ARTIFACT_SCHEMA_VERSION,
        "status": "pilot_candidate_not_authoritative",
        "backend": "recurrent_vllm_compat_paged_cache",
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "model_config_sha256": report.config_sha256,
        "export_receipt": {
            "path": str(Path(export_receipt).expanduser().resolve()),
            "sha256": sha256_file(export_receipt),
            "checkpoint_metadata_sha256": receipt["checkpoint"]["metadata_sha256"],
            "checkpoint_variant": receipt["variant"],
        },
        "request_sha256": _request_digest(requests),
        "recurrent_contract": validate_recurrent_artifact_contract(
            {
                "chunk_size": report.contract.chunk_size,
                "v_gap": report.contract.v_gap,
                "deep_rescale_mode": report.contract.deep_rescale_mode,
            }
        ),
        "engine": {
            **asdict(engine_config),
            # The 0.11 port uses torch-2.8's native F.linear and therefore has
            # a distinct diagnostic numeric contract from the 0.25 backend.
            "numeric_mode": numeric_alignment["contract_id"],
            "vllm_version": importlib.metadata.version("vllm"),
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "max_num_batched_tokens": report.contract.chunk_size,
            "long_prefill_token_threshold": report.contract.chunk_size,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "numeric_alignment": numeric_alignment,
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "prompt_tokens": total_prompt,
            "generated_tokens": total_generated,
            "tokens_per_second": (
                (total_prompt + total_generated) / elapsed if elapsed else None
            ),
        },
        "results": rows,
    }


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


__all__ = [
    "PilotEngineConfig",
    "CANDIDATE_NUMERIC_ALIGNMENT_CONTRACT",
    "VLLM_0_11_CANDIDATE_NUMERIC_ALIGNMENT_CONTRACT",
    "VLLM_0_11_DIAGNOSTIC_NUMERIC_MODE",
    "CANONICAL_HF_NUMERIC_CONTRACT",
    "PARITY_ARTIFACT_SCHEMA_VERSION",
    "TRAINING_NUMERIC_MODE",
    "engine_numeric_alignment_metadata",
    "force_in_process_vllm",
    "install_training_semantic_linear_mode",
    "numeric_alignment_metadata",
    "build_engine",
    "ensure_fresh_report",
    "load_requests",
    "run_requests",
    "validate_recurrent_artifact_contract",
    "verify_export_receipt",
    "write_json_atomic",
]
