"""Fail-closed contract for the recurrent Marin vLLM pilot.

This probe validates only facts that can be established without executing
model code.  A successful probe is permission to attempt a bounded parity
pilot, never evidence that the backend is numerically correct.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .vllm_compat import (
    VLLM_0_25_0,
    require_runtime_profile,
)


SCHEMA_VERSION = 1
RECURRENT_MODEL_TYPE = "custom_recurrent_marin_akv254_reusekv"
RECURRENT_ARCHITECTURE = "CustomRecurrentMarinAKV254ReuseKVForCausalLM"
VLLM_MODEL_CLASS = "RecurrentMarinAKV254ReuseKVForCausalLM"
# Kept for callers that bind the production 0.25 runtime explicitly.
SUPPORTED_VLLM_VERSION = VLLM_0_25_0
STATIC_CANDIDATE_STATUS = "recurrent_vllm_native_candidate_unverified"
BLOCKED_STATUS = "blocked"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    evidence: str | None = None


@dataclass(frozen=True)
class RecurrentModelContract:
    chunk_size: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    v_gap: int
    source_layer_by_target: tuple[int, ...]
    source_layers: tuple[int, ...]
    use_qk_norm: bool
    use_nope: bool
    deep_kv_rescale: bool
    deep_rescale_mode: str
    deep_scale_frozen: bool
    deep_v_scale_init: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_layer_by_target"] = list(self.source_layer_by_target)
        value["source_layers"] = list(self.source_layers)
        value["deep_v_scale_init"] = [list(row) for row in self.deep_v_scale_init]
        return value


@dataclass
class RecurrentProbeReport:
    schema_version: int
    model_dir: str
    config_sha256: str | None
    static_status: str
    runtime_status: str
    runtime_dependencies: dict[str, str | None]
    ready_for_pilot: bool
    runtime_verified: bool
    ready_for_authoritative_evaluation: bool
    contract: RecurrentModelContract | None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contract"] = self.contract.to_dict() if self.contract else None
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecurrentProbeReport":
        raw_contract = value.get("contract")
        contract = None
        if raw_contract is not None:
            contract = RecurrentModelContract(
                chunk_size=raw_contract["chunk_size"],
                num_hidden_layers=raw_contract["num_hidden_layers"],
                hidden_size=raw_contract["hidden_size"],
                num_attention_heads=raw_contract["num_attention_heads"],
                num_key_value_heads=raw_contract["num_key_value_heads"],
                head_dim=raw_contract["head_dim"],
                v_gap=raw_contract["v_gap"],
                source_layer_by_target=tuple(raw_contract["source_layer_by_target"]),
                source_layers=tuple(raw_contract["source_layers"]),
                use_qk_norm=raw_contract["use_qk_norm"],
                use_nope=raw_contract["use_nope"],
                deep_kv_rescale=raw_contract["deep_kv_rescale"],
                deep_rescale_mode=raw_contract["deep_rescale_mode"],
                deep_scale_frozen=raw_contract["deep_scale_frozen"],
                deep_v_scale_init=tuple(
                    tuple(row) for row in raw_contract["deep_v_scale_init"]
                ),
            )
        return cls(
            schema_version=value["schema_version"],
            model_dir=value["model_dir"],
            config_sha256=value.get("config_sha256"),
            static_status=value["static_status"],
            runtime_status=value["runtime_status"],
            runtime_dependencies=dict(value.get("runtime_dependencies", {})),
            ready_for_pilot=bool(value["ready_for_pilot"]),
            runtime_verified=bool(value["runtime_verified"]),
            ready_for_authoritative_evaluation=bool(
                value["ready_for_authoritative_evaluation"]
            ),
            contract=contract,
            findings=[Finding(**item) for item in value.get("findings", [])],
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RecurrentProbeReport":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _boolean(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _scale_matrix(
    value: Any,
    *,
    num_layers: int,
    num_kv_heads: int,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scalar = _finite_float(value, "deep_v_scale_init")
        return tuple(
            tuple(scalar for _ in range(num_kv_heads)) for _ in range(num_layers)
        )
    if not isinstance(value, list) or len(value) != num_layers:
        raise ValueError(
            "deep_v_scale_init must be a scalar or one entry per hidden layer"
        )
    rows: list[tuple[float, ...]] = []
    for layer_index, entry in enumerate(value):
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            scalar = _finite_float(entry, f"deep_v_scale_init[{layer_index}]")
            rows.append(tuple(scalar for _ in range(num_kv_heads)))
            continue
        if not isinstance(entry, list) or len(entry) != num_kv_heads:
            raise ValueError(
                f"deep_v_scale_init[{layer_index}] must contain "
                f"{num_kv_heads} KV-head values"
            )
        rows.append(
            tuple(
                _finite_float(item, f"deep_v_scale_init[{layer_index}][{head}]")
                for head, item in enumerate(entry)
            )
        )
    return tuple(rows)


def contract_from_config(config: Mapping[str, Any]) -> RecurrentModelContract:
    if config.get("model_type") != RECURRENT_MODEL_TYPE:
        raise ValueError(f"model_type must be exactly {RECURRENT_MODEL_TYPE!r}")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or RECURRENT_ARCHITECTURE not in architectures
    ):
        raise ValueError(f"architectures must contain {RECURRENT_ARCHITECTURE!r}")

    chunk_size = _positive_int(config, "chunk_size")
    num_layers = _positive_int(config, "num_hidden_layers")
    hidden_size = _positive_int(config, "hidden_size")
    num_heads = _positive_int(config, "num_attention_heads")
    num_kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    if hidden_size != num_heads * head_dim:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim")
    if num_heads % num_kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    v_gap = config.get("v_gap")
    if isinstance(v_gap, bool) or not isinstance(v_gap, int):
        raise ValueError("the native pilot requires integer v_gap")
    source_map = tuple(
        max(min(layer + v_gap, num_layers - 1), 0) for layer in range(num_layers)
    )

    use_qk_norm = _boolean(config, "use_qk_norm", False)
    use_nope = _boolean(config, "use_nope", False)
    if use_nope:
        raise ValueError("use_nope=true has not passed the native pilot contract")

    deep_rescale = _boolean(config, "deep_kv_rescale", False)
    deep_mode = config.get("deep_rescale_mode", "v")
    if deep_mode not in {"v", "vnorm", "vnormqk"}:
        raise ValueError("native pilot supports deep_rescale_mode v, vnorm, or vnormqk")
    deep_frozen = _boolean(config, "deep_scale_frozen", False)
    scale = _scale_matrix(
        config.get("deep_v_scale_init", 1.0),
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
    )

    return RecurrentModelContract(
        chunk_size=chunk_size,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        v_gap=v_gap,
        source_layer_by_target=source_map,
        source_layers=tuple(sorted(set(source_map))),
        use_qk_norm=use_qk_norm,
        use_nope=use_nope,
        deep_kv_rescale=deep_rescale,
        deep_rescale_mode=deep_mode,
        deep_scale_frozen=deep_frozen,
        deep_v_scale_init=scale,
    )


def _has_weights(model_dir: Path) -> bool:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((model_dir / name).is_file() for name in names) or any(
        model_dir.glob("model-*.safetensors")
    )


def _runtime_versions() -> tuple[str, dict[str, str | None], list[Finding]]:
    versions: dict[str, str | None] = {}
    for name in ("vllm", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    findings: list[Finding] = []
    if not all(versions.values()):
        findings.append(
            Finding(
                "runtime_dependencies_missing",
                "error",
                "vLLM, torch, and transformers must all be installed",
                json.dumps(versions, sort_keys=True),
            )
        )
        return "dependencies_missing", versions, findings
    try:
        profile = require_runtime_profile(
            vllm_version=str(versions["vllm"]),
            torch_version=str(versions["torch"]),
        )
    except RuntimeError as error:
        findings.append(
            Finding(
                "vllm_runtime_unpinned",
                "error",
                "pilot runtime is outside the exact supported matrix",
                str(error),
            )
        )
        return "vllm_version_mismatch", versions, findings
    versions["recurrent_api_profile"] = profile.vllm_version
    return "dependencies_present_unverified", versions, findings


def probe_model_directory(
    model_dir: str | Path,
    *,
    check_runtime: bool = False,
) -> RecurrentProbeReport:
    directory = Path(model_dir).expanduser().resolve()
    findings: list[Finding] = []
    config_path = directory / "config.json"
    config_digest = None
    contract = None

    try:
        config_digest = sha256_file(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("config.json root must be an object")
        contract = contract_from_config(config)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        findings.append(
            Finding(
                "model_config_invalid",
                "error",
                "recurrent model config failed the native contract",
                str(error),
            )
        )

    if not _has_weights(directory):
        findings.append(
            Finding(
                "weights_missing",
                "error",
                "prepared HF weight payload was not found",
                str(directory),
            )
        )

    runtime_status = "not_checked"
    versions = {"vllm": None, "torch": None, "transformers": None}
    if check_runtime:
        runtime_status, versions, runtime_findings = _runtime_versions()
        findings.extend(runtime_findings)

    static_ok = not any(
        item.severity == "error"
        for item in findings
        if item.code not in {"runtime_dependencies_missing", "vllm_runtime_unpinned"}
    )
    runtime_ok = (
        not check_runtime or runtime_status == "dependencies_present_unverified"
    )
    ready = static_ok and runtime_ok
    if static_ok:
        findings.append(
            Finding(
                "gpu_parity_required",
                "warning",
                "static compatibility does not establish recurrent numerical parity",
            )
        )

    return RecurrentProbeReport(
        schema_version=SCHEMA_VERSION,
        model_dir=str(directory),
        config_sha256=config_digest,
        static_status=STATIC_CANDIDATE_STATUS if static_ok else BLOCKED_STATUS,
        runtime_status=runtime_status,
        runtime_dependencies=versions,
        ready_for_pilot=ready,
        runtime_verified=False,
        ready_for_authoritative_evaluation=False,
        contract=contract,
        findings=findings,
    )


def validate_engine_contract(
    *,
    chunk_size: int,
    block_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    speculative_model: str | None,
) -> None:
    """Reject engine settings that invalidate the cache rewrite algorithm."""

    if block_size < 1 or block_size % 16 or chunk_size % block_size:
        raise ValueError(
            "block_size must be a positive multiple of 16 that divides chunk_size"
        )
    if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
        raise ValueError("P0 recurrent parity is restricted to TP=1 and PP=1")
    if enable_prefix_caching:
        raise ValueError("prefix caching is unsafe because recurrent KV pages mutate")
    if not enforce_eager:
        raise ValueError("P0 recurrent cache rewrite requires enforce_eager=true")
    if speculative_model is not None:
        raise ValueError("speculative decoding is not supported by the recurrent pilot")
