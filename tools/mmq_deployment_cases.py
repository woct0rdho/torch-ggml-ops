"""Neutral metadata for exact MMQ deployment correctness cases.

This module intentionally has no dependency on ``bench``.  The selected keys
come from the checked-in catalogs and deployment inventory; tensor names are
the stable checkpoint representatives used to materialize those exact shapes.
"""

import os
import sysconfig
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tools.ggtensile.deployment import DeploymentKey
from tools.mmq_deployment_spec import kernels

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_KERNEL_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"


@dataclass(frozen=True)
class TensorSource:
    model_family: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class DeploymentCase:
    operation: str
    identity: str
    symbol: str
    artifact: str
    quant_type: str
    rows: int
    out_features: int
    in_features: int
    tensor_source: TensorSource
    key: DeploymentKey

    def to_mapping(self) -> dict[str, object]:
        """Return report-safe route metadata without importing ``bench``."""
        return {
            "operation": self.operation,
            "identity": self.identity,
            "symbol": self.symbol,
            "artifact": self.artifact,
            "quant_type": self.quant_type,
            "rows": self.rows,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "model_family": self.tensor_source.model_family,
            "tensors": list(self.tensor_source.names),
            "ggtensile": "direct deployed HSACO",
            "hip": "optional legacy HIP-control HSACO",
        }


_DENSE_TENSORS = {
    ("Q3_K", 512, 2048): TensorSource("qwen", ("blk.3.attn_k.weight",)),
    ("Q3_K", 8192, 2048): TensorSource("qwen", ("blk.3.attn_q.weight",)),
    ("Q4_K", 512, 2048): TensorSource("qwen", ("blk.5.ffn_gate_shexp.weight",)),
    ("Q4_K", 2048, 512): TensorSource("qwen", ("blk.5.ffn_down_shexp.weight",)),
    ("Q4_K", 2048, 4096): TensorSource("qwen", ("blk.3.attn_output.weight",)),
    ("Q4_K", 8192, 2048): TensorSource("qwen", ("blk.39.attn_q.weight",)),
    ("Q5_K", 512, 2048): TensorSource("qwen", ("blk.0.ffn_gate_shexp.weight",)),
    ("Q5_K", 2048, 512): TensorSource("qwen", ("blk.0.ffn_down_shexp.weight",)),
    ("Q6_K", 248320, 2048): TensorSource("qwen", ("output.weight",)),
    ("Q8_0", 1024, 4096): TensorSource("deepseek", ("blk.0.attn_q_a.weight",)),
    ("Q8_0", 32768, 1024): TensorSource("deepseek", ("blk.0.attn_q_b.weight",)),
    ("Q8_0", 512, 4096): TensorSource("deepseek", ("blk.0.attn_kv.weight",)),
    ("Q8_0", 4096, 8192): TensorSource("deepseek", ("blk.0.attn_output_b.weight",)),
    ("Q8_0", 2048, 4096): TensorSource("deepseek", ("blk.0.ffn_gate_shexp.weight",)),
    ("Q8_0", 4096, 2048): TensorSource("deepseek", ("blk.0.ffn_down_shexp.weight",)),
    ("Q8_0", 129280, 4096): TensorSource("deepseek", ("output.weight",)),
}

_GROUPED_TENSORS = {
    ("single", "Q4_K", 2048, 512): TensorSource(
        "qwen", ("blk.2.ffn_down_exps.weight",)
    ),
    ("single", "Q5_K", 2048, 512): TensorSource(
        "qwen", ("blk.0.ffn_down_exps.weight",)
    ),
    ("single", "IQ2_S", 2048, 512): TensorSource(
        "qwen", ("blk.10.ffn_down_exps.weight",)
    ),
    ("single", "Q2_K", 4096, 2048): TensorSource(
        "deepseek", ("blk.0.ffn_down_exps.weight",)
    ),
    ("pair", "Q3_K", 512, 2048): TensorSource(
        "qwen", ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight")
    ),
    ("pair", "IQ2_S", 512, 2048): TensorSource(
        "qwen", ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight")
    ),
    ("pair", "IQ2_XXS", 2048, 4096): TensorSource(
        "deepseek", ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight")
    ),
    ("fixed", "Q8_0", 1024, 4096): TensorSource(
        "deepseek", ("blk.0.attn_output_a.weight",)
    ),
}


def _integer(mapping: Mapping[str, object], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int:
        raise ValueError(f"deployment key field {name} is not an integer")
    return value


def _geometry(item) -> tuple[str, int, int, int]:
    mapping = item.key.to_mapping()
    contract = mapping["ProblemContract"]
    problem = mapping["Problem"]
    if not isinstance(contract, Mapping) or not isinstance(problem, Mapping):
        raise TypeError("deployment key has invalid problem mappings")
    quant = str(contract["quant_type"])
    if item.operation == "OrdinaryForward":
        return (
            quant,
            _integer(problem, "m"),
            _integer(problem, "n"),
            _integer(problem, "k"),
        )
    if item.operation == "OrdinaryBackward":
        return (
            quant,
            _integer(problem, "m"),
            _integer(problem, "k"),
            _integer(problem, "n"),
        )
    if item.operation == "GroupedBackward":
        return (
            quant,
            _integer(problem, "m"),
            _integer(problem, "k"),
            _integer(problem, "n"),
        )
    row_name = "tokens" if item.operation.startswith("Fixed") else "aggregate_rows"
    rows = _integer(problem, row_name)
    out_name = "output_features" if "output_features" in contract else "out_features"
    in_name = "input_features" if "input_features" in contract else "in_features"
    return quant, rows, _integer(contract, out_name), _integer(contract, in_name)


def _source(item, quant: str, out_features: int, in_features: int) -> TensorSource:
    if item.operation in {"OrdinaryForward", "OrdinaryBackward"}:
        result = _DENSE_TENSORS.get((quant, out_features, in_features))
    else:
        kind = (
            "fixed"
            if item.operation.startswith("Fixed")
            else "pair"
            if item.operation.endswith("Pair")
            else "single"
        )
        result = _GROUPED_TENSORS.get((kind, quant, out_features, in_features))
    if result is None:
        raise ValueError(
            f"no tensor source for {item.operation} {quant} "
            f"N={out_features} K={in_features}"
        )
    return result


@cache
def public_deployment_cases() -> tuple[DeploymentCase, ...]:
    result: list[DeploymentCase] = []
    for item in kernels():
        if item.key is None or item.operation is None:
            continue
        quant, rows, out_features, in_features = _geometry(item)
        result.append(
            DeploymentCase(
                item.operation,
                item.key.hash,
                item.symbol,
                item.filename,
                quant,
                rows,
                out_features,
                in_features,
                _source(item, quant, out_features, in_features),
                item.key,
            )
        )
    return tuple(result)


def case_for_key(operation: str, key: DeploymentKey) -> DeploymentCase:
    for case in public_deployment_cases():
        if case.operation == operation and case.identity == key.hash:
            return case
    raise ValueError(f"{operation} solution {key.hash} is not a public deployment key")


def public_artifact_path(case: DeploymentCase, root: Path | None = None) -> Path:
    configured = root or Path(
        os.environ.get("GGTENSILE_PUBLIC_CODE_OBJECT_ROOT", PUBLIC_KERNEL_DIR)
    )
    if configured.is_file():
        return configured
    roots = (configured, configured / "gfx1151")
    for candidate_root in roots:
        candidate = candidate_root / case.artifact
        if candidate.is_file():
            return candidate
    return roots[0] / case.artifact


def model_path(source: TensorSource) -> Path:
    if source.model_family == "qwen":
        return Path(
            os.environ.get(
                "GGUF_MMQ_TEST_MODEL",
                "~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf",
            )
        ).expanduser()
    return Path(
        os.environ.get(
            "GGUF_DEEPSEEK_MMQ_TEST_MODEL", "~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"
        )
    ).expanduser()


def operation_counts() -> dict[str, int]:
    return {
        operation: sum(
            case.operation == operation for case in public_deployment_cases()
        )
        for operation in sorted({case.operation for case in public_deployment_cases()})
    }


def hip_control_root() -> Path | None:
    configured = os.environ.get("GGTENSILE_HIP_CONTROL_ROOT")
    if configured:
        return Path(configured)
    root = ROOT / "build/mmq_hip_controls/gfx1151"
    if root.is_dir():
        return root
    purelib = Path(sysconfig.get_paths()["purelib"])
    installed = purelib / "torch_ggml_ops/kernels/gfx1151"
    return installed if installed.is_dir() else None
