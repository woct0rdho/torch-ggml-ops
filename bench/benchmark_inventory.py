"""Benchmark-side view of the complete public MMQ deployment inventory."""

import argparse
import json
import os
import sys
import sysconfig
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ggtensile.deployment import DeploymentKey
from tools.mmq_deployment_spec import kernels

KERNEL_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"


@dataclass(frozen=True)
class OperationBenchmark:
    script: str
    public_api: str
    baseline: str


OPERATIONS = {
    "OrdinaryForward": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_mmq_fwd.py",
        "torch_ggml_ops.mmq",
        "torch.mm",
    ),
    "OrdinaryBackward": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_mmq_bwd.py",
        "torch.autograd.grad(torch_ggml_ops.mmq(...))",
        "torch.mm",
    ),
    "GroupedForward": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_grouped_mmq_fwd.py",
        "torch_ggml_ops.grouped_mmq",
        "BF16 routed matmul",
    ),
    "GroupedForwardPair": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_grouped_mmq_fwd_pair.py",
        "torch_ggml_ops.grouped_mmq_pair",
        "BF16 routed matmul pair",
    ),
    "GroupedBackward": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_grouped_mmq_bwd.py",
        "torch.autograd.grad(torch_ggml_ops.grouped_mmq(...))",
        "BF16 routed matmul",
    ),
    "GroupedBackwardPair": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_grouped_mmq_bwd_pair.py",
        "torch.autograd.grad(torch_ggml_ops.grouped_mmq_pair(...))",
        "BF16 routed matmul pair",
    ),
    "FixedGroupedForward": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_fixed_grouped_mmq_fwd.py",
        "torch_ggml_ops.fixed_grouped_mmq",
        "torch.bmm",
    ),
    "FixedGroupedBackward": OperationBenchmark(
        "bench/ggtensile/benchmark_ggtensile_fixed_grouped_mmq_bwd.py",
        "torch.autograd.grad(torch_ggml_ops.fixed_grouped_mmq(...))",
        "torch.bmm",
    ),
}


@dataclass(frozen=True)
class TensorSource:
    model_family: str
    names: tuple[str, ...]


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


@dataclass(frozen=True)
class BenchmarkCase:
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

    @property
    def benchmark(self) -> OperationBenchmark:
        return OPERATIONS[self.operation]

    def to_mapping(self) -> dict[str, object]:
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
            "script": self.benchmark.script,
            "public_api": self.benchmark.public_api,
            "ggtensile": "direct deployed HSACO",
            "hip": "optional legacy HIP-control HSACO",
            "baseline": self.benchmark.baseline,
        }


def _mapping_value(mapping: object, name: str) -> object:
    if not isinstance(mapping, dict) or name not in mapping:
        raise ValueError(f"deployment key is missing {name}")
    return mapping[name]


def _integer_value(mapping: object, name: str) -> int:
    value = _mapping_value(mapping, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"deployment key field {name} is not an integer")
    return value


def _case_geometry(item) -> tuple[str, int, int, int]:
    mapping = item.key.to_mapping()
    contract = _mapping_value(mapping, "ProblemContract")
    problem = _mapping_value(mapping, "Problem")
    if not isinstance(contract, dict) or not isinstance(problem, dict):
        raise TypeError("deployment key has invalid problem mappings")
    quant_type = _mapping_value(contract, "quant_type")
    if item.operation in {"OrdinaryForward", "OrdinaryBackward"}:
        if item.operation == "OrdinaryBackward":
            # Backward ProblemSize stores the transposed packed-weight contract:
            # N is grad-input width and K is grad-output width.
            out_features = _integer_value(problem, "k")
            in_features = _integer_value(problem, "n")
        else:
            out_features = _integer_value(problem, "n")
            in_features = _integer_value(problem, "k")
        return (
            str(quant_type),
            _integer_value(problem, "m"),
            out_features,
            in_features,
        )
    rows_name = (
        "tokens"
        if item.operation.startswith("Fixed")
        else ("m" if item.operation == "GroupedBackward" else "aggregate_rows")
    )
    rows = _integer_value(problem, rows_name)
    if item.operation == "GroupedBackward":
        # The serial grouped-backward key follows the same transposed
        # ProblemSize convention as ordinary backward.
        out_features = _integer_value(problem, "k")
        in_features = _integer_value(problem, "n")
    else:
        out_name = (
            "output_features" if "output_features" in contract else "out_features"
        )
        in_name = "input_features" if "input_features" in contract else "in_features"
        out_features = _integer_value(contract, out_name)
        in_features = _integer_value(contract, in_name)
    return str(quant_type), rows, out_features, in_features


def _tensor_source(
    item, quant_type: str, out_features: int, in_features: int
) -> TensorSource:
    if item.operation in {"OrdinaryForward", "OrdinaryBackward"}:
        source = _DENSE_TENSORS.get((quant_type, out_features, in_features))
    else:
        kind = (
            "fixed"
            if item.operation.startswith("Fixed")
            else "pair"
            if item.operation.endswith("Pair")
            else "single"
        )
        source = _GROUPED_TENSORS.get((kind, quant_type, out_features, in_features))
    if source is None:
        raise ValueError(
            f"no benchmark tensor source for {item.operation} "
            f"{quant_type} N={out_features} K={in_features}"
        )
    return source


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    result = []
    for item in kernels():
        if item.key is None or item.operation is None:
            continue
        quant_type, rows, out_features, in_features = _case_geometry(item)
        result.append(
            BenchmarkCase(
                item.operation,
                item.key.hash,
                item.symbol,
                item.filename,
                quant_type,
                rows,
                out_features,
                in_features,
                _tensor_source(item, quant_type, out_features, in_features),
                item.key,
            )
        )
    return tuple(result)


def case_for_key(operation: str, key: DeploymentKey) -> BenchmarkCase:
    matches = [case for case in benchmark_cases() if case.operation == operation]
    for case in matches:
        if case.identity == key.hash:
            return case
    raise ValueError(
        f"{operation} solution {key.hash} is not a checked-in public deployment key"
    )


def validate_direct_artifact(
    operation: str, key: DeploymentKey, code_object: Path
) -> BenchmarkCase:
    case = case_for_key(operation, key)
    expected = KERNEL_DIR / case.artifact
    if not code_object.is_file():
        raise ValueError(f"GGTensile artifact does not exist: {code_object}")
    if code_object.resolve() != expected.resolve():
        raise ValueError(
            f"{code_object} is not the deployed artifact for {case.identity}; "
            f"expected {expected}"
        )
    return case


def hip_control_status() -> dict[str, object]:
    """Describe optional legacy HIP controls without treating public MMQ as HIP."""
    configured = os.environ.get("GGTENSILE_HIP_CONTROL_ROOT")
    roots = []
    if configured:
        path = Path(configured)
        roots.extend(
            (
                path,
                path / "gfx1151",
                path / "hip_controls",
                path / "gfx1151" / "hip_controls",
            )
        )
    purelib = Path(sysconfig.get_paths()["purelib"])
    roots.extend(
        (
            ROOT / "build/mmq_hip_controls/gfx1151",
            ROOT / "torch_ggml_ops/kernels/gfx1151/hip_controls",
            purelib / "torch_ggml_ops/kernels/gfx1151/hip_controls",
        )
    )
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        count = len(tuple(root.glob("*.hsaco")))
        if count:
            return {"available": True, "source": str(root), "artifacts": count}
    return {
        "available": False,
        "source": None,
        "reason": (
            "no legacy HIP controls were found; run "
            "python tools/build_mmq_hip_controls.py"
        ),
    }


def validate_manifest(*, require_artifacts: bool = False) -> tuple[BenchmarkCase, ...]:
    cases = benchmark_cases()
    expected_operations = {
        "OrdinaryForward": 50,
        "OrdinaryBackward": 50,
        "GroupedForward": 12,
        "GroupedForwardPair": 9,
        "GroupedBackward": 12,
        "GroupedBackwardPair": 9,
        "FixedGroupedForward": 3,
        "FixedGroupedBackward": 3,
    }
    counts = Counter(case.operation for case in cases)
    if counts != expected_operations:
        raise ValueError(f"deployment benchmark counts differ: {counts}")
    if len(cases) != 148 or len({case.identity for case in cases}) != len(cases):
        raise ValueError("deployment benchmark identities are incomplete or duplicated")
    scripts = {case.benchmark.script for case in cases}
    missing_scripts = sorted(
        script for script in scripts if not (ROOT / script).is_file()
    )
    if missing_scripts:
        raise ValueError(f"benchmark scripts are missing: {missing_scripts}")
    required_markers = (
        "public_api",
        "ggtensile",
        "hip",
        "bf16_baseline",
        "correctness_before_timing",
    )
    incomplete_scripts = []
    for script in scripts:
        source = (ROOT / script).read_text(encoding="utf-8")
        missing = [marker for marker in required_markers if marker not in source]
        if missing:
            incomplete_scripts.append((script, missing))
    if incomplete_scripts:
        raise ValueError(
            f"benchmark scripts lack required surfaces: {incomplete_scripts}"
        )
    missing_artifacts = sorted(
        case.artifact for case in cases if not (KERNEL_DIR / case.artifact).is_file()
    )
    if require_artifacts and missing_artifacts:
        raise ValueError(
            f"deployed GGTensile artifacts are missing: {missing_artifacts}"
        )
    return cases


def manifest_mapping(*, require_artifacts: bool = False) -> dict[str, object]:
    cases = validate_manifest(require_artifacts=require_artifacts)
    return {
        "inventory": "tools/ggtensile/configs/mmq_deployment.json and ordinary catalogs",
        "case_count": len(cases),
        "operation_counts": dict(Counter(case.operation for case in cases)),
        "hip_control": hip_control_status(),
        "cases": [case.to_mapping() for case in cases],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full manifest")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="also require every checked-in GGTensile HSACO",
    )
    args = parser.parse_args()
    manifest = manifest_mapping(require_artifacts=args.require_artifacts)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            f"validated {manifest['case_count']} deployment cases: "
            f"{manifest['operation_counts']}"
        )


if __name__ == "__main__":
    main()
