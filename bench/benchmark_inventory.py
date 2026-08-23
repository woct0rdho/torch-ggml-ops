"""Benchmark consumer for the neutral public MMQ deployment metadata."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ggtensile.deployment import DeploymentKey
from tools.mmq_deployment_cases import (
    DeploymentCase,
    case_for_key,
    hip_control_root,
    public_artifact_path,
    public_deployment_cases,
)


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

# Compatibility name for benchmark callers. The object itself is owned by tools.
BenchmarkCase = DeploymentCase


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    return public_deployment_cases()


def _operation_mapping(case: BenchmarkCase) -> dict[str, object]:
    operation = OPERATIONS[case.operation]
    return {
        "script": operation.script,
        "public_api": operation.public_api,
        "baseline": operation.baseline,
    }


def case_for_benchmark_key(operation: str, key: DeploymentKey) -> BenchmarkCase:
    return case_for_key(operation, key)


def validate_direct_artifact(
    operation: str, key: DeploymentKey, code_object: Path
) -> BenchmarkCase:
    case = case_for_benchmark_key(operation, key)
    expected = public_artifact_path(case)
    if not code_object.is_file():
        raise ValueError(f"GGTensile artifact does not exist: {code_object}")
    if code_object.resolve() != expected.resolve():
        raise ValueError(
            f"{code_object} is not the deployed artifact for {case.identity}; "
            f"expected {expected}"
        )
    return case


def hip_control_status() -> dict[str, object]:
    root = hip_control_root()
    if root is None:
        return {
            "available": False,
            "source": None,
            "reason": (
                "no legacy HIP controls were found; run "
                "python tools/build_mmq_hip_controls.py"
            ),
        }
    return {
        "available": True,
        "source": str(root),
        "artifacts": len(tuple(root.glob("*.hsaco"))),
    }


def validate_manifest(*, require_artifacts: bool = False) -> tuple[BenchmarkCase, ...]:
    cases = benchmark_cases()
    expected = {
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
    if counts != expected or len(cases) != 148:
        raise ValueError(f"deployment benchmark counts differ: {counts}")
    if len({case.identity for case in cases}) != len(cases):
        raise ValueError("deployment benchmark identities are incomplete or duplicated")
    scripts = {OPERATIONS[case.operation].script for case in cases}
    missing_scripts = sorted(
        script for script in scripts if not (ROOT / script).is_file()
    )
    if missing_scripts:
        raise ValueError(f"benchmark scripts are missing: {missing_scripts}")
    missing_artifacts = sorted(
        case.artifact for case in cases if not public_artifact_path(case).is_file()
    )
    if require_artifacts and missing_artifacts:
        raise ValueError(
            f"deployed GGTensile artifacts are missing: {missing_artifacts}"
        )
    return cases


def manifest_mapping(*, require_artifacts: bool = False) -> dict[str, object]:
    cases = validate_manifest(require_artifacts=require_artifacts)
    mapped = []
    for case in cases:
        item = case.to_mapping()
        item.update(_operation_mapping(case))
        mapped.append(item)
    return {
        "inventory": "tools/ggtensile/configs/mmq_deployment.json and ordinary catalogs",
        "case_count": len(cases),
        "operation_counts": dict(Counter(case.operation for case in cases)),
        "hip_control": hip_control_status(),
        "cases": mapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full manifest")
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()
    manifest = manifest_mapping(require_artifacts=args.require_artifacts)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            f"validated {manifest['case_count']} deployment cases: {manifest['operation_counts']}"
        )


if __name__ == "__main__":
    main()
