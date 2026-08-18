import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .mmq_bwd_spec import BackwardKernelSpec, BackwardProblemContract
from .mmq_fwd_spec import (
    ForwardKernelCandidate,
    ForwardKernelSpec,
    ForwardProblemContract,
)
from .model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from .quant_formats import QUANT_FORMATS


class CatalogError(ValueError):
    """A deployment catalog is malformed or internally inconsistent."""


def _mapping(value: object, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CatalogError(f"{name} must be a JSON object")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CatalogError(f"{name} keys must be strings")
        normalized[key] = item
    actual = set(normalized)
    if actual != keys:
        raise CatalogError(
            f"invalid {name} keys: expected {sorted(keys)}, got {sorted(actual)}"
        )
    return normalized


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise CatalogError(f"{name} must be an integer")
    return value


Solution = BackwardSolution | ForwardSolution


@dataclass(frozen=True)
class CatalogEntry:
    """One exact deployment key and its current selected kernel specification."""

    solution_key: SolutionKey

    @property
    def problem_size(self) -> ProblemSize:
        return self.solution_key.problem_size

    @property
    def solution(self) -> Solution:
        return cast(Solution, self.solution_key.solution)

    @property
    def operation_type(self) -> str:
        return self.solution_key.problem_type.operation_type

    @property
    def quant_data_type(self) -> str:
        return self.solution_key.problem_type.quant_data_type

    @property
    def slug(self) -> str:
        size = self.problem_size
        return f"m{size.m}_n{size.n}_k{size.k}"

    @property
    def expected_logical_weight_shape(self) -> tuple[int, int]:
        if self.operation_type == "MMQForward":
            return (self.problem_size.n, self.problem_size.k)
        return (self.problem_size.k, self.problem_size.n)

    @property
    def expected_physical_weight_shape(self) -> tuple[int, int]:
        size = self.problem_size
        quant_format = QUANT_FORMATS[self.quant_data_type]
        if self.operation_type == "MMQForward":
            return (
                size.n,
                size.k // quant_format.block_values * quant_format.block_bytes,
            )
        return (
            size.k,
            size.n // quant_format.block_values * quant_format.block_bytes,
        )


@dataclass(frozen=True)
class DeploymentCatalog:
    """Grid-based exact deployment logic for one MMQ problem type."""

    problem_type: ProblemType
    solutions: tuple[Solution, ...]
    entries: tuple[CatalogEntry, ...]

    @property
    def solution_keys(self) -> tuple[SolutionKey, ...]:
        return tuple(entry.solution_key for entry in self.entries)

    def entry_for(self, problem_size: ProblemSize) -> CatalogEntry:
        for entry in self.entries:
            if entry.problem_size == problem_size:
                return entry
        raise CatalogError(
            f"problem size is not in the deployment catalog: {problem_size.to_mapping()}"
        )

    def selected(
        self,
        *,
        sizes: tuple[ProblemSize, ...] = (),
    ) -> tuple[CatalogEntry, ...]:
        if not sizes:
            return self.entries
        requested = set(sizes)
        available = {entry.problem_size for entry in self.entries}
        unknown = [size.to_mapping() for size in sizes if size not in available]
        if unknown:
            raise CatalogError(f"sizes are not in the deployment catalog: {unknown}")
        selected = tuple(
            entry for entry in self.entries if entry.problem_size in requested
        )
        if not selected:
            raise CatalogError("catalog selection is empty")
        return selected

    def to_mapping(self) -> dict[str, object]:
        solution_indices = {
            solution: index for index, solution in enumerate(self.solutions)
        }
        first_key = self.entries[0].solution_key
        if isinstance(self.solutions[0], ForwardSolution):
            family = "OrdinaryForward"
            contract = ForwardProblemContract.from_solution(
                self.problem_type.quant_data_type, self.solutions[0]
            )
            specs = [
                ForwardKernelSpec.from_solution(solution).to_mapping()
                for solution in self.solutions
                if isinstance(solution, ForwardSolution)
            ]
        else:
            family = "OrdinaryBackward"
            contract = BackwardProblemContract.from_solution_key(first_key)
            specs = [
                BackwardKernelSpec.from_solution(solution).to_mapping(
                    contract.quant_type
                )
                for solution in self.solutions
                if isinstance(solution, BackwardSolution)
            ]
        return {
            "ArtifactKind": "DeploymentCatalog",
            "KernelFamily": family,
            "ProblemContract": contract.to_mapping(),
            "KernelSpecs": specs,
            "ExactLogic": [
                {
                    "Problem": entry.problem_size.to_canonical_mapping(),
                    "KernelSpecIndex": solution_indices[entry.solution],
                }
                for entry in self.entries
            ],
        }


def load_solution(
    path: Path,
    *,
    problem_type: ProblemType,
) -> Solution:
    value = json.loads(path.read_text(encoding="utf-8"))
    if problem_type.operation_type == "MMQForward":
        candidate = ForwardKernelCandidate.from_mapping(value)
        if candidate.problem_contract.quant_type != problem_type.quant_data_type:
            raise CatalogError("candidate ProblemContract does not match the campaign")
        return candidate.to_solution()
    item = _mapping(
        value,
        "BackwardKernelCandidate",
        frozenset({"ArtifactKind", "KernelFamily", "ProblemContract", "KernelSpec"}),
    )
    if item["ArtifactKind"] != "KernelCandidate":
        raise CatalogError("candidate ArtifactKind must be KernelCandidate")
    if item["KernelFamily"] != "OrdinaryBackward":
        raise CatalogError("backward candidate KernelFamily must be OrdinaryBackward")
    contract = BackwardProblemContract.from_mapping(
        item["ProblemContract"], ProblemSize(1, 1, 1)
    )
    if contract.quant_type != problem_type.quant_data_type:
        raise CatalogError("candidate ProblemContract does not match the campaign")
    return BackwardKernelSpec.from_mapping(
        item["KernelSpec"], contract.quant_type
    ).to_solution(contract)


def load_catalog(path: Path) -> DeploymentCatalog:
    root = _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "deployment catalog",
        frozenset(
            {
                "ArtifactKind",
                "KernelFamily",
                "ProblemContract",
                "KernelSpecs",
                "ExactLogic",
            }
        ),
    )
    if root["ArtifactKind"] != "DeploymentCatalog":
        raise CatalogError("catalog ArtifactKind must be DeploymentCatalog")
    family = root["KernelFamily"]
    if family == "OrdinaryForward":
        contract = ForwardProblemContract.from_mapping(root["ProblemContract"])
        problem_type = ProblemType.mmq_forward(contract.quant_type)
        parse_spec = ForwardKernelSpec.from_mapping
        to_solution = lambda spec: spec.to_solution(contract)
    elif family == "OrdinaryBackward":
        contract = BackwardProblemContract.from_mapping(
            root["ProblemContract"], ProblemSize(1, 1, 1)
        )
        problem_type = ProblemType.mmq_backward(contract.quant_type)
        parse_spec = lambda value: BackwardKernelSpec.from_mapping(
            value, contract.quant_type
        )
        to_solution = lambda spec: spec.to_solution(contract)
    else:
        raise CatalogError(f"unsupported catalog KernelFamily {family!r}")

    raw_specs = root["KernelSpecs"]
    if type(raw_specs) is not list or not raw_specs:
        raise CatalogError("KernelSpecs must be a nonempty JSON list")
    kernel_specs = tuple(parse_spec(value) for value in raw_specs)
    if len(kernel_specs) != len(set(kernel_specs)):
        raise CatalogError("KernelSpecs must not contain duplicate specifications")
    solutions = tuple(to_solution(spec) for spec in kernel_specs)
    if len(solutions) != len(set(solutions)):
        raise CatalogError("KernelSpecs must map to distinct lowering specifications")

    raw_logic = root["ExactLogic"]
    if type(raw_logic) is not list or not raw_logic:
        raise CatalogError("ExactLogic must be a nonempty JSON list")
    entries: list[CatalogEntry] = []
    selected_indices: set[int] = set()
    for index, raw_entry in enumerate(raw_logic):
        item = _mapping(
            raw_entry,
            f"ExactLogic[{index}]",
            frozenset({"Problem", "KernelSpecIndex"}),
        )
        size = ProblemSize.from_canonical_mapping(item["Problem"])
        if min(size.m, size.n, size.k) <= 0:
            raise CatalogError(f"ExactLogic[{index}].Problem must be positive")
        solution_index = _integer(
            item["KernelSpecIndex"], f"ExactLogic[{index}].KernelSpecIndex"
        )
        if not 0 <= solution_index < len(solutions):
            raise CatalogError(
                f"ExactLogic[{index}].KernelSpecIndex is outside KernelSpecs"
            )
        selected_indices.add(solution_index)
        entries.append(
            CatalogEntry(
                SolutionKey(problem_type, size, solutions[solution_index]),
            )
        )

    sizes = [entry.problem_size for entry in entries]
    if len(sizes) != len(set(sizes)):
        raise CatalogError("ExactLogic must not contain duplicate problem sizes")
    unused = sorted(set(range(len(solutions))) - selected_indices)
    if unused:
        raise CatalogError(f"KernelSpecs contains unreferenced indices: {unused}")

    return DeploymentCatalog(problem_type, solutions, tuple(entries))
