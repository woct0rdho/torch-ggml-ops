import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
        return self.solution_key.solution

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
        return {
            "ProblemType": self.problem_type.to_mapping(),
            "Solutions": [solution.to_mapping() for solution in self.solutions],
            "ExactLogic": [
                {
                    "ProblemSize": entry.problem_size.to_mapping(),
                    "SolutionIndex": solution_indices[entry.solution],
                }
                for entry in self.entries
            ],
        }


def load_solution(
    path: Path,
    *,
    problem_type: ProblemType,
) -> Solution:
    solution_type = (
        ForwardSolution
        if problem_type.operation_type == "MMQForward"
        else BackwardSolution
    )
    return solution_type.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _catalog_problem_type(value: object) -> ProblemType:
    problem_type = ProblemType.from_mapping(value)
    if problem_type.operation_type == "MMQForward":
        factory = ProblemType.mmq_forward
    elif problem_type.operation_type == "MMQBackward":
        factory = ProblemType.mmq_backward
    else:
        raise CatalogError(
            f"unsupported catalog operation {problem_type.operation_type!r}"
        )
    try:
        expected = factory(problem_type.quant_data_type)
    except ValueError as error:
        raise CatalogError(str(error)) from error
    if problem_type != expected:
        raise CatalogError(
            "ProblemType does not match the canonical MMQ deployment contract"
        )
    return problem_type


def load_catalog(path: Path) -> DeploymentCatalog:
    root = _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "deployment catalog",
        frozenset({"ProblemType", "Solutions", "ExactLogic"}),
    )
    problem_type = _catalog_problem_type(root["ProblemType"])
    solution_type = (
        ForwardSolution
        if problem_type.operation_type == "MMQForward"
        else BackwardSolution
    )

    raw_solutions = root["Solutions"]
    if type(raw_solutions) is not list or not raw_solutions:
        raise CatalogError("Solutions must be a nonempty JSON list")
    solutions = tuple(solution_type.from_mapping(value) for value in raw_solutions)
    if len(solutions) != len(set(solutions)):
        raise CatalogError("Solutions must not contain duplicate specifications")

    raw_logic = root["ExactLogic"]
    if type(raw_logic) is not list or not raw_logic:
        raise CatalogError("ExactLogic must be a nonempty JSON list")
    entries: list[CatalogEntry] = []
    selected_indices: set[int] = set()
    for index, raw_entry in enumerate(raw_logic):
        item = _mapping(
            raw_entry,
            f"ExactLogic[{index}]",
            frozenset({"ProblemSize", "SolutionIndex"}),
        )
        size = ProblemSize.from_mapping(item["ProblemSize"])
        if min(size.m, size.n, size.k) <= 0:
            raise CatalogError(f"ExactLogic[{index}].ProblemSize must be positive")
        solution_index = _integer(
            item["SolutionIndex"], f"ExactLogic[{index}].SolutionIndex"
        )
        if not 0 <= solution_index < len(solutions):
            raise CatalogError(
                f"ExactLogic[{index}].SolutionIndex is outside Solutions"
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
        raise CatalogError(f"Solutions contains unreferenced indices: {unused}")

    return DeploymentCatalog(problem_type, solutions, tuple(entries))
