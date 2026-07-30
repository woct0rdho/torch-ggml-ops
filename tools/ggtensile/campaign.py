import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .model import ProblemSize, ProblemType, SchemaError, Solution, SolutionKey

DEFAULT_INVENTORY = Path(__file__).with_name("q4_k_dense_inventory.json")
DEFAULT_RETAINED_SOLUTION = Path(__file__).with_name("q4_k_retained_solution.json")
DEFAULT_SELECTED_SOLUTIONS = Path(__file__).with_name("q4_k_selected_solutions.json")

_EXPECTED_M = (2048, 8192, 32768)
_CAMPAIGN_SPECS = {
    "Q4_K": {
        "families": {
            "narrow": ((2048, 512), "blk.5.ffn_gate_shexp.weight", 70),
            "shared_down": ((512, 2048), "blk.5.ffn_down_shexp.weight", 30),
            "attention_output": ((4096, 2048), "blk.3.attn_output.weight", 10),
            "query": ((2048, 8192), "blk.39.attn_q.weight", 1),
        },
        "block_bytes": 144,
    },
    "Q5_K": {
        "families": {
            "narrow": ((2048, 512), "blk.0.ffn_gate_shexp.weight", 21),
            "shared_down": ((512, 2048), "blk.0.ffn_down_shexp.weight", 10),
        },
        "block_bytes": 176,
    },
}


def _campaign_spec(quant_type: str) -> Mapping[str, object]:
    try:
        return _CAMPAIGN_SPECS[quant_type]
    except KeyError as error:
        raise CampaignError(
            f"unsupported campaign quant type {quant_type!r}"
        ) from error


class CampaignError(ValueError):
    """A dense MMQ campaign input is malformed or internally inconsistent."""


def _mapping(value: object, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{name} must be a JSON object")
    actual = {str(key) for key in value}
    if actual != keys:
        raise CampaignError(
            f"invalid {name} keys: expected {sorted(keys)}, got {sorted(actual)}"
        )
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str:
        raise CampaignError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise CampaignError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class CampaignEntry:
    quant_data_type: str
    family: str
    problem_size: ProblemSize
    representative_tensor: str
    call_count: int
    historical_hip_median_ms: float
    current_status: str
    selected_solution: str

    @property
    def slug(self) -> str:
        size = self.problem_size
        return f"m{size.m}_n{size.n}_k{size.k}"

    @property
    def weighted_historical_latency_ms(self) -> float:
        return self.call_count * self.historical_hip_median_ms

    @property
    def expected_logical_weight_shape(self) -> tuple[int, int]:
        return (self.problem_size.k, self.problem_size.n)

    @property
    def expected_physical_weight_shape(self) -> tuple[int, int]:
        size = self.problem_size
        spec = _campaign_spec(self.quant_data_type)
        return (size.k, size.n // 256 * int(spec["block_bytes"]))

    def solution_key(
        self, problem_type: ProblemType, solution: Solution
    ) -> SolutionKey:
        return SolutionKey(problem_type, self.problem_size, solution)


@dataclass(frozen=True)
class CampaignInventory:
    problem_type: ProblemType
    model_file: str
    entries: tuple[CampaignEntry, ...]

    def selected(
        self,
        *,
        families: tuple[str, ...] = (),
        sizes: tuple[ProblemSize, ...] = (),
    ) -> tuple[CampaignEntry, ...]:
        spec = _campaign_spec(self.problem_type.quant_data_type)
        family_specs = spec["families"]
        assert isinstance(family_specs, Mapping)
        unknown_families = sorted(set(families) - family_specs.keys())
        if unknown_families:
            raise CampaignError(f"unknown families: {unknown_families}")
        inventory_sizes = {entry.problem_size for entry in self.entries}
        unknown_sizes = [
            size.to_mapping() for size in sizes if size not in inventory_sizes
        ]
        if unknown_sizes:
            raise CampaignError(f"sizes are not in the inventory: {unknown_sizes}")
        family_filter = set(families)
        size_filter = set(sizes)
        selected = tuple(
            entry
            for entry in self.entries
            if (not family_filter or entry.family in family_filter)
            and (not size_filter or entry.problem_size in size_filter)
        )
        if not selected:
            raise CampaignError("campaign selection is empty")
        return selected


def _load_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {name} {path}: {error}") from error


def load_solution(path: Path = DEFAULT_RETAINED_SOLUTION) -> Solution:
    try:
        return Solution.from_mapping(_load_json(path, "solution"))
    except SchemaError as error:
        raise CampaignError(f"invalid campaign solution: {error}") from error


def load_solution_catalog(
    path: Path = DEFAULT_SELECTED_SOLUTIONS,
) -> dict[str, Solution]:
    root = _mapping(
        _load_json(path, "solution catalog"),
        "solution catalog",
        frozenset({"Solutions"}),
    )
    raw_solutions = root["Solutions"]
    if not isinstance(raw_solutions, Mapping) or not raw_solutions:
        raise CampaignError("Solutions must be a nonempty JSON object")
    solutions: dict[str, Solution] = {}
    for name, value in raw_solutions.items():
        selected_name = _string(name, "solution catalog key")
        try:
            solutions[selected_name] = Solution.from_mapping(value)
        except SchemaError as error:
            raise CampaignError(
                f"invalid catalog solution {selected_name!r}: {error}"
            ) from error
    return solutions


def load_inventory(path: Path = DEFAULT_INVENTORY) -> CampaignInventory:
    root = _mapping(
        _load_json(path, "inventory"),
        "inventory",
        frozenset({"ProblemType", "ModelFile", "Validation", "Keys"}),
    )
    try:
        problem_type = ProblemType.from_mapping(root["ProblemType"])
    except SchemaError as error:
        raise CampaignError(f"invalid inventory ProblemType: {error}") from error
    spec = _campaign_spec(problem_type.quant_data_type)
    validation_expected = {
        "QuantType": problem_type.quant_data_type,
        "ExactDispatch": True,
        "RequireBitExactHip": True,
        "RequireGradOutputMutation": True,
        "RequirePackedWeightMutation": True,
    }
    validation = _mapping(
        root["Validation"], "Validation", frozenset(validation_expected)
    )
    if dict(validation) != validation_expected:
        raise CampaignError(
            f"Validation must be exactly {validation_expected}, got {dict(validation)}"
        )
    raw_entries = root["Keys"]
    if type(raw_entries) is not list:
        raise CampaignError("Keys must be a list")

    entries: list[CampaignEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        item = _mapping(
            raw_entry,
            f"Keys[{index}]",
            frozenset(
                {
                    "Family",
                    "ProblemSize",
                    "RepresentativeTensor",
                    "CallCount",
                    "HistoricalHipMedianMs",
                    "CurrentStatus",
                    "SelectedSolution",
                }
            ),
        )
        family = _string(item["Family"], f"Keys[{index}].Family")
        family_specs = spec["families"]
        assert isinstance(family_specs, Mapping)
        if family not in family_specs:
            raise CampaignError(f"unknown family {family!r}")
        try:
            size = ProblemSize.from_mapping(item["ProblemSize"])
        except SchemaError as error:
            raise CampaignError(
                f"invalid Keys[{index}].ProblemSize: {error}"
            ) from error
        historical = item["HistoricalHipMedianMs"]
        if type(historical) not in (int, float) or historical <= 0:
            raise CampaignError(f"Keys[{index}].HistoricalHipMedianMs must be positive")
        status = _string(item["CurrentStatus"], f"Keys[{index}].CurrentStatus")
        if status not in ("open", "selected"):
            raise CampaignError(f"invalid CurrentStatus {status!r}")
        entry = CampaignEntry(
            quant_data_type=problem_type.quant_data_type,
            family=family,
            problem_size=size,
            representative_tensor=_string(
                item["RepresentativeTensor"],
                f"Keys[{index}].RepresentativeTensor",
            ),
            call_count=_integer(item["CallCount"], f"Keys[{index}].CallCount"),
            historical_hip_median_ms=float(historical),
            current_status=status,
            selected_solution=_string(
                item["SelectedSolution"], f"Keys[{index}].SelectedSolution"
            ),
        )
        if entry.call_count <= 0:
            raise CampaignError(f"Keys[{index}].CallCount must be positive")
        if not entry.selected_solution:
            raise CampaignError(f"Keys[{index}].SelectedSolution must not be empty")
        entries.append(entry)

    family_specs = spec["families"]
    assert isinstance(family_specs, Mapping)
    expected_sizes = {
        ProblemSize(m, n, k)
        for (n, k), _, _ in family_specs.values()
        for m in _EXPECTED_M
    }
    actual_sizes = [entry.problem_size for entry in entries]
    if len(actual_sizes) != len(set(actual_sizes)):
        raise CampaignError("inventory has duplicate problem sizes")
    if set(actual_sizes) != expected_sizes:
        raise CampaignError(
            f"inventory does not contain the exact {len(expected_sizes)} production keys"
        )
    for entry in entries:
        (n, k), tensor, calls = family_specs[entry.family]
        if (entry.problem_size.n, entry.problem_size.k) != (n, k):
            raise CampaignError(f"{entry.slug} does not match family {entry.family}")
        if entry.representative_tensor != tensor or entry.call_count != calls:
            raise CampaignError(
                f"{entry.slug} has inconsistent tensor or call count for {entry.family}"
            )

    return CampaignInventory(
        problem_type=problem_type,
        model_file=_string(root["ModelFile"], "ModelFile"),
        entries=tuple(entries),
    )
