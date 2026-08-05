import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from typing_extensions import NotRequired

from .model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
)
from .quant_formats import QUANT_FORMATS

_EXPECTED_M = (2048, 8192, 32768)
_FamilySpec = tuple[tuple[int, int], str, int]


class _CampaignSpec(TypedDict):
    families: dict[str, _FamilySpec]
    m_values: NotRequired[dict[str, tuple[int, ...]]]


_MMQ_BWD_CAMPAIGN_SPECS: dict[str, _CampaignSpec] = {
    "Q3_K": {
        "families": {
            "narrow": ((2048, 512), "blk.3.attn_k.weight", 9),
            "query": ((2048, 8192), "blk.3.attn_q.weight", 9),
        },
    },
    "Q4_K": {
        "families": {
            "narrow": ((2048, 512), "blk.5.ffn_gate_shexp.weight", 70),
            "shared_down": ((512, 2048), "blk.5.ffn_down_shexp.weight", 30),
            "attention_output": ((4096, 2048), "blk.3.attn_output.weight", 10),
            "query": ((2048, 8192), "blk.39.attn_q.weight", 1),
        },
    },
    "Q5_K": {
        "families": {
            "narrow": ((2048, 512), "blk.0.ffn_gate_shexp.weight", 21),
            "shared_down": ((512, 2048), "blk.0.ffn_down_shexp.weight", 10),
        },
    },
    "Q6_K": {
        "families": {
            "lm_head": ((2048, 248320), "output.weight", 1),
        },
        "m_values": {
            "lm_head": (64, 128, 256),
        },
    },
    "Q8_0": {
        "families": {
            "attention_q_a": ((4096, 1024), "blk.0.attn_q_a.weight", 43),
            "attention_q_b": ((1024, 32768), "blk.0.attn_q_b.weight", 43),
            "attention_kv": ((4096, 512), "blk.0.attn_kv.weight", 43),
            "attention_output_b": ((8192, 4096), "blk.0.attn_output_b.weight", 43),
            "shared_gate_up": ((4096, 2048), "blk.0.ffn_gate_shexp.weight", 86),
            "shared_down": ((2048, 4096), "blk.0.ffn_down_shexp.weight", 43),
            "lm_head": ((4096, 129280), "output.weight", 1),
        },
        "m_values": {
            "lm_head": (32, 64, 128, 256, 512),
        },
    },
}

_MMQ_FWD_CAMPAIGN_SPECS: dict[str, _CampaignSpec] = {
    "Q4_K": {
        "families": {
            "narrow": ((512, 2048), "blk.5.ffn_gate_shexp.weight", 70),
            "shared_down": ((2048, 512), "blk.5.ffn_down_shexp.weight", 30),
            "attention_output": ((2048, 4096), "blk.3.attn_output.weight", 10),
            "query": ((8192, 2048), "blk.39.attn_q.weight", 1),
        },
    },
    "Q5_K": {
        "families": {
            "narrow": ((512, 2048), "blk.0.ffn_gate_shexp.weight", 21),
            "shared_down": ((2048, 512), "blk.0.ffn_down_shexp.weight", 10),
        },
    },
    "Q6_K": {
        "families": {
            "lm_head": ((248320, 2048), "output.weight", 1),
        },
        "m_values": {
            "lm_head": (64, 128, 256),
        },
    },
    "Q8_0": {
        "families": {
            "attention_q_a": ((1024, 4096), "blk.0.attn_q_a.weight", 43),
            "attention_q_b": ((32768, 1024), "blk.0.attn_q_b.weight", 43),
            "attention_kv": ((512, 4096), "blk.0.attn_kv.weight", 43),
            "attention_output_b": (
                (4096, 8192),
                "blk.0.attn_output_b.weight",
                43,
            ),
            "shared_gate_up": (
                (2048, 4096),
                "blk.0.ffn_gate_shexp.weight",
                86,
            ),
            "shared_down": ((4096, 2048), "blk.0.ffn_down_shexp.weight", 43),
            "lm_head": ((129280, 4096), "output.weight", 1),
        },
        "m_values": {
            "lm_head": (32, 64, 128, 256, 512),
        },
    },
}


def _campaign_spec(operation_type: str, quant_type: str) -> _CampaignSpec:
    specs = (
        _MMQ_FWD_CAMPAIGN_SPECS
        if operation_type == "MMQForward"
        else _MMQ_BWD_CAMPAIGN_SPECS
    )
    return specs[quant_type]


class CampaignError(ValueError):
    """A MMQ campaign input is malformed or internally inconsistent."""


def _mapping(value: object, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{name} must be a JSON object")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CampaignError(f"{name} keys must be strings")
        normalized[key] = item
    actual = set(normalized)
    if actual != keys:
        raise CampaignError(
            f"invalid {name} keys: expected {sorted(keys)}, got {sorted(actual)}"
        )
    return normalized


def _string(value: object, name: str) -> str:
    if type(value) is not str:
        raise CampaignError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise CampaignError(f"{name} must be an integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise CampaignError(f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class CampaignEntry:
    operation_type: str
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
        spec = _campaign_spec(
            self.problem_type.operation_type,
            self.problem_type.quant_data_type,
        )
        family_specs = spec["families"]
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


def load_solution(
    path: Path,
    *,
    problem_type: ProblemType,
) -> BackwardSolution | ForwardSolution:
    solution_type = (
        ForwardSolution
        if problem_type.operation_type == "MMQForward"
        else BackwardSolution
    )
    return solution_type.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def load_solution_catalog(
    path: Path,
    *,
    problem_type: ProblemType,
) -> dict[str, BackwardSolution | ForwardSolution]:
    root = _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "solution catalog",
        frozenset({"Solutions"}),
    )
    raw_solutions = root["Solutions"]
    if not isinstance(raw_solutions, Mapping) or not raw_solutions:
        raise CampaignError("Solutions must be a nonempty JSON object")
    solution_type = (
        ForwardSolution
        if problem_type.operation_type == "MMQForward"
        else BackwardSolution
    )
    solutions: dict[str, BackwardSolution | ForwardSolution] = {}
    for name, value in raw_solutions.items():
        selected_name = _string(name, "solution catalog key")
        solutions[selected_name] = solution_type.from_mapping(value)
    return solutions


def load_inventory(path: Path) -> CampaignInventory:
    root = _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "inventory",
        frozenset({"ProblemType", "ModelFile", "Validation", "Keys"}),
    )
    problem_type = ProblemType.from_mapping(root["ProblemType"])
    spec = _campaign_spec(
        problem_type.operation_type,
        problem_type.quant_data_type,
    )
    if problem_type.operation_type == "MMQForward":
        validation_expected = {
            "QuantType": problem_type.quant_data_type,
            "ExactDispatch": True,
            "MaxCandidateToHipNormalizedRmse": 0.0005,
            "MaxCandidateToHipAbsoluteError": 0.015625,
            "MaxIndependentNormalizedRmse": 0.04,
            "RequireInputMutation": True,
            "RequirePackedWeightMutation": True,
            "RequireWorkspaceMutation": True,
            "FixedActivationProducer": (
                "HIP_Q8_1_F32_D4"
                if problem_type.quant_data_type in ("Q6_K", "Q8_0")
                else "HIP_Q8_1_F16_D4S4"
            ),
        }
    else:
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
        size = ProblemSize.from_mapping(item["ProblemSize"])
        historical = _positive_float(
            item["HistoricalHipMedianMs"],
            f"Keys[{index}].HistoricalHipMedianMs",
        )
        status = _string(item["CurrentStatus"], f"Keys[{index}].CurrentStatus")
        if status not in ("open", "selected"):
            raise CampaignError(f"invalid CurrentStatus {status!r}")
        entry = CampaignEntry(
            operation_type=problem_type.operation_type,
            quant_data_type=problem_type.quant_data_type,
            family=family,
            problem_size=size,
            representative_tensor=_string(
                item["RepresentativeTensor"],
                f"Keys[{index}].RepresentativeTensor",
            ),
            call_count=_integer(item["CallCount"], f"Keys[{index}].CallCount"),
            historical_hip_median_ms=historical,
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
    m_values = spec.get("m_values", {})
    expected_sizes = {
        ProblemSize(m, n, k)
        for family, ((n, k), _, _) in family_specs.items()
        for m in m_values.get(family, _EXPECTED_M)
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
