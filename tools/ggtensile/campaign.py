"""Strict canonical selected-kernel catalogs and campaign candidates."""

import json
from dataclasses import dataclass
from pathlib import Path

from .family_registry import (
    family_for_instance,
    mapping_for_instance,
    parse_instance,
    problem_size_for_instance,
    problem_type_for_family,
)
from .identity import (
    GFX1151_TARGET,
    KernelFamily,
    KernelTarget,
    problem_type_mapping,
    quant_type_from_problem_type,
)
from .kernel_instance import KernelInstance
from .mmq_bwd_spec import BackwardKernelSpec
from .mmq_fwd_spec import ForwardKernelSpec
from .model import ProblemSize, ProblemType
from .quant_formats import BACKWARD_QUANT_FORMATS, GROUPED_QUANT_FORMATS, QUANT_FORMATS
from .schema import integer, strict_mapping


class CatalogError(ValueError):
    """A selected-kernel catalog is malformed or internally inconsistent."""


@dataclass(frozen=True)
class CatalogEntry:
    """One exact selected passive kernel instance."""

    instance: KernelInstance

    @property
    def problem_size(self) -> ProblemSize:
        return problem_size_for_instance(self.instance)

    @property
    def operation_type(self) -> str:
        return family_for_instance(self.instance).value

    @property
    def quant_data_type(self) -> str:
        return self.instance.problem_type.quant_data_type

    @property
    def slug(self) -> str:
        size = self.problem_size
        return f"m{size.m}_n{size.n}_k{size.k}"

    @property
    def expected_logical_weight_shape(self) -> tuple[int, int]:
        size = self.problem_size
        return (
            (size.n, size.k)
            if family_for_instance(self.instance).is_forward
            else (size.k, size.n)
        )

    @property
    def expected_physical_weight_shape(self) -> tuple[int, int]:
        size = self.problem_size
        formats = {**QUANT_FORMATS, **GROUPED_QUANT_FORMATS, **BACKWARD_QUANT_FORMATS}
        quant = formats[self.quant_data_type]
        if family_for_instance(self.instance).is_forward:
            return size.n, size.k // quant.block_values * quant.block_bytes
        return size.k, size.n // quant.block_values * quant.block_bytes


@dataclass(frozen=True)
class DeploymentCatalog:
    """Selected exact logic for one family, target, and problem type."""

    family: KernelFamily
    problem_type: ProblemType
    entries: tuple[CatalogEntry, ...]

    @property
    def instances(self) -> tuple[KernelInstance, ...]:
        return tuple(entry.instance for entry in self.entries)

    def entry_for(self, problem_size: ProblemSize) -> CatalogEntry:
        for entry in self.entries:
            if entry.problem_size == problem_size:
                return entry
        raise CatalogError(
            f"problem size is not in the deployment catalog: {problem_size.to_mapping()}"
        )

    def selected(
        self, *, sizes: tuple[ProblemSize, ...] = ()
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
        first = mapping_for_instance(self.entries[0].instance)
        specs: list[object] = []
        logic: list[dict[str, object]] = []
        for entry in self.entries:
            mapping = mapping_for_instance(entry.instance)
            for name in ("KernelFamily", "Target", "ProblemType"):
                if mapping[name] != first[name]:
                    raise CatalogError(f"catalog entries disagree on {name}")
            spec = mapping["KernelSpec"]
            if spec in specs:
                index = specs.index(spec)
            else:
                index = len(specs)
                specs.append(spec)
            logic.append({"Problem": mapping["Problem"], "KernelSpecIndex": index})
        return {
            "KernelFamily": first["KernelFamily"],
            "Target": first["Target"],
            "ProblemType": first["ProblemType"],
            "KernelSpecs": specs,
            "ExactLogic": logic,
        }


def unique_kernel_specs(catalog: DeploymentCatalog) -> tuple[object, ...]:
    """Return each typed kernel specification in catalog order once."""
    specs: list[object] = []
    for entry in catalog.entries:
        spec = entry.instance.kernel_spec
        if spec not in specs:
            specs.append(spec)
    return tuple(specs)


def load_kernel_spec(path: Path, *, problem_type: ProblemType) -> object:
    """Load one canonical candidate KernelSpec for a catalog problem type."""
    item = strict_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "kernel candidate",
        frozenset({"KernelFamily", "Target", "ProblemType", "KernelSpec"}),
        error_type=CatalogError,
    )
    family = (
        KernelFamily.OrdinaryForward
        if problem_type.operation_type == "MMQForward"
        else KernelFamily.OrdinaryBackward
    )
    if item["KernelFamily"] != family.value:
        raise CatalogError("candidate KernelFamily does not match the campaign")
    KernelTarget.from_mapping(item["Target"])
    quant = quant_type_from_problem_type(item["ProblemType"], family)
    if quant != problem_type.quant_data_type:
        raise CatalogError("candidate ProblemType does not match the campaign")
    if family is KernelFamily.OrdinaryForward:
        return ForwardKernelSpec.from_mapping(item["KernelSpec"])
    return BackwardKernelSpec.from_mapping(item["KernelSpec"], quant)


def load_catalog(path: Path) -> DeploymentCatalog:
    root = strict_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "deployment catalog",
        frozenset(
            {"KernelFamily", "Target", "ProblemType", "KernelSpecs", "ExactLogic"}
        ),
        error_type=CatalogError,
    )
    family = KernelFamily(root["KernelFamily"])
    KernelTarget.from_mapping(root["Target"])
    quant_type = quant_type_from_problem_type(root["ProblemType"], family)
    if root["Target"] != GFX1151_TARGET.to_mapping():
        raise CatalogError("catalog Target is not canonical")
    if root["ProblemType"] != problem_type_mapping(family, quant_type):
        raise CatalogError("catalog ProblemType is not canonical")

    raw_specs = root["KernelSpecs"]
    if type(raw_specs) is not list or not raw_specs:
        raise CatalogError("KernelSpecs must be a nonempty JSON list")
    if len({json.dumps(spec, sort_keys=True) for spec in raw_specs}) != len(raw_specs):
        raise CatalogError("KernelSpecs must not contain duplicate specifications")

    raw_logic = root["ExactLogic"]
    if type(raw_logic) is not list or not raw_logic:
        raise CatalogError("ExactLogic must be a nonempty JSON list")
    entries: list[CatalogEntry] = []
    selected_indices: set[int] = set()
    problem_identities: set[str] = set()
    for index, raw_entry in enumerate(raw_logic):
        item = strict_mapping(
            raw_entry,
            f"ExactLogic[{index}]",
            frozenset({"Problem", "KernelSpecIndex"}),
            error_type=CatalogError,
        )
        spec_index = integer(item, "KernelSpecIndex", error_type=CatalogError)
        if not 0 <= spec_index < len(raw_specs):
            raise CatalogError(
                f"ExactLogic[{index}].KernelSpecIndex is outside KernelSpecs"
            )
        exact_mapping = {
            "KernelFamily": family.value,
            "Target": root["Target"],
            "ProblemType": root["ProblemType"],
            "Problem": item["Problem"],
            "KernelSpec": raw_specs[spec_index],
        }
        instance = parse_instance(exact_mapping)
        problem_identity = json.dumps(
            item["Problem"], sort_keys=True, separators=(",", ":")
        )
        if problem_identity in problem_identities:
            raise CatalogError("ExactLogic must not contain duplicate problems")
        problem_identities.add(problem_identity)
        selected_indices.add(spec_index)
        entries.append(CatalogEntry(instance))
    unused = sorted(set(range(len(raw_specs))) - selected_indices)
    if unused:
        raise CatalogError(f"KernelSpecs contains unreferenced indices: {unused}")
    return DeploymentCatalog(
        family,
        problem_type_for_family(family, quant_type),
        tuple(entries),
    )
