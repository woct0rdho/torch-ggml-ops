"""Closed family registry for passive kernel instances."""

from collections.abc import Callable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, cast

from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from .fixed_grouped_mmq_bwd_spec import FixedBackwardKernelSpec
from .fixed_grouped_mmq_fwd_model import FixedForwardProblem
from .fixed_grouped_mmq_fwd_spec import (
    FixedForwardKernelSpec,
)
from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from .grouped_mmq_bwd_pair_spec import (
    GroupedBackwardPairContract,
    GroupedBackwardPairKernelSpec,
)
from .grouped_mmq_bwd_spec import GroupedBackwardKernelSpec
from .grouped_mmq_fwd_model import GroupedForwardProblem
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairRouteOwnership,
)
from .grouped_mmq_fwd_pair_spec import (
    GroupedForwardPairContract,
    GroupedForwardPairKernelSpec,
)
from .grouped_mmq_fwd_spec import (
    GroupedForwardKernelSpec,
    GroupedForwardProblemContract,
)
from .identity import (
    KernelFamily,
    KernelTarget,
    exact_key_hash,
    problem_type_mapping,
    quant_type_from_problem_type,
)
from .kernel_instance import KernelInstance
from .mmq_bwd_spec import BackwardKernelSpec
from .mmq_fwd_spec import ForwardKernelSpec
from .model import ProblemSize, ProblemType
from .schema import SchemaError, enum_value, integer, strict_mapping
from .toolchain import Toolchain

if TYPE_CHECKING:
    from .kernel_abi import KernelAbi
    from .kernel_writer_assembly import AssemblyKernelWriter


def _exact_item(value: object, name: str) -> Mapping[str, object]:
    return strict_mapping(
        value,
        name,
        frozenset({"KernelFamily", "Target", "ProblemType", "Problem", "KernelSpec"}),
    )


def problem_type_for_family(family: KernelFamily, quant_type: str) -> ProblemType:
    if family is KernelFamily.OrdinaryForward:
        return ProblemType.mmq_forward(quant_type)
    if family is KernelFamily.OrdinaryBackward:
        return ProblemType.mmq_backward(quant_type)
    if family is KernelFamily.GroupedBackward:
        return ProblemType.grouped_mmq_backward(quant_type)
    mapping = problem_type_mapping(family, quant_type)
    return ProblemType(
        operation_type=family.value,
        quant_data_type=quant_type,
        data_type_a=cast(str, mapping["DataTypeA"]),
        data_type_b=cast(str, mapping["DataTypeB"]),
        dest_data_type=cast(str, mapping["DestinationDataType"]),
        compute_data_type=cast(str, mapping["ComputeDataType"]),
        transpose_a=cast(bool, mapping["TransposeA"]),
        transpose_b=cast(bool, mapping["TransposeB"]),
    )


def _common(
    value: object, family: KernelFamily
) -> tuple[Mapping[str, object], KernelTarget, str]:
    item = _exact_item(value, f"{family.value} kernel instance")
    if item["KernelFamily"] != family.value:
        raise SchemaError(
            f"kernel instance has the wrong KernelFamily for {family.value}"
        )
    target = KernelTarget.from_mapping(item["Target"])
    quant_type = quant_type_from_problem_type(item["ProblemType"], family)
    return item, target, quant_type


def _ordinary_forward(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.OrdinaryForward)
    return KernelInstance(
        KernelFamily.OrdinaryForward,
        target,
        problem_type_for_family(KernelFamily.OrdinaryForward, quant_type),
        ProblemSize.from_mapping(item["Problem"]),
        ForwardKernelSpec.from_mapping(item["KernelSpec"]),
    )


def _ordinary_backward(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.OrdinaryBackward)
    return KernelInstance(
        KernelFamily.OrdinaryBackward,
        target,
        problem_type_for_family(KernelFamily.OrdinaryBackward, quant_type),
        ProblemSize.from_mapping(item["Problem"]),
        BackwardKernelSpec.from_mapping(item["KernelSpec"], quant_type),
    )


def _grouped_backward(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.GroupedBackward)
    problem_item = strict_mapping(
        item["Problem"],
        "GroupedBackwardProblem",
        frozenset({"M", "N", "K", "PhysicalExpertCount", "MaxRouteEntries"}),
    )
    if (
        problem_item["PhysicalExpertCount"] != 256
        or problem_item["MaxRouteEntries"] != 256
    ):
        raise SchemaError("grouped backward routing bounds must be 256")
    problem = ProblemSize.from_mapping(
        {name: problem_item[name] for name in ("M", "N", "K")}
    )
    spec = GroupedBackwardKernelSpec.from_mapping(item["KernelSpec"], quant_type)
    return KernelInstance(
        KernelFamily.GroupedBackward,
        target,
        problem_type_for_family(KernelFamily.GroupedBackward, quant_type),
        problem,
        spec,
    )


def _grouped_forward(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.GroupedForward)
    raw = strict_mapping(
        item["Problem"],
        "GroupedForwardProblem",
        frozenset({"M", "N", "K", "PhysicalExpertCount", "MaxRouteEntries"}),
    )
    problem = GroupedForwardProblem(
        quant_type,
        integer(raw, "M"),
        integer(raw, "N"),
        integer(raw, "K"),
        integer(raw, "PhysicalExpertCount"),
        integer(raw, "MaxRouteEntries"),
    )
    return KernelInstance(
        KernelFamily.GroupedForward,
        target,
        problem_type_for_family(KernelFamily.GroupedForward, quant_type),
        problem,
        GroupedForwardKernelSpec.from_mapping(
            item["KernelSpec"], GroupedForwardProblemContract.for_problem(problem)
        ),
    )


def _grouped_forward_pair(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.GroupedForwardPair)
    raw = strict_mapping(
        item["Problem"],
        "GroupedForwardPairProblem",
        frozenset(
            {
                "M",
                "N",
                "K",
                "PhysicalExpertCount",
                "MaxRouteEntries",
                "ProjectionCount",
            }
        ),
    )
    problem = GroupedForwardPairProblem(
        quant_type,
        integer(raw, "M"),
        integer(raw, "N"),
        integer(raw, "K"),
        integer(raw, "PhysicalExpertCount"),
        integer(raw, "MaxRouteEntries"),
        integer(raw, "ProjectionCount"),
    )
    contract = GroupedForwardPairContract.for_problem(problem)
    return KernelInstance(
        KernelFamily.GroupedForwardPair,
        target,
        problem_type_for_family(KernelFamily.GroupedForwardPair, quant_type),
        problem,
        GroupedForwardPairKernelSpec.from_mapping(item["KernelSpec"], contract),
    )


def _grouped_backward_pair(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.GroupedBackwardPair)
    raw = strict_mapping(
        item["Problem"],
        "GroupedBackwardPairProblem",
        frozenset(
            {
                "M",
                "N",
                "K",
                "PhysicalExpertCount",
                "MaxRouteEntries",
                "ProjectionCount",
            }
        ),
    )
    problem = GroupedBackwardPairProblem(
        quant_type,
        integer(raw, "M"),
        integer(raw, "K"),
        integer(raw, "N"),
        integer(raw, "PhysicalExpertCount"),
        integer(raw, "MaxRouteEntries"),
        integer(raw, "ProjectionCount"),
    )
    contract = GroupedBackwardPairContract.for_problem(problem)
    return KernelInstance(
        KernelFamily.GroupedBackwardPair,
        target,
        problem_type_for_family(KernelFamily.GroupedBackwardPair, quant_type),
        problem,
        GroupedBackwardPairKernelSpec.from_mapping(item["KernelSpec"], contract),
    )


def _fixed_forward(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.FixedGroupedForward)
    raw = strict_mapping(
        item["Problem"],
        "FixedForwardProblem",
        frozenset({"M", "N", "K", "GroupCount"}),
    )
    problem = FixedForwardProblem(
        quant_type,
        integer(raw, "M"),
        integer(raw, "N"),
        integer(raw, "K"),
        integer(raw, "GroupCount"),
    )
    return KernelInstance(
        KernelFamily.FixedGroupedForward,
        target,
        problem_type_for_family(KernelFamily.FixedGroupedForward, quant_type),
        problem,
        FixedForwardKernelSpec.from_mapping(item["KernelSpec"]),
    )


def _fixed_backward(value: object) -> KernelInstance:
    item, target, quant_type = _common(value, KernelFamily.FixedGroupedBackward)
    raw = strict_mapping(
        item["Problem"],
        "FixedBackwardProblem",
        frozenset({"M", "N", "K", "GroupCount"}),
    )
    problem = FixedBackwardProblem(
        quant_type,
        integer(raw, "M"),
        integer(raw, "K"),
        integer(raw, "N"),
        integer(raw, "GroupCount"),
    )
    return KernelInstance(
        KernelFamily.FixedGroupedBackward,
        target,
        problem_type_for_family(KernelFamily.FixedGroupedBackward, quant_type),
        problem,
        FixedBackwardKernelSpec.from_mapping(item["KernelSpec"]),
    )


_PARSERS: Mapping[KernelFamily, Callable[[object], KernelInstance]] = {
    KernelFamily.OrdinaryForward: _ordinary_forward,
    KernelFamily.OrdinaryBackward: _ordinary_backward,
    KernelFamily.GroupedForward: _grouped_forward,
    KernelFamily.GroupedBackward: _grouped_backward,
    KernelFamily.GroupedForwardPair: _grouped_forward_pair,
    KernelFamily.GroupedBackwardPair: _grouped_backward_pair,
    KernelFamily.FixedGroupedForward: _fixed_forward,
    KernelFamily.FixedGroupedBackward: _fixed_backward,
}


def parse_instance(value: object) -> KernelInstance:
    if not isinstance(value, Mapping):
        raise SchemaError("kernel instance must be a mapping")
    family = enum_value(value, "KernelFamily", KernelFamily)
    instance = _PARSERS[family](value)
    from .validation import validate_instance

    validate_instance(instance)
    return instance


def family_for_instance(instance: KernelInstance) -> KernelFamily:
    return instance.family


def _problem_mapping(instance: KernelInstance) -> dict[str, object]:
    family = instance.family
    problem = instance.problem
    if family in {
        KernelFamily.OrdinaryForward,
        KernelFamily.OrdinaryBackward,
        KernelFamily.GroupedBackward,
    }:
        assert isinstance(problem, ProblemSize)
        mapping: dict[str, object] = dict(problem.to_mapping())
        if family is KernelFamily.GroupedBackward:
            mapping.update(PhysicalExpertCount=256, MaxRouteEntries=256)
        return mapping
    if family is KernelFamily.GroupedForward:
        assert isinstance(problem, GroupedForwardProblem)
        return {
            "M": problem.aggregate_rows,
            "N": problem.output_features,
            "K": problem.input_features,
            "PhysicalExpertCount": problem.physical_experts,
            "MaxRouteEntries": problem.max_route_entries,
        }
    if family is KernelFamily.GroupedForwardPair:
        assert isinstance(problem, GroupedForwardPairProblem)
        return {
            "M": problem.aggregate_rows,
            "N": problem.output_features,
            "K": problem.input_features,
            "PhysicalExpertCount": problem.physical_experts,
            "MaxRouteEntries": problem.max_route_entries,
            "ProjectionCount": problem.projection_count,
        }
    if family is KernelFamily.GroupedBackwardPair:
        assert isinstance(problem, GroupedBackwardPairProblem)
        return {
            "M": problem.aggregate_rows,
            "N": problem.in_features,
            "K": problem.out_features,
            "PhysicalExpertCount": problem.physical_experts,
            "MaxRouteEntries": problem.max_route_entries,
            "ProjectionCount": problem.projection_count,
        }
    if family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem)
        return {
            "M": problem.tokens,
            "N": problem.output_features,
            "K": problem.input_features,
            "GroupCount": problem.groups,
        }
    if family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem)
        return {
            "M": problem.tokens,
            "N": problem.input_features,
            "K": problem.output_features,
            "GroupCount": problem.groups,
        }
    raise TypeError(f"unsupported kernel family {family!r}")


def _spec_mapping(instance: KernelInstance) -> dict[str, object]:
    family = instance.family
    spec = instance.kernel_spec
    quant_type = instance.problem_type.quant_data_type
    if family is KernelFamily.OrdinaryForward:
        assert isinstance(spec, ForwardKernelSpec)
        return spec.to_mapping()
    if family is KernelFamily.OrdinaryBackward:
        assert isinstance(spec, BackwardKernelSpec)
        return spec.to_mapping(quant_type)
    if family is KernelFamily.GroupedBackward:
        assert isinstance(spec, GroupedBackwardKernelSpec)
        return spec.to_mapping(quant_type)
    if family is KernelFamily.GroupedForward:
        assert isinstance(spec, GroupedForwardKernelSpec)
        return spec.to_mapping()
    if family is KernelFamily.GroupedForwardPair:
        assert isinstance(spec, GroupedForwardPairKernelSpec)
        return spec.to_mapping()
    if family is KernelFamily.FixedGroupedForward:
        assert isinstance(spec, FixedForwardKernelSpec)
        return spec.to_mapping()
    if family is KernelFamily.FixedGroupedBackward:
        assert isinstance(spec, FixedBackwardKernelSpec)
        return spec.to_mapping()
    if family is KernelFamily.GroupedBackwardPair:
        assert isinstance(instance.problem, GroupedBackwardPairProblem)
        assert isinstance(spec, GroupedBackwardPairKernelSpec)
        contract = GroupedBackwardPairContract.for_problem(instance.problem)
        return spec.to_mapping(contract)
    raise TypeError(f"unsupported kernel family {family!r}")


def mapping_for_instance(instance: KernelInstance) -> dict[str, object]:
    family = instance.family
    quant_type = instance.problem_type.quant_data_type
    return {
        "KernelFamily": family.value,
        "Target": instance.target.to_mapping(),
        "ProblemType": problem_type_mapping(family, quant_type),
        "Problem": _problem_mapping(instance),
        "KernelSpec": _spec_mapping(instance),
    }


def _identity_prefix(family: KernelFamily) -> str:
    if family is KernelFamily.GroupedForwardPair:
        return "ggpair"
    if family in {KernelFamily.GroupedBackwardPair}:
        return "ggbpair"
    return "ggsol"


def instance_hash(instance: KernelInstance) -> str:
    return exact_key_hash(
        mapping_for_instance(instance), prefix=_identity_prefix(instance.family)
    )


def problem_size_for_instance(instance: KernelInstance) -> ProblemSize:
    family = instance.family
    problem = instance.problem
    if family in {
        KernelFamily.OrdinaryForward,
        KernelFamily.OrdinaryBackward,
        KernelFamily.GroupedBackward,
    }:
        assert isinstance(problem, ProblemSize)
        return problem
    if family is KernelFamily.GroupedForward:
        assert isinstance(problem, GroupedForwardProblem)
        return ProblemSize(
            problem.aggregate_rows, problem.output_features, problem.input_features
        )
    if family is KernelFamily.GroupedForwardPair:
        assert isinstance(problem, GroupedForwardPairProblem)
        return ProblemSize(
            problem.aggregate_rows, problem.output_features, problem.input_features
        )
    if family is KernelFamily.GroupedBackwardPair:
        assert isinstance(problem, GroupedBackwardPairProblem)
        return ProblemSize(
            problem.aggregate_rows, problem.in_features, problem.out_features
        )
    if family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem)
        return ProblemSize(
            problem.tokens, problem.output_features, problem.input_features
        )
    if family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem)
        return ProblemSize(
            problem.tokens, problem.input_features, problem.output_features
        )
    raise TypeError(f"unsupported kernel family {family!r}")


def instance_name(instance: KernelInstance) -> str:
    family = instance.family
    problem = instance.problem
    quant = instance.problem_type.quant_data_type.lower()
    digest = instance_hash(instance)
    size = problem_size_for_instance(instance)
    if family is KernelFamily.OrdinaryForward:
        prefix, suffix = "mmq_fwd", digest[6:]
        return f"{prefix}_{quant}_m{size.m}_n{size.n}_k{size.k}_{suffix}"
    if family is KernelFamily.OrdinaryBackward:
        prefix, suffix = "mmq_bwd", digest[6:]
        return f"{prefix}_{quant}_m{size.m}_n{size.n}_k{size.k}_{suffix}"
    if family is KernelFamily.GroupedBackward:
        return f"grouped_mmq_bwd_{quant}_m{size.m}_n{size.n}_k{size.k}_{digest[6:]}"
    if family is KernelFamily.GroupedForward:
        assert isinstance(problem, GroupedForwardProblem)
        return (
            f"grouped_mmq_fwd_{quant}_r{problem.aggregate_rows}_"
            f"n{problem.output_features}_k{problem.input_features}_{digest[6:]}"
        )
    if family is KernelFamily.GroupedForwardPair:
        assert isinstance(problem, GroupedForwardPairProblem)
        return (
            f"grouped_mmq_fwd_pair_{quant}_r{problem.aggregate_rows}_"
            f"n{problem.output_features}_k{problem.input_features}_{digest[7:]}"
        )
    if family is KernelFamily.GroupedBackwardPair:
        assert isinstance(problem, GroupedBackwardPairProblem)
        return (
            f"grouped_mmq_bwd_pair_{quant}_r{problem.aggregate_rows}_"
            f"n{problem.in_features}_k{problem.out_features}_{digest[8:]}"
        )
    if family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem)
        return (
            f"fixed_grouped_mmq_fwd_q8_0_t{problem.tokens}_"
            f"n{problem.output_features}_k{problem.input_features}_{digest[6:]}"
        )
    if family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem)
        return (
            f"fixed_grouped_mmq_bwd_q8_0_t{problem.tokens}_"
            f"n{problem.input_features}_k{problem.output_features}_{digest[6:]}"
        )
    raise TypeError(f"unsupported kernel family {family!r}")


class KernelAbiName(str, Enum):
    OrdinaryForward = "OrdinaryForward"
    OrdinaryBackward = "OrdinaryBackward"
    GroupedForward = "GroupedForward"
    GroupedBackward = "GroupedBackward"
    GroupedForwardPair = "GroupedForwardPair"
    GroupedForwardPairRowTask = "GroupedForwardPairRowTask"
    GroupedBackwardPair = "GroupedBackwardPair"
    FixedGroupedForward = "FixedGroupedForward"
    FixedGroupedBackward = "FixedGroupedBackward"


def abi_for_instance(instance: KernelInstance) -> tuple[KernelAbiName, "KernelAbi"]:
    from .kernel_abi import (
        FIXED_GROUPED_BACKWARD_ABI,
        FIXED_GROUPED_FORWARD_ABI,
        GROUPED_BACKWARD_ABI,
        GROUPED_BACKWARD_PAIR_ABI,
        GROUPED_FORWARD_ABI,
        GROUPED_FORWARD_PAIR_ABI,
        GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
        ORDINARY_BACKWARD_ABI,
        ORDINARY_FORWARD_ABI,
    )

    if (
        instance.family is KernelFamily.GroupedForwardPair
        and isinstance(instance.kernel_spec, GroupedForwardPairKernelSpec)
        and instance.kernel_spec.route_ownership
        is GroupedPairRouteOwnership.DeviceRowTasks
    ):
        return (
            KernelAbiName.GroupedForwardPairRowTask,
            GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
        )
    values = {
        KernelFamily.OrdinaryForward: ORDINARY_FORWARD_ABI,
        KernelFamily.OrdinaryBackward: ORDINARY_BACKWARD_ABI,
        KernelFamily.GroupedForward: GROUPED_FORWARD_ABI,
        KernelFamily.GroupedBackward: GROUPED_BACKWARD_ABI,
        KernelFamily.GroupedForwardPair: GROUPED_FORWARD_PAIR_ABI,
        KernelFamily.GroupedBackwardPair: GROUPED_BACKWARD_PAIR_ABI,
        KernelFamily.FixedGroupedForward: FIXED_GROUPED_FORWARD_ABI,
        KernelFamily.FixedGroupedBackward: FIXED_GROUPED_BACKWARD_ABI,
    }
    return KernelAbiName(instance.family.value), values[instance.family]


def launch_for_instance(instance: KernelInstance):
    from .launch import _derive_launch_metadata

    return _derive_launch_metadata(instance)


def writer_for_instance(
    instance: KernelInstance, toolchain: Toolchain
) -> "AssemblyKernelWriter":
    from .kernel_writer_assembly_fixed_grouped_mmq_bwd import (
        FixedGroupedBackwardKernelWriterAssembly,
    )
    from .kernel_writer_assembly_fixed_grouped_mmq_fwd import (
        FixedGroupedForwardKernelWriterAssembly,
    )
    from .kernel_writer_assembly_grouped_mmq_bwd import (
        GroupedBackwardKernelWriterAssembly,
    )
    from .kernel_writer_assembly_grouped_mmq_bwd_pair import (
        GroupedBackwardPairKernelWriterAssembly,
    )
    from .kernel_writer_assembly_grouped_mmq_fwd import (
        GroupedForwardKernelWriterAssembly,
    )
    from .kernel_writer_assembly_grouped_mmq_fwd_pair import (
        GroupedForwardPairKernelWriterAssembly,
    )
    from .kernel_writer_assembly_mmq_bwd import BackwardKernelWriterAssembly
    from .kernel_writer_assembly_mmq_fwd import ForwardKernelWriterAssembly

    problem = instance.problem
    spec = instance.kernel_spec
    quant_type = instance.problem_type.quant_data_type
    kernel_name = instance_name(instance)
    if instance.family is KernelFamily.OrdinaryForward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, ForwardKernelSpec)
        return ForwardKernelWriterAssembly(
            problem, quant_type, spec, kernel_name, toolchain
        )
    if instance.family is KernelFamily.OrdinaryBackward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, BackwardKernelSpec)
        return BackwardKernelWriterAssembly(
            problem, quant_type, spec, kernel_name, toolchain
        )
    if instance.family is KernelFamily.GroupedForward:
        assert isinstance(problem, GroupedForwardProblem) and isinstance(
            spec, GroupedForwardKernelSpec
        )
        return GroupedForwardKernelWriterAssembly(problem, spec, kernel_name, toolchain)
    if instance.family is KernelFamily.GroupedBackward:
        assert isinstance(problem, ProblemSize) and isinstance(
            spec, GroupedBackwardKernelSpec
        )
        return GroupedBackwardKernelWriterAssembly(
            problem, quant_type, spec, kernel_name, toolchain
        )
    if instance.family is KernelFamily.GroupedForwardPair:
        assert isinstance(problem, GroupedForwardPairProblem) and isinstance(
            spec, GroupedForwardPairKernelSpec
        )
        return GroupedForwardPairKernelWriterAssembly(
            problem, spec, kernel_name, toolchain
        )
    if instance.family is KernelFamily.GroupedBackwardPair:
        assert isinstance(problem, GroupedBackwardPairProblem) and isinstance(
            spec, GroupedBackwardPairKernelSpec
        )
        return GroupedBackwardPairKernelWriterAssembly(
            problem, spec, kernel_name, toolchain
        )
    if instance.family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem) and isinstance(
            spec, FixedForwardKernelSpec
        )
        return FixedGroupedForwardKernelWriterAssembly(
            problem, spec, kernel_name, toolchain
        )
    if instance.family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem) and isinstance(
            spec, FixedBackwardKernelSpec
        )
        return FixedGroupedBackwardKernelWriterAssembly(
            problem, spec, kernel_name, toolchain
        )
    raise TypeError(f"unsupported kernel family {instance.family!r}")
