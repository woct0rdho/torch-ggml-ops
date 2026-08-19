"""Typed kernel argument layouts shared by emission, launch, and inspection."""

import ctypes
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class KernelArgumentKind(Enum):
    GlobalBuffer = "global_buffer"
    ByValue = "by_value"


class KernelValueType(Enum):
    Struct = "struct"
    BFloat16 = "bf16"
    Int32 = "i32"
    Int64 = "i64"
    UInt32 = "u32"
    UInt64 = "u64"


@dataclass(frozen=True)
class KernelArgument:
    name: str
    kind: KernelArgumentKind
    value_type: KernelValueType

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("kernel argument name must not be empty")
        if self.kind is KernelArgumentKind.ByValue and self.value_type not in {
            KernelValueType.Int32,
            KernelValueType.Int64,
            KernelValueType.UInt32,
            KernelValueType.UInt64,
        }:
            raise ValueError("by-value kernel arguments require an integer type")

    @property
    def size(self) -> int:
        if self.kind is KernelArgumentKind.GlobalBuffer:
            return 8
        return {
            KernelValueType.Int32: 4,
            KernelValueType.Int64: 8,
            KernelValueType.UInt32: 4,
            KernelValueType.UInt64: 8,
        }[self.value_type]

    @property
    def alignment(self) -> int:
        return self.size


@dataclass(frozen=True)
class KernelArgumentLayout:
    argument: KernelArgument
    offset: int

    @property
    def metadata_tuple(self) -> tuple[str, int, int, str, str]:
        return (
            self.argument.name,
            self.offset,
            self.argument.size,
            self.argument.kind.value,
            self.argument.value_type.value,
        )


@dataclass(frozen=True)
class PackedKernelArguments:
    arguments: tuple[Any, ...]
    parameters: Any


@dataclass(frozen=True)
class KernelAbi:
    arguments: tuple[KernelArgument, ...]
    segment_alignment: int = 8

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("kernel ABI requires at least one argument")
        names = tuple(argument.name for argument in self.arguments)
        if len(names) != len(set(names)):
            raise ValueError("kernel ABI argument names must be unique")
        if self.segment_alignment <= 0:
            raise ValueError("kernel ABI segment alignment must be positive")
        if any(
            argument.alignment > self.segment_alignment for argument in self.arguments
        ):
            raise ValueError("kernel argument alignment exceeds segment alignment")

    @property
    def layout(self) -> tuple[KernelArgumentLayout, ...]:
        offset = 0
        result: list[KernelArgumentLayout] = []
        for argument in self.arguments:
            offset = _align(offset, argument.alignment)
            result.append(KernelArgumentLayout(argument, offset))
            offset += argument.size
        return tuple(result)

    @property
    def segment_size(self) -> int:
        final = self.layout[-1]
        return _align(
            final.offset + final.argument.size,
            self.segment_alignment,
        )

    @property
    def metadata_arguments(self) -> tuple[tuple[str, int, int, str, str], ...]:
        return tuple(item.metadata_tuple for item in self.layout)

    def argument(self, name: str) -> KernelArgumentLayout:
        for item in self.layout:
            if item.argument.name == name:
                return item
        raise KeyError(name)

    def offset(self, name: str) -> int:
        return self.argument(name).offset

    def pack(self, values: Mapping[str, int]) -> PackedKernelArguments:
        expected = {argument.name for argument in self.arguments}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"kernel ABI values do not match: missing={missing}, unknown={unknown}"
            )
        arguments = tuple(
            _ctype_value(argument, values[argument.name]) for argument in self.arguments
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        return PackedKernelArguments(arguments, parameters)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ctype_value(argument: KernelArgument, value: int) -> Any:
    if argument.kind is KernelArgumentKind.GlobalBuffer:
        return ctypes.c_uint64(value)
    constructor = {
        KernelValueType.Int32: ctypes.c_int32,
        KernelValueType.Int64: ctypes.c_int64,
        KernelValueType.UInt32: ctypes.c_uint32,
        KernelValueType.UInt64: ctypes.c_uint64,
    }[argument.value_type]
    return constructor(value)


def _pointer(name: str, value_type: KernelValueType) -> KernelArgument:
    return KernelArgument(name, KernelArgumentKind.GlobalBuffer, value_type)


def _value(name: str, value_type: KernelValueType) -> KernelArgument:
    return KernelArgument(name, KernelArgumentKind.ByValue, value_type)


ORDINARY_FORWARD_ABI = KernelAbi(
    (
        _pointer("packed_weight", KernelValueType.Struct),
        _pointer("activations", KernelValueType.Struct),
        _pointer("output", KernelValueType.BFloat16),
        _value("nrows_weight", KernelValueType.UInt32),
        _value("nrows_activation", KernelValueType.UInt32),
        _value("nrows_activation_padded", KernelValueType.UInt32),
        _value("blocks_per_weight_row", KernelValueType.UInt32),
    )
)

ORDINARY_BACKWARD_ABI = KernelAbi(
    (
        _pointer("grad_output", KernelValueType.BFloat16),
        _pointer("packed_weight", KernelValueType.Struct),
        _pointer("grad_input", KernelValueType.BFloat16),
        _value("rows", KernelValueType.UInt32),
        _value("out_features", KernelValueType.UInt32),
        _value("in_features", KernelValueType.UInt32),
        _value("blocks_per_weight_row", KernelValueType.UInt32),
    )
)

GROUPED_BACKWARD_ABI = KernelAbi(
    (
        _pointer("grad_output", KernelValueType.BFloat16),
        _pointer("packed_weight", KernelValueType.Struct),
        _pointer("grad_input", KernelValueType.BFloat16),
        _pointer("expert_indices", KernelValueType.Int64),
        _pointer("expert_offsets", KernelValueType.Int32),
        _value("num_experts", KernelValueType.Int32),
        _value("rows", KernelValueType.Int32),
        _value("bytes_per_expert", KernelValueType.Int64),
    )
)

GROUPED_BACKWARD_PAIR_ABI = KernelAbi(
    (
        _pointer("first_grad_output", KernelValueType.BFloat16),
        _pointer("second_grad_output", KernelValueType.BFloat16),
        _pointer("first_packed_weight", KernelValueType.Struct),
        _pointer("second_packed_weight", KernelValueType.Struct),
        _pointer("grad_input", KernelValueType.BFloat16),
        _pointer("expert_indices", KernelValueType.Int64),
        _pointer("expert_offsets", KernelValueType.Int32),
        _value("num_experts", KernelValueType.Int32),
        _value("rows", KernelValueType.Int32),
        _value("bytes_per_expert", KernelValueType.Int64),
    )
)

GROUPED_FORWARD_ABI = KernelAbi(
    (
        _pointer("weights", KernelValueType.Struct),
        _pointer("activations", KernelValueType.Struct),
        _pointer("dst", KernelValueType.BFloat16),
        _pointer("expert_indices", KernelValueType.Int64),
        _pointer("expert_offsets", KernelValueType.Int32),
        _value("num_experts", KernelValueType.UInt32),
        _value("nrows_weight", KernelValueType.UInt32),
        _value("nrows_activation", KernelValueType.UInt32),
        _value("blocks_per_weight_row", KernelValueType.UInt32),
        _value("bytes_per_expert", KernelValueType.UInt64),
    )
)

GROUPED_FORWARD_PAIR_ABI = KernelAbi(
    (
        _pointer("weights_first", KernelValueType.Struct),
        _pointer("weights_second", KernelValueType.Struct),
        _pointer("activations", KernelValueType.Struct),
        _pointer("dst_first", KernelValueType.BFloat16),
        _pointer("dst_second", KernelValueType.BFloat16),
        _pointer("expert_indices", KernelValueType.Int64),
        _pointer("expert_offsets", KernelValueType.Int32),
        _value("num_experts", KernelValueType.UInt32),
        _value("nrows_weight", KernelValueType.UInt32),
        _value("nrows_activation", KernelValueType.UInt32),
        _value("blocks_per_weight_row", KernelValueType.UInt32),
        _value("bytes_per_expert", KernelValueType.UInt64),
    )
)

GROUPED_FORWARD_PAIR_ROW_TASK_ABI = KernelAbi(
    (
        _pointer("weights_first", KernelValueType.Struct),
        _pointer("weights_second", KernelValueType.Struct),
        _pointer("activations", KernelValueType.Struct),
        _pointer("dst_first", KernelValueType.BFloat16),
        _pointer("dst_second", KernelValueType.BFloat16),
        _pointer("task_count", KernelValueType.Int32),
        _pointer("task_experts", KernelValueType.Int32),
        _pointer("task_row_starts", KernelValueType.Int32),
        _pointer("task_row_ends", KernelValueType.Int32),
        _value("num_experts", KernelValueType.UInt32),
        _value("nrows_weight", KernelValueType.UInt32),
        _value("nrows_activation", KernelValueType.UInt32),
        _value("blocks_per_weight_row", KernelValueType.UInt32),
        _value("bytes_per_expert", KernelValueType.UInt64),
    )
)

GROUPED_FORWARD_ROW_TASK_ABI = KernelAbi(
    (
        _pointer("weights", KernelValueType.Struct),
        _pointer("activations", KernelValueType.Struct),
        _pointer("dst", KernelValueType.BFloat16),
        _pointer("task_count", KernelValueType.Int32),
        _pointer("task_experts", KernelValueType.Int32),
        _pointer("task_row_starts", KernelValueType.Int32),
        _pointer("task_row_ends", KernelValueType.Int32),
        _value("nrows_activation", KernelValueType.Int32),
        _value("bytes_per_expert", KernelValueType.Int64),
    )
)

GROUPED_ROW_TASK_SETUP_ABI = KernelAbi(
    (
        _pointer("expert_indices", KernelValueType.Int64),
        _pointer("expert_offsets", KernelValueType.Int32),
        _pointer("task_count", KernelValueType.Int32),
        _pointer("task_experts", KernelValueType.Int32),
        _pointer("task_row_starts", KernelValueType.Int32),
        _pointer("task_row_ends", KernelValueType.Int32),
        _value("num_experts", KernelValueType.Int32),
        _value("num_groups", KernelValueType.Int32),
        _value("nrows_activation", KernelValueType.Int32),
        _value("row_tile", KernelValueType.Int32),
    )
)

Q8_1_QUANTIZER_ABI = KernelAbi(
    (
        _pointer("input", KernelValueType.BFloat16),
        _pointer("output", KernelValueType.Struct),
        _value("rows", KernelValueType.Int64),
        _value("rows_padded", KernelValueType.Int64),
        _value("k", KernelValueType.Int64),
    )
)

FIXED_GROUPED_FORWARD_ABI = KernelAbi(
    (
        _pointer("packed_weight", KernelValueType.Struct),
        _pointer("activations", KernelValueType.Struct),
        _pointer("output", KernelValueType.BFloat16),
        _value("tokens", KernelValueType.UInt32),
        _value("out_features", KernelValueType.UInt32),
        _value("bytes_per_group", KernelValueType.UInt64),
    )
)
