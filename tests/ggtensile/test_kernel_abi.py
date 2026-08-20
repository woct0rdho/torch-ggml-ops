import ctypes

import pytest

from tools.ggtensile.kernel_abi import (
    FIXED_GROUPED_BACKWARD_ABI,
    FIXED_GROUPED_FORWARD_ABI,
    GROUPED_FORWARD_ABI,
    GROUPED_FORWARD_PAIR_ABI,
    GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
    GROUPED_FORWARD_ROW_TASK_ABI,
    GROUPED_ROW_TASK_SETUP_ABI,
    ORDINARY_BACKWARD_ABI,
    ORDINARY_FORWARD_ABI,
    Q8_1_QUANTIZER_ABI,
    KernelAbi,
    KernelArgumentKind,
    KernelValueType,
)


@pytest.mark.parametrize(
    ("abi", "names", "offsets", "segment_size"),
    (
        (
            ORDINARY_FORWARD_ABI,
            (
                "packed_weight",
                "activations",
                "output",
                "nrows_weight",
                "nrows_activation",
                "nrows_activation_padded",
                "blocks_per_weight_row",
            ),
            (0, 8, 16, 24, 28, 32, 36),
            40,
        ),
        (
            ORDINARY_BACKWARD_ABI,
            (
                "grad_output",
                "packed_weight",
                "grad_input",
                "rows",
                "out_features",
                "in_features",
                "blocks_per_weight_row",
            ),
            (0, 8, 16, 24, 28, 32, 36),
            40,
        ),
        (
            GROUPED_FORWARD_ABI,
            (
                "weights",
                "activations",
                "dst",
                "expert_indices",
                "expert_offsets",
                "num_experts",
                "nrows_weight",
                "nrows_activation",
                "blocks_per_weight_row",
                "bytes_per_expert",
            ),
            (0, 8, 16, 24, 32, 40, 44, 48, 52, 56),
            64,
        ),
        (
            GROUPED_FORWARD_PAIR_ABI,
            (
                "weights_first",
                "weights_second",
                "activations",
                "dst_first",
                "dst_second",
                "expert_indices",
                "expert_offsets",
                "num_experts",
                "nrows_weight",
                "nrows_activation",
                "blocks_per_weight_row",
                "bytes_per_expert",
            ),
            (0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 68, 72),
            80,
        ),
        (
            GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
            (
                "weights_first",
                "weights_second",
                "activations",
                "dst_first",
                "dst_second",
                "task_count",
                "task_experts",
                "task_row_starts",
                "task_row_ends",
                "num_experts",
                "nrows_weight",
                "nrows_activation",
                "blocks_per_weight_row",
                "bytes_per_expert",
            ),
            (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 76, 80, 84, 88),
            96,
        ),
        (
            GROUPED_FORWARD_ROW_TASK_ABI,
            (
                "weights",
                "activations",
                "dst",
                "task_count",
                "task_experts",
                "task_row_starts",
                "task_row_ends",
                "nrows_activation",
                "bytes_per_expert",
            ),
            (0, 8, 16, 24, 32, 40, 48, 56, 64),
            72,
        ),
        (
            GROUPED_ROW_TASK_SETUP_ABI,
            (
                "expert_indices",
                "expert_offsets",
                "task_count",
                "task_experts",
                "task_row_starts",
                "task_row_ends",
                "num_experts",
                "num_groups",
                "nrows_activation",
                "row_tile",
            ),
            (0, 8, 16, 24, 32, 40, 48, 52, 56, 60),
            64,
        ),
        (
            Q8_1_QUANTIZER_ABI,
            ("input", "output", "rows", "rows_padded", "k"),
            (0, 8, 16, 24, 32),
            40,
        ),
        (
            FIXED_GROUPED_FORWARD_ABI,
            (
                "packed_weight",
                "activations",
                "output",
                "tokens",
                "out_features",
                "bytes_per_group",
            ),
            (0, 8, 16, 24, 28, 32),
            40,
        ),
        (
            FIXED_GROUPED_BACKWARD_ABI,
            (
                "grad_output",
                "packed_weight",
                "grad_input",
                "tokens",
                "out_features",
                "bytes_per_group",
            ),
            (0, 8, 16, 24, 28, 32),
            40,
        ),
    ),
)
def test_kernel_abi_layout_is_exact(
    abi: KernelAbi,
    names: tuple[str, ...],
    offsets: tuple[int, ...],
    segment_size: int,
) -> None:
    assert tuple(item.argument.name for item in abi.layout) == names
    assert tuple(item.offset for item in abi.layout) == offsets
    assert abi.segment_size == segment_size
    assert abi.segment_alignment == 8


def test_kernel_abi_packs_runtime_arguments_in_layout_order() -> None:
    values = {
        argument.name: index + 1
        for index, argument in enumerate(FIXED_GROUPED_FORWARD_ABI.arguments)
    }
    packed = FIXED_GROUPED_FORWARD_ABI.pack(values)
    assert tuple(argument.value for argument in packed.arguments) == tuple(
        values[argument.name] for argument in FIXED_GROUPED_FORWARD_ABI.arguments
    )
    assert all(
        isinstance(argument, ctypes.c_uint64)
        for argument, value in zip(
            packed.arguments,
            FIXED_GROUPED_FORWARD_ABI.arguments,
            strict=True,
        )
        if value.kind is KernelArgumentKind.GlobalBuffer
    )
    assert len(packed.parameters) == len(FIXED_GROUPED_FORWARD_ABI.arguments)


def test_installed_helper_abis_preserve_signed_source_types() -> None:
    assert tuple(
        item.argument.value_type for item in GROUPED_FORWARD_ROW_TASK_ABI.layout[7:]
    ) == (KernelValueType.Int32, KernelValueType.Int64)
    assert (
        tuple(
            item.argument.value_type for item in GROUPED_ROW_TASK_SETUP_ABI.layout[6:]
        )
        == (KernelValueType.Int32,) * 4
    )
    assert (
        tuple(item.argument.value_type for item in Q8_1_QUANTIZER_ABI.layout[2:])
        == (KernelValueType.Int64,) * 3
    )


def test_kernel_abi_runtime_packing_rejects_missing_and_unknown_values() -> None:
    with pytest.raises(ValueError, match="missing=.*output"):
        FIXED_GROUPED_FORWARD_ABI.pack(
            {
                "packed_weight": 1,
                "activations": 2,
                "tokens": 3,
                "out_features": 4,
                "bytes_per_group": 5,
            }
        )
    with pytest.raises(ValueError, match="unknown=.*extra"):
        FIXED_GROUPED_FORWARD_ABI.pack(
            {
                "packed_weight": 1,
                "activations": 2,
                "output": 3,
                "tokens": 4,
                "out_features": 5,
                "bytes_per_group": 6,
                "extra": 7,
            }
        )
