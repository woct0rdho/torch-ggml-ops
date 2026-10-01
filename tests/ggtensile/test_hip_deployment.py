"""The HIP deployment catalog is the single source of truth for selection."""

import json

import pytest

from tools.ggtensile.hip_deployment import (
    CONFIG_PATH,
    SCHEMA,
    deployment_table,
    select_hip_control,
)
from tools.mmq_deployment_cases import public_deployment_cases
from tools.mmq_hip_control_spec import hip_control_specs

_CATALOGUED_OPERATIONS = (
    "OrdinaryForward",
    "OrdinaryBackward",
    "FixedGroupedForward",
    "FixedGroupedBackward",
)


def _public_keys() -> set[tuple[str, str, int, int, int]]:
    keys = set()
    for case in public_deployment_cases():
        if case.operation not in _CATALOGUED_OPERATIONS:
            continue
        if case.operation == "OrdinaryBackward":
            # Problem axes: the backward gradient contracts the forward output.
            keys.add(
                (
                    case.operation,
                    case.quant_type,
                    case.rows,
                    case.in_features,
                    case.out_features,
                )
            )
        else:
            keys.add(
                (
                    case.operation,
                    case.quant_type,
                    case.rows,
                    case.out_features,
                    case.in_features,
                )
            )
    return keys


def test_catalog_declares_its_schema() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA


def test_catalog_keys_are_exactly_the_public_keys() -> None:
    assert set(deployment_table()) == _public_keys()


def test_every_catalogued_symbol_is_a_built_control() -> None:
    inventory = {spec.symbol for spec in hip_control_specs()}
    for key, symbol in sorted(deployment_table().items()):
        assert symbol in inventory, f"{key} selects unbuilt control {symbol}"


def test_every_public_key_selects_a_built_control() -> None:
    inventory = {spec.symbol for spec in hip_control_specs()}
    for operation, quant_type, m, n, k in sorted(_public_keys()):
        control = select_hip_control(operation, quant_type, m, n, k)
        assert control.symbol in inventory


def test_unknown_key_fails_closed() -> None:
    with pytest.raises(ValueError, match="no deployed HIP control"):
        select_hip_control("OrdinaryForward", "Q4_K", 7, 512, 2048)


def test_dense_forward_launch_geometry_follows_the_control_config() -> None:
    control = select_hip_control("OrdinaryForward", "Q4_K", 2048, 512, 2048)
    grid, block, shared = control.launch_configuration(2048, 512, 2048)
    assert grid == (8, 16, 1)
    assert block == (32, 4, 1)
    assert shared == 38_400

    small = select_hip_control("OrdinaryForward", "Q8_0", 32, 129280, 4096)
    assert small.symbol.endswith("j64_bounded")
    grid, block, shared = small.launch_configuration(32, 129280, 4096)
    assert grid == (2020, 1, 1)
    assert block == (32, 4, 1)
    assert shared == 28_928


def test_dense_backward_launch_geometry_follows_the_control_config() -> None:
    control = select_hip_control("OrdinaryBackward", "Q4_K", 2048, 2048, 512)
    grid, block, shared = control.launch_configuration(2048, 2048, 512)
    assert grid == (1, 16, 16)
    assert block == (128, 1, 1)
    assert shared == 0


def test_fixed_grouped_selection_uses_the_tuned_bodies() -> None:
    forward = select_hip_control("FixedGroupedForward", "Q8_0", 2048, 1024, 4096)
    assert forward.symbol == "grouped_fwd_fixed_q8_0_g8_k4096_j64_full"
    backward = select_hip_control("FixedGroupedBackward", "Q8_0", 2048, 1024, 4096)
    assert backward.symbol == "grouped_bwd_tuned_fixed_q8_0_g8_k4096_mt192_nt64"
    assert backward.fixed_m_tile() == 192


def test_launch_configuration_rejects_underived_families() -> None:
    forward = select_hip_control("FixedGroupedForward", "Q8_0", 2048, 1024, 4096)
    with pytest.raises(ValueError, match="no derived launch configuration"):
        forward.launch_configuration(2048, 1024, 4096)
