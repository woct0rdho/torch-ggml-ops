"""The HIP deployment catalog is the single source of truth for selection."""

import pytest

from tools.mmq_deployment_cases import public_deployment_cases
from tools.mmq_hip_control_spec import hip_control_specs
from tools.mmq_hip_deployment import (
    RoutedControl,
    control_inventory,
    deployment_table,
    routed_table,
    select_grouped_forward_control,
    select_hip_control,
    select_routed_control,
)

_CATALOGUED_OPERATIONS = (
    "OrdinaryForward",
    "OrdinaryBackward",
    "FixedGroupedForward",
    "FixedGroupedBackward",
)

_ROUTED_OPERATIONS = (
    "GroupedForward",
    "GroupedForwardPair",
    "GroupedBackward",
    "GroupedBackwardPair",
)

_ROUTED_SYMBOL_PREFIXES = {
    "GroupedForward": "grouped_fwd_",
    "GroupedForwardPair": "grouped_fwd_",
    "GroupedBackward": "grouped_bwd_",
    "GroupedBackwardPair": "grouped_bwd_",
}


def _routed_cases() -> list:
    return [
        case
        for case in public_deployment_cases()
        if case.operation in _ROUTED_OPERATIONS
    ]


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


def test_catalog_keys_cover_the_public_keys() -> None:
    missing = _public_keys() - set(deployment_table())
    assert not missing, f"public keys without a deployed control: {sorted(missing)}"


def _hip_only_keys() -> set[tuple[str, str, int, int, int]]:
    """Return deployment keys whose shapes have no GGTensile problem key yet.

    The Qwen4-Exp and GatedDeltaNet forward shapes get HIP controls first, so
    they are selectable before any GGTensile kernel covers them. Every other
    deployment key must be a public key, and a HIP-only key must still select a
    built control for a whole number of 64-row tiles.
    """

    return set(deployment_table()) - _public_keys()


def test_extra_catalog_keys_are_hip_only_forward_shapes() -> None:
    inventory = {spec.symbol for spec in hip_control_specs()}
    for operation, quant_type, m, n, k in sorted(_hip_only_keys()):
        assert operation == "OrdinaryForward"
        assert quant_type in {
            "IQ4_NL",
            "IQ4_XS",
            "Q2_0",
            "Q3_K",
            "Q4_0",
            "Q4_K",
            "Q5_0",
            "Q5_K",
            "Q6_K",
            "Q8_0",
        }
        control = select_hip_control(operation, quant_type, m, n, k)
        assert control.symbol in inventory
        assert n % 64 == 0


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


def test_grouped_forward_rules_select_only_built_controls() -> None:
    inventory = control_inventory()
    families: dict[tuple[str, int, int], list[tuple[int, str]]] = {}
    for control in routed_table():
        if control.operation != "GroupedForward":
            continue
        assert control.symbol in inventory, (
            f"{control.quant_type} selects unbuilt control {control.symbol}"
        )
        families.setdefault(
            (control.quant_type, control.out_features, control.in_features), []
        ).append((control.value or 0, control.symbol))
    for key, rules in families.items():
        assert rules[-1][0] == 0, f"{key} has no unconditional default rule"
        defaults = {symbol for value, symbol in rules if value == 0}
        assert len(defaults) == 1
        assert defaults.isdisjoint({symbol for value, symbol in rules if value != 0}), (
            f"{key} reuses its default body in a conditional rule"
        )


def test_every_routed_symbol_is_a_built_control() -> None:
    inventory = {spec.symbol for spec in hip_control_specs()}
    for control in routed_table():
        assert control.symbol in inventory, (
            f"{control.key} selects unbuilt control {control.symbol}"
        )


def test_routed_families_cover_the_public_routed_keys() -> None:
    catalogued = {control.key for control in routed_table()}
    public = {
        (case.operation, case.quant_type, case.out_features, case.in_features)
        for case in _routed_cases()
    }
    assert catalogued == public


def test_routed_rules_shape_and_defaults_are_consistent() -> None:
    families: dict[tuple[str, str, int, int], list[RoutedControl]] = {}
    for control in routed_table():
        families.setdefault(control.key, []).append(control)
    for key, rules in families.items():
        assert rules, key
        for rule in rules:
            assert rule.symbol.startswith(_ROUTED_SYMBOL_PREFIXES[key[0]]), rule
            if rule.when == "always":
                assert rule.value is None, rule
            else:
                assert rule.value is not None and rule.value > 0, rule
        if all(rule.when == "rows_eq" for rule in rules):
            assert len({rule.value for rule in rules}) == len(rules), key
        assert [rule.when for rule in rules].count("always") <= 1, key


def test_every_public_routed_case_selects_a_built_control() -> None:
    inventory = {spec.symbol for spec in hip_control_specs()}
    for case in _routed_cases():
        choice = select_routed_control(
            case.operation,
            case.quant_type,
            case.out_features,
            case.in_features,
            case.rows,
            256,
        )
        assert choice.symbol in inventory, (
            f"{case.name} selects unbuilt control {choice.symbol}"
        )


def test_paired_controls_carry_their_tile() -> None:
    for control in routed_table():
        if control.operation == "GroupedBackwardPair":
            assert control.tile in {64, 128}, control
        if (
            control.operation == "GroupedForwardPair"
            and control.quant_type == "IQ2_XXS"
        ):
            assert control.tile in {64, 80}, control


@pytest.mark.parametrize(
    ("quant_type", "out_features", "in_features", "rows", "expected"),
    (
        ("Q4_K", 2048, 512, 16_384, "grouped_fwd_serial_q4_k_n2048_k512_j32"),
        ("Q4_K", 2048, 512, 65_536, "grouped_fwd_serial_q4_k_n2048_k512_j64"),
        ("Q4_K", 2048, 512, 262_144, "grouped_fwd_serial_q4_k_n2048_k512_j64"),
        ("Q5_K", 2048, 512, 16_384, "grouped_fwd_serial_q5_k_n2048_k512_j32"),
        ("Q5_K", 2048, 512, 65_536, "grouped_fwd_serial_q5_k_n2048_k512_j64"),
        ("IQ2_S", 2048, 512, 16_384, "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32"),
        ("IQ2_S", 2048, 512, 65_536, "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32"),
        ("IQ2_S", 2048, 512, 262_144, "grouped_fwd_serial_iq2_s_n2048_k512_j64"),
        ("Q2_K", 4096, 2048, 12_288, "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16"),
        ("Q2_K", 4096, 2048, 49_152, "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16"),
        ("Q2_K", 4096, 2048, 196_608, "grouped_fwd_serial_q2_k_n4096_k2048_j32"),
    ),
)
def test_grouped_forward_policy_matches_the_deployed_bodies(
    quant_type: str, out_features: int, in_features: int, rows: int, expected: str
) -> None:
    control = select_grouped_forward_control(
        quant_type, out_features, in_features, rows, 256
    )
    assert control.symbol == expected
    grid, block, shared = control.launch_configuration(out_features, 256)
    assert grid == (out_features // 64, 256, 1)
    assert block == (32, 4, 1)
    assert shared == control.lds_bytes


def test_grouped_forward_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="no deployed HIP routed control"):
        select_grouped_forward_control("Q6_K", 2048, 512, 16_384, 256)
    with pytest.raises(ValueError, match="no deployed HIP routed control"):
        select_grouped_forward_control("Q4_K", 1024, 512, 16_384, 256)
    with pytest.raises(ValueError, match="at least one route entry"):
        select_grouped_forward_control("Q4_K", 2048, 512, 16_384, 0)
    control = select_grouped_forward_control("Q4_K", 2048, 512, 16_384, 256)
    with pytest.raises(ValueError, match="at least one route entry"):
        control.launch_configuration(2048, 0)
