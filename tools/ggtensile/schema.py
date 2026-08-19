"""Strict primitives for GGTensile's unversioned serialized boundary."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Literal, TypeVar, overload


class SchemaError(ValueError):
    """A GGTensile input does not match its strict schema."""


EnumT = TypeVar("EnumT", bound=Enum)


def strict_mapping(
    value: object,
    name: str,
    keys: frozenset[str],
    *,
    error_type: type[ValueError] = SchemaError,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error_type(f"{name} must be a mapping")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise error_type(f"{name} keys must be strings")
        normalized[key] = item
    actual = set(normalized)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise error_type(f"invalid {name}: {', '.join(details)}")
    return normalized


def strict_mapping_optional(
    value: object,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SchemaError(f"{name} keys must be strings")
        normalized[key] = item
    actual = set(normalized)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise SchemaError(f"invalid {name}: {', '.join(details)}")
    return normalized


def canonical_value(value: object) -> object:
    """Project typed records without null inactive-policy sentinels."""
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for field in fields(value):
            item = getattr(value, field.name)
            if item is None:
                continue
            projected = canonical_value(item)
            if isinstance(projected, dict) and not projected:
                continue
            result[field.name] = projected
        return result
    if isinstance(value, tuple | list):
        return [canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in value.items()}
    return value


def string(value: object, name: str) -> str:
    if type(value) is not str:
        raise SchemaError(f"{name} must be str, not {type(value).__name__}")
    return value


def integer(
    value: object,
    name: str,
    *,
    error_type: type[ValueError] = SchemaError,
) -> int:
    if type(value) is not int:
        raise error_type(f"{name} must be int, not {type(value).__name__}")
    return value


def boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{name} must be bool, not {type(value).__name__}")
    return value


@overload
def integer_tuple(
    value: object, name: str, length: Literal[3]
) -> tuple[int, int, int]: ...


@overload
def integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]: ...


def integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError(f"{name} must be a {length}-element list")
    return tuple(integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def integer_triple(value: object, name: str) -> tuple[int, int, int]:
    values = integer_tuple(value, name, 3)
    return values[0], values[1], values[2]


def enum_value(value: object, name: str, enum_type: type[EnumT]) -> EnumT:
    serialized = string(value, name)
    if serialized not in enum_type._value2member_map_:
        raise SchemaError(f"{name} has unsupported value {serialized!r}") from None
    return enum_type(serialized)
