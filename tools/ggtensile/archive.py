"""Strict, lossless loaders for the two research archive schemas.

Archives describe candidate evidence.  They are deliberately separate from the
checked-in deployment inventory and never select a runtime kernel.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from .grouped_mmq_fwd_model import GroupedForwardSolutionKey
from .grouped_mmq_fwd_pair_model import GroupedForwardPairSolutionKey
from .model import SolutionKey
from .schema import SchemaError, integer, strict_mapping


class ArchiveError(SchemaError):
    """An archive manifest does not match its declared schema."""


_FINAL_FAMILIES = frozenset(
    {
        "ordinary_forward",
        "ordinary_backward",
        "grouped_forward",
        "grouped_forward_pair",
        "grouped_backward",
        "fixed_grouped_forward",
    }
)
_FINAL_RECORD_KEYS = frozenset(
    {
        "artifact_stem",
        "code_object_sha256",
        "diagnostic_mode",
        "exact_key",
        "family",
        "identity_hash",
        "kernel_name",
        "metadata",
        "source_sha256",
        "static_counts",
    }
)
_EXTENDED_ROOT_KEYS = frozenset(
    {
        "fixed_count",
        "grouped_count",
        "ordinary_backward_count",
        "ordinary_forward_count",
        "paired_count",
        "record_count",
        "records",
    }
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ArchiveError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ArchiveError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _records(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise ArchiveError(f"{name} must be a JSON list")
    return cast(list[object], value)


def _open_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArchiveError(f"{name} must be a mapping")
    return strict_mapping(value, name, frozenset(value), error_type=ArchiveError)


def _exact_key(family: str, value: object) -> object:
    if family in {"ordinary_forward", "ordinary_backward", "grouped_backward"}:
        return SolutionKey.from_mapping(value)
    if family == "grouped_forward":
        return GroupedForwardSolutionKey.from_mapping(value)
    if family == "grouped_forward_pair":
        return GroupedForwardPairSolutionKey.from_mapping(value)
    if family == "fixed_grouped_forward":
        return FixedForwardSolutionKey.from_mapping(value)
    raise ArchiveError(f"unsupported final archive family {family!r}")


@dataclass(frozen=True)
class FinalArchiveRecord:
    family: str
    identity_hash: str
    kernel_name: str
    artifact_stem: str
    exact_key_mapping: Mapping[str, object]
    exact_key: object | None
    exact_key_rejection: str | None
    diagnostic_mode: str | None
    source_sha256: str
    code_object_sha256: str
    metadata: Mapping[str, object]
    static_counts: Mapping[str, object]


@dataclass(frozen=True)
class FinalArchive:
    assembler_path: str
    assembler_sha256: str
    assembler_version: str
    records: tuple[FinalArchiveRecord, ...]


def load_final_archive(path: Path) -> FinalArchive:
    root = strict_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "final archive",
        frozenset(
            {
                "assembler_path",
                "assembler_sha256",
                "assembler_version",
                "record_count",
                "records",
            }
        ),
        error_type=ArchiveError,
    )
    raw_records = _records(root["records"], "final archive records")
    if integer(root["record_count"], "record_count", error_type=ArchiveError) != len(
        raw_records
    ):
        raise ArchiveError("final archive record_count does not match records")
    records: list[FinalArchiveRecord] = []
    for index, value in enumerate(raw_records):
        item = strict_mapping(
            value, f"records[{index}]", _FINAL_RECORD_KEYS, error_type=ArchiveError
        )
        family = _text(item["family"], f"records[{index}].family")
        if family not in _FINAL_FAMILIES:
            raise ArchiveError(f"records[{index}].family is unsupported: {family!r}")
        diagnostic = item["diagnostic_mode"]
        if diagnostic is not None:
            diagnostic = _text(diagnostic, f"records[{index}].diagnostic_mode")
        key_mapping = _open_mapping(item["exact_key"], f"records[{index}].exact_key")
        # Diagnostic records intentionally preserve rejected identities verbatim.
        typed_key = None
        key_rejection = None
        spec = key_mapping.get("KernelSpec")
        decode = spec.get("decode") if isinstance(spec, Mapping) else None
        contract = key_mapping.get("ProblemContract")
        known_inactive_q3 = (
            family == "ordinary_backward"
            and isinstance(decode, Mapping)
            and decode.get("pairing") == "Inactive"
            and isinstance(contract, Mapping)
            and contract.get("quant_type") == "Q3_K"
        )
        if diagnostic is None and known_inactive_q3:
            key_rejection = "Q3_K pairing must be an active policy"
        elif diagnostic is None:
            typed_key = _exact_key(family, key_mapping)
        metadata = _open_mapping(item["metadata"], f"records[{index}].metadata")
        counts = _open_mapping(item["static_counts"], f"records[{index}].static_counts")
        records.append(
            FinalArchiveRecord(
                family,
                _text(item["identity_hash"], f"records[{index}].identity_hash"),
                _text(item["kernel_name"], f"records[{index}].kernel_name"),
                _text(item["artifact_stem"], f"records[{index}].artifact_stem"),
                key_mapping,
                typed_key,
                key_rejection,
                diagnostic,
                _sha256(item["source_sha256"], f"records[{index}].source_sha256"),
                _sha256(
                    item["code_object_sha256"], f"records[{index}].code_object_sha256"
                ),
                metadata,
                counts,
            )
        )
    return FinalArchive(
        _text(root["assembler_path"], "assembler_path"),
        _sha256(root["assembler_sha256"], "assembler_sha256"),
        _text(root["assembler_version"], "assembler_version"),
        tuple(records),
    )


@dataclass(frozen=True)
class ExtendedArchiveRecord:
    family: str
    key_hash: str
    path: str
    source_bytes: int
    source_sha256: str
    key: Mapping[str, object]
    factory: str | None


@dataclass(frozen=True)
class ExtendedArchive:
    records: tuple[ExtendedArchiveRecord, ...]


def load_extended_archive(path: Path) -> ExtendedArchive:
    root = strict_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "extended archive",
        _EXTENDED_ROOT_KEYS,
        error_type=ArchiveError,
    )
    raw_records = _records(root["records"], "extended archive records")
    if integer(root["record_count"], "record_count", error_type=ArchiveError) != len(
        raw_records
    ):
        raise ArchiveError("extended archive record_count does not match records")
    records: list[ExtendedArchiveRecord] = []
    for index, value in enumerate(raw_records):
        if not isinstance(value, Mapping):
            raise ArchiveError(f"records[{index}] must be a mapping")
        keys = frozenset(value)
        expected = frozenset(
            {"family", "key", "key_hash", "path", "source_bytes", "source_sha256"}
        )
        if keys not in {expected, expected | {"factory"}}:
            strict_mapping(
                value, f"records[{index}]", expected, error_type=ArchiveError
            )
        item = value
        key_mapping = _open_mapping(item["key"], f"records[{index}].key")
        if key_mapping.get("ArtifactKind") != "ExactKernel":
            raise ArchiveError(
                f"records[{index}].key must retain ArtifactKind='ExactKernel'"
            )
        source_bytes = integer(
            item["source_bytes"],
            f"records[{index}].source_bytes",
            error_type=ArchiveError,
        )
        if source_bytes <= 0:
            raise ArchiveError(f"records[{index}].source_bytes must be positive")
        factory = item.get("factory")
        records.append(
            ExtendedArchiveRecord(
                _text(item["family"], f"records[{index}].family"),
                _text(item["key_hash"], f"records[{index}].key_hash"),
                _text(item["path"], f"records[{index}].path"),
                source_bytes,
                _sha256(item["source_sha256"], f"records[{index}].source_sha256"),
                key_mapping,
                None
                if factory is None
                else _text(factory, f"records[{index}].factory"),
            )
        )
    return ExtendedArchive(tuple(records))
