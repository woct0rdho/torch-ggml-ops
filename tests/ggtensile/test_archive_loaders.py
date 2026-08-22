import json
from pathlib import Path
from typing import Any, cast

import pytest

from tools.ggtensile.archive import (
    ArchiveError,
    load_extended_archive,
    load_final_archive,
)
from tools.ggtensile.campaign import load_catalog

ROOT = Path(__file__).resolve().parents[2]


def _final_manifest() -> dict[str, Any]:
    catalog = load_catalog(ROOT / "tools/ggtensile/configs/mmq_fwd_q4_k_catalog.json")
    key = catalog.entries[0].solution_key
    return {
        "assembler_path": "/opt/llvm/bin/clang",
        "assembler_sha256": "a" * 64,
        "assembler_version": "clang test",
        "record_count": 1,
        "records": [
            {
                "artifact_stem": "000-test",
                "code_object_sha256": "b" * 64,
                "diagnostic_mode": None,
                "exact_key": key.to_mapping(),
                "family": "ordinary_forward",
                "identity_hash": key.hash,
                "kernel_name": key.kernel_name,
                "metadata": {".kernarg_segment_size": 40},
                "source_sha256": "c" * 64,
                "static_counts": {"wmma": 1},
            }
        ],
    }


def _write(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_final_archive_loads_typed_exact_key(tmp_path: Path) -> None:
    archive = load_final_archive(_write(tmp_path, _final_manifest()))
    assert len(archive.records) == 1
    assert archive.records[0].exact_key is not None
    assert archive.records[0].exact_key_rejection is None


def test_final_archive_rejects_unknown_record_field(tmp_path: Path) -> None:
    manifest = _final_manifest()
    manifest["records"][0]["ArtifactKind"] = "ExactKernel"
    with pytest.raises(ArchiveError, match="unknown"):
        load_final_archive(_write(tmp_path, manifest))


def test_extended_archive_preserves_extended_fields(tmp_path: Path) -> None:
    manifest = {
        "fixed_count": 0,
        "grouped_count": 0,
        "ordinary_backward_count": 0,
        "ordinary_forward_count": 1,
        "paired_count": 0,
        "record_count": 1,
        "records": [
            {
                "family": "ordinary-fwd-q4_k",
                "key": {
                    "ArtifactKind": "ExactKernel",
                    "KernelFamily": "OrdinaryForward",
                    "ProblemContract": {},
                    "Problem": {"m": 1, "n": 1, "k": 1},
                    "KernelSpec": {"resource_limits": {"max_vgprs": 256}},
                },
                "key_hash": "ggsol_test",
                "path": "sources/test.s",
                "source_bytes": 1,
                "source_sha256": "d" * 64,
            }
        ],
    }
    archive = load_extended_archive(_write(tmp_path, manifest))
    key = archive.records[0].key
    spec = cast(dict[str, object], key["KernelSpec"])
    assert key["ArtifactKind"] == "ExactKernel"
    assert spec["resource_limits"] == {"max_vgprs": 256}


def test_extended_archive_is_not_accepted_as_final(tmp_path: Path) -> None:
    manifest = {
        "fixed_count": 0,
        "grouped_count": 0,
        "ordinary_backward_count": 0,
        "ordinary_forward_count": 0,
        "paired_count": 0,
        "record_count": 0,
        "records": [],
    }
    with pytest.raises(ArchiveError, match="invalid final archive"):
        load_final_archive(_write(tmp_path, manifest))
