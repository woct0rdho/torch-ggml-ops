import json
from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.equivalence import capture_structural_evidence
from tools.ggtensile.toolchain import Toolchain

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "tests" / "ggtensile" / "data" / "canonical_structural_evidence.json"
)
CATALOG_ROOT = ROOT / "tools" / "ggtensile" / "configs"


def _jsonable(value: object) -> dict[str, object]:
    return json.loads(json.dumps(value))


def test_catalogs_preserve_canonical_structural_evidence() -> None:
    expected: list[dict[str, object]] = json.loads(
        EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    assert type(expected) is list

    toolchain = Toolchain.discover()
    actual: list[dict[str, object]] = []
    catalog_paths = sorted(CATALOG_ROOT.glob("*catalog.json")) + sorted(
        (CATALOG_ROOT / "research").glob("*catalog.json")
    )
    for path in catalog_paths:
        catalog = load_catalog(path)
        for instance in catalog.instances:
            actual.append(
                _jsonable(capture_structural_evidence(instance, toolchain).to_mapping())
            )

    assert len(actual) == len(expected) == 154
    unmatched = []
    for record in expected:
        matches = [
            candidate
            for candidate in actual
            if candidate["KernelFamily"] == record["KernelFamily"]
            and candidate["QuantDataType"] == record["QuantDataType"]
            and candidate["Problem"] == record["Problem"]
            and candidate["NormalizedSourceSHA256"] == record["NormalizedSourceSHA256"]
        ]
        if len(matches) != 1:
            unmatched.append((record["KernelFamily"], record["Problem"], len(matches)))
            continue
        assert matches[0] == record

    assert not unmatched
