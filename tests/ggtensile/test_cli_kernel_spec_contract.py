import json
from pathlib import Path

import pytest

from tools.ggtensile import campaign, cli
from tools.ggtensile.family_registry import (
    family_for_instance,
    instance_hash,
    instance_name,
    mapping_for_instance,
)


def test_cli_parser_requires_kernel_spec_key(tmp_path: Path) -> None:
    arguments = cli._parser().parse_args(
        [
            "generate",
            "--kernel-spec-key",
            str(tmp_path / "key.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert arguments.kernel_spec_key == tmp_path / "key.json"
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "generate",
                "--kernel-instance",
                str(tmp_path / "key.json"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_forward_enumeration_writes_kernel_spec_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "candidates"
    assert (
        cli.main(
            [
                "enumerate-forward",
                "--quant-type",
                "Q6_K",
                "--m",
                "64",
                "--n",
                "248320",
                "--k",
                "2048",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    candidate_dirs = [path for path in output.iterdir() if path.is_dir()]
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert candidate_dirs
    assert len(candidate_dirs) == index["CandidateCount"]
    assert (candidate_dirs[0] / "kernel-spec-key.json").is_file()
    candidate = json.loads(
        (candidate_dirs[0] / "candidate.json").read_text(encoding="utf-8")
    )
    assert candidate["KernelSpecKeyPath"] == "kernel-spec-key.json"
    assert "KernelSpecHash" in candidate
    assert all("Solution" not in key for key in candidate)


@pytest.mark.parametrize(
    "catalog_name",
    (
        "mmq_fwd_q3_k_catalog.json",
        "mmq_bwd_q3_k_catalog.json",
        "mmq_grouped_fwd_q4_k_catalog.json",
        "mmq_grouped_bwd_q4_k_catalog.json",
        "mmq_grouped_fwd_pair_q3_k_catalog.json",
        "mmq_grouped_fwd_pair_iq2_s_catalog.json",
        "mmq_grouped_bwd_pair_q3_k_catalog.json",
        "mmq_fixed_grouped_fwd_q8_0_catalog.json",
        "mmq_fixed_grouped_bwd_q8_0_catalog.json",
    ),
)
def test_cli_generate_accepts_every_typed_kernel_family(
    tmp_path: Path, catalog_name: str
) -> None:
    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "ggtensile"
        / "configs"
        / catalog_name
    )
    instance = campaign.load_catalog(catalog_path).entries[0].instance
    key_path = tmp_path / "kernel-spec-key.json"
    key_path.write_text(json.dumps(mapping_for_instance(instance)), encoding="utf-8")
    output = tmp_path / "generated"

    assert (
        cli.main(
            [
                "generate",
                "--kernel-spec-key",
                str(key_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "generate.json").read_text(encoding="utf-8"))
    assert manifest["KernelSpecHash"] == instance_hash(instance)
    assert manifest["KernelName"] == instance_name(instance)
    generated_key = json.loads(
        (output / "kernel-spec.json").read_text(encoding="utf-8")
    )
    assert generated_key["KernelFamily"] == family_for_instance(instance).value
    assert (output / "kernel.s").is_file()
