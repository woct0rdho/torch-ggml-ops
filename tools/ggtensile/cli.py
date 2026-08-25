import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from .family_registry import (
    family_for_instance,
    instance_hash,
    instance_name,
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from .identity import KernelFamily
from .inspection import ArtifactInspection, inspect_artifact
from .kernel_instance import KernelInstance
from .mmq_fwd_search import (
    ForwardExactPairManifest,
    ForwardSearchKnobGroup,
    Q6ExactPairManifest,
    Q6SearchKnobGroup,
    candidate_domains,
    candidate_neighbors,
    q6_schedule_neighbors,
    q6_schedule_seed,
)
from .model import ProblemSize
from .toolchain import Toolchain
from .validation import validate_instance


class ManifestError(ValueError):
    """A phase manifest is missing, inconsistent, or no longer immutable."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, build, and inspect exact gfx1151 GGTensile kernels"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--kernel-spec-key", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--generate-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--build-manifest", type=Path, required=True)
    inspect.add_argument("--output", type=Path)

    enumerate_forward = subparsers.add_parser("enumerate-forward")
    enumerate_forward.add_argument(
        "--quant-type",
        choices=("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"),
        required=True,
    )
    enumerate_forward.add_argument("--m", type=int, required=True)
    enumerate_forward.add_argument("--n", type=int, required=True)
    enumerate_forward.add_argument("--k", type=int, required=True)
    enumerate_forward.add_argument(
        "--knob-group",
        action="append",
        choices=("InstructionPolicy", "Epilogue", "Metadata"),
    )
    enumerate_forward.add_argument("--output-dir", type=Path, required=True)

    enumerate_q6 = subparsers.add_parser("enumerate-forward-q6")
    enumerate_q6.add_argument(
        "--macro-tile", type=int, choices=(64, 128), required=True
    )
    enumerate_q6.add_argument("--m", type=int, required=True)
    enumerate_q6.add_argument("--n", type=int, required=True)
    enumerate_q6.add_argument("--k", type=int, required=True)
    enumerate_q6.add_argument(
        "--knob-group",
        action="append",
        choices=("InstructionPolicy", "Epilogue"),
    )
    enumerate_q6.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _copy_exclusive(source: Path, destination: Path) -> None:
    with (
        source.open("rb") as source_handle,
        destination.open("xb") as destination_handle,
    ):
        shutil.copyfileobj(source_handle, destination_handle)


def _load_mapping(path: Path, name: str) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must contain a JSON object")
    return value


def _accepted_manifest(
    path: Path,
    *,
    phase: str,
    keys: frozenset[str],
) -> Mapping[str, object]:
    manifest = _load_mapping(path, f"{phase} manifest")
    actual = {str(key) for key in manifest}
    if actual != keys:
        raise ManifestError(
            f"invalid {phase} manifest keys: expected {sorted(keys)}, got {sorted(actual)}"
        )
    if manifest["Phase"] != phase or manifest["Status"] != "Accepted":
        raise ManifestError(f"{phase} manifest is not accepted")
    return manifest


def _field(value: object, name: str) -> object:
    return value[name] if isinstance(value, Mapping) else value


def _path(value: object, name: str) -> Path:
    item = _field(value, name)
    if type(item) is not str:
        raise ManifestError(f"{name} must be a path string")
    return Path(item)


def _string(value: object, name: str) -> str:
    item = _field(value, name)
    if type(item) is not str:
        raise ManifestError(f"{name} must be a string")
    return item


def _load_instance(path: Path) -> KernelInstance:
    return parse_instance(json.loads(path.read_text(encoding="utf-8")))


def _generate(kernel_spec_key_path: Path, output_dir: Path) -> int:
    manifest_path = output_dir / "generate.json"
    if manifest_path.exists():
        raise ManifestError(f"refusing to overwrite {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    instance = _load_instance(kernel_spec_key_path)

    kernel_spec_output = (output_dir / "kernel-spec.json").resolve()
    assembly = (output_dir / "kernel.s").resolve()
    if kernel_spec_output.exists() or assembly.exists():
        raise ManifestError("refusing to overwrite generated KernelSpec or assembly")
    toolchain = Toolchain.discover()
    source = writer_for_instance(instance, toolchain).source()
    mapping = mapping_for_instance(instance)
    kernel_hash = instance_hash(instance)
    kernel_name = instance_name(instance)
    _write_json_exclusive(kernel_spec_output, mapping)
    assembly.parent.mkdir(parents=True, exist_ok=True)
    with assembly.open("x", encoding="utf-8") as handle:
        handle.write(source)
    _write_json_exclusive(
        manifest_path,
        {
            "Phase": "Generate",
            "Status": "Accepted",
            "KernelSpecKey": mapping,
            "KernelSpecHash": kernel_hash,
            "KernelName": kernel_name,
            "KernelSpecPath": str(kernel_spec_output),
            "AssemblyPath": str(assembly),
            "AssemblySHA256": _sha256(assembly),
        },
    )
    return 0


_GENERATE_KEYS = frozenset(
    {
        "Phase",
        "Status",
        "KernelSpecKey",
        "KernelSpecHash",
        "KernelName",
        "KernelSpecPath",
        "AssemblyPath",
        "AssemblySHA256",
    }
)


def _instance_from_manifest(manifest: Mapping[str, object]) -> KernelInstance:
    instance = parse_instance(manifest["KernelSpecKey"])
    if manifest.get("KernelSpecHash") != instance_hash(instance):
        raise ManifestError("manifest KernelSpecHash does not match KernelSpecKey")
    if manifest.get("KernelName") != instance_name(instance):
        raise ManifestError("manifest KernelName does not match KernelSpecKey")
    return instance


def _build(generate_manifest: Path, output_dir: Path | None) -> int:
    manifest = _accepted_manifest(
        generate_manifest,
        phase="Generate",
        keys=_GENERATE_KEYS,
    )
    instance = _instance_from_manifest(manifest)
    assembly = _path(manifest, "AssemblyPath")
    expected_assembly_hash = _string(manifest, "AssemblySHA256")
    destination = output_dir or generate_manifest.parent
    destination.mkdir(parents=True, exist_ok=True)
    build_manifest = destination / "build.json"
    object_path = (destination / "kernel.o").resolve()
    code_object = (destination / "kernel.hsaco").resolve()
    for path in (build_manifest, object_path, code_object):
        if path.exists():
            raise ManifestError(f"refusing to overwrite {path}")
    assert _sha256(assembly) == expected_assembly_hash

    toolchain = Toolchain.discover()
    with tempfile.TemporaryDirectory(prefix="ggtensile-build-") as temporary:
        temporary_path = Path(temporary)
        temporary_object = temporary_path / "kernel.o"
        temporary_code_object = temporary_path / "kernel.hsaco"
        toolchain.assemble(assembly, temporary_object)
        toolchain.link(temporary_object, temporary_code_object)
        _copy_exclusive(temporary_object, object_path)
        _copy_exclusive(temporary_code_object, code_object)

    _write_json_exclusive(
        build_manifest,
        {
            "Phase": "Build",
            "Status": "Accepted",
            "KernelSpecKey": mapping_for_instance(instance),
            "KernelSpecHash": instance_hash(instance),
            "KernelName": instance_name(instance),
            "GenerateManifestPath": str(generate_manifest.resolve()),
            "AssemblyPath": str(assembly.resolve()),
            "AssemblySHA256": expected_assembly_hash,
            "ObjectPath": str(object_path),
            "CodeObjectPath": str(code_object),
        },
    )
    return 0


_BUILD_KEYS = frozenset(
    {
        "Phase",
        "Status",
        "KernelSpecKey",
        "KernelSpecHash",
        "KernelName",
        "GenerateManifestPath",
        "AssemblyPath",
        "AssemblySHA256",
        "ObjectPath",
        "CodeObjectPath",
    }
)


def _inspect_instance_artifact(
    instance: KernelInstance, code_object: Path, toolchain: Toolchain
) -> ArtifactInspection:
    family = family_for_instance(instance)
    problem = instance.problem
    spec = instance.kernel_spec
    kernel_name = instance_name(instance)
    if family is KernelFamily.GroupedForward:
        from .grouped_mmq_fwd_inspection import inspect_grouped_forward_artifact
        from .grouped_mmq_fwd_model import GroupedForwardProblem
        from .grouped_mmq_fwd_spec import GroupedForwardKernelSpec

        assert isinstance(problem, GroupedForwardProblem)
        assert isinstance(spec, GroupedForwardKernelSpec)
        return inspect_grouped_forward_artifact(
            problem, spec, kernel_name, code_object, toolchain
        )
    if family is KernelFamily.GroupedForwardPair:
        from .grouped_mmq_fwd_pair_inspection import (
            inspect_grouped_forward_pair_artifact,
        )
        from .grouped_mmq_fwd_pair_model import GroupedForwardPairProblem
        from .grouped_mmq_fwd_pair_spec import GroupedForwardPairKernelSpec

        assert isinstance(problem, GroupedForwardPairProblem)
        assert isinstance(spec, GroupedForwardPairKernelSpec)
        return inspect_grouped_forward_pair_artifact(
            problem, spec, kernel_name, code_object, toolchain
        )
    if family is KernelFamily.GroupedBackwardPair:
        from .grouped_mmq_bwd_pair_inspection import (
            inspect_grouped_backward_pair_artifact,
        )
        from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
        from .grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec

        assert isinstance(problem, GroupedBackwardPairProblem)
        assert isinstance(spec, GroupedBackwardPairKernelSpec)
        return inspect_grouped_backward_pair_artifact(
            problem, spec, kernel_name, code_object, toolchain
        )
    return inspect_artifact(instance, code_object, toolchain)


def _inspect(build_manifest: Path, output: Path | None) -> int:
    manifest = _accepted_manifest(
        build_manifest,
        phase="Build",
        keys=_BUILD_KEYS,
    )
    instance = _instance_from_manifest(manifest)
    code_object = _path(manifest, "CodeObjectPath")
    inspection_manifest = output or build_manifest.with_name("inspect.json")
    if inspection_manifest.exists():
        raise ManifestError(f"refusing to overwrite {inspection_manifest}")

    inspection = _inspect_instance_artifact(instance, code_object, Toolchain.discover())
    _write_json_exclusive(
        inspection_manifest,
        {
            "Phase": "Inspect",
            "Status": "Accepted",
            "BuildManifestPath": str(build_manifest.resolve()),
            "Inspection": inspection.to_mapping(),
        },
    )
    return 0


def _enumerate_forward(
    quant_type: str,
    problem_size: ProblemSize,
    knob_groups: tuple[ForwardSearchKnobGroup, ...],
    output_dir: Path,
) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ManifestError(f"refusing to populate nonempty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for domain in candidate_domains(quant_type, problem_size):
        groups = knob_groups or domain.knob_groups
        for candidate in candidate_neighbors(domain.kernel_spec, quant_type, groups):
            pair = ForwardExactPairManifest(quant_type, problem_size, candidate)
            instance = pair.kernel_instance
            validate_instance(instance)
            candidate_dir = output_dir / pair.candidate_hash
            candidate_dir.mkdir()
            _write_json_exclusive(
                candidate_dir / "kernel-spec-key.json",
                mapping_for_instance(instance),
            )
            _write_json_exclusive(
                candidate_dir / "candidate.json",
                {
                    **pair.to_mapping(),
                    "ExactPairHash": pair.exact_pair_hash,
                    "KernelSpecKeyPath": "kernel-spec-key.json",
                },
            )
            entries.append(
                {
                    "CandidateHash": pair.candidate_hash,
                    "ExactPairHash": pair.exact_pair_hash,
                    "KernelSpecHash": instance_hash(instance),
                    "KernelName": instance_name(instance),
                    "Directory": pair.candidate_hash,
                }
            )
    _write_json_exclusive(
        output_dir / "index.json",
        {
            "KernelFamily": quant_type,
            "ProblemSize": problem_size.to_mapping(),
            "KnobGroups": list(knob_groups),
            "CandidateCount": len(entries),
            "Candidates": entries,
        },
    )
    return 0


def _enumerate_forward_q6(
    macro_tile: int,
    problem_size: ProblemSize,
    knob_groups: tuple[Q6SearchKnobGroup, ...],
    output_dir: Path,
) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ManifestError(f"refusing to populate nonempty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = q6_schedule_seed(macro_tile)
    candidates = q6_schedule_neighbors(seed, knob_groups)
    entries: list[dict[str, object]] = []
    for schedule in candidates:
        pair = Q6ExactPairManifest(problem_size, schedule)
        instance = pair.kernel_instance
        validate_instance(instance)
        candidate_dir = output_dir / pair.candidate_hash
        candidate_dir.mkdir()
        kernel_spec_key_path = candidate_dir / "kernel-spec-key.json"
        manifest_path = candidate_dir / "candidate.json"
        _write_json_exclusive(kernel_spec_key_path, mapping_for_instance(instance))
        _write_json_exclusive(
            manifest_path,
            {
                **pair.to_mapping(),
                "ExactPairHash": pair.exact_pair_hash,
                "KernelSpecKeyPath": "kernel-spec-key.json",
            },
        )
        entries.append(
            {
                "CandidateHash": pair.candidate_hash,
                "ExactPairHash": pair.exact_pair_hash,
                "KernelSpecHash": instance_hash(instance),
                "KernelName": instance_name(instance),
                "Directory": pair.candidate_hash,
            }
        )
    _write_json_exclusive(
        output_dir / "index.json",
        {
            "KernelFamily": "Q6StructuredDecoded",
            "ProblemSize": problem_size.to_mapping(),
            "MacroTile0": macro_tile,
            "KnobGroups": list(knob_groups),
            "CandidateCount": len(entries),
            "Candidates": entries,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "generate":
        return _generate(arguments.kernel_spec_key, arguments.output_dir)
    if arguments.command == "build":
        return _build(arguments.generate_manifest, arguments.output_dir)
    if arguments.command == "inspect":
        return _inspect(arguments.build_manifest, arguments.output)
    if arguments.command == "enumerate-forward":
        raw_groups = arguments.knob_group
        knob_groups = cast(tuple[ForwardSearchKnobGroup, ...], tuple(raw_groups or ()))
        return _enumerate_forward(
            arguments.quant_type,
            ProblemSize(arguments.m, arguments.n, arguments.k),
            knob_groups,
            arguments.output_dir,
        )
    if arguments.command == "enumerate-forward-q6":
        raw_groups = arguments.knob_group or ["InstructionPolicy", "Epilogue"]
        knob_groups = cast(tuple[Q6SearchKnobGroup, ...], tuple(raw_groups))
        return _enumerate_forward_q6(
            arguments.macro_tile,
            ProblemSize(arguments.m, arguments.n, arguments.k),
            knob_groups,
            arguments.output_dir,
        )
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
