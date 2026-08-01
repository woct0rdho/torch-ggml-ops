import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .inspection import InspectionError, inspect_artifact
from .kernel_writer_assembly_mmq_bwd import KernelWriterAssembly
from .kernel_writer_assembly_mmq_fwd import DenseForwardKernelWriterAssembly
from .model import SchemaError, SolutionKey
from .toolchain import Toolchain, ToolchainError
from .validation import validate_solution


class ManifestError(ValueError):
    """A phase manifest is missing, inconsistent, or no longer immutable."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, build, and inspect exact gfx1151 GGTensile kernels"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--solution-key", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--generate-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--build-manifest", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def _copy_exclusive(source: Path, destination: Path) -> None:
    with (
        source.open("rb") as source_handle,
        destination.open("xb") as destination_handle,
    ):
        shutil.copyfileobj(source_handle, destination_handle)


def _load_mapping(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {name} {path}: {error}") from error
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


def _path(value: object, name: str) -> Path:
    if type(value) is not str:
        raise ManifestError(f"{name} must be a path string")
    return Path(value)


def _string(value: object, name: str) -> str:
    if type(value) is not str:
        raise ManifestError(f"{name} must be a string")
    return value


def _reject(
    manifest_path: Path,
    *,
    phase: str,
    error: Exception | None = None,
    reasons: Sequence[Mapping[str, object]] = (),
) -> int:
    manifest: dict[str, Any] = {
        "Phase": phase,
        "Status": "Rejected",
        "RejectReasons": list(reasons),
    }
    if error is not None:
        manifest["Error"] = f"{type(error).__name__}: {error}"
    _write_json_exclusive(manifest_path, manifest)
    return 2


def _generate(solution_path: Path, output_dir: Path) -> int:
    manifest_path = output_dir / "generate.json"
    if manifest_path.exists():
        raise ManifestError(f"refusing to overwrite {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        key = SolutionKey.from_json_file(solution_path)
    except (OSError, SchemaError) as error:
        return _reject(manifest_path, phase="Generate", error=error)

    reasons = validate_solution(key)
    if reasons:
        return _reject(
            manifest_path,
            phase="Generate",
            reasons=tuple(reason.to_mapping() for reason in reasons),
        )

    solution_output = (output_dir / "solution.json").resolve()
    assembly = (output_dir / "kernel.s").resolve()
    if solution_output.exists() or assembly.exists():
        raise ManifestError("refusing to overwrite generated solution or assembly")
    toolchain = Toolchain.discover()
    writer_type = (
        DenseForwardKernelWriterAssembly
        if key.problem_type.operation_type == "DenseMMQForward"
        else KernelWriterAssembly
    )
    source = writer_type(key, toolchain).source()
    _write_json_exclusive(solution_output, key.to_mapping())
    _write_text_exclusive(assembly, source)
    _write_json_exclusive(
        manifest_path,
        {
            "Phase": "Generate",
            "Status": "Accepted",
            "SolutionKey": key.to_mapping(),
            "SolutionHash": key.hash,
            "KernelName": key.kernel_name,
            "SolutionPath": str(solution_output),
            "AssemblyPath": str(assembly),
            "AssemblySHA256": _sha256(assembly),
        },
    )
    return 0


_GENERATE_KEYS = frozenset(
    {
        "Phase",
        "Status",
        "SolutionKey",
        "SolutionHash",
        "KernelName",
        "SolutionPath",
        "AssemblyPath",
        "AssemblySHA256",
    }
)


def _key_from_manifest(manifest: Mapping[str, object]) -> SolutionKey:
    try:
        key = SolutionKey.from_mapping(manifest["SolutionKey"])
    except (KeyError, SchemaError) as error:
        raise ManifestError(f"invalid manifest SolutionKey: {error}") from error
    if manifest.get("SolutionHash") != key.hash:
        raise ManifestError("manifest SolutionHash does not match SolutionKey")
    if manifest.get("KernelName") != key.kernel_name:
        raise ManifestError("manifest KernelName does not match SolutionKey")
    return key


def _build(generate_manifest: Path, output_dir: Path | None) -> int:
    manifest = _accepted_manifest(
        generate_manifest,
        phase="Generate",
        keys=_GENERATE_KEYS,
    )
    key = _key_from_manifest(manifest)
    assembly = _path(manifest["AssemblyPath"], "AssemblyPath")
    expected_assembly_hash = _string(manifest["AssemblySHA256"], "AssemblySHA256")
    destination = output_dir or generate_manifest.parent
    destination.mkdir(parents=True, exist_ok=True)
    build_manifest = destination / "build.json"
    object_path = (destination / "kernel.o").resolve()
    code_object = (destination / "kernel.hsaco").resolve()
    for path in (build_manifest, object_path, code_object):
        if path.exists():
            raise ManifestError(f"refusing to overwrite {path}")
    if _sha256(assembly) != expected_assembly_hash:
        return _reject(
            build_manifest,
            phase="Build",
            error=ManifestError("assembly hash does not match generate manifest"),
        )

    toolchain = Toolchain.discover()
    try:
        with tempfile.TemporaryDirectory(prefix="ggtensile-build-") as temporary:
            temporary_path = Path(temporary)
            temporary_object = temporary_path / "kernel.o"
            temporary_code_object = temporary_path / "kernel.hsaco"
            toolchain.assemble(assembly, temporary_object)
            toolchain.link(temporary_object, temporary_code_object)
            _copy_exclusive(temporary_object, object_path)
            _copy_exclusive(temporary_code_object, code_object)
    except (OSError, ToolchainError) as error:
        return _reject(build_manifest, phase="Build", error=error)

    _write_json_exclusive(
        build_manifest,
        {
            "Phase": "Build",
            "Status": "Accepted",
            "SolutionKey": key.to_mapping(),
            "SolutionHash": key.hash,
            "KernelName": key.kernel_name,
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
        "SolutionKey",
        "SolutionHash",
        "KernelName",
        "GenerateManifestPath",
        "AssemblyPath",
        "AssemblySHA256",
        "ObjectPath",
        "CodeObjectPath",
    }
)


def _inspect(build_manifest: Path, output: Path | None) -> int:
    manifest = _accepted_manifest(
        build_manifest,
        phase="Build",
        keys=_BUILD_KEYS,
    )
    key = _key_from_manifest(manifest)
    code_object = _path(manifest["CodeObjectPath"], "CodeObjectPath")
    inspection_manifest = output or build_manifest.with_name("inspect.json")
    if inspection_manifest.exists():
        raise ManifestError(f"refusing to overwrite {inspection_manifest}")

    try:
        inspection = inspect_artifact(key, code_object, Toolchain.discover())
    except (InspectionError, OSError, ToolchainError) as error:
        return _reject(inspection_manifest, phase="Inspect", error=error)
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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "generate":
            return _generate(arguments.solution_key, arguments.output_dir)
        if arguments.command == "build":
            return _build(arguments.generate_manifest, arguments.output_dir)
        if arguments.command == "inspect":
            return _inspect(arguments.build_manifest, arguments.output)
    except (ManifestError, FileExistsError, ToolchainError) as error:
        print(f"ggtensile {arguments.command}: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
