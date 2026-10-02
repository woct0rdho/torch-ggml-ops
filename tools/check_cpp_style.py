"""Check the mechanically enforceable C++/CUDA style rules of `csrc/`.

Composable Kernel and llama.cpp both set `IndentWidth: 4` with `UseTab: Never`
and `NamespaceIndentation: None`, and the code under `csrc/` follows the same
convention, so the mechanical rules below are exactly the shared part of those
conventions:
- the first statement of a control-flow block (`if`, `else`, `for`, `while`,
  `do`, `switch`, `case`, `default`) is indented four spaces past the line that
  opens it, when that line starts the statement;
- no code line is indented less than four spaces per enclosing block, counting
  namespaces as no indentation (both projects leave namespace contents at the
  outer level);
- no tabs;
- a local include (`"name.cuh"`) comes before an include from the parent
  directory (`"../name.cuh"`), and the quoted project includes come before the
  system includes.

Continuation lines, line width, line breaks and vertical alignment stay
deliberately unconstrained: those are chosen semantically, so this check only
looks at the block width that can be enforced mechanically.
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
INDENT_WIDTH = 4
SOURCE_PATTERNS = ("*.cuh", "*.h", "*.cpp", "*.cu")
# `mmq_bundle_table.cuh` is generated from the typed inventory.
GENERATED = ("mmq_bundle_table.cuh",)
CONTROL_FLOW = re.compile(r"\b(if|else|for|while|switch|do)\b")
CASE_LABEL = re.compile(r"^\s*(case\b.*|default)\s*:")
DIRECTIVE = re.compile(r"^\s*#")
QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
SYSTEM_INCLUDE = re.compile(r"^\s*#\s*include\s*<")
NAMESPACE = re.compile(r"^\s*namespace\b")


def mask(text: str) -> list[str]:
    """Blank comments and string contents, keeping the line structure."""
    out: list[str] = []
    index = 0
    state = "code"
    length = len(text)
    while index < length:
        char = text[index]
        pair = text[index : index + 2]
        if state == "code":
            if pair == "//":
                state = "line"
                out.append("  ")
                index += 2
            elif pair == "/*":
                state = "block"
                out.append("  ")
                index += 2
            elif char in "\"'":
                state = "string" if char == '"' else "char"
                out.append(" ")
                index += 1
            else:
                out.append(char)
                index += 1
            continue
        if state == "line":
            out.append("\n" if char == "\n" else " ")
            state = "code" if char == "\n" else state
            index += 1
            continue
        if state == "block":
            if pair == "*/":
                state = "code"
                out.append("  ")
                index += 2
            else:
                out.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if char == "\\" and index + 1 < length:
            out.append("  ")
            index += 2
            continue
        closing = '"' if state == "string" else "'"
        if char == closing:
            state = "code"
            out.append(" ")
        else:
            out.append("\n" if char == "\n" else " ")
        index += 1
    return "".join(out).splitlines()


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _starts_statement(previous: str | None) -> bool:
    """Whether a line begins a statement rather than continuing an expression."""
    if previous is None:
        return True
    stripped = previous.strip()
    if not stripped or DIRECTIVE.match(previous) or stripped.startswith("//"):
        return True
    return stripped.endswith((";", "{", "}", ":"))


def _is_control_block(line: str, previous: str | None) -> bool:
    if not line.strip().endswith("{"):
        return False
    head = line.strip()[:-1].strip()
    if not (CONTROL_FLOW.search(head) or CASE_LABEL.match(line.strip())):
        return False
    return _starts_statement(previous)


def _include_problems(name: str, raw: list[str]) -> list[str]:
    """Check the include prologue: local first, then parent directory, then system.

    Only the first include block is checked. Includes that appear inside the file
    are placed deliberately (the vendored llama.cpp fragments are configured that
    way), so they are left alone.
    """
    problems: list[str] = []
    seen_parent = False
    seen_system = False
    started = False
    for number, line in enumerate(raw, start=1):
        stripped = line.strip()
        if not stripped:
            if started:
                continue
            continue
        if stripped.startswith("//"):
            continue
        quoted = QUOTED_INCLUDE.match(line)
        system = SYSTEM_INCLUDE.match(line)
        if quoted is None and system is None:
            break
        started = True
        if quoted is not None:
            if seen_system:
                problems.append(
                    f"{name}:{number}: quoted project include after a system include"
                )
            elif quoted.group(1).startswith("../"):
                seen_parent = True
            elif seen_parent:
                problems.append(
                    f"{name}:{number}: local include after a parent-directory include"
                )
        else:
            seen_system = True
    return problems


def analyse(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    raw = text.splitlines()
    masked = mask(text)
    problems: list[str] = []
    if path.is_relative_to(ROOT):
        name = path.relative_to(ROOT).as_posix()
    else:
        name = str(path)
    problems.extend(_include_problems(name, raw))
    depth = 0
    non_indenting = 0  # namespace blocks leave their contents at the outer level
    stack: list[bool] = []
    previous: str | None = None

    for number, line in enumerate(masked, start=1):
        source = raw[number - 1]
        if "\t" in source:
            problems.append(f"{name}:{number}: tab indentation")
        stripped = line.strip()
        directive = bool(DIRECTIVE.match(line)) if stripped else False
        comment_only = not stripped and bool(source.strip())

        if stripped and not directive:
            leading_closers = len(stripped) - len(stripped.lstrip("}"))
            level = depth - non_indenting - leading_closers
            target = INDENT_WIDTH * max(level, 0)
            if indent_of(source) < target:
                problems.append(
                    f"{name}:{number}: indent {indent_of(source)} is shallower than "
                    f"{target} for block level {max(level, 0)}"
                )
            if _is_control_block(line, previous):
                body = _first_statement(masked, number)
                if body is not None:
                    body_number, body_line = body
                    width = indent_of(body_line) - indent_of(source)
                    if width != INDENT_WIDTH:
                        problems.append(
                            f"{name}:{body_number}: block body indented by {width}, "
                            f"expected {INDENT_WIDTH} (opener at line {number})"
                        )
        elif comment_only:
            level = depth - non_indenting
            target = INDENT_WIDTH * max(level, 0)
            if indent_of(source) < target:
                problems.append(
                    f"{name}:{number}: comment indent {indent_of(source)} is shallower "
                    f"than {target} for block level {max(level, 0)}"
                )

        if stripped and not directive and not comment_only:
            namespace = bool(NAMESPACE.match(line))
            for char in line:
                if char == "{":
                    stack.append(namespace)
                    depth += 1
                    if namespace:
                        non_indenting += 1
                elif char == "}":
                    if stack and stack.pop():
                        non_indenting -= 1
                    depth -= 1
        if stripped and not directive:
            previous = line

    return problems


def _first_statement(masked: list[str], opener: int) -> tuple[int, str] | None:
    for number in range(opener + 1, len(masked) + 1):
        line = masked[number - 1]
        stripped = line.strip()
        if not stripped or DIRECTIVE.match(line):
            continue
        if stripped.startswith("}"):
            return None
        return number, line
    return None


def sources(paths: list[Path] | None = None) -> list[Path]:
    if paths:
        return sorted(paths)
    return sorted(
        path
        for pattern in SOURCE_PATTERNS
        for path in CSRC.rglob(pattern)
        if path.name not in GENERATED
    )


def find_problems(paths: list[Path] | None = None) -> list[str]:
    problems: list[str] = []
    for path in sources(paths):
        problems.extend(analyse(path))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = sources(args.paths or None)
    problems = find_problems(paths)
    if problems:
        raise SystemExit("block indentation problems:\n" + "\n".join(problems))
    print(
        f"{len(paths)} files use {INDENT_WIDTH}-space block indentation "
        "and the local-before-parent include order"
    )


if __name__ == "__main__":
    main()
