"""The mechanically enforceable C++/CUDA style rules of `csrc/`.

Composable Kernel and llama.cpp both set `IndentWidth: 4` with `UseTab: Never`,
and this project follows them, so the block indentation of every checked-in
C++/CUDA file is enforced mechanically. Line breaks, line width and vertical
alignment are chosen semantically and stay unconstrained.
"""

from pathlib import Path

from tools.check_cpp_style import INDENT_WIDTH, analyse, find_problems


def _written(tmp_path: Path, text: str) -> list[str]:
    source = tmp_path / "sample.cpp"
    source.write_text(text, encoding="utf-8")
    return analyse(source)


def test_checked_in_sources_follow_the_shared_convention() -> None:
    assert find_problems() == []


def test_block_indentation_width_is_enforced(tmp_path: Path) -> None:
    problems = _written(
        tmp_path,
        "void f() {\n    for (int i = 0; i < 2; ++i) {\n      g(i);\n    }\n}\n",
    )
    assert any("block body indented by 2" in problem for problem in problems)
    assert any("shallower than" in problem for problem in problems)

    problems = _written(
        tmp_path,
        "void f() {\n    if (x) {\n            g();\n    }\n}\n",
    )
    assert any(
        f"block body indented by {2 * INDENT_WIDTH}" in problem for problem in problems
    )


def test_correct_nesting_and_namespaces_are_accepted(tmp_path: Path) -> None:
    assert (
        _written(
            tmp_path,
            "namespace outer::inner {\n"
            "\n"
            "void f() {\n"
            "    if (x) {\n"
            "        // a comment may sit at the block level\n"
            "        for (int i = 0; i < 2; ++i) {\n"
            "            g(i);\n"
            "        }\n"
            "    } else {\n"
            "        h();\n"
            "    }\n"
            "}\n"
            "\n"
            "} // namespace outer::inner\n",
        )
        == []
    )


def test_shallow_and_tab_indentation_are_rejected(tmp_path: Path) -> None:
    problems = _written(
        tmp_path,
        "void f() {\n  if (x) {\n\tg();\n  }\n}\n",
    )
    assert any("tab indentation" in problem for problem in problems)
    assert any("shallower than" in problem for problem in problems)


def test_include_order_is_enforced(tmp_path: Path) -> None:
    problems = _written(
        tmp_path,
        '#include "../vendor/llama_cpp/common.cuh"\n'
        '#include "detail.cuh"\n'
        "\n"
        "int f();\n",
    )
    assert problems == [
        f"{tmp_path / 'sample.cpp'}:2: local include after a parent-directory include"
    ]

    assert (
        _written(
            tmp_path,
            '#include "detail.cuh"\n'
            '#include "../vendor/llama_cpp/common.cuh"\n'
            "\n"
            "#include <cstdint>\n"
            "\n"
            "int f();\n",
        )
        == []
    )
