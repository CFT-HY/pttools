"""Tests for the LaTeX utilities."""

from pathlib import Path

from pttools.docs import latex

#: An excerpt of the console output of a failed LaTeX build
OUTPUT: list[str] = [
    "This is XeTeX, Version 3.141592653",
    "! Undefined control sequence.",
    "<argument> \\foo",
    "l.123 \\foo",
    "                                   x",
    "Some other line",
    "! Missing $ inserted.",
    "<inserted text>",
    "                $",
    "<to be read again>",
    "                   \\par",
    "Latexmk: Errors, so I did not complete making targets",
]


def test_latex_errors() -> None:
    """The error lines should be found with their context, which ends at the line number or after three lines."""
    assert latex.latex_errors(OUTPUT) == [
        "! Undefined control sequence.",
        "<argument> \\foo",
        "l.123 \\foo",
        "! Missing $ inserted.",
        "<inserted text>",
        "                $",
        "<to be read again>",
        "Latexmk: Errors, so I did not complete making targets",
    ]
    assert latex.latex_errors(["No errors here", "LaTeX Warning: Reference `x' undefined"]) == []


def test_latex_log_messages(tmp_path: Path) -> None:
    """The errors and the warnings of the final pass should be read from the log files."""
    build_dir = tmp_path / latex.LATEX_SUBDIR_NAME
    build_dir.mkdir()
    (build_dir / "b.log").write_text(
        "LaTeX Warning: Reference `x' on page 1 undefined on input line 5.\n"
        "LaTeX Warning: Some other warning.\n"
        "LaTeX Warning: There were undefined references.\n"
    )
    (build_dir / "a.log").write_text("\n".join(OUTPUT[:4]) + "\n")
    (build_dir / "notes.txt").write_text("! Not a log file\n")

    assert latex.latex_log_paths(build_dir) == [build_dir / "a.log", build_dir / "b.log"]
    assert latex.latex_log_messages(build_dir / "a.log") == (OUTPUT[1:4], [])
    assert latex.latex_log_messages(build_dir / "missing.log") == ([], [])
    paths, errors, warnings = latex.all_latex_log_messages(build_dir)
    assert paths == [build_dir / "a.log", build_dir / "b.log"]
    assert errors == OUTPUT[1:4]
    assert warnings == [
        "LaTeX Warning: Reference `x' on page 1 undefined on input line 5.",
        "LaTeX Warning: There were undefined references.",
    ]
    assert latex.all_latex_log_messages(tmp_path / "missing") == ([], [], [])
