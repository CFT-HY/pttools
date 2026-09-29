"""Test the plotting utilities."""

from pathlib import Path

from matplotlib.figure import Figure
import pytest

from pttools.analysis.utils import save_fig

FORMATS: tuple[str, ...] = ("pdf", "svg")


def saved_files(directory: Path) -> list[str]:
    """Relative paths of the files in a directory tree."""
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())


def test_save_fig_fig_dir(tmp_path: Path) -> None:
    """A relative path without an extension is saved in a subdirectory of fig_dir for each format."""
    save_fig(Figure(), "fig", fig_dir=tmp_path, formats=FORMATS)
    assert saved_files(tmp_path) == ["pdf/fig.pdf", "svg/fig.svg"]


def test_save_fig_fig_dir_subdirectory(tmp_path: Path) -> None:
    """The subdirectories of a relative path are created within the directory of each format."""
    save_fig(Figure(), Path("model") / "model_fig", fig_dir=tmp_path, formats=FORMATS)
    assert saved_files(tmp_path) == ["pdf/model/model_fig.pdf", "svg/model/model_fig.svg"]


def test_save_fig_fig_dir_extension_subdirectory(tmp_path: Path) -> None:
    """A relative path with an extension is saved as is within fig_dir, creating its subdirectories."""
    save_fig(Figure(), "model/fig.png", fig_dir=tmp_path, formats=FORMATS)
    assert saved_files(tmp_path) == ["model/fig.png"]


def test_save_fig_force_formats_subdirectory(tmp_path: Path) -> None:
    """With force_formats, a path with a dot in its name gets a directory and an extension for each format."""
    save_fig(Figure(), "model/fig.v1", fig_dir=tmp_path, formats=FORMATS, force_formats=True)
    assert saved_files(tmp_path) == ["pdf/model/fig.v1.pdf", "svg/model/fig.v1.svg"]


@pytest.mark.parametrize("use_fig_dir", [False, True])
def test_save_fig_absolute(tmp_path: Path, use_fig_dir: bool) -> None:
    """An absolute path is saved next to itself for each format, regardless of fig_dir."""
    save_fig(
        Figure(), tmp_path / "abs" / "fig",
        fig_dir=tmp_path / "unused" if use_fig_dir else None, formats=FORMATS
    )
    assert saved_files(tmp_path) == ["abs/fig.pdf", "abs/fig.svg"]


def test_save_fig_no_makedirs(tmp_path: Path) -> None:
    """With makedirs=False, missing directories are not created."""
    with pytest.raises(FileNotFoundError):
        save_fig(Figure(), "fig", fig_dir=tmp_path, formats=FORMATS, makedirs=False)
    assert saved_files(tmp_path) == []
