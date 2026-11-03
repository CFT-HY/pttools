"""Tests for the archiving utilities."""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zlib

import pytest

from pttools.utils.archive import archive, optimize_pngs

requires_optipng: pytest.MarkDecorator = pytest.mark.skipif(
    shutil.which("optipng") is None, reason="optipng is not installed"
)
requires_xz: pytest.MarkDecorator = pytest.mark.skipif(shutil.which("xz") is None, reason="xz is not installed")


@requires_xz
def test_archive(tmp_path: Path) -> None:
    """The archive should contain the given paths with their new names, without the excluded files.

    An existing archive should not be overwritten.
    """
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "data.h5").write_text("h5")
    (src / "sub" / "data.h5.sha256").write_text("sha")
    log_file = tmp_path / "run.log"
    log_file.write_text("log")
    paths = {src: "renamed", log_file: "run.log"}
    archive_path = tmp_path / "out" / "repo_nightly_2026-10-04.tar.xz"

    first = archive(archive_path, paths, exclude=("*.h5",))
    assert first == archive_path
    with tarfile.open(first) as tar:
        assert set(tar.getnames()) == {
            "renamed", "renamed/a.txt", "renamed/sub", "renamed/sub/data.h5.sha256", "run.log"
        }
    second = archive(archive_path, paths)
    assert second.name == "repo_nightly_2026-10-04_2.tar.xz"
    with tarfile.open(second) as tar:
        assert "renamed/sub/data.h5" in tar.getnames()
    assert archive(archive_path, paths).name == "repo_nightly_2026-10-04_3.tar.xz"

    with pytest.raises(FileNotFoundError):
        archive(archive_path, {tmp_path / "missing": "missing"})


def test_archive_xz_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If xz fails, the error should be raised and the partial archive removed."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xz = fake_bin / "xz"
    fake_xz.write_text("#!/bin/sh\ncat > /dev/null\nexit 2\n")
    fake_xz.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    src = tmp_path / "a.txt"
    src.write_text("a")
    archive_path = tmp_path / "repo_nightly_2026-10-04.tar.xz"
    with pytest.raises(subprocess.CalledProcessError):
        archive(archive_path, {src: "a.txt"})
    assert not archive_path.exists()



def create_png(path: Path, width: int = 64, height: int = 64) -> None:
    """Create an uncompressed grayscale PNG file, which optipng can make smaller.

    :param path: the path of the file
    :param width: the width of the image in pixels
    :param height: the height of the image in pixels
    """
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big") + chunk_type + data
            + zlib.crc32(chunk_type + data).to_bytes(4, "big")
        )

    pixels = b"".join(b"\x00" + bytes((x * y) % 256 for x in range(width)) for y in range(height))
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes((8, 0, 0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels, 0)) + chunk(b"IEND", b"")
    )


@requires_optipng
def test_optimize_pngs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The PNG files should be made smaller, and identical files should stay identical.

    Invalid files should be logged and left as they are.
    """
    build_dir = tmp_path / "_build"
    fig_dir = tmp_path / "fig"
    create_png(build_dir / "html" / "a.png")
    create_png(build_dir / "epub" / "a.png")
    create_png(fig_dir / "b.png", width=32)
    invalid = fig_dir / "invalid.png"
    invalid.write_text("not a PNG")
    (fig_dir / "notes.txt").write_text("text")
    sizes_before = {path: path.stat().st_size for path in tmp_path.rglob("*.png")}

    optimize_pngs([build_dir, fig_dir, tmp_path / "missing"])

    output = capsys.readouterr().out
    assert f"optipng failed for {invalid}" in output
    assert "Optimized 4 PNG files (3 unique) with optipng -o7 -preserve -quiet" in output
    assert f"{build_dir} " in output
    assert f"{fig_dir} " in output
    assert invalid.read_text() == "not a PNG"
    for path, size in sizes_before.items():
        if path != invalid:
            assert path.stat().st_size < size, path
    assert (build_dir / "html" / "a.png").read_bytes() == (build_dir / "epub" / "a.png").read_bytes()


def test_optimize_pngs_without_optipng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    """Without optipng, the PNG files should be left as they are."""
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    png = tmp_path / "fig" / "a.png"
    create_png(png)
    data = png.read_bytes()
    optimize_pngs([png.parent])
    assert "optipng was not found in PATH" in capsys.readouterr().out
    assert png.read_bytes() == data
