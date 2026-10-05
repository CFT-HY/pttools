"""Unit tests for documentation paths."""

from pathlib import Path
import tempfile
import typing as tp
import unittest
from unittest import mock

from pttools.docs import paths


class DocsPathsTest(unittest.TestCase):
    """Tests for finding the documentation directory of the project being documented."""

    @tp.override
    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.root: Path = Path(self.tmp_dir.name).resolve()

    @tp.override
    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def make_docs_dir(self, *parts: str) -> Path:
        """Create a directory with the files of a documentation directory under the temporary directory."""
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        for name in paths.DOCS_DIR_FILES:
            (path / name).write_text("")
        return path

    def test_is_docs_dir(self) -> None:
        """Test that a directory is recognised as a docs directory only if it has all the required files."""
        assert not paths.is_docs_dir(self.root)
        docs = self.make_docs_dir("docs")
        assert paths.is_docs_dir(docs)
        (docs / "Makefile").unlink()
        assert not paths.is_docs_dir(docs)

    def test_find_docs_dir_env(self) -> None:
        """The docs directory alongside the virtual environment is preferred."""
        docs = self.make_docs_dir("project", "docs")
        env = self.root / "project" / "venv"
        env.mkdir()
        with mock.patch.object(paths, "env_dir", return_value=env):
            assert paths.find_docs_dir(cwd=self.root) == docs

    def test_find_docs_dir_cwd_subdir(self) -> None:
        """The docs subdirectory of the working directory is found."""
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            assert paths.find_docs_dir(cwd=self.root / "project") == docs

    def test_find_docs_dir_cwd(self) -> None:
        """The working directory is found if it is itself a docs directory."""
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            assert paths.find_docs_dir(cwd=docs) == docs

    def test_find_docs_dir_pttools_repo(self) -> None:
        """When run from the PTtools repository, its docs are found even if not in a virtual environment."""
        with mock.patch.object(paths, "env_dir", return_value=None):
            found = paths.find_docs_dir(cwd=self.root)
        pttools_docs = paths.PTTOOLS_DIR.parent / paths.DOCS_DIR_NAME
        if paths.is_docs_dir(pttools_docs):
            assert found == pttools_docs
        else:
            assert found is None

    def test_find_docs_dir_not_found(self) -> None:
        """None is returned if no docs directory is found."""
        with mock.patch.object(paths, "env_dir", return_value=None), \
                mock.patch.object(paths, "PTTOOLS_DIR", self.root / "site-packages" / "pttools"):
            assert paths.find_docs_dir(cwd=self.root) is None

    def test_default_log_dir(self) -> None:
        """The log directory is next to the docs directory, or in the working directory if there are no docs."""
        docs = self.make_docs_dir("project", "docs")
        assert paths.default_log_dir(docs) == self.root / "project" / "logs"
        with mock.patch.object(paths, "find_docs_dir", return_value=None):
            assert paths.default_log_dir() == Path.cwd() / "logs"
