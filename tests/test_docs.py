"""Documentation tests."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pttools.docs import paths
from pttools.utils import IS_GITHUB_ACTIONS


class DocsTest(unittest.TestCase):
    @unittest.skipIf(IS_GITHUB_ACTIONS, "Docs dependencies are not installed for CI test job")
    def test_docs_conf(self) -> None:
        from docs import conf  # noqa: PLC0415
        self.assertEqual(conf.project, "PTtools")


class DocsPathsTest(unittest.TestCase):
    """Tests for finding the documentation directory of the project being documented."""

    def setUp(self) -> None:
        self.tmp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.root: Path = Path(self.tmp_dir.name).resolve()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def make_docs_dir(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        for name in paths.DOCS_DIR_FILES:
            (path / name).write_text("")
        return path

    def test_is_docs_dir(self) -> None:
        self.assertFalse(paths.is_docs_dir(self.root))
        docs = self.make_docs_dir("docs")
        self.assertTrue(paths.is_docs_dir(docs))
        (docs / "Makefile").unlink()
        self.assertFalse(paths.is_docs_dir(docs))

    def test_find_docs_dir_env(self) -> None:
        """The docs directory alongside the virtual environment is preferred."""
        docs = self.make_docs_dir("project", "docs")
        env = self.root / "project" / "venv"
        env.mkdir()
        with mock.patch.object(paths, "env_dir", return_value=env):
            self.assertEqual(paths.find_docs_dir(cwd=self.root), docs)

    def test_find_docs_dir_cwd_subdir(self) -> None:
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            self.assertEqual(paths.find_docs_dir(cwd=self.root / "project"), docs)

    def test_find_docs_dir_cwd(self) -> None:
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            self.assertEqual(paths.find_docs_dir(cwd=docs), docs)

    def test_find_docs_dir_pttools_repo(self) -> None:
        """When run from the PTtools repository, its docs are found even if not in a virtual environment."""
        with mock.patch.object(paths, "env_dir", return_value=None):
            found = paths.find_docs_dir(cwd=self.root)
        pttools_docs = paths.PTTOOLS_DIR.parent / paths.DOCS_DIR_NAME
        if paths.is_docs_dir(pttools_docs):
            self.assertEqual(found, pttools_docs)
        else:
            self.assertIsNone(found)

    def test_find_docs_dir_not_found(self) -> None:
        with mock.patch.object(paths, "env_dir", return_value=None), \
                mock.patch.object(paths, "PTTOOLS_DIR", self.root / "site-packages" / "pttools"):
            self.assertIsNone(paths.find_docs_dir(cwd=self.root))

    def test_default_log_dir(self) -> None:
        docs = self.make_docs_dir("project", "docs")
        self.assertEqual(paths.default_log_dir(docs), self.root / "project" / "logs")
        with mock.patch.object(paths, "find_docs_dir", return_value=None):
            self.assertEqual(paths.default_log_dir(), Path.cwd() / "logs")


if __name__ == "__main__":
    unittest.main()
