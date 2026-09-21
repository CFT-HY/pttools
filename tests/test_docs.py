"""Documentation tests."""

import os
import tempfile
import unittest
from unittest import mock

from pttools.docs import paths
from pttools.utils import IS_GITHUB_ACTIONS


class DocsTest(unittest.TestCase):
    @unittest.skipIf(IS_GITHUB_ACTIONS, "Docs dependencies are not installed for CI test job")
    def test_docs_conf(self):
        from docs import conf  # noqa: PLC0415
        self.assertEqual(conf.project, "PTtools")


class DocsPathsTest(unittest.TestCase):
    """Tests for finding the documentation directory of the project being documented."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def make_docs_dir(self, *parts: str) -> str:
        path = os.path.join(self.root, *parts)
        os.makedirs(path, exist_ok=True)
        for name in paths.DOCS_DIR_FILES:
            with open(os.path.join(path, name), "w") as file:
                file.write("")
        return path

    def test_is_docs_dir(self):
        self.assertFalse(paths.is_docs_dir(self.root))
        docs = self.make_docs_dir("docs")
        self.assertTrue(paths.is_docs_dir(docs))
        os.remove(os.path.join(docs, "Makefile"))
        self.assertFalse(paths.is_docs_dir(docs))

    def test_find_docs_dir_env(self):
        """The docs directory alongside the virtual environment is preferred."""
        docs = self.make_docs_dir("project", "docs")
        env = os.path.join(self.root, "project", "venv")
        os.makedirs(env)
        with mock.patch.object(paths, "env_dir", return_value=env):
            self.assertEqual(paths.find_docs_dir(cwd=self.root), docs)

    def test_find_docs_dir_cwd_subdir(self):
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            self.assertEqual(paths.find_docs_dir(cwd=os.path.join(self.root, "project")), docs)

    def test_find_docs_dir_cwd(self):
        docs = self.make_docs_dir("project", "docs")
        with mock.patch.object(paths, "env_dir", return_value=None):
            self.assertEqual(paths.find_docs_dir(cwd=docs), docs)

    def test_find_docs_dir_pttools_repo(self):
        """When run from the PTtools repository, its docs are found even if not in a virtual environment."""
        with mock.patch.object(paths, "env_dir", return_value=None):
            found = paths.find_docs_dir(cwd=self.root)
        pttools_docs = os.path.join(os.path.dirname(paths.PTTOOLS_DIR), paths.DOCS_DIR_NAME)
        if paths.is_docs_dir(pttools_docs):
            self.assertEqual(found, pttools_docs)
        else:
            self.assertIsNone(found)

    def test_find_docs_dir_not_found(self):
        with mock.patch.object(paths, "env_dir", return_value=None), \
                mock.patch.object(paths, "PTTOOLS_DIR", os.path.join(self.root, "site-packages", "pttools")):
            self.assertIsNone(paths.find_docs_dir(cwd=self.root))

    def test_default_log_dir(self):
        docs = self.make_docs_dir("project", "docs")
        self.assertEqual(paths.default_log_dir(docs), os.path.join(self.root, "project", "logs"))
        with mock.patch.object(paths, "find_docs_dir", return_value=None):
            self.assertEqual(paths.default_log_dir(), os.path.join(os.getcwd(), "logs"))


if __name__ == "__main__":
    unittest.main()
