"""Documentation tests."""

import unittest

from pttools.utils import IS_GITHUB_ACTIONS


class DocsTest(unittest.TestCase):
    """Tests for the Sphinx configuration."""

    @unittest.skipIf(IS_GITHUB_ACTIONS, "Docs dependencies are not installed for CI test job")
    def test_docs_conf(self) -> None:
        """Test that the Sphinx configuration can be imported and has the correct project name."""
        from docs import conf  # noqa: PLC0415
        assert conf.project == "PTtools"


if __name__ == "__main__":
    unittest.main()
