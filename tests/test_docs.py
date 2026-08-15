"""Documentation tests"""

import unittest

from pttools.utils import IS_GITHUB_ACTIONS


class DocsTest(unittest.TestCase):
    @unittest.skipIf(IS_GITHUB_ACTIONS, "Docs dependencies are not installed for CI test job")
    def test_docs_conf(self):
        from docs import conf
        self.assertEqual(conf.project, "PTtools")


if __name__ == "__main__":
    unittest.main()
