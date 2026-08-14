"""Documentation tests"""

import unittest


class DocsTest(unittest.TestCase):
    def test_docs_conf(self):
        from docs import conf
        self.assertEqual(conf.project, "PTtools")


if __name__ == "__main__":
    unittest.main()
