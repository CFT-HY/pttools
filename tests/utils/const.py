"""Constants used by the unit tests"""

import os

TEST_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH: str = os.path.join(TEST_PATH, "test_data")
TEST_RESULT_PATH: str = os.path.join(os.path.dirname(TEST_PATH), "test-results")
TEST_FIGURE_PATH: str = os.path.join(TEST_RESULT_PATH, "figures")

for path in [TEST_PATH, TEST_DATA_PATH, TEST_RESULT_PATH, TEST_FIGURE_PATH]:
    os.makedirs(path, exist_ok=True)
