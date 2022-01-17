import os.path

TEST_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(TEST_PATH, "test_data")
TEST_RESULT_PATH = os.path.join(os.path.dirname(TEST_PATH), "test-results")
TEST_FIGURE_PATH = os.path.join(TEST_RESULT_PATH, "figures")
