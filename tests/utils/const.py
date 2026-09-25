"""Constants used by the unit tests."""

from pathlib import Path

TEST_PATH: Path = Path(__file__).resolve().parent.parent
REPO_DIR: Path = TEST_PATH.parent
TEST_DATA_PATH: Path = TEST_PATH / "test_data"
TEST_RESULT_PATH: Path = REPO_DIR / "test-results"
TEST_FIGURE_PATH: Path = TEST_RESULT_PATH / "figures"
TEST_JSON_PATH: Path = TEST_RESULT_PATH / "json"

path: Path
for path in [TEST_PATH, TEST_DATA_PATH, TEST_JSON_PATH, TEST_RESULT_PATH, TEST_FIGURE_PATH]:
    path.mkdir(parents=True, exist_ok=True)
