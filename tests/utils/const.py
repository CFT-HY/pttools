"""Constants used by the unit tests."""

from pathlib import Path
import typing as tp

#: Low accuracy settings of the spectra for the tests that do not compare the values of the spectra
#: with reference values, see :py:mod:`examples.props.n_points`
LOW_ACCURACY_SPECTRUM_KWARGS: dict[str, tp.Any] = {"nT": 300, "n_z_lookup": 500}

TEST_PATH: Path = Path(__file__).resolve().parent.parent
REPO_DIR: Path = TEST_PATH.parent
TEST_DATA_PATH: Path = TEST_PATH / "test_data"
TEST_RESULT_PATH: Path = REPO_DIR / "test-results"
TEST_FIGURE_PATH: Path = TEST_RESULT_PATH / "figures"
TEST_JSON_PATH: Path = TEST_RESULT_PATH / "json"

path: Path
for path in [TEST_PATH, TEST_DATA_PATH, TEST_JSON_PATH, TEST_RESULT_PATH, TEST_FIGURE_PATH]:
    path.mkdir(parents=True, exist_ok=True)
