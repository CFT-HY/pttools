"""Tests for the parallel creation of bubbles and spectra."""

from concurrent.futures import ThreadPoolExecutor
import typing as tp
import unittest
from unittest import mock

import numpy as np
import pytest

from pttools.analysis import parallel
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.speedup import parallel as speedup_parallel
from tests.utils.mark import uses_parallel


class TestCreateSpectra(unittest.TestCase):
    """Tests for creating spectra in parallel with processes or threads."""

    model: tp.ClassVar[ConstCSModel]
    #: Low-resolution settings for faster tests
    SPECTRUM_KWARGS: tp.ClassVar[dict[str, tp.Any]] = {
        "r_star": 0.1, "y": np.logspace(-1, 3, 300), "nT": 2000, "n_z_lookup": 2000
    }

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)

    def test_create_spectrum_parallel_in_kwargs(self) -> None:
        """Giving parallel in spectrum_kwargs should raise an error, as it's an argument of create_spectrum()."""
        with pytest.raises(ValueError, match="parallel"):
            parallel.create_spectrum(np.array([0.5, 0.2]), self.model, spectrum_kwargs={"parallel": True})

    @uses_parallel
    def test_create_spectra_threads(self) -> None:
        """Spectra created with use_threads should be created in a thread pool, and match those of a single thread."""
        v_walls = np.array([0.5, 0.7])
        alpha_ns = np.array([0.1, 0.2])
        ref = parallel.create_spectra(
            self.model, v_walls, alpha_ns, single_thread=True, spectrum_kwargs=self.SPECTRUM_KWARGS)
        with mock.patch.object(speedup_parallel, "ThreadPoolExecutor", wraps=ThreadPoolExecutor) as thread_pool, \
                mock.patch.object(speedup_parallel, "create_process_pool") as process_pool, \
                mock.patch.object(speedup_parallel, "get_global_process_pool") as global_process_pool:
            spectra = parallel.create_spectra(
                self.model, v_walls, alpha_ns, use_threads=True, max_workers=4, spectrum_kwargs=self.SPECTRUM_KWARGS)
        thread_pool.assert_called_once_with(max_workers=4)
        process_pool.assert_not_called()
        global_process_pool.assert_not_called()
        assert isinstance(spectra, np.ndarray)
        assert isinstance(ref, np.ndarray)
        assert spectra.shape == ref.shape == (alpha_ns.size, v_walls.size)
        for spectrum, ref_spectrum in zip(spectra.flat, ref.flat, strict=True):
            assert isinstance(spectrum, Spectrum)
            np.testing.assert_allclose(spectrum.omgw0(), ref_spectrum.omgw0(), rtol=1e-12)

    def test_create_spectra_parallel_flag(self) -> None:
        """The Numba parallelism of the spectra should be enabled only with threads and concurrent parallelism."""
        cases = ((False, True, False), (True, False, False), (True, True, True))
        for use_threads, concurrent, expected in cases:
            with self.subTest(use_threads=use_threads, concurrent=concurrent), \
                    mock.patch.object(parallel.options, "CONCURRENT_PARALLELISM", concurrent), \
                    mock.patch.object(parallel, "create_bubbles") as create_bubbles:
                parallel.create_spectra(self.model, np.array([0.5]), np.array([0.1]), use_threads=use_threads)
                kwargs = create_bubbles.call_args.kwargs
                assert kwargs["kwargs"]["parallel"] is expected
                assert kwargs["use_threads"] is use_threads
                assert kwargs["bubble_func"] is parallel.create_spectrum


if __name__ == "__main__":
    unittest.main()
