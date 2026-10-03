"""Tests for plotting examples."""

import unittest

from matplotlib.pyplot import close

from examples.basic import basic, datamodel, parallel, spectra
from examples.const_cs import const_cs, const_cs_bag_comparison, const_cs_find, const_cs_gw, const_cs_xi_v
from examples.entropy import entropy_comparison, entropy_grid, entropy_old, entropy_profile
from examples.gksvdv import (
    gksvdv_bubble,
    gksvdv_comparison,
    gksvdv_fig2,
    gksvdv_testing,
    gksvdv_testing2,
    gksvdv_testing3,
)
from examples.low_k import low_k
from examples.props import chapman_jouguet, ke_frac, noise, reference_props, suppression, vp_vm_plane, w_by_w, xi_kappa
from examples.reverse import reverse, reverse_approx
from examples.solvers import bag, old_new, xi_kappa_bag
from examples.standard_model import standard_model_xi_v
from pttools.analysis import close_figs
from tests.utils.mark import mark_xfail_multiprocessing_jit, skip_slow, uses_multiprocessing


class ExampleTest(unittest.TestCase):
    """Tests that run the examples and close the resulting figures."""

    # Basic
    @staticmethod
    def test_basic() -> None:
        """Test that the example ``examples.basic.basic`` runs without errors."""
        close_figs(*basic.main())

    @staticmethod
    def test_datamodel() -> None:
        """Test that the example ``examples.basic.datamodel`` runs without errors."""
        close_figs(datamodel.main())

    @staticmethod
    def test_parallel() -> None:
        """Test that the example ``examples.basic.parallel`` runs without errors."""
        close(parallel.main().fig)

    @staticmethod
    def test_spectra() -> None:
        """Test that the example ``examples.basic.spectra`` runs without errors."""
        close(spectra.main())

    # ConstCS
    @staticmethod
    def test_const_cs() -> None:
        """Test that the example ``examples.const_cs.const_cs`` runs without errors."""
        plot, fig1, fig2 = const_cs.main()
        close_figs(plot.fig, fig1, fig2)

    @staticmethod
    def test_const_cs_bag_comparison() -> None:
        """Test that the example ``examples.const_cs.const_cs_bag_comparison`` runs without errors."""
        close(const_cs_bag_comparison.main().fig)

    @staticmethod
    def test_const_cs_find() -> None:
        """Test that the example ``examples.const_cs.const_cs_find`` runs without errors."""
        const_cs_find.main()

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_const_cs_gw() -> None:
        """Test that the example ``examples.const_cs.const_cs_gw`` runs without errors."""
        figs1, figs2, _table = const_cs_gw.main()
        close_figs(*figs1)
        close_figs(*figs2.flat)

    @staticmethod
    def test_const_cs_xi_v() -> None:
        """Test that the example ``examples.const_cs.const_cs_xi_v`` runs without errors."""
        close(const_cs_xi_v.main())

    @staticmethod
    def test_plot_const_cs_xi_v_w() -> None:
        """Test that the example ``examples.const_cs.const_cs_xi_v_w`` runs without errors."""
        import examples.const_cs.const_cs_xi_v_w as script  # noqa: PLC0415
        script.plot.fig()

    # Entropy
    @staticmethod
    def test_entropy_comparison() -> None:
        """Test that the example ``examples.entropy.entropy_comparison`` runs without errors."""
        entropy_comparison.main()

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_entropy_grid() -> None:
        """Test that the example ``examples.entropy.entropy_grid`` runs without errors."""
        close(entropy_grid.main())

    @staticmethod
    @unittest.expectedFailure
    def test_entropy_old() -> None:
        """Run the example ``examples.entropy.entropy_old``, which is currently expected to fail."""
        close(entropy_old.main())

    @staticmethod
    def test_entropy_profile() -> None:
        """Test that the example ``examples.entropy.entropy_profile`` runs without errors."""
        close(entropy_profile.main())

    # GKSVDV
    @staticmethod
    def test_gksvdv_bubble() -> None:
        """Test that the example ``examples.gksvdv.gksvdv_bubble`` runs without errors."""
        close(gksvdv_bubble.main())

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_gksvdv_comparison() -> None:
        """Test that the example ``examples.gksvdv.gksvdv_comparison`` runs without errors."""
        close_figs(*gksvdv_comparison.main())

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_gksvdv_fig2() -> None:
        """Test that the example ``examples.gksvdv.gksvdv_fig2`` runs without errors."""
        close_figs(*gksvdv_fig2.main())

    @staticmethod
    def test_gksvdv_testing() -> None:
        """Test that the example ``examples.gksvdv.gksvdv_testing`` runs without errors."""
        close(gksvdv_testing.main())

    @staticmethod
    def test_gksvdv_testing2() -> None:
        """Test that the example ``examples.gksvdv.gksvdv_testing2`` runs without errors."""
        close(gksvdv_testing2.main())

    @staticmethod
    def test_gksvdv_testing3() -> None:
        """Test that the example ``examples.gksvdv.gksvdv_testing3`` runs without errors."""
        close(gksvdv_testing3.main())

    # Low-k
    @staticmethod
    def test_low_k() -> None:
        """Test that the example ``examples.low_k.low_k`` runs without errors."""
        close(low_k.main())

    @staticmethod
    def test_plot_chapman_jouguet() -> None:
        """Test that the example ``examples.props.chapman_jouguet`` runs without errors."""
        close(chapman_jouguet.main().fig)

    @staticmethod
    def test_delta_theta() -> None:
        """Test that the example ``examples.props.delta_theta`` runs without errors."""
        from examples.props import delta_theta  # noqa: PLC0415
        delta_theta.plot.fig()

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_ke_frac() -> None:
        """Test that the example ``examples.props.ke_frac`` runs without errors."""
        close(ke_frac.main())

    @staticmethod
    def test_noise() -> None:
        """Test that the example ``examples.props.noise`` runs without errors."""
        close(noise.main())

    @staticmethod
    def test_reference_props() -> None:
        """Test that the example ``examples.props.reference_props`` runs without errors."""
        close(reference_props.main())

    @staticmethod
    def test_suppression() -> None:
        """Test that the example ``examples.props.suppression`` runs without errors."""
        close_figs(*[plot.fig for plot in suppression.main()])

    @staticmethod
    def test_vm_vp_plane() -> None:
        """Test that the example ``examples.props.vp_vm_plane`` runs without errors."""
        close(vp_vm_plane.main())

    @staticmethod
    def test_w_by_w() -> None:
        """Test that the example ``examples.props.w_by_w`` runs without errors."""
        close(w_by_w.main())

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_xi_kappa() -> None:
        """Test that the example ``examples.props.xi_kappa`` runs without errors."""
        close(xi_kappa.main())

    # Reverse
    @staticmethod
    def test_reverse() -> None:
        """Test that the example ``examples.reverse.reverse`` runs without errors."""
        reverse.main()

    @staticmethod
    def test_reverse_approx() -> None:
        """Test that the example ``examples.reverse.reverse_approx`` runs without errors."""
        close(reverse_approx.main())

    # Solvers
    @staticmethod
    def test_bag() -> None:
        """Test that the example ``examples.solvers.bag`` runs without errors."""
        close(bag.main())

    @staticmethod
    def test_old_new() -> None:
        """Test that the example ``examples.solvers.old_new`` runs without errors."""
        close(old_new.main())

    @staticmethod
    def test_xi_kappa_bag() -> None:
        """Test that the example ``examples.solvers.xi_kappa_bag`` runs without errors."""
        close(xi_kappa_bag.main())

    # Standard Model
    @staticmethod
    def test_standard_model() -> None:
        """Test that the example ``examples.standard_model.standard_model`` runs without errors."""
        import examples.standard_model.standard_model as script  # noqa: PLC0415
        close(script.fig)
        close(script.plot.fig)
        close(script.plot2.fig)

    @staticmethod
    def test_standard_model_xi_v() -> None:
        """Test that the example ``examples.standard_model.standard_model_xi_v`` runs without errors."""
        close(standard_model_xi_v.main())


if __name__ == "__main__":
    unittest.main()
