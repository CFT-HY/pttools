"""Tests for plotting examples"""

import unittest

from matplotlib.pyplot import close

from examples.basic import basic, datamodel, parallel, spectra
from examples.const_cs import const_cs, const_cs_bag_comparison, const_cs_find, const_cs_gw, const_cs_xi_v
from examples.entropy import entropy_comparison, entropy_grid, entropy_old, entropy_profile
from examples.gksvdv import gksvdv_bubble, gksvdv_comparison, gksvdv_fig2, \
    gksvdv_testing, gksvdv_testing2, gksvdv_testing3
from examples.low_k import low_k
from examples.props import chapman_jouguet, ke_frac, noise, reference_props, suppression, vp_vm_plane, w_by_w, xi_kappa
from examples.reverse import reverse, reverse_approx
from examples.solvers import bag, old_new, xi_kappa_bag
from examples.standard_model import standard_model_xi_v
from pttools.analysis import close_figs
from tests.utils.mark import mark_xfail_multiprocessing_jit, skip_slow, uses_multiprocessing


class ExampleTest(unittest.TestCase):
    # Basic
    @staticmethod
    def test_basic():
        close_figs(*basic.main())

    @staticmethod
    def test_datamodel():
        close_figs(datamodel.main())

    @staticmethod
    def test_parallel():
        close(parallel.main().fig)

    @staticmethod
    def test_spectra():
        close(spectra.main())

    # ConstCS
    @staticmethod
    def test_const_cs():
        plot, fig1, fig2 = const_cs.main()
        close_figs(plot.fig, fig1, fig2)

    @staticmethod
    def test_const_cs_bag_comparison():
        close(const_cs_bag_comparison.main().fig)

    @staticmethod
    def test_const_cs_find():
        const_cs_find.main()

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_const_cs_gw():
        figs1, figs2, table = const_cs_gw.main()
        close_figs(*figs1)
        close_figs(*figs2.flat)

    @staticmethod
    def test_const_cs_xi_v():
        close(const_cs_xi_v.main())

    @staticmethod
    def test_plot_const_cs_xi_v_w():
        import examples.const_cs.const_cs_xi_v_w as script
        script.plot.fig()

    # Entropy
    @staticmethod
    def test_entropy_comparison():
        entropy_comparison.main()

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_entropy_grid():
        close(entropy_grid.main())

    @staticmethod
    @unittest.expectedFailure
    def test_entropy_old():
        close(entropy_old.main())

    @staticmethod
    def test_entropy_profile():
        close(entropy_profile.main())

    # GKSVDV
    @staticmethod
    def test_gksvdv_bubble():
        close(gksvdv_bubble.main())

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_gksvdv_comparison():
        close_figs(*gksvdv_comparison.main())

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_gksvdv_fig2():
        close_figs(*gksvdv_fig2.main())

    @staticmethod
    def test_gksvdv_testing():
        close(gksvdv_testing.main())

    @staticmethod
    def test_gksvdv_testing2():
        close(gksvdv_testing2.main())

    @staticmethod
    def test_gksvdv_testing3():
        close(gksvdv_testing3.main())

    # Low-k
    @staticmethod
    def test_low_k():
        close(low_k.main())

    @staticmethod
    def test_plot_chapman_jouguet():
        close(chapman_jouguet.main().fig)

    @staticmethod
    def test_delta_theta():
        from examples.props import delta_theta
        delta_theta.plot.fig()

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_ke_frac():
        close(ke_frac.main())

    @staticmethod
    def test_noise():
        close(noise.main())

    @staticmethod
    def test_reference_props():
        close(reference_props.main())

    @staticmethod
    def test_suppression():
        close_figs(*[plot.fig for plot in suppression.main()])

    @staticmethod
    def test_vm_vp_plane():
        close(vp_vm_plane.main())

    @staticmethod
    def test_w_by_w():
        close(w_by_w.main())

    @staticmethod
    @mark_xfail_multiprocessing_jit
    @skip_slow
    @uses_multiprocessing
    def test_xi_kappa():
        close(xi_kappa.main())

    # Reverse
    @staticmethod
    def test_reverse():
        reverse.main()

    @staticmethod
    def test_reverse_approx():
        close(reverse_approx.main())

    # Solvers
    @staticmethod
    def test_bag():
        close(bag.main())

    @staticmethod
    def test_old_new():
        close(old_new.main())

    @staticmethod
    def test_xi_kappa_bag():
        close(xi_kappa_bag.main())

    # Standard Model
    @staticmethod
    def test_standard_model():
        import examples.standard_model.standard_model as script
        close(script.fig)
        close(script.plot.fig)
        close(script.plot2.fig)

    @staticmethod
    def test_standard_model_xi_v():
        close(standard_model_xi_v.main())


if __name__ == "__main__":
    unittest.main()
