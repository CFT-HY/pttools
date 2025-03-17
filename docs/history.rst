Version history
===============

1.0.0 (planned changes)
-----------------------
- GW power spectrum for low wavenumbers by using the formulae by :giombi_2024_cs:`\ `.
- More polished documentation and examples.
- Improve unit testing by using more comprehensive reference data, which are stored externally as HDF5.
  Test whether `Git LFS <https://git-lfs.com/>`_ on GitLab would be a suitable storage.
- Improve integration with Cobaya and add examples.

0.9.0 (upcoming)
----------------
These modifications were `Mika's master's thesis <https://github.com/AgenttiX/msc-thesis2>`_.

- PTtools published as open source with the MIT license
- Documentation published at Read the Docs.
- New fluid velocity profile solver and object-oriented interface.
- Support for equations of state beyond the bag model.
    - Temperature-dependent degrees of freedom and sound speed.
- Support for computing the gravitational wave spectrum today ($\Omega_\text{gw,0}$).


0.0.1
-----
These modifications were Mika's summer project in 2021.

- Improve code structure by splitting :mod:`pttools.bubble` and :mod:`pttools.ssmtools` to multiple submodules.
- Improve code quality.
    - Lint each commit automatically with `Pylint <https://pylint.pycqa.org/en/latest/>`_.
    - Improve compliance with PEP8.
- Add support for other integrators in addition to
  :meth:`scipy.integrate.odeint`
  such as
  :meth:`scipy.integrate.solve_ivp`
  and NumbaLSODA.
- Speed up the simulations with Numba and NumbaLSODA.
    - Full GW power spectrum calculations: 5x for 1 CPU, 7x for 4 CPUs
    - Sine transform: Nx for N CPU cores (trivially parallelisable, minus some overhead)
    - ODE integration: 20x faster than :meth:`scipy.integrate.odeint` with pure Python
- Add unit testing.
    - Set up `CI/CD pipeline on GitHub Actions <https://github.com/CFT-HY/pttools/actions>`_.
    - Set up automatic testing with Python versions from 3.6 to 3.9
      and with multiple versions of Numba and other libraries.
    - Set up automatic testing with all major operating systems (GNU/Linux, Windows, macOS).
    - Set up automatic performance testing using `timeit <https://docs.python.org/3/library/timeit.html>`_.
    - Set up automatic profiling (:mod:`tests.profiling`) of the performance-critical parts of the code such as
      :meth:`pttools.ssmtools.spectrum.power_gw_scaled` and
      :meth:`tests.paper.ssm_paper_utils.do_all_plot_ps_compare_nuc` using
      `cProfile <https://docs.python.org/3/library/profile.html>`_,
      `Pyinstrument <https://github.com/joerick/pyinstrument>`_ and
      `YAPPI <https://github.com/sumerc/yappi>`_.
- `Package <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_
  PTtools with
  `setuptools <https://pypi.org/project/setuptools/>`_
  so that it can be installed with pip.
  - This prepares the project for being published on `PyPI <https://pypi.org/>`_.
- Add example scripts for running on `Slurm <https://slurm.schedmd.com/>`_ clusters.
- Add `Sphinx <https://www.sphinx-doc.org/en/master/>`_ documentation (the one you're currently reading).


Previous development
--------------------

2020-06
^^^^^^^
Bubble

- Small improvements to docstrints.
- Start introducing checks for physical ($v_\text{wall}, \alpha_n$): check_wall_speed, check_physical_parameters

SSMtools

- use analytic formula for high-k sin transforms.
  Should eliminate spurious high-k signal in GWPS from numerical error.
- sin_transform now handles array z, simplifying its calling elsewhere
- resample_uniform_xi function introduced to simply coding for sin_transform of lam
- Allow calls to power spectra and spectral density functions
  with 2-component params list, i.e. params = [v_wall, alpha_n] (parse_params)
  exponential nucleation with parameters (1,) assumed.
- reduced NQDEFAULT from 2000 to 320, to reduce high-k numerical error when using numerical sin transform

Planned changes
"""""""""""""""
Bubble

- Include bubble nucleation calculations of beta (from $V(T,\phi)$)

SSMtools

- Check default nucleation type for nu function.
