# PTtools
[![CI](https://github.com/hindmars-org/pttools/actions/workflows/main.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/main.yml)
[![Docs](https://github.com/hindmars-org/pttools/actions/workflows/docs.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/docs.yml)
[![macOS](https://github.com/hindmars-org/pttools/actions/workflows/mac.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/mac.yml)
[![Windows](https://github.com/hindmars-org/pttools/actions/workflows/windows.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/windows.yml)

PTtools is Python library for calculating hydrodynamical quantities
around expanding bubbles of the new phase in an early universe phase transition,
and the resulting gravitational wave power spectrum in the Sound Shell Model.

![Types of solutions](https://raw.githubusercontent.com/AgenttiX/msc-thesis2/refs/heads/main/msc2-python/fig/relativistic_combustion.png)

### Documentation
The documentation is available online at [Read the Docs](https://pttools.readthedocs.io/).
The documentation for previous releases can be found at the
[releases](https://github.com/hindmars-org/pttools/releases) page.
The documentation can also be downloaded from the
[GitHub Actions results](https://github.com/hindmars-org/pttools/actions)
by selecting the latest successful *docs* workflow and then scrolling down to the *artifacts* section.
There you can find a zip file containing the documentation in various formats.

### References
- [Mäki: The effect of sound speed on the gravitational wave spectrum of first order phase transitions in the early universe (2024)](https://github.com/AgenttiX/msc-thesis2)
- [Hindmarsh et al.: Phase transitions in the early universe (2021)](https://arxiv.org/abs/2008.09136)
- [Hindmarsh & Hijazi: Gravitational waves from first order cosmological phase transitions in the Sound Shell Model (2019)](https://arxiv.org/abs/1909.10040)
- [Hindmarsh: Sound shell model for acoustic gravitational wave production at a first-order phase transition in the early Universe (2018)](https://arxiv.org/abs/1608.04735)

### Submodules
- bubble: Tools for computing the fluid shells (velocity and enthalpy as a function of scaled radius).
  Also includes some scripts for plotting.
- ssmttools: Tools for computing the GW spectra from the fluid shells.
- speedup: Computational utilities used by the other modules.
- omgw0: Tools for converting the GW spectra to frequencies and amplitudes today.

### Branches
- main: Stable version, from which releases are created.
- dev: Developments for later inclusion in the main branch.
- droplet-dev: Development of different solution types (legacy).
- eos-dev: Development of support for different equations of state (legacy).

### Requirements
Python 3.10 - 3.12

### Who do I talk to?
- Repo owner: [Mark Hindmarsh](https://github.com/hindmars/)

### Example figures
![Fluid velocity profiles](https://raw.githubusercontent.com/AgenttiX/msc-thesis2/refs/heads/main/msc2-python/fig/const_cs_gw_v.png)

![Gravitational wave spectra](https://raw.githubusercontent.com/AgenttiX/msc-thesis2/refs/heads/main/msc2-python/fig/const_cs_gw_omgw0.png)
