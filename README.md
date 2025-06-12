# PTtools
[![DOI](https://zenodo.org/badge/373413324.svg)](https://zenodo.org/badge/latestdoi/373413324)
[![ReadTheDocs](https://readthedocs.org/projects/pttools/badge/)](https://pttools.readthedocs.io/)
[![CI](https://github.com/CFT-HY/pttools/actions/workflows/main.yml/badge.svg)](https://github.com/CFT-HY/pttools/actions/workflows/main.yml)
[![Windows](https://github.com/CFT-HY/pttools/actions/workflows/windows.yml/badge.svg)](https://github.com/CFT-HY/pttools/actions/workflows/windows.yml)
[![macOS](https://github.com/CFT-HY/pttools/actions/workflows/mac.yml/badge.svg)](https://github.com/CFT-HY/pttools/actions/workflows/mac.yml)
[![codecov](https://codecov.io/gh/CFT-HY/pttools/graph/badge.svg?token=ALFVWC1LZR)](https://codecov.io/gh/CFT-HY/pttools)

PTtools is a Python library for calculating hydrodynamical quantities
around expanding bubbles of the new phase in an early universe phase transition,
and the resulting gravitational wave power spectrum in the Sound Shell Model.

![Types of solutions](https://raw.githubusercontent.com/AgenttiX/msc-thesis2/refs/heads/main/msc2-python/fig/relativistic_combustion.png)

### Installation
PTtools is [available on PyPI](https://pypi.org/project/pttools-gw/) and can be installed with pip:
``` bash
pip3 install --upgrade pttools-gw[numbalsoda,performance]
```
The `[numbalsoda]` and `[performance]` dependencies are optional,
and you can omit them if they are not available on your platform.

PTtools is also [available on Docker Hub](https://hub.docker.com/r/cfthy/pttools) and can be installed with:
``` bash
docker pull cfthy/pttools:main
```

### Documentation
The documentation is available online at [Read the Docs](https://pttools.readthedocs.io/).
The documentation for previous releases can be found at the
[releases](https://github.com/CFT-HY/pttools/releases) page.
The documentation can also be downloaded from the
[GitHub Actions results](https://github.com/CFT-HY/pttools/actions)
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
- omgw0: Tools for converting the GW spectra to frequencies and amplitudes today. Includes utilities for approximations and noise.

### Who do I talk to?
- Repo owner: [Mark Hindmarsh](https://github.com/hindmars/)
- Main developer: [Mika Mäki](https://github.com/AgenttiX)

### Example figures
Fluid velocity profiles
![Fluid velocity profiles](https://raw.githubusercontent.com/AgenttiX/msc-thesis2/refs/heads/main/msc2-python/fig/const_cs_gw_v.png)

Gravitational wave power spectra
![Gravitational wave power spectra](https://raw.githubusercontent.com/AgenttiX/msc-thesis2/refs/heads/main/msc2-python/fig/const_cs_gw_omgw0.png)
