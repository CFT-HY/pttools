# README
[![CI](https://github.com/hindmars-org/pttools/actions/workflows/main.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/main.yml)
[![Docs](https://github.com/hindmars-org/pttools/actions/workflows/docs.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/docs.yml)
[![macOS](https://github.com/hindmars-org/pttools/actions/workflows/mac.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/mac.yml)
[![Windows](https://github.com/hindmars-org/pttools/actions/workflows/windows.yml/badge.svg)](https://github.com/hindmars-org/pttools/actions/workflows/windows.yml)

PTtools is Python library for calculating hydrodynamical quantities
around expanding bubbles of the new phase in an early universe phase transition,
and the resulting gravitational wave power spectrum in the sound shell model.

### References
- [Hindmarsh et al.: Phase transitions in the early universe (2021)](https://arxiv.org/abs/2008.09136)
- [Hindmarsh & Hijazi: Gravitational waves from first order cosmological phase transitions in the Sound Shell Model (2019)](https://arxiv.org/abs/1909.10040)
- [Hindmarsh: Sound shell model for acoustic gravitational wave production at a first-order phase transition in the early Universe (2018)](https://arxiv.org/abs/1608.04735)

### Submodules
- bubble: Tools for computing the fluid shells (velocity and enthalpy as a function of scaled radius).
  Also includes some scripts for plotting.
- speedup: Computational utilities used by the other modules.
- ssmttools: Tools for computing the GW spectra from the fluid shells.

### Branches
- main: Stable version, from which releases are created.
- dev: Developments for later inclusion in the main branch.
- droplet-dev: Development of different solution types.
- eos-dev: Development of support for different equations of state.

### Documentation
As this repository is not yet public, the documentation is not yet hosted directly on the web.
Documentation for the releases can be found at the
[releases](https://github.com/hindmars-org/pttools/releases) page.

The documentation is automatically generated for new commits, and the resulting documentation can be found at the
[GitHub Actions results](https://github.com/hindmars-org/pttools/actions)
by selecting the latest successful *docs* workflow and then scrolling down to the *artifacts* section.
There you can find a zip file containing the documentation in various formats.

The recommended way to browse the documentation is to unpack the docs zip file and then open
`html/index.html` with a web browser.

### Requirements
Python 3.10 - 3.12

### Who do I talk to?
- Repo owner: [Mark Hindmarsh](https://github.com/hindmars/)
