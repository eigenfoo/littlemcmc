<p align="center"><img src="docs/_static/logo/default-cropped.png"></p>

> *Warning:* `littlemcmc`'s behavior is unstable in Jupyter notebooks - for best
> results and support, please use `littlemcmc` in Python scripts. Furthermore,
> despite best efforts, `littlemcmc` is not guaranteed to be up-to-date with the
> [current PyMC3 HMC/NUTS
> samplers](https://github.com/pymc-devs/pymc3/tree/master/pymc3/step_methods/hmc) -
> please consult [our GitHub
> issues](https://github.com/eigenfoo/littlemcmc/issues).

---

![Tests Status](https://github.com/eigenfoo/littlemcmc/workflows/tests/badge.svg)
![Lint Status](https://github.com/eigenfoo/littlemcmc/workflows/lint/badge.svg)
![Up to date with PyMC3 Status](https://github.com/eigenfoo/littlemcmc/workflows/even-with-pymc3/badge.svg)
[![Coverage Status](https://codecov.io/gh/eigenfoo/littlemcmc/branch/master/graph/badge.svg)](https://codecov.io/gh/eigenfoo/littlemcmc)
[![Documentation Status](https://readthedocs.org/projects/littlemcmc/badge/?version=latest)](https://littlemcmc.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/eigenfoo/littlemcmc)](https://github.com/eigenfoo/littlemcmc/blob/master/LICENSE.littlemcmc.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4710368.svg)](https://doi.org/10.5281/zenodo.4710368)

> littlemcmc &nbsp; &nbsp; /lɪtəl ɛm si ɛm si/ &nbsp; &nbsp; _noun_
>
> A lightweight and performant implementation of HMC and NUTS in Python, spun
> out of [the PyMC project](https://github.com/pymc-devs). Not to be confused
> with [minimc](https://github.com/ColCarroll/minimc).

## Installation

The latest release of LittleMCMC can be installed from PyPI using `pip`:

```bash
pip install littlemcmc
```

The current development branch of LittleMCMC can be installed directly from
GitHub, also using `pip`:

```bash
pip install git+https://github.com/eigenfoo/littlemcmc.git
```

## Contributors

LittleMCMC is developed by [George Ho](https://eigenfoo.xyz/). For a full list
of contributors, please see the [GitHub contributor
graph](https://github.com/eigenfoo/littlemcmc/graphs/contributors).

## License

LittleMCMC is modified from [the PyMC3 and PyMC4
projects](https://github.com/pymc-devs/), both of which are distributed under
the Apache-2.0 license. A copy of both projects' license files are distributed
with LittleMCMC. All modifications from PyMC are distributed under [an identical
Apache-2.0 license](https://github.com/eigenfoo/littlemcmc/blob/master/LICENSE).

## Citation

Please cite using the [Zenodo DOI](https://doi.org/10.5281/zenodo.4710367).
