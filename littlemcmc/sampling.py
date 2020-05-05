#  Copyright 2019-2020 George Ho
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Sampling driver functions (unrelated to PyMC3's `sampling.py`)."""

import os
from collections.abc import Iterable
from typing import Callable, Tuple, Optional, Union, List
import logging
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm, tqdm_notebook
from .nuts import NUTS
from .hmc import HamiltonianMC
from .quadpotential import QuadPotential, QuadPotentialDiagAdapt, QuadPotentialFullAdapt
from .report import SamplerWarning

_log = logging.getLogger("littlemcmc")


def _sample_one_chain(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_ndim: int,
    draws: int,
    tune: int,
    step: Union[NUTS, HamiltonianMC],
    start: np.ndarray,
    random_seed: Union[None, int, List[int]] = None,
    discard_tuned_samples: bool = True,
    progressbar: Union[bool, str] = True,
    progressbar_position: Optional[int] = None,
):
    """Sample one chain in one process."""
    if random_seed is not None:
        np.random.seed(random_seed)

    if progressbar_position is None:
        progressbar_position = 0

    q = start
    trace = np.zeros([model_ndim, tune + draws])
    stats: List[SamplerWarning] = []

    if progressbar == "notebook":
        iterator = tqdm_notebook(range(tune + draws), position=progressbar_position)
    elif progressbar == "console" or progressbar:
        iterator = tqdm(range(tune + draws), position=progressbar_position)
    else:
        iterator = range(tune + draws)

    for i in iterator:
        q, step_stats = step._astep(q)
        trace[:, i] = q
        stats.extend(step_stats)
        if i == tune - 1:  # Draws are 0-indexed, not 1-indexed
            step.stop_tuning()

    if discard_tuned_samples:
        trace = trace[:, tune:]
        stats = stats[tune:]

    return trace, stats


def sample(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_ndim: int,
    draws: int = 1000,
    tune: int = 1000,
    step: Union[NUTS, HamiltonianMC] = None,
    init: str = "auto",
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    start: Optional[np.ndarray] = None,
    progressbar: Union[bool, str] = True,
    random_seed: Optional[Union[int, List[int]]] = None,
    discard_tuned_samples: bool = True,
    **kwargs,
):
    """
    Draw samples from the posterior using the given step methods.

    Parameters
    ----------
    logp_dlogp_func: Python callable
        Python callable that returns a tuple of the model joint log probability and its
        derivative, in that order.
    model_ndim: int
        The number of parameters of the model.
    draws: int
        The number of samples to draw. Defaults to 1000. The number of tuned samples are
        discarded by default. See ``discard_tuned_samples``.
    tune: int
        Number of iterations to tune, defaults to 1000. Samplers adjust the step sizes,
        scalings or similar during tuning. Tuning samples will be drawn in addition to
        the number specified in the ``draws`` argument, and will be discarded unless
        ``discard_tuned_samples`` is set to False.
    step: function
        A step function. By default the NUTS step method will be used.
    init: str
        Initialization method to use for auto-assigned NUTS samplers.
            * auto: Choose a default initialization method automatically. Currently,
              this is ``jitter+adapt_diag``, but this can change in the future. If you
              depend on the exact behaviour, choose an initialization method explicitly.
            * adapt_diag: Start with a identity mass matrix and then adapt a diagonal
              based on the variance of the tuning samples.
            * jitter+adapt_diag: Same as ``adapt_diag``, but add uniform jitter in
              [-1, 1] to the starting point in each chain.
            * adapt_full: Same as `'adapt_diag'`, but adapt a dense mass matrix using
              the sample covariances.
    chains: int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics and can also reveal multiple modes in the posterior. If
        ``None``, then set to either ``cores`` or 2, whichever is larger.
    cores: int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs
        in the system, but at most 4.
    start: dict, or array of dict
        Starting point in parameter space. Initialization methods for NUTS (see ``init``
        keyword) can overwrite the default.
    progressbar: bool, optional default=True
        Whether or not to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    random_seed: int or list of ints
        A list is accepted if ``cores`` is greater than one.
    discard_tuned_samples: bool
        Whether to discard posterior samples of the tune interval.

    Returns
    -------
    trace: np.array
        An array that contains the samples.
    stats: dict
        A dictionary that contains sampler statistics.

    Notes
    -----
    Optional keyword arguments can be passed to ``sample`` to be delivered to the
    ``step_method``s used during sampling. In particular, the NUTS step method accepts a
    number of arguments. You can find a full list of arguments in the docstring of the
    step methods. Common options are:
        * target_accept: float in [0, 1]. The step size is tuned such that we
          approximate this acceptance rate. Higher values like 0.9 or 0.95 often work
          better for problematic posteriors.
        * max_treedepth: The maximum depth of the trajectory tree.
        * step_scale: float, default 0.25. The initial guess for the step size scaled
        down by :math:`1/n**(1/4)`.
    """
    if cores is None:
        cores = min(4, os.cpu_count())
    if chains is None:
        chains = max(2, cores)

    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]  # type: ignore
    elif isinstance(random_seed, Iterable) and len(random_seed) != chains:
        random_seed = random_seed[:chains]
    elif not isinstance(random_seed, Iterable):
        raise TypeError("Invalid value for `random_seed`. Must be tuple, list or int")

    # Small trace warning
    if draws == 0:
        msg = "Tuning was enabled throughout the whole trace."
        _log.warning(msg)
    elif draws < 500:
        msg = "Only {} samples in chain.".format(draws)
        _log.warning(msg)

    if step is None or start is None:
        start_, step_ = init_nuts(
            logp_dlogp_func=logp_dlogp_func,
            model_ndim=model_ndim,
            init=init,
            random_seed=random_seed,
            **kwargs,
        )
        if step is None:
            step = step_
        if start is None:
            start = start_

    _log.info("Multiprocess sampling ({} chains in {} jobs)".format(chains, cores))
    results = Parallel(n_jobs=cores, backend="multiprocessing")(
        delayed(_sample_one_chain)(
            logp_dlogp_func=logp_dlogp_func,
            model_ndim=model_ndim,
            draws=draws,
            tune=tune,
            step=step,
            start=start,
            random_seed=seed,
            discard_tuned_samples=discard_tuned_samples,
            progressbar=progressbar,
            progressbar_position=i,
        )
        for i, seed in enumerate(random_seed)
    )

    # Flatten `trace` to have shape [num_variables, num_chains * num_samples]
    trace = np.hstack([np.atleast_2d(chain_trace) for (chain_trace, _) in results])

    # Reshape `stats` to a dictionary
    stats_ = [iter_stats for (_, chain_stats) in results for iter_stats in chain_stats]
    stats = {
        name: np.squeeze(np.array([iter_stats[name] for iter_stats in stats_])).astype(dtype)
        for (name, dtype) in step.stats_dtypes[0].items()
    }

    return trace, stats


def init_nuts(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_ndim: int,
    init: str = "auto",
    random_seed: Union[None, int, List[int]] = None,
    **kwargs,
):
    """Set up the mass matrix initialization for NUTS.

    NUTS convergence and sampling speed is extremely dependent on the
    choice of mass/scaling matrix. This function implements different
    methods for choosing or adapting the mass matrix.

    Parameters
    ----------
    init: str
        Initialization method to use.
        * auto : Choose a default initialization method automatically.
          Currently, this is `'jitter+adapt_diag'`, but this can change in the
          future. If you depend on the exact behaviour, choose an initialization
          method explicitly.
        * adapt_diag : Start with a identity mass matrix and then adapt a diagonal based
          on the variance of the tuning samples.
        * jitter+adapt_diag : Same as `'adapt_diag'`, but use uniform jitter in [-1, 1]
          as starting point in each chain.
        * adapt_full: Same as `'adapt_diag'`, but adapts a dense mass matrix using the
          sample covariances.
        * jitter+adapt_full: Same as `'adapt_full'`, but use uniform jitter in [-1, 1]
          as starting point in each chain.
    **kwargs: keyword arguments
        Extra keyword arguments are forwarded to littlemcmc.NUTS.

    Returns
    -------
    start: np.array
        Starting point for sampler.
    nuts_sampler: NUTS
        Instantiated and initialized NUTS sampler object.
    """
    if not isinstance(init, str):
        raise TypeError("init must be a string.")

    if init is not None:
        init = init.lower()

    if init == "auto":
        init = "jitter+adapt_diag"

    _log.info("Initializing NUTS using {}...".format(init))

    if random_seed is not None:
        random_seed = int(np.atleast_1d(random_seed)[0])
        np.random.seed(random_seed)

    if init == "adapt_diag":
        start = np.zeros(model_ndim)
        mean = start
        var = np.ones(model_ndim)
        potential: QuadPotential = QuadPotentialDiagAdapt(model_ndim, mean, var, 10)
    elif init == "jitter+adapt_diag":
        start = 2 * np.random.rand(model_ndim) - 1
        mean = start
        var = np.ones(model_ndim)
        potential = QuadPotentialDiagAdapt(model_ndim, mean, var, 10)
    elif init == "adapt_full":
        start = np.zeros(model_ndim)
        mean = start
        cov = np.eye(model_ndim)
        potential = QuadPotentialFullAdapt(model_ndim, mean, cov, 10)
    elif init == "jitter+adapt_full":
        start = 2 * np.random.rand(model_ndim) - 1
        mean = start
        cov = np.eye(model_ndim)
        potential = QuadPotentialFullAdapt(model_ndim, mean, cov, 10)
    else:
        raise ValueError("Unknown initializer: {}.".format(init))

    step = NUTS(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim, potential=potential, **kwargs)

    return start, step
