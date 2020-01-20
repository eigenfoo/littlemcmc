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
from .quadpotential import QuadPotentialDiagAdapt
from .report import SamplerWarning

_log = logging.getLogger("littlemcmc")


def _sample_one_chain(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    size: int,
    draws: int,
    tune: int,
    step=None,
    init: str = "auto",
    start: Optional[np.ndarray] = None,
    random_seed: Union[None, int, List[int]] = None,
    discard_tuned_samples: bool = True,
    progressbar: Union[bool, str] = True,
    progressbar_position: Optional[int] = None,
    **kwargs,
):
    """Sample one chain in one process."""
    if random_seed is not None:
        np.random.seed(random_seed)

    start_, step_ = init_nuts(
        logp_dlogp_func=logp_dlogp_func, size=size, init=init, random_seed=random_seed, **kwargs,
    )

    if start is not None:
        q = start
    else:
        q = start_

    if step is None:
        step = step_

    if progressbar_position is None:
        progressbar_position = 0

    trace = np.zeros([size, tune + draws])
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
    size: int,
    draws: int,
    tune: int,
    step=None,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    start: Optional[np.ndarray] = None,
    random_seed: Optional[Union[int, List[int]]] = None,
    discard_tuned_samples: bool = True,
    progressbar: Union[bool, str] = True,
):
    """Sample."""
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

    _log.info("Multiprocess sampling ({} chains in {} jobs)".format(chains, cores))
    results = Parallel(n_jobs=cores, backend="multiprocessing")(
        delayed(_sample_one_chain)(
            logp_dlogp_func=logp_dlogp_func,
            size=size,
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

    # Flatten `trace` to be 1-dimensional
    trace = np.reshape([chain_trace for (chain_trace, _) in results], [-1])

    # Reshape `stats` to a dictionary
    # TODO: we should target an ArviZ-like data structure...
    stats_ = [iter_stats for (_, chain_stats) in results for iter_stats in chain_stats]
    stats = {
        name: np.reshape(np.array([iter_stats[name] for iter_stats in stats_]), [-1]).astype(dtype)
        for (name, dtype) in step.stats_dtypes[0].items()
    }

    return trace, stats


def init_nuts(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    size: int,
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
        * adapt_diag : Start with a identity mass matrix and then adapt a
          diagonal based on the variance of the tuning samples. Uses the test
          value (usually the prior mean) as starting point.
        * jitter+adapt_diag : Same as `'adapt_diag'`, but use uniform jitter in
          [-1, 1] as starting point in each chain.
    **kwargs: keyword arguments
        Extra keyword arguments are forwarded to littlemcmc.NUTS.

    Returns
    -------
    start: np.array
        Starting point for sampler.
    nuts_sampler: pymc3.step_methods.NUTS
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
        start = np.zeros(size)
        mean = start
        var = np.ones(size)
        potential = QuadPotentialDiagAdapt(size, mean, var, 10)
    elif init == "jitter+adapt_diag":
        start = 2 * np.random.rand(size) - 1
        mean = start
        var = np.ones(size)
        potential = QuadPotentialDiagAdapt(size, mean, var, 10)
    else:
        raise ValueError("Unknown initializer: {}.".format(init))

    step = NUTS(logp_dlogp_func=logp_dlogp_func, size=size, potential=potential, **kwargs)

    return start, step
