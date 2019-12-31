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
import logging
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm, tqdm_notebook

_log = logging.getLogger("littlemcmc")


def _sample_one_chain(
    logp_dlogp_func,
    size,
    stepper,
    draws,
    tune,
    init=None,
    random_seed=None,
    discard_tuned_samples=True,
    progressbar=True,
    progressbar_position=None,
):
    """Sample one chain in one process."""
    if random_seed is not None:
        np.random.seed(random_seed)

    if init is not None:
        q = init
    else:
        q = np.zeros(size)

    if progressbar_position is None:
        progressbar_position = 0

    trace = np.zeros([size, tune + draws])
    stats = []

    if progressbar or progressbar == "console":
        iterator = tqdm(range(tune + draws), position=progressbar_position)
    elif progressbar == "notebook":
        iterator = tqdm_notebook(range(tune + draws), position=progressbar_position)
    else:
        iterator = range(tune + draws)

    for i in iterator:
        q, step_stats = stepper._astep(q)
        trace[:, i] = q
        stats.extend(step_stats)
        if i == tune:
            stepper.stop_tuning()

    if discard_tuned_samples:
        trace = trace[:, tune:]
        stats = stats[tune:]

    return trace, stats


def sample(
    logp_dlogp_func,
    size,
    stepper,
    draws,
    tune,
    chains=None,
    cores=None,
    init=None,
    random_seed=None,
    discard_tuned_samples=True,
):
    """Sample."""
    if cores is None:
        cores = min(4, os.cpu_count())
    if chains is None:
        chains = max(2, cores)

    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]
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
    results = Parallel(n_jobs=cores)(
        delayed(_sample_one_chain)(
            logp_dlogp_func=logp_dlogp_func,
            size=size,
            stepper=stepper,
            draws=draws,
            tune=tune,
            init=init,
            random_seed=seed,
            discard_tuned_samples=discard_tuned_samples,
            progressbar_position=i,
        )
        for i, seed in enumerate(random_seed)
    )

    trace = np.hstack([chain_trace for (chain_trace, _) in results])
    stats = [chain_stats for (_, chain_stats) in results]  # FIXME reshape stats

    return trace, stats
