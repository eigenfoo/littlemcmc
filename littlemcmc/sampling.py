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
import pickle
from typing import Callable, Tuple, Optional, Union, List
import logging
import numpy as np
from fastprogress.fastprogress import progress_bar

from . import parallel_sampling as ps
from .nuts import NUTS
from .hmc import HamiltonianMC
from .quadpotential import QuadPotential, QuadPotentialDiagAdapt, QuadPotentialFullAdapt
from .report import SamplerWarning


_log = logging.getLogger("littlemcmc")


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
    chain_idx: int = 0,
    callback=None,
    mp_ctx=None,
    pickle_backend: str = "pickle",
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
        assert cores is not None  # To make mypy happy
    if chains is None:
        chains = max(2, cores)
        assert chains is not None  # To make mypy happy

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

    if start is None:
        start = {}
    if isinstance(start, np.ndarray) or isinstance(start, dict):
        start = [start] * chains

    sample_args = {
        "logp_dlogp_func": logp_dlogp_func,
        "model_ndim": model_ndim,
        "draws": draws,
        "tune": tune,
        "step": step,
        "start": start,
        "chain": chain_idx,
        "chains": chains,
        "progressbar": progressbar,
        "random_seed": random_seed,
        "cores": cores,
        "callback": callback,
        "discard_tuned_samples": discard_tuned_samples,
    }
    parallel_args = {
        "pickle_backend": pickle_backend,
        "mp_ctx": mp_ctx,
    }

    parallel = cores > 1 and chains > 1
    if parallel:
        _log.info("Multiprocess sampling ({} chains in {} jobs)".format(chains, cores))
        try:
            traces, stats = _mp_sample(**sample_args, **parallel_args)  # type: ignore
        except pickle.PickleError:
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exec_info=True)
            parallel = False
        except AttributeError as e:
            if str(e).startswith("AttributeError: Can't pickle"):
                _log.warning("Could not pickle model, sampling singlethreaded.")
                _log.debug("Pickling error:", exec_info=True)
                parallel = False
            else:
                raise

    if not parallel:
        _log.info("Sequential sampling ({} chains in 1 job)".format(chains))
        traces, stats = _sample_many(**sample_args)  # type: ignore

    # Reshape `trace` to have shape [num_chains, num_samples, num_variables]
    trace = np.array([np.atleast_2d(chain_trace).T for chain_trace in traces])

    # Reshape `stats` to a dictionary with keys = string of sampling stat name,
    # values = np.array with shape [num_chains, num_samples, num_variables]
    stats = {
        name: np.array(
            [
                [np.atleast_1d(iter_stats[name]) for iter_stats in chain_stats]
                for chain_stats in stats
            ]
        ).astype(dtype)
        for (name, dtype) in step.stats_dtypes[0].items()
    }

    return trace, stats


def _mp_sample(
    model_ndim: int,
    draws: int,
    tune: int,
    step,
    chains: int,
    cores: int,
    chain: int,
    random_seed: list,
    start: list,
    progressbar=True,
    trace=None,
    callback=None,
    discard_tuned_samples=True,
    mp_ctx=None,
    pickle_backend="pickle",
    **kwargs,
):
    """
    Sample multiple chains in multiple processes.

    Main iteration for multiprocess sampling.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    step : function
        Step function
    chains : int
        The number of chains to sample.
    cores : int
        The number of chains to run in parallel.
    chain : int
        Number of the first chain.
    random_seed : list of ints
        Random seeds for each chain.
    start : list
        Starting points for each chain.
    progressbar : bool
        Whether or not to display a progress bar in the command line.
    trace : backend, list, MultiTrace or None
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    callback : Callable
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        A ``MultiTrace`` object that contains the samples for all chains.
    """
    trace = np.zeros([chains, model_ndim, tune + draws])
    stats: List[List[SamplerWarning]] = [[] for _ in range(chains)]

    sampler = ps.ParallelSampler(
        model_ndim,
        draws,
        tune,
        chains,
        cores,
        random_seed,
        start,
        step,
        chain,
        progressbar,
        mp_ctx=mp_ctx,
        pickle_backend=pickle_backend,
    )
    try:
        try:
            with sampler:
                for draw in sampler:
                    trace[draw.chain, :, draw.draw_idx] = draw.point
                    stats[draw.chain].append(draw.stats[0])
                    if callback is not None:
                        callback(trace=trace, draw=draw)

        except ps.ParallelSamplingError as error:
            # trace = traces[error._chain - chain]
            # trace._add_warnings(error._warnings)
            # for trace in traces:
            #     trace.close()

            # multitrace = MultiTrace(traces)
            # multitrace._report._log_summary()
            raise
        if discard_tuned_samples and trace is not None and stats is not None:
            # pylint: disable=W0631
            trace = trace[:, :, tune:]
            stats = [chain_stats[tune:] for chain_stats in stats]
        return trace, stats
    except KeyboardInterrupt:
        if discard_tuned_samples and trace is not None and stats is not None:
            # pylint: disable=W0631
            trace = trace[:, :, tune:]
            stats = [chain_stats[tune:] for chain_stats in stats]


def _sample_many(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_ndim: int,
    draws: int,
    tune: int,
    chain: int,
    chains: int,
    start: list,
    random_seed: list,
    step,
    discard_tuned_samples=True,
    callback=None,
    **kwargs,
):
    """
    Sample all chains sequentially.

    Parameters
    ----------
    draws: int
        The number of samples to draw
    chain: int
        Number of the first chain in the sequence.
    chains: int
        Total number of chains to sample.
    start: list
        Starting points for each chain
    random_seed: list
        A list of seeds, one for each chain
    step: function
        Step function

    Returns
    -------
    trace, stats
    """
    traces: List[np.ndarray] = []
    stats = []

    for i in range(chains):
        trace, stats_ = _sample(
            logp_dlogp_func=logp_dlogp_func,
            model_ndim=model_ndim,
            draws=draws,
            tune=tune,
            random_seed=random_seed[i],
            chain=chain + i,
            start=start[i],
            step=step,
            discard_tuned_samples=discard_tuned_samples,
            callback=callback,
            **kwargs,
        )
        if trace is None:
            if len(traces) == 0:
                raise ValueError("Sampling stopped before a sample was created.")
            else:
                break
        # TODO: bring this back for easy keyboard interrupts...?
        # elif len(trace) < tune + draws:
        #     if len(traces) == 0:
        #         traces.append(trace)
        #         stats.append(stats_)
        #     break
        else:
            traces.append(trace)
            stats.append(stats_)

    return traces, stats


def _sample(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_ndim: int,
    chain: int,
    progressbar: bool,
    random_seed,
    start,
    draws: int,
    step=None,
    trace=None,
    tune=None,
    discard_tuned_samples=True,
    callback=None,
    **kwargs,
):
    """
    Sample one chain in one process.

    Parameters
    ----------
    logp_dlogp_func : Python callable
    model_ndim : int
    chain : int
        Number of the chain that the samples will belong to.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    random_seed : int or list of ints
        A list is accepted if ``cores`` is greater than one.
    start : dict
        Starting point in parameter space (or partial point)
    draws : int
        The number of samples to draw
    step : function
        Step function
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    tune : int, optional
        Number of iterations to tune, if applicable (defaults to None)
    callback : Python callable, optional

    Returns
    -------
    trace, stats
    """
    skip_first = kwargs.get("skip_first", 0)

    sampling = _iter_sample(
        logp_dlogp_func, model_ndim, draws, tune, step, start, random_seed, callback
    )
    _pbar_data = {"chain": chain, "divergences": 0}
    _desc = "Sampling chain {chain:d}, {divergences:,d} divergences"
    if progressbar:
        sampling = progress_bar(sampling, total=tune + draws, display=progressbar)
        sampling.comment = _desc.format(**_pbar_data)

    trace = None
    stats = None
    try:
        for it, (trace, stats) in enumerate(sampling):
            if it >= skip_first:  # and diverging:
                # FIXME: surface num divergences from `stats`
                # _pbar_data["divergences"] += 1
                if progressbar:
                    sampling.comment = _desc.format(**_pbar_data)
    except KeyboardInterrupt:
        pass

    if discard_tuned_samples and trace is not None and stats is not None:
        # pylint: disable=W0631
        trace = trace[:, tune:]
        stats = stats[tune:]

    return trace, stats


def _iter_sample(
    logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model_ndim: int,
    draws: int,
    tune: int,
    step: Union[NUTS, HamiltonianMC],
    start: np.ndarray,
    random_seed: Union[None, int, List[int]] = None,
    callback=None,
):
    """
    Yield one chain in one process.

    Main iterator for singleprocess sampling.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    q = start
    trace = np.zeros([model_ndim, tune + draws])
    stats: List[SamplerWarning] = []

    step.tune = bool(tune)
    if hasattr(step, "reset_tuning"):
        step.reset_tuning()

    for i in range(tune + draws):
        if i == 0 and hasattr(step, "iter_count"):
            step.iter_count = 0
        if i == tune:
            step.stop_tuning()
        q, step_stats = step._astep(q)
        trace[:, i] = q
        stats.extend(step_stats)
        if callback is not None:
            warns = getattr(step, "warnings", None)
            # FIXME: implement callbacks
            # callback(
            #     trace=trace, draw=(chain, i == draws, i, i < tune, stats, point, warns),
            # )
        yield trace, stats


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

    step = NUTS(
        logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim, potential=potential, **kwargs
    )

    return start, step
