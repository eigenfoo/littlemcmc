from typing import Dict, List, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any, Tuple
from typing import Iterable as TIterable
from collections.abc import Iterable
from collections import defaultdict
from copy import copy
import pickle
import logging
import warnings

import numpy as np
# TODO: add progress bar
# from fastprogress import progress_bar
from .nuts import NUTS
from .hmc import HamiltonianMC

import sys

sys.setrecursionlimit(10000)

__all__ = [
    "sample",
    "iter_sample",
    "sample_posterior_predictive",
    "sample_posterior_predictive_w",
    "init_nuts",
    "sample_prior_predictive",
    "sample_ppc",
    "sample_ppc_w",
]

STEP_METHODS = (
    NUTS,
    HamiltonianMC,
    # Metropolis,
    # BinaryMetropolis,
    # BinaryGibbsMetropolis,
    # Slice,
    # CategoricalGibbsMetropolis,
)


# _log = logging.getLogger("pymc3")


def instantiate_steppers(model, steps, selected_steps, step_kwargs=None):
    """Instantiates steppers assigned to the model variables.

    This function is intended to be called automatically from ``sample()``, but
    may be called manually.

    Parameters
    ----------
    model : Model object
        A fully-specified model object
    step : step function or vector of step functions
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to None (no assigned variables).
    selected_steps: dictionary of step methods and variables
        The step methods and the variables that have were assigned to them.
    step_kwargs : dict
        Parameters for the samplers. Keys are the lower case names of
        the step method, values a dict of arguments.

    Returns
    -------
    methods : list
        List of step methods associated with the model's variables.
    """
    if step_kwargs is None:
        step_kwargs = {}

    used_keys = set()
    for step_class, vars in selected_steps.items():
        if len(vars) == 0:
            continue
        args = step_kwargs.get(step_class.name, {})
        used_keys.add(step_class.name)
        step = step_class(vars=vars, **args)
        steps.append(step)

    unused_args = set(step_kwargs).difference(used_keys)
    if unused_args:
        raise ValueError("Unused step method arguments: %s" % unused_args)

    if len(steps) == 1:
        steps = steps[0]

    return steps


def assign_step_methods(model, step=None, methods=STEP_METHODS, step_kwargs=None):
    """Assign model variables to appropriate step methods.

    Passing a specified model will auto-assign its constituent stochastic
    variables to step methods based on the characteristics of the variables.
    This function is intended to be called automatically from ``sample()``, but
    may be called manually. Each step method passed should have a
    ``competence()`` method that returns an ordinal competence value
    corresponding to the variable passed to it. This value quantifies the
    appropriateness of the step method for sampling the variable.

    Parameters
    ----------
    model : Model object
        A fully-specified model object
    step : step function or vector of step functions
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to ``None`` (no assigned variables).
    methods : vector of step method classes
        The set of step methods from which the function may choose. Defaults
        to the main step methods provided by PyMC3.
    step_kwargs : dict
        Parameters for the samplers. Keys are the lower case names of
        the step method, values a dict of arguments.

    Returns
    -------
    methods : list
        List of step methods associated with the model's variables.
    """
    steps = []
    assigned_vars = set()

    if step is not None:
        try:
            steps += list(step)
        except TypeError:
            steps.append(step)
        for step in steps:
            try:
                assigned_vars = assigned_vars.union(set(step.vars))
            except AttributeError:
                for method in step.methods:
                    assigned_vars = assigned_vars.union(set(method.vars))

    # Use competence classmethods to select step methods for remaining
    # variables
    selected_steps = defaultdict(list)
    for var in model.free_RVs:
        if var not in assigned_vars:
            # determine if a gradient can be computed
            has_gradient = var.dtype not in discrete_types
            if has_gradient:
                try:
                    tg.grad(model.logpt, var)
                except (AttributeError, NotImplementedError, tg.NullTypeGradError):
                    has_gradient = False
            # select the best method
            selected = max(
                methods,
                key=lambda method, var=var, has_gradient=has_gradient: method._competence(
                    var, has_gradient
                ),
            )
            selected_steps[selected].append(var)

    return instantiate_steppers(model, steps, selected_steps, step_kwargs)


def _print_step_hierarchy(s, level=0):
    if isinstance(s, (list, tuple)):
        _log.info(">" * level + "list")
        for i in s:
            _print_step_hierarchy(i, level + 1)
    elif isinstance(s, CompoundStep):
        _log.info(">" * level + "CompoundStep")
        for i in s.methods:
            _print_step_hierarchy(i, level + 1)
    else:
        varnames = ", ".join(
            [
                get_untransformed_name(v.name)
                if is_transformed_name(v.name)
                else v.name
                for v in s.vars
            ]
        )
        _log.info(">" * level + "{}: [{}]".format(s.__class__.__name__, varnames))


def sample(
    draws=500,
    step=None,
    init="auto",
    n_init=200000,
    start=None,
    trace=None,
    chain_idx=0,
    chains=None,
    cores=None,
    tune=500,
    progressbar=True,
    model=None,
    random_seed=None,
    discard_tuned_samples=True,
    compute_convergence_checks=True,
    **kwargs,
):
    """Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 500. The number of tuned samples are discarded
        by default. See ``discard_tuned_samples``.
    init : str
        Initialization method to use for auto-assigned NUTS samplers.

        * auto : Choose a default initialization method automatically.
          Currently, this is ``'jitter+adapt_diag'``, but this can change in the future.
          If you depend on the exact behaviour, choose an initialization method explicitly.
        * adapt_diag : Start with a identity mass matrix and then adapt a diagonal based on the
          variance of the tuning samples. All chains use the test value (usually the prior mean)
          as starting point.
        * jitter+adapt_diag : Same as ``adapt_diag``\, but add uniform jitter in [-1, 1] to the
          starting point in each chain.
        * advi+adapt_diag : Run ADVI and then adapt the resulting diagonal mass matrix based on the
          sample variance of the tuning samples.
        * advi+adapt_diag_grad : Run ADVI and then adapt the resulting diagonal mass matrix based
          on the variance of the gradients during tuning. This is **experimental** and might be
          removed in a future release.
        * advi : Run ADVI to estimate posterior mean and diagonal mass matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map : Use the MAP as starting point. This is discouraged.
        * nuts : Run NUTS and estimate posterior mean and mass matrix from the trace.
    step : function or iterable of functions
        A step function or collection of functions. If there are variables without step methods,
        step methods for those variables will be assigned automatically.  By default the NUTS step
        method will be used, if appropriate to the model; this is a good default for beginning
        users.
    n_init : int
        Number of iterations of initializer. Only works for 'nuts' and 'ADVI'.
        If 'ADVI', number of iterations, if 'nuts', number of draws.
    start : dict, or array of dict
        Starting point in parameter space (or partial point)
        Defaults to ``trace.point(-1))`` if there is a trace provided and model.test_point if not
        (defaults to empty dict). Initialization methods for NUTS (see ``init`` keyword) can
        overwrite the default.
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
        Passing either "text" or "sqlite" is taken as a shortcut to set up the corresponding
        backend (with "mcmc" used as the base name).
    chain_idx : int
        Chain number used to store sample in backend. If ``chains`` is greater than one, chain
        numbers will start here.
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics and can also reveal multiple modes in the posterior. If ``None``,
        then set to either ``cores`` or 2, whichever is larger.
    cores : int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system, but at most 4.
    tune : int
        Number of iterations to tune, defaults to 500. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number specified in
        the ``draws`` argument, and will be discarded unless ``discard_tuned_samples`` is set to
        False.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
    model : Model (optional if in ``with`` context)
    random_seed : int or list of ints
        A list is accepted if ``cores`` is greater than one.
    discard_tuned_samples : bool
        Whether to discard posterior samples of the tune interval.
    compute_convergence_checks : bool, default=True
        Whether to compute sampler statistics like Gelman-Rubin and ``effective_n``.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        A ``MultiTrace`` object that contains the samples.

    Notes
    -----
    Optional keyword arguments can be passed to ``sample`` to be delivered to the
    ``step_method``s used during sampling. In particular, the NUTS step method accepts
    a number of arguments. Common options are:

        * target_accept: float in [0, 1]. The step size is tuned such that we approximate this
          acceptance rate. Higher values like 0.9 or 0.95 often work better for problematic
          posteriors.
        * max_treedepth: The maximum depth of the trajectory tree.
        * step_scale: float, default 0.25
          The initial guess for the step size scaled down by :math:`1/n**(1/4)`

    You can find a full list of arguments in the docstring of the step methods.

    Examples
    --------
    .. code:: ipython

        >>> import pymc3 as pm
        ... n = 100
        ... h = 61
        ... alpha = 2
        ... beta = 2

    .. code:: ipython

        >>> with pm.Model() as model: # context management
        ...     p = pm.Beta('p', alpha=alpha, beta=beta)
        ...     y = pm.Binomial('y', n=n, p=p, observed=h)
        ...     trace = pm.sample(2000, tune=1000, cores=4)
        >>> pm.summary(trace)
               mean        sd  mc_error   hpd_2.5  hpd_97.5
        p  0.604625  0.047086   0.00078  0.510498  0.694774
    """
    model = modelcontext(model)

    nuts_kwargs = kwargs.pop("nuts_kwargs", None)
    if nuts_kwargs is not None:
        warnings.warn(
            "The nuts_kwargs argument has been deprecated. Pass step "
            "method arguments directly to sample instead",
            DeprecationWarning,
        )
        kwargs.update(nuts_kwargs)
    step_kwargs = kwargs.pop("step_kwargs", None)
    if step_kwargs is not None:
        warnings.warn(
            "The step_kwargs argument has been deprecated. Pass step "
            "method arguments directly to sample instead",
            DeprecationWarning,
        )
        kwargs.update(step_kwargs)

    if cores is None:
        cores = min(4, _cpu_count())

    if "njobs" in kwargs:
        cores = kwargs["njobs"]
        warnings.warn(
            "The njobs argument has been deprecated. Use cores instead.",
            DeprecationWarning,
        )
    if "nchains" in kwargs:
        chains = kwargs["nchains"]
        warnings.warn(
            "The nchains argument has been deprecated. Use chains instead.",
            DeprecationWarning,
        )
    if chains is None:
        chains = max(2, cores)
    if isinstance(start, dict):
        start = [start] * chains
    if random_seed == -1:
        random_seed = None
    if chains == 1 and isinstance(random_seed, int):
        random_seed = [random_seed]
    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]
    if not isinstance(random_seed, Iterable):
        raise TypeError("Invalid value for `random_seed`. Must be tuple, list or int")
    if "chain" in kwargs:
        chain_idx = kwargs["chain"]
        warnings.warn(
            "The chain argument has been deprecated. Use chain_idx instead.",
            DeprecationWarning,
        )

    if start is not None:
        for start_vals in start:
            _check_start_shape(model, start_vals)

    # small trace warning
    if draws == 0:
        msg = "Tuning was enabled throughout the whole trace."
        _log.warning(msg)
    elif draws < 500:
        msg = "Only %s samples in chain." % draws
        _log.warning(msg)

    draws += tune

    if model.ndim == 0:
        raise ValueError("The model does not contain any free variables.")

    if step is None and init is not None and all_continuous(model.vars):
        try:
            # By default, try to use NUTS
            _log.info("Auto-assigning NUTS sampler...")
            start_, step = init_nuts(
                init=init,
                chains=chains,
                n_init=n_init,
                model=model,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )
            if start is None:
                start = start_
        except (AttributeError, NotImplementedError, tg.NullTypeGradError):
            # gradient computation failed
            _log.info(
                "Initializing NUTS failed. "
                "Falling back to elementwise auto-assignment."
            )
            _log.debug("Exception in init nuts", exec_info=True)
            step = assign_step_methods(model, step, step_kwargs=kwargs)
    else:
        step = assign_step_methods(model, step, step_kwargs=kwargs)

    if isinstance(step, list):
        step = CompoundStep(step)
    if start is None:
        start = {}
    if isinstance(start, dict):
        start = [start] * chains

    sample_args = {
        "draws": draws,
        "step": step,
        "start": start,
        "trace": trace,
        "chain": chain_idx,
        "chains": chains,
        "tune": tune,
        "progressbar": progressbar,
        "model": model,
        "random_seed": random_seed,
        "cores": cores,
    }

    sample_args.update(kwargs)

    has_population_samplers = np.any(
        [
            isinstance(m, arraystep.PopulationArrayStepShared)
            for m in (step.methods if isinstance(step, CompoundStep) else [step])
        ]
    )

    parallel = cores > 1 and chains > 1 and not has_population_samplers
    if parallel:
        _log.info("Multiprocess sampling ({} chains in {} jobs)".format(chains, cores))
        _print_step_hierarchy(step)
        try:
            trace = _mp_sample(**sample_args)
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
        if has_population_samplers:
            has_demcmc = np.any(
                [
                    isinstance(m, DEMetropolis)
                    for m in (
                        step.methods if isinstance(step, CompoundStep) else [step]
                    )
                ]
            )
            _log.info("Population sampling ({} chains)".format(chains))
            if has_demcmc and chains < 3:
                raise ValueError(
                    "DEMetropolis requires at least 3 chains. "
                    "For this {}-dimensional model you should use ≥{} chains".format(
                        model.ndim, model.ndim + 1
                    )
                )
            if has_demcmc and chains <= model.ndim:
                warnings.warn(
                    "DEMetropolis should be used with more chains than dimensions! "
                    "(The model has {} dimensions.)".format(model.ndim),
                    UserWarning,
                )
            _print_step_hierarchy(step)
            trace = _sample_population(**sample_args, parallelize=cores > 1)
        else:
            _log.info("Sequential sampling ({} chains in 1 job)".format(chains))
            _print_step_hierarchy(step)
            trace = _sample_many(**sample_args)

    discard = tune if discard_tuned_samples else 0
    trace = trace[discard:]

    if compute_convergence_checks:
        if draws - tune < 100:
            warnings.warn(
                "The number of samples is too small to check convergence reliably."
            )
        else:
            trace.report._run_convergence_checks(trace, model)

    trace.report._log_summary()

    return trace


def _check_start_shape(model, start):
    if not isinstance(start, dict):
        raise TypeError("start argument must be a dict or an array-like of dicts")
    e = ""
    for var in model.vars:
        if var.name in start.keys():
            var_shape = var.shape.tag.test_value
            start_var_shape = np.shape(start[var.name])
            if start_var_shape:
                if not np.array_equal(var_shape, start_var_shape):
                    e += "\nExpected shape {} for var '{}', got: {}".format(
                        tuple(var_shape), var.name, start_var_shape
                    )
            # if start var has no shape
            else:
                # if model var has a specified shape
                if var_shape.size > 0:
                    e += "\nExpected shape {} for var " "'{}', got scalar {}".format(
                        tuple(var_shape), var.name, start[var.name]
                    )

    if e != "":
        raise ValueError("Bad shape for start argument:{}".format(e))


def _sample_many(draws, chain, chains, start, random_seed, step, **kwargs):
    traces = []
    for i in range(chains):
        trace = _sample(
            draws=draws,
            chain=chain + i,
            start=start[i],
            step=step,
            random_seed=random_seed[i],
            **kwargs,
        )
        if trace is None:
            if len(traces) == 0:
                raise ValueError("Sampling stopped before a sample was created.")
            else:
                break
        elif len(trace) < draws:
            if len(traces) == 0:
                traces.append(trace)
            break
        else:
            traces.append(trace)
    return MultiTrace(traces)


def _sample_population(
    draws,
    chain,
    chains,
    start,
    random_seed,
    step,
    tune,
    model,
    progressbar=None,
    parallelize=False,
    **kwargs,
):
    # create the generator that iterates all chains in parallel
    chains = [chain + c for c in range(chains)]
    sampling = _prepare_iter_population(
        draws,
        chains,
        step,
        start,
        parallelize,
        tune=tune,
        model=model,
        random_seed=random_seed,
    )

    sampling = progress_bar(sampling, total=draws, display=progressbar)

    latest_traces = None
    for it, traces in enumerate(sampling):
        latest_traces = traces
    return MultiTrace(latest_traces)


def _sample(
    chain,
    progressbar,
    random_seed,
    start,
    draws=None,
    step=None,
    trace=None,
    tune=None,
    model=None,
    **kwargs,
):
    skip_first = kwargs.get("skip_first", 0)

    sampling = _iter_sample(draws, step, start, trace, chain, tune, model, random_seed)
    _pbar_data = None
    _pbar_data = {"chain": chain, "divergences": 0}
    _desc = "Sampling chain {chain:d}, {divergences:,d} divergences"
    sampling = progress_bar(sampling, total=draws, display=progressbar)
    sampling.comment = _desc.format(**_pbar_data)
    try:
        strace = None
        for it, (strace, diverging) in enumerate(sampling):
            if it >= skip_first:
                trace = MultiTrace([strace])
                if diverging and _pbar_data is not None:
                    _pbar_data["divergences"] += 1
                    sampling.comment = _desc.format(**_pbar_data)
    except KeyboardInterrupt:
        pass
    return strace


def iter_sample(
    draws,
    step,
    start=None,
    trace=None,
    chain=0,
    tune=None,
    model=None,
    random_seed=None,
):
    """Generator that returns a trace on each iteration using the given
    step method.  Multiple step methods supported via compound step
    method returns the amount of time taken.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function
    start : dict
        Starting point in parameter space (or partial point). Defaults to trace.point(-1)) if
        there is a trace provided and model.test_point if not (defaults to empty dict)
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track, or a MultiTrace object
        with past values. If a MultiTrace object is given, it must contain samples for the chain
        number ``chain``. If None or a list of variables, the NDArray backend is used.
    chain : int
        Chain number used to store sample in backend. If ``cores`` is greater than one, chain numbers
        will start here.
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in ``with`` context)
    random_seed : int or list of ints
        A list is accepted if more if ``cores`` is greater than one.

    Examples
    --------
    ::

        for trace in iter_sample(500, step):
            ...
    """
    sampling = _iter_sample(draws, step, start, trace, chain, tune, model, random_seed)
    for i, (strace, _) in enumerate(sampling):
        yield MultiTrace([strace[: i + 1]])


def _iter_sample(
    draws,
    step,
    start=None,
    trace=None,
    chain=0,
    tune=None,
    model=None,
    random_seed=None,
):
    model = modelcontext(model)
    draws = int(draws)
    if random_seed is not None:
        np.random.seed(random_seed)
    if draws < 1:
        raise ValueError("Argument `draws` must be greater than 0.")

    if start is None:
        start = {}

    strace = _choose_backend(trace, chain, model=model)

    if len(strace) > 0:
        update_start_vals(start, strace.point(-1), model)
    else:
        update_start_vals(start, model.test_point, model)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    if step.generates_stats and strace.supports_sampler_stats:
        strace.setup(draws, chain, step.stats_dtypes)
    else:
        strace.setup(draws, chain)

    try:
        step.tune = bool(tune)
        for i in range(draws):
            if i == 0 and hasattr(step, "iter_count"):
                step.iter_count = 0
            if i == tune:
                step = stop_tuning(step)
            if step.generates_stats:
                point, stats = step.step(point)
                if strace.supports_sampler_stats:
                    strace.record(point, stats)
                    diverging = i > tune and stats and stats[0].get("diverging")
                else:
                    strace.record(point)
            else:
                point = step.step(point)
                strace.record(point)
                diverging = False
            yield strace, diverging
    except KeyboardInterrupt:
        strace.close()
        if hasattr(step, "warnings"):
            warns = step.warnings()
            strace._add_warnings(warns)
        raise
    except BaseException:
        strace.close()
        raise
    else:
        strace.close()
        if hasattr(step, "warnings"):
            warns = step.warnings()
            strace._add_warnings(warns)


class PopulationStepper:
    def __init__(self, steppers, parallelize):
        """Tries to use multiprocessing to parallelize chains.

        Falls back to sequential evaluation if multiprocessing fails.

        In the multiprocessing mode of operation, a new process is started for each
        chain/stepper and Pipes are used to communicate with the main process.

        Parameters
        ----------
        steppers : list
            A collection of independent step methods, one for each chain.
        parallelize : bool
            Indicates if chain parallelization is desired
        """
        self.nchains = len(steppers)
        self.is_parallelized = False
        self._master_ends = []
        self._processes = []
        self._steppers = steppers
        if parallelize:
            try:
                # configure a child process for each stepper
                _log.info(
                    "Attempting to parallelize chains to all cores. You can turn this off with `pm.sample(cores=1)`."
                )
                import multiprocessing

                for c, stepper in enumerate(progress_bar(steppers)):
                    slave_end, master_end = multiprocessing.Pipe()
                    stepper_dumps = pickle.dumps(stepper, protocol=4)
                    process = multiprocessing.Process(
                        target=self.__class__._run_slave,
                        args=(c, stepper_dumps, slave_end),
                        name="ChainWalker{}".format(c),
                    )
                    # we want the child process to exit if the parent is terminated
                    process.daemon = True
                    # Starting the process might fail and takes time.
                    # By doing it in the constructor, the sampling progress bar
                    # will not be confused by the process start.
                    process.start()
                    self._master_ends.append(master_end)
                    self._processes.append(process)
                self.is_parallelized = True
            except Exception:
                _log.info(
                    "Population parallelization failed. "
                    "Falling back to sequential stepping of chains."
                )
                _log.debug("Error was: ", exec_info=True)
        else:
            _log.info(
                "Chains are not parallelized. You can enable this by passing "
                "`pm.sample(cores=n)`, where n > 1."
            )
        return super().__init__()

    def __enter__(self):
        """Does nothing because processes are already started in __init__."""
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self._processes) > 0:
            try:
                for master_end in self._master_ends:
                    master_end.send(None)
                for process in self._processes:
                    process.join(timeout=3)
            except Exception:
                _log.warning("Termination failed.")
        return

    @staticmethod
    def _run_slave(c, stepper_dumps, slave_end):
        """Started on a separate process to perform stepping of a chain.

        Parameters
        ----------
        c : int
            number of this chain
        stepper : BlockedStep
            a step method such as CompoundStep
        slave_end : multiprocessing.connection.PipeConnection
            This is our connection to the main process
        """
        # re-seed each child process to make them unique
        np.random.seed(None)
        try:
            stepper = pickle.loads(stepper_dumps)
            # the stepper is not necessarily a PopulationArraySharedStep itself,
            # but rather a CompoundStep. PopulationArrayStepShared.population
            # has to be updated, therefore we identify the substeppers first.
            population_steppers = []
            for sm in (
                stepper.methods if isinstance(stepper, CompoundStep) else [stepper]
            ):
                if isinstance(sm, arraystep.PopulationArrayStepShared):
                    population_steppers.append(sm)
            while True:
                incoming = slave_end.recv()
                # receiving a None is the signal to exit
                if incoming is None:
                    break
                tune_stop, population = incoming
                if tune_stop:
                    stop_tuning(stepper)
                # forward the population to the PopulationArrayStepShared objects
                # This is necessary because due to the process fork, the population
                # object is no longer shared between the steppers.
                for popstep in population_steppers:
                    popstep.population = population
                update = stepper.step(population[c])
                slave_end.send(update)
        except Exception:
            _log.exception("ChainWalker{}".format(c))
        return

    def step(self, tune_stop, population):
        """Steps the entire population of chains.

        Parameters
        ----------
        tune_stop : bool
            Indicates if the condition (i == tune) is fulfilled
        population : list
            Current Points of all chains

        Returns
        -------
        update : Point
            The new positions of the chains
        """
        updates = [None] * self.nchains
        if self.is_parallelized:
            for c in range(self.nchains):
                self._master_ends[c].send((tune_stop, population))
            # Blockingly get the step outcomes
            for c in range(self.nchains):
                updates[c] = self._master_ends[c].recv()
        else:
            for c in range(self.nchains):
                if tune_stop:
                    self._steppers[c] = stop_tuning(self._steppers[c])
                updates[c] = self._steppers[c].step(population[c])
        return updates


def _prepare_iter_population(
    draws, chains, step, start, parallelize, tune=None, model=None, random_seed=None
):
    """Prepares a PopulationStepper and traces for population sampling.

    Returns
    -------
    _iter_population : generator
        The generator the yields traces of all chains at the same time
    """
    # chains contains the chain numbers, but for indexing we need indices...
    nchains = len(chains)
    model = modelcontext(model)
    draws = int(draws)
    if random_seed is not None:
        np.random.seed(random_seed)
    if draws < 1:
        raise ValueError("Argument `draws` should be above 0.")

    # The initialization of traces, samplers and points must happen in the right order:
    # 1. traces are initialized and update_start_vals configures variable transforms
    # 2. population of points is created
    # 3. steppers are initialized and linked to the points object
    # 4. traces are configured to track the sampler stats
    # 5. a PopulationStepper is configured for parallelized stepping

    # 1. prepare a BaseTrace for each chain
    traces = [_choose_backend(None, chain, model=model) for chain in chains]
    for c, strace in enumerate(traces):
        # initialize the trace size and variable transforms
        if len(strace) > 0:
            update_start_vals(start[c], strace.point(-1), model)
        else:
            update_start_vals(start[c], model.test_point, model)

    # 2. create a population (points) that tracks each chain
    # it is updated as the chains are advanced
    population = [Point(start[c], model=model) for c in range(nchains)]

    # 3. Set up the steppers
    steppers = [None] * nchains
    for c in range(nchains):
        # need indepenent samplers for each chain
        # it is important to copy the actual steppers (but not the delta_logp)
        if isinstance(step, CompoundStep):
            chainstep = CompoundStep([copy(m) for m in step.methods])
        else:
            chainstep = copy(step)
        # link population samplers to the shared population state
        for sm in chainstep.methods if isinstance(step, CompoundStep) else [chainstep]:
            if isinstance(sm, arraystep.PopulationArrayStepShared):
                sm.link_population(population, c)
        steppers[c] = chainstep

    # 4. configure tracking of sampler stats
    for c in range(nchains):
        if steppers[c].generates_stats and traces[c].supports_sampler_stats:
            traces[c].setup(draws, c, steppers[c].stats_dtypes)
        else:
            traces[c].setup(draws, c)

    # 5. configure the PopulationStepper (expensive call)
    popstep = PopulationStepper(steppers, parallelize)

    # Because the preparations above are expensive, the actual iterator is
    # in another method. This way the progbar will not be disturbed.
    return _iter_population(draws, tune, popstep, steppers, traces, population)


def _iter_population(draws, tune, popstep, steppers, traces, points):
    """Generator that iterates a PopulationStepper.

    Parameters
    ----------
    draws : int
        number of draws per chain
    tune : int
        number of tuning steps
    popstep : PopulationStepper
        the helper object for (parallelized) stepping of chains
    steppers : list
        The step methods for each chain
    traces : list
        Traces for each chain
    points : list
        population of chain states
    """
    try:
        with popstep:
            # iterate draws of all chains
            for i in range(draws):
                updates = popstep.step(i == tune, points)

                # apply the update to the points and record to the traces
                for c, strace in enumerate(traces):
                    if steppers[c].generates_stats:
                        points[c], stats = updates[c]
                        if strace.supports_sampler_stats:
                            strace.record(points[c], stats)
                        else:
                            strace.record(points[c])
                    else:
                        points[c] = updates[c]
                        strace.record(points[c])
                # yield the state of all chains in parallel
                yield traces
    except KeyboardInterrupt:
        for c, strace in enumerate(traces):
            strace.close()
            if hasattr(steppers[c], "report"):
                steppers[c].report._finalize(strace)
        raise
    except BaseException:
        for c, strace in enumerate(traces):
            strace.close()
        raise
    else:
        for c, strace in enumerate(traces):
            strace.close()
            if hasattr(steppers[c], "report"):
                steppers[c].report._finalize(strace)


def _choose_backend(trace, chain, shortcuts=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]
    if trace is None:
        return NDArray(**kwds)

    if shortcuts is None:
        shortcuts = pm.backends._shortcuts

    try:
        backend = shortcuts[trace]["backend"]
        name = shortcuts[trace]["name"]
        return backend(name, **kwds)
    except TypeError:
        return NDArray(vars=trace, **kwds)
    except KeyError:
        raise ValueError("Argument `trace` is invalid.")


def _mp_sample(
    draws,
    tune,
    step,
    chains,
    cores,
    chain,
    random_seed,
    start,
    progressbar,
    trace=None,
    model=None,
    **kwargs,
):

    import pymc3.parallel_sampling as ps

    # We did draws += tune in pm.sample
    draws -= tune

    traces = []
    for idx in range(chain, chain + chains):
        if trace is not None:
            strace = _choose_backend(copy(trace), idx, model=model)
        else:
            strace = _choose_backend(None, idx, model=model)
        # for user supply start value, fill-in missing value if the supplied
        # dict does not contain all parameters
        update_start_vals(start[idx - chain], model.test_point, model)
        if step.generates_stats and strace.supports_sampler_stats:
            strace.setup(draws + tune, idx + chain, step.stats_dtypes)
        else:
            strace.setup(draws + tune, idx + chain)
        traces.append(strace)

    sampler = ps.ParallelSampler(
        draws, tune, chains, cores, random_seed, start, step, chain, progressbar
    )
    try:
        try:
            with sampler:
                for draw in sampler:
                    trace = traces[draw.chain - chain]
                    if trace.supports_sampler_stats and draw.stats is not None:
                        trace.record(draw.point, draw.stats)
                    else:
                        trace.record(draw.point)
                    if draw.is_last:
                        trace.close()
                        if draw.warnings is not None:
                            trace._add_warnings(draw.warnings)
        except ps.ParallelSamplingError as error:
            trace = traces[error._chain - chain]
            trace._add_warnings(error._warnings)
            for trace in traces:
                trace.close()

            multitrace = MultiTrace(traces)
            multitrace._report._log_summary()
            raise
        return MultiTrace(traces)
    except KeyboardInterrupt:
        traces, length = _choose_chains(traces, tune)
        return MultiTrace(traces)[:length]
    finally:
        for trace in traces:
            trace.close()


def _choose_chains(traces, tune):
    if tune is None:
        tune = 0

    if not traces:
        return []

    lengths = [max(0, len(trace) - tune) for trace in traces]
    if not sum(lengths):
        raise ValueError("Not enough samples to build a trace.")

    idxs = np.argsort(lengths)[::-1]
    l_sort = np.array(lengths)[idxs]

    final_length = l_sort[0]
    last_total = 0
    for i, length in enumerate(l_sort):
        total = (i + 1) * length
        if total < last_total:
            use_until = i
            break
        last_total = total
        final_length = length
    else:
        use_until = len(lengths)

    return [traces[idx] for idx in idxs[:use_until]], final_length + tune


def stop_tuning(step):
    """ stop tuning the current step method """

    step.stop_tuning()
    return step


class _DefaultTrace:
    """
    This class is a utility for collecting a number of samples
    into a dictionary. Name comes from its similarity to `defaultdict` --
    entries are lazily created.

    Parameters
    ----------
    samples : int
        The number of samples that will be collected, per variable,
        into the trace.

    Attributes
    ----------
    trace_dict : Dict[str, np.ndarray]
        A dictionary constituting a trace.  Should be extracted
        after a procedure has filled the `_DefaultTrace` using the
        `insert()` method
    """

    trace_dict = {}  # type: Dict[str, np.ndarray]
    _len = None  # type: int

    def __init__(self, samples):
        self._len = samples
        self.trace_dict = {}

    def insert(self, k: str, v, idx: int):
        """
        Insert `v` as the value of the `idx`th sample for the variable `k`.

        Parameters
        ----------
        k : str
            Name of the variable.
        v : anything that can go into a numpy array (including a numpy array)
            The value of the `idx`th sample from variable `k`
        ids : int
            The index of the sample we are inserting into the trace.
        """
        if hasattr(v, "shape"):
            value_shape = tuple(v.shape)  # type: Tuple[int, ...]
        else:
            value_shape = ()

        # initialize if necessary
        if k not in self.trace_dict:
            array_shape = (self._len,) + value_shape
            self.trace_dict[k] = np.empty(array_shape, dtype=np.array(v).dtype)

        # do the actual insertion
        if value_shape == ():
            self.trace_dict[k][idx] = v
        else:
            self.trace_dict[k][idx, :] = v


def init_nuts(
    init="auto",
    chains=1,
    n_init=500000,
    model=None,
    random_seed=None,
    progressbar=True,
    **kwargs,
):
    """Set up the mass matrix initialization for NUTS.

    NUTS convergence and sampling speed is extremely dependent on the
    choice of mass/scaling matrix. This function implements different
    methods for choosing or adapting the mass matrix.

    Parameters
    ----------
    init : str
        Initialization method to use.

        * auto : Choose a default initialization method automatically.
          Currently, this is `'jitter+adapt_diag'`, but this can change in the future. If you
          depend on the exact behaviour, choose an initialization method explicitly.
        * adapt_diag : Start with a identity mass matrix and then adapt a diagonal based on the
          variance of the tuning samples. All chains use the test value (usually the prior mean)
          as starting point.
        * jitter+adapt_diag : Same as ``adapt_diag``, but use uniform jitter in [-1, 1] as starting
          point in each chain.
        * advi+adapt_diag : Run ADVI and then adapt the resulting diagonal mass matrix based on the
          sample variance of the tuning samples.
        * advi+adapt_diag_grad : Run ADVI and then adapt the resulting diagonal mass matrix based
          on the variance of the gradients during tuning. This is **experimental** and might be
          removed in a future release.
        * advi : Run ADVI to estimate posterior mean and diagonal mass matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map : Use the MAP as starting point. This is discouraged.
        * nuts : Run NUTS and estimate posterior mean and mass matrix from
          the trace.
    chains : int
        Number of jobs to start.
    n_init : int
        Number of iterations of initializer
        If 'ADVI', number of iterations, if 'nuts', number of draws.
    model : Model (optional if in ``with`` context)
    progressbar : bool
        Whether or not to display a progressbar for advi sampling.
    **kwargs : keyword arguments
        Extra keyword arguments are forwarded to pymc3.NUTS.

    Returns
    -------
    start : pymc3.model.Point
        Starting point for sampler
    nuts_sampler : pymc3.step_methods.NUTS
        Instantiated and initialized NUTS sampler object
    """
    model = modelcontext(model)

    vars = kwargs.get("vars", model.vars)
    if set(vars) != set(model.vars):
        raise ValueError("Must use init_nuts on all variables of a model.")
    if not all_continuous(vars):
        raise ValueError(
            "init_nuts can only be used for models with only " "continuous variables."
        )

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

    cb = [
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="absolute"),
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff="relative"),
    ]

    if init == "adapt_diag":
        start = [model.test_point] * chains
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(model.ndim, mean, var, 10)
    elif init == "jitter+adapt_diag":
        start = []
        for _ in range(chains):
            mean = {var: val.copy() for var, val in model.test_point.items()}
            for val in mean.values():
                val[...] += 2 * np.random.rand(*val.shape) - 1
            start.append(mean)
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(model.ndim, mean, var, 10)
    elif init == "advi+adapt_diag_grad":
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )  # type: pm.MeanField
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        mean = approx.bij.rmap(approx.mean.get_value())
        mean = model.dict_to_array(mean)
        weight = 50
        potential = quadpotential.QuadPotentialDiagAdaptGrad(
            model.ndim, mean, cov, weight
        )
    elif init == "advi+adapt_diag":
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )  # type: pm.MeanField
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        mean = approx.bij.rmap(approx.mean.get_value())
        mean = model.dict_to_array(mean)
        weight = 50
        potential = quadpotential.QuadPotentialDiagAdapt(model.ndim, mean, cov, weight)
    elif init == "advi":
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init,
            method="advi",
            model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )  # type: pm.MeanField
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == "advi_map":
        start = pm.find_MAP(include_transformed=True)
        approx = pm.MeanField(model=model, start=start)
        pm.fit(
            random_seed=random_seed,
            n=n_init,
            method=pm.KLqp(approx),
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == "map":
        start = pm.find_MAP(include_transformed=True)
        cov = pm.find_hessian(point=start)
        start = [start] * chains
        potential = quadpotential.QuadPotentialFull(cov)
    elif init == "nuts":
        init_trace = pm.sample(
            draws=n_init, step=pm.NUTS(), tune=n_init // 2, random_seed=random_seed
        )
        cov = np.atleast_1d(pm.trace_cov(init_trace))
        start = list(np.random.choice(init_trace, chains))
        potential = quadpotential.QuadPotentialFull(cov)
    else:
        raise ValueError("Unknown initializer: {}.".format(init))

    step = pm.NUTS(potential=potential, model=model, **kwargs)

    return start, step
