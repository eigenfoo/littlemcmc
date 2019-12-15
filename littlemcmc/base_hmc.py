"""Base class for Hamiltonian Monte Carlo samplers."""

from collections import namedtuple
import numpy as np

from . import integration, step_sizes
from .quadpotential import quad_potential, QuadPotentialDiagAdapt
from .report import SamplerWarning, WarningType

HMCStepData = namedtuple("HMCStepData", "end, accept_stat, divergence_info, stats")
DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state")


def metropolis_select(log_accept_rate, q, q0):
    """Perform rejection/acceptance step for Metropolis class samplers.

    Returns the new sample q if a uniform random number is less than the
    Metropolis acceptance rate (`mr`), and the old sample otherwise, along
    with a boolean indicating whether the sample was accepted.

    Parameters
    ----------
    log_accept_rate : float
        Log of Metropolis acceptance rate
    q : Proposed sample
    q0 : Current sample

    Returns
    -------
    q or q0, boolean
    """
    if np.isfinite(log_accept_rate) and np.log(np.random.uniform()) < log_accept_rate:
        return q, True
    else:
        return q0, False


class BaseHMC:
    """Superclass to implement Hamiltonian Monte Carlo."""

    def __init__(
        self,
        vars=None,
        scaling=None,
        step_scale=0.25,
        is_cov=False,
        logp_dlogp_func=None,
        size=None,
        potential=None,
        integrator="leapfrog",
        dtype=None,
        Emax=1000,
        target_accept=0.8,
        gamma=0.05,
        k=0.75,
        t0=10,
        adapt_step_size=True,
        step_rand=None,
    ):
        """Set up Hamiltonian samplers with common structures.

        Parameters
        ----------
        vars : list of Theano variables
            FIXME: this can't be correct, right?
        scaling : 1 or 2-dimensional array-like
            Scaling for momentum distribution. 1 dimensional arrays are
            interpreted as a matrix diagonal.
        step_scale : float, default=0.25
            Size of steps to take, automatically scaled down by 1 / (size ** 0.25)
        is_cov : bool, default=False
            Treat scaling as a covariance matrix/vector if True, else treat
            it as a precision matrix/vector
        logp_dlog_func : Python callable
            TODO: document this!
        size : tuple
            TODO: document this!
        potential: littlemcmc.quadpotential.Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        integrator
        dtype
        Emax
        target_accept
        gamma : float, default .05
        k : float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0 : int, default 10
            Parameter for dual averaging. Higher values slow initial adaptation.
        adapt_step_size : bool, default=True
            If True, performs dual averaging step size adaptation. If False,
            `k`, `t0`, `gamma` and `target_accept` are ignored.
        step_rand : Python callable
            Called on step size to randomize, immediately before adapting step
            size.
        """
        self._logp_dlogp_func = logp_dlogp_func
        self.adapt_step_size = adapt_step_size
        self.Emax = Emax
        self.iter_count = 0
        self.size = size
        self.step_size = step_scale / (size ** 0.25)
        self.target_accept = target_accept
        # FIXME: find a better name that step_adapt
        self.step_adapt = step_sizes.DualAverageAdaptation(
            self.step_size, target_accept, gamma, k, t0
        )
        self.integrator = integration.CpuLeapfrogIntegrator(
            self.potential, self._logp_dlogp_func
        )
        self.tune = True

        if scaling is None and potential is None:
            # Default to diagonal quadpotential
            mean = np.zeros(size)
            var = np.ones(size)
            potential = QuadPotentialDiagAdapt(size, mean, var, 10)

        if scaling is not None and potential is not None:
            raise ValueError("Cannot specify both `potential` and `scaling`.")
        elif potential is not None:
            self.potential = potential
        else:
            self.potential = quad_potential(scaling, is_cov)

        self._step_rand = step_rand
        self._warnings = []
        self._samples_after_tune = 0
        self._num_divs_sample = 0

    def step(self, array):
        """Perform a single HMC iteration.

        Generates sampler statistics if the sampler supports it.

        Parameters
        ----------
        array : array-like
            TODO: document this!
        """
        # FIXME where does generates_stats come from?
        if self.generates_stats:
            apoint, stats = self._astep(array)
            # point = self._logp_dlogp_func.array_to_full_dict(apoint)
            return apoint, stats
        else:
            apoint = self._astep(array)
            # point = self._logp_dlogp_func.array_to_full_dict(apoint)
            return apoint

    def stop_tuning(self):
        """Stop tuning."""
        if hasattr(self, "tune"):
            self.tune = False

    def _hamiltonian_step(self, start, p0, step_size):
        """Compute one Hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.

        Parameters
        ----------
        start
        p0
        step_size
            TODO: document these!
        """
        raise NotImplementedError("Abstract method")

    def _astep(self, q0):
        """Perform a single HMC iteration.

        Parameters
        ----------
        q0
            TODO: document these!
        """
        p0 = self.potential.random()
        start = self.integrator.compute_state(q0, p0)

        if not np.isfinite(start.energy):
            # self.potential.raise_ok(self._logp_dlogp_func._ordering.vmap)
            raise ValueError(
                "Bad initial energy: {}. The model might be misspecified.".format(
                    start.energy
                )
            )

        # Adapt step size
        adapt_step = self.tune and self.adapt_step_size
        step_size = self.step_adapt.current(adapt_step)
        self.step_size = step_size
        if self._step_rand is not None:
            step_size = self._step_rand(step_size)

        # Take the Hamiltonian step
        hmc_step = self._hamiltonian_step(start, p0, step_size)

        # Update step size and quadpotential
        self.step_adapt.update(hmc_step.accept_stat, adapt_step)
        self.potential.update(hmc_step.end.q, hmc_step.end.q_grad, self.tune)

        if hmc_step.divergence_info:
            info = hmc_step.divergence_info
            if self.tune:
                kind = WarningType.TUNING_DIVERGENCE
                point = None
            else:
                kind = WarningType.DIVERGENCE
                self._num_divs_sample += 1
                # We don't want to fill up all memory with divergence info
                if self._num_divs_sample < 100:
                    point = self._logp_dlogp_func.array_to_dict(info.state.q)
                else:
                    point = None
            warning = SamplerWarning(
                kind, info.message, "debug", self.iter_count, info.exec_info, point
            )

            self._warnings.append(warning)

        self.iter_count += 1
        if not self.tune:
            self._samples_after_tune += 1

        stats = {"tune": self.tune, "diverging": bool(hmc_step.divergence_info)}

        stats.update(hmc_step.stats)
        stats.update(self.step_adapt.stats())

        return hmc_step.end.q, [stats]

    def warnings(self):
        """Generate warnings from HMC sampler."""
        # list.copy() is only available in Python 3
        warnings = self._warnings.copy()

        # Generate a global warning for divergences
        message = ""
        n_divs = self._num_divs_sample
        if n_divs and self._samples_after_tune == n_divs:
            message = "The chain contains only diverging samples. The model is probably misspecified."
        elif n_divs == 1:
            message = (
                "There was 1 divergence after tuning. Increase "
                "`target_accept` or reparameterize."
            )
        elif n_divs > 1:
            message = (
                "There were %s divergences after tuning. Increase "
                "`target_accept` or reparameterize." % n_divs
            )

        if message:
            warning = SamplerWarning(
                WarningType.DIVERGENCES, message, "error", None, None, None
            )
            warnings.append(warning)

        warnings.extend(self.step_adapt.warnings())
        return warnings
