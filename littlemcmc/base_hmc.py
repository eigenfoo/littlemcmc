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

"""Base class for Hamiltonian Monte Carlo samplers."""

from collections import namedtuple
from typing import Callable, Tuple, List, Optional
import numpy as np

from . import integration, step_sizes
from .quadpotential import quad_potential, QuadPotential, QuadPotentialDiagAdapt
from .report import SamplerWarning, WarningType

HMCStepData = namedtuple("HMCStepData", "end, accept_stat, divergence_info, stats")
DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state")


class BaseHMC:
    """Superclass to implement Hamiltonian Monte Carlo."""

    def __init__(
        self,
        logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        model_ndim: int,
        scaling: Optional[np.ndarray],
        is_cov: bool,
        potential: QuadPotential,
        target_accept: float,
        Emax: float,
        adapt_step_size: bool,
        step_scale: float,
        gamma: float,
        k: float,
        t0: int,
        step_rand: Optional[Callable[[float], float]],
    ) -> None:
        """Set up Hamiltonian samplers with common structures.

        Parameters
        ----------
        logp_dlogp_func : Python callable
            Python callable that returns the log-probability and derivative of
            the log-probability, respectively.
        model_ndim : int
            Total number of parameters. Dimensionality of the output of
            ``logp_dlogp_func``.
        scaling : 1 or 2-dimensional array-like
            Scaling for momentum distribution. 1 dimensional arrays are
            interpreted as a matrix diagonal. Only one of ``scaling`` or
            ``potential`` may be non-None.
        is_cov : bool
            Treat scaling as a covariance matrix/vector if True, else treat
            it as a precision matrix/vector
        potential : littlemcmc.quadpotential.Potential, optional
            An object that represents the Hamiltonian with methods ``velocity``,
            ``energy``, and ``random`` methods. Only one of ``scaling`` or
            ``potential`` may be non-None.
        target_accept : float
            Adapt the step size such that the average acceptance probability
            across the trajectories are close to target_accept. Higher values
            for target_accept lead to smaller step sizes. Setting this to higher
            values like 0.9 or 0.99 can help with sampling from difficult
            posteriors. Valid values are between 0 and 1 (exclusive).
        Emax : float
            The maximum allowable change in the value of the Hamiltonian. Any
            trajectories that result in changes in the value of the Hamiltonian
            larger than ``Emax`` will be declared divergent.
        adapt_step_size : bool, default=True
            If True, performs dual averaging step size adaptation. If False,
            ``k``, ``t0``, ``gamma`` and ``target_accept`` are ignored.
        step_scale : float
            Size of steps to take, automatically scaled down by 1 / (``size`` **
            0.25).
        gamma : float, default .05
        k : float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0 : int, default 10
            Parameter for dual averaging. Higher values slow initial adaptation.
        step_rand : Python callable
            Callback for step size adaptation. Called on the step size at each
            iteration immediately before performing dual-averaging step size
            adaptation.
        """
        self._logp_dlogp_func = logp_dlogp_func
        self.adapt_step_size = adapt_step_size
        self.Emax = Emax
        self.iter_count = 0
        self.model_ndim = model_ndim
        self.step_size = step_scale / (model_ndim ** 0.25)
        self.target_accept = target_accept
        self.step_adapt = step_sizes.DualAverageAdaptation(
            self.step_size, target_accept, gamma, k, t0
        )
        self.tune = True

        if scaling is None and potential is None:
            # Default to diagonal quadpotential
            mean = np.zeros(model_ndim)
            var = np.ones(model_ndim)
            potential = QuadPotentialDiagAdapt(model_ndim, mean, var, 10)

        if scaling is not None and potential is not None:
            raise ValueError("Cannot specify both `potential` and `scaling`.")
        elif potential is not None:
            self.potential = potential
        else:
            self.potential = quad_potential(scaling, is_cov)

        self.integrator = integration.CpuLeapfrogIntegrator(self.potential, self._logp_dlogp_func)
        self._step_rand = step_rand
        self._warnings: List[SamplerWarning] = []
        self._samples_after_tune = 0
        self._num_divs_sample = 0

    def stop_tuning(self) -> None:
        """Stop tuning."""
        if hasattr(self, "tune"):
            self.tune = False

    def _hamiltonian_step(self, start: np.ndarray, p0: np.ndarray, step_size: float) -> HMCStepData:
        """Compute one Hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.
        """
        raise NotImplementedError("Abstract method")

    def _astep(self, q0: np.ndarray):
        """Perform a single HMC iteration."""
        p0 = self.potential.random()
        start = self.integrator.compute_state(q0, p0)

        if not np.isfinite(start.energy):
            raise ValueError(
                "Bad initial energy: {}. The model might be misspecified.".format(start.energy)
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
                # We don't want to fill up all memory, so do not return points
                # with divergence info
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

    def reset_tuning(self, start: np.ndarray = None) -> None:
        """Reset quadpotential and step size adaptation, and begin retuning."""
        self.step_adapt.reset()
        self.reset(start=None)

    def reset(self, start: np.ndarray = None) -> None:
        """Reset quadpotential and begin retuning."""
        self.tune = True
        self.potential.reset()

    def warnings(self) -> List[SamplerWarning]:
        """Generate warnings from HMC sampler."""
        # list.copy() is only available in Python 3
        warnings = self._warnings.copy()

        # Generate a global warning for divergences
        message = ""
        n_divs = self._num_divs_sample
        if n_divs and self._samples_after_tune == n_divs:
            message = (
                "The chain contains only diverging samples. The model is probably misspecified."
            )
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
            warning = SamplerWarning(WarningType.DIVERGENCES, message, "error", None, None, None)
            warnings.append(warning)

        warnings.extend(self.step_adapt.warnings())
        return warnings
