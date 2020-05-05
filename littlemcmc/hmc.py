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

"""Hamiltonian Monte Carlo sampler."""

from typing import Callable, Tuple, Optional
import numpy as np

from .integration import IntegrationError
from .base_hmc import BaseHMC, HMCStepData, DivergenceInfo


__all__ = ["HamiltonianMC"]


class HamiltonianMC(BaseHMC):
    r"""A sampler for continuous variables based on Hamiltonian mechanics.

    See NUTS sampler for automatically tuned stopping time and step size
    scaling.
    """

    name = "hmc"
    generates_stats = True
    stats_dtypes = [
        {
            "step_size": np.float64,
            "n_steps": np.int64,
            "tune": np.bool,
            "step_size_bar": np.float64,
            "accept": np.float64,
            "diverging": np.bool,
            "energy_error": np.float64,
            "energy": np.float64,
            "path_length": np.float64,
            "accepted": np.bool,
            "model_logp": np.float64,
        }
    ]

    def __init__(
        self,
        logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        model_ndim: int,
        scaling: Optional[np.ndarray] = None,
        is_cov: bool = False,
        potential=None,
        target_accept: float = 0.8,
        Emax: float = 1000,
        adapt_step_size: bool = True,
        step_scale: float = 0.25,
        gamma: float = 0.05,
        k: float = 0.75,
        t0: int = 10,
        step_rand: Optional[Callable[[float], float]] = None,
        path_length: float = 2.0,
        max_steps: int = 1024,
    ):
        """Set up the Hamiltonian Monte Carlo sampler.

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
            interpreted as a matrix diagonal.
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
        path_length : float, default=2
            total length to travel
        max_steps : int, default=1024
            The maximum number of leapfrog steps.
        """
        super(HamiltonianMC, self).__init__(
            scaling=scaling,
            step_scale=step_scale,
            is_cov=is_cov,
            logp_dlogp_func=logp_dlogp_func,
            model_ndim=model_ndim,
            potential=potential,
            Emax=Emax,
            target_accept=target_accept,
            gamma=gamma,
            k=k,
            t0=t0,
            adapt_step_size=adapt_step_size,
            step_rand=step_rand,
        )
        self.path_length = path_length
        self.max_steps = max_steps

    def _hamiltonian_step(self, start: np.ndarray, p0: np.ndarray, step_size: float) -> HMCStepData:
        path_length = np.random.rand() * self.path_length
        n_steps = max(1, int(path_length / step_size))
        n_steps = min(self.max_steps, n_steps)

        energy_change = -np.inf
        state = start
        div_info = None
        try:
            for _ in range(n_steps):
                state = self.integrator.step(step_size, state)
        except IntegrationError as e:
            div_info = DivergenceInfo("Divergence encountered.", e, state)
        else:
            if not np.isfinite(state.energy):
                div_info = DivergenceInfo("Divergence encountered, bad energy.", None, state)
            energy_change = start.energy - state.energy
            if np.isnan(energy_change):
                energy_change = -np.inf
            if np.abs(energy_change) > self.Emax:
                div_info = DivergenceInfo(
                    "Divergence encountered, large integration error.", None, state
                )

        accept_stat = min(1, np.exp(energy_change))

        if div_info is not None or np.random.rand() >= accept_stat:
            end = start
            accepted = False
        else:
            end = state
            accepted = True

        stats = {
            "path_length": path_length,
            "n_steps": n_steps,
            "accept": accept_stat,
            "energy_error": energy_change,
            "energy": state.energy,
            "accepted": accepted,
            "model_logp": state.model_logp,
        }
        return HMCStepData(end, accept_stat, div_info, stats)
