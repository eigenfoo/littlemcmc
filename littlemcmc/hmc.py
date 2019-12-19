#  Copyright 2019 George Ho
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

import numpy as np

from .integration import IntegrationError
from .base_hmc import BaseHMC, HMCStepData, DivergenceInfo


__all__ = ["HamiltonianMC"]


def unif(step_size, elow=0.85, ehigh=1.15):
    return np.random.uniform(elow, ehigh) * step_size


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
            "max_energy_error": np.float64,
            "path_length": np.float64,
            "accepted": np.bool,
            "model_logp": np.float64,
        }
    ]

    def __init__(
        self,
        logp_dlogp_func=None,
        size=None,
        scaling=None,
        is_cov=False,
        potential=None,
        target_accept=0.8,
        Emax=1000,
        adapt_step_size=True,
        step_scale=0.25,
        gamma=0.05,
        k=0.75,
        t0=10,
        step_rand=None,
        path_length=2.0,
    ):
        """Set up the Hamiltonian Monte Carlo sampler.

        Parameters
        ----------
        logp_dlogp_func : Python callable
            Python callable that returns the log-probability and derivative of
            the log-probability, respectively.
        size : int
            Total number of parameters. Dimensionality of the output of
            `logp_dlogp_func`.
        scaling : 1 or 2-dimensional array-like
            Scaling for momentum distribution. 1 dimensional arrays are
            interpreted as a matrix diagonal.
        is_cov : bool
            Treat scaling as a covariance matrix/vector if True, else treat
            it as a precision matrix/vector
        potential : littlemcmc.quadpotential.Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods.
        target_accept : float
        Emax : float
        adapt_step_size : bool, default=True
            If True, performs dual averaging step size adaptation. If False,
            `k`, `t0`, `gamma` and `target_accept` are ignored.
        step_scale : float
            Size of steps to take, automatically scaled down by 1 / (size ** 0.25)
        gamma : float, default .05
        k : float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0 : int, default 10
            Parameter for dual averaging. Higher values slow initial adaptation.
        step_rand : Python callable
            # FIXME rename this to callback or something
            Called on step size to randomize, immediately before adapting step
            size.
        path_length : float, default=2
            total length to travel
        """
        super(HamiltonianMC, self).__init__(
            scaling=scaling,
            step_scale=step_scale,
            is_cov=is_cov,
            logp_dlogp_func=logp_dlogp_func,
            size=size,
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

    def _hamiltonian_step(self, start, p0, step_size):
        path_length = np.random.rand() * self.path_length
        n_steps = max(1, int(path_length / step_size))

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
                div_info = DivergenceInfo(
                    "Divergence encountered, bad energy.", None, state
                )
            energy_change = start.energy - state.energy
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
