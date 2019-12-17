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
        path_length=2.0,
    ):
        """Set up the Hamiltonian Monte Carlo sampler.

        Parameters
        ----------
        vars : list of theano variables
            FIXME: this can't be correct, right?
        path_length : float, default=2
            total length to travel
        step_rand : function float -> float, default=unif
            A function which takes the step size and returns an new one used to
            randomize the step size at each iteration.
        step_scale : float, default=0.25
            Initial size of steps to take, automatically scaled down
            by 1/n**(1/4).
        scaling : array_like, ndim = {1,2}
            The inverse mass, or precision matrix. One dimensional arrays are
            interpreted as diagonal matrices. If `is_cov` is set to True,
            this will be interpreded as the mass or covariance matrix.
        is_cov : bool, default=False
            Treat the scaling as mass or covariance matrix.
        potential : Potential, optional
            An object that represents the Hamiltonian with methods `velocity`,
            `energy`, and `random` methods. It can be specified instead
            of the scaling matrix.
        target_accept : float, default .8
            Adapt the step size such that the average acceptance probability
            across the trajectories are close to target_accept. Higher values
            for target_accept lead to smaller step sizes. Setting this to higher
            values like 0.9 or 0.99 can help with sampling from difficult
            posteriors. Valid values are between 0 and 1 (exclusive).
        gamma : float, default .05
        k : float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0 : int, default 10
            Parameter for dual averaging. Higher values slow initial
            adaptation.
        adapt_step_size : bool, default=True
            If True, performs dual averaging step size adaptation. If False,
            `k`, `t0`, `gamma` and `target_accept` are ignored.
        **kwargs : passed to BaseHMC
        """
        super(HamiltonianMC, self).__init__(
            scaling=scaling,
            step_scale=step_scale,
            is_cov=is_cov,
            logp_dlogp_func=logp_dlogp_func,
            size=size,
            potential=potential,
            integrator=integrator,
            dtype=dtype,
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
