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

"""Leapfrog integrators."""

from collections import namedtuple
from typing import Callable, Tuple

import numpy as np
from scipy import linalg
from .quadpotential import QuadPotential


State = namedtuple("State", "q, p, v, q_grad, energy, model_logp")


class IntegrationError(RuntimeError):
    """Numerical errors during leapfrog integration."""

    pass


class CpuLeapfrogIntegrator(object):
    """Leapfrog integrator using the CPU."""

    def __init__(
        self,
        potential: QuadPotential,
        logp_dlogp_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Instantiate a CPU leapfrog integrator.

        Parameters
        ----------
        potential
        logp_dlogp_func
        """
        self._potential = potential
        self._logp_dlogp_func = logp_dlogp_func

    def compute_state(self, q: np.ndarray, p: np.ndarray) -> State:
        """Compute Hamiltonian functions using a position and momentum.

        Parameters
        ----------
        q
            Position.
        p
            Momentum
        """
        logp, dlogp = self._logp_dlogp_func(q)
        v = self._potential.velocity(p)
        kinetic = self._potential.energy(p, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, dlogp, energy, logp)

    def step(self, epsilon, state: State, out=None):
        """Leapfrog integrator step.

        Half a momentum update, full position update, half momentum update.

        Parameters
        ----------
        epsilon: float, > 0
            step scale
        state: State namedtuple,
            current position data
        out: (optional) State namedtuple,
            preallocated arrays to write to in place

        Returns
        -------
        None if `out` is provided, else a State namedtuple
        """
        try:
            return self._step(epsilon, state)
        except linalg.LinAlgError as err:
            msg = "LinAlgError during leapfrog step."
            raise IntegrationError(msg)
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                msg = "Infs or nans in scipy.linalg during leapfrog step."
                raise IntegrationError(msg)
            else:
                raise

    def _step(self, epsilon, state: State) -> State:
        """Perform one leapfrog step."""
        pot = self._potential
        q, p, v, q_grad, energy, logp = state

        dt = 0.5 * epsilon

        # Half momentum step
        p_new = p + dt * q_grad

        # Whole position step
        v_new = pot.velocity(p_new)
        q_new = (q + epsilon * v_new).astype(q.dtype)

        # Half momentum step
        logp, q_new_grad = self._logp_dlogp_func(q_new)
        p_new = p_new + dt * q_new_grad

        kinetic = pot.velocity_energy(p_new, v_new)
        energy = kinetic - logp

        return State(q_new, p_new, v_new, q_new_grad, energy, logp)
