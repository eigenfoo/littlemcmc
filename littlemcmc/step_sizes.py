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

"""Dual averaging step size adaptation."""

import numpy as np
from scipy import stats

from .report import SamplerWarning, WarningType


class DualAverageAdaptation(object):
    """Dual averaging step size adaptation."""

    def __init__(self, initial_step, target, gamma, k, t0):
        """Class for dual averaging step size adaptation.

        Parameters
        ----------
        initial_step
        target
        gamma : float, default .05
        k : float, default .75
            Parameter for dual averaging for step size adaptation. Values
            between 0.5 and 1 (exclusive) are admissible. Higher values
            correspond to slower adaptation.
        t0 : int, default 10
            Parameter for dual averaging. Higher values slow initial
            adaptation.
        """
        self._initial_step = initial_step
        self._target = target
        self._k = k
        self._t0 = t0
        self._gamma = gamma
        self.reset()

    def reset(self):
        """Reset step size adaptation routine."""
        self._log_step = np.log(self._initial_step)
        self._log_bar = self._log_step
        self._hbar = 0.0
        self._count = 1
        self._mu = np.log(10 * self._initial_step)
        self._tuned_stats = []

    def current(self, tune):
        """Get current step size.

        Parameters
        ----------
        tune : bool
            True during tuning, else False.
        """
        if tune:
            return np.exp(self._log_step)
        else:
            return np.exp(self._log_bar)

    def update(self, accept_stat, tune):
        """Update step size.

        Parameters
        ----------
        accept_stat : float
            HMC step acceptance statistic.
        tune : bool
            True during tuning, else False.
        """
        if not tune:
            self._tuned_stats.append(accept_stat)
            return

        count, k, t0 = self._count, self._k, self._t0
        w = 1.0 / (count + t0)
        self._hbar = (1 - w) * self._hbar + w * (self._target - accept_stat)

        self._log_step = self._mu - self._hbar * np.sqrt(count) / self._gamma
        mk = count ** -k
        self._log_bar = mk * self._log_step + (1 - mk) * self._log_bar
        self._count += 1

    def stats(self):
        """Get step size adaptation statistics."""
        return {
            "step_size": np.exp(self._log_step),
            "step_size_bar": np.exp(self._log_bar),
        }

    def warnings(self):
        """Generate warnings from dual averaging step size adaptation."""
        accept = np.array(self._tuned_stats)
        mean_accept = np.mean(accept)
        target_accept = self._target
        # Try to find a reasonable interval for acceptable acceptance
        # probabilities. Finding this was mostly trial and error.
        n_bound = min(100, len(accept))
        n_good, n_bad = mean_accept * n_bound, (1 - mean_accept) * n_bound
        lower, upper = stats.beta(n_good + 1, n_bad + 1).interval(0.95)
        if target_accept < lower or target_accept > upper:
            msg = (
                "The acceptance probability does not match the target. It "
                "is %s, but should be close to %s. Try to increase the "
                "number of tuning steps." % (mean_accept, target_accept)
            )
            info = {"target": target_accept, "actual": mean_accept}
            warning = SamplerWarning(WarningType.BAD_ACCEPTANCE, msg, "warn", None, None, info)
            return [warning]
        else:
            return []
