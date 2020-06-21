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

"""Utility mathematical functions."""

import numpy as np
import numpy.random as nr


def logbern(log_p: float) -> bool:
    """Perform a Bernoulli trial given a log probability."""
    if np.isnan(log_p):
        raise FloatingPointError("log_p can't be nan.")
    return np.log(nr.uniform()) < log_p


def log1mexp_numpy(x: float) -> float:
    """
    Compute log(1 - exp(-x)).

    This function is numerically more stable than the naive approach. For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    return np.where(x < 0.683, np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))


def logdiffexp_numpy(a: float, b: float) -> float:
    """Compute log(exp(a) - exp(b))."""
    return a + log1mexp_numpy(a - b)
