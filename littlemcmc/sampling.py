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

"""Sampling driver functions (unrelated to PyMC3's `sampling.py`)."""

import numpy as np


def sample(logp_dlogp_func, size, stepper, draws, tune, init=None):
    """Sample."""
    if init is not None:
        q = init
    else:
        q = np.zeros(size)

    trace = np.zeros([size, tune + draws])
    stats = []

    for i in range(tune + draws):
        q, step_stats = stepper._astep(q)
        trace[:, i] = q
        stats.extend(step_stats)
        if i == tune:
            stepper.stop_tuning()

    return trace, stats
