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

import numpy as np
import scipy.stats
import littlemcmc as lmc


def logp_func(x, loc=0, scale=1):
    return np.log(scipy.stats.norm.pdf(x, loc=loc, scale=scale))


def dlogp_func(x, loc=0, scale=1):
    return -(x - loc) / scale


def logp_dlogp_func(x, loc=0, scale=1):
    return logp_func(x, loc=loc, scale=scale), dlogp_func(x, loc=loc, scale=scale)


def test_sampling():
    size = 1
    stepper = lmc.HamiltonianMC(logp_dlogp_func=logp_dlogp_func, size=size)
    draws = 1000
    tune = 1000
    init = None
    trace, stats = lmc.sample(logp_dlogp_func, size, stepper, draws, tune, init)

    assert np.allclose(np.mean(trace[:, 1000:]), 0, atol=0.05)
    assert np.allclose(np.std(trace[:, 1000:]), 1, atol=0.05)
