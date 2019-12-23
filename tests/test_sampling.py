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
from littlemcmc.sampling import _sample_one_chain


def logp_func(x, loc=0, scale=1):
    return np.log(scipy.stats.norm.pdf(x, loc=loc, scale=scale))


def dlogp_func(x, loc=0, scale=1):
    return -(x - loc) / scale


def logp_dlogp_func(x, loc=0, scale=1):
    return logp_func(x, loc=loc, scale=scale), dlogp_func(x, loc=loc, scale=scale)


def test_hmc_sampling_runs():
    size = 1
    stepper = lmc.HamiltonianMC(logp_dlogp_func=logp_dlogp_func, size=size)
    draws = 1
    tune = 1
    trace, stats = _sample_one_chain(logp_dlogp_func, size, stepper, draws, tune)


def test_nuts_sampling_runs():
    size = 1
    stepper = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, size=size)
    draws = 1
    tune = 1
    trace, stats = _sample_one_chain(logp_dlogp_func, size, stepper, draws, tune)


def test_hmc_recovers_1d_normal():
    size = 1
    stepper = lmc.HamiltonianMC(logp_dlogp_func=logp_dlogp_func, size=size)
    draws = 1000
    tune = 1000
    trace, stats = _sample_one_chain(logp_dlogp_func, size, stepper, draws, tune)

    assert np.allclose(np.mean(trace), 0, atol=1)
    assert np.allclose(np.std(trace), 1, atol=1)


def test_nuts_recovers_1d_normal():
    size = 1
    stepper = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, size=size)
    draws = 1000
    tune = 1000
    trace, stats = _sample_one_chain(logp_dlogp_func, size, stepper, draws, tune)

    assert np.allclose(np.mean(trace), 0, atol=1)
    assert np.allclose(np.std(trace), 1, atol=1)
