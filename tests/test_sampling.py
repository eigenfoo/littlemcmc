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

import numpy as np
import pytest
import littlemcmc as lmc
from test_utils import logp_dlogp_func


@pytest.mark.parametrize(
    "method", ["adapt_diag", "jitter+adapt_diag", "adapt_full", "jitter+adapt_full",],
)
def test_init_nuts(method):
    start, step = lmc.init_nuts(logp_dlogp_func=logp_dlogp_func, model_ndim=1, init=method)
    assert isinstance(start, np.ndarray)
    assert len(start) == 1
    assert isinstance(step, lmc.NUTS)


def test_hmc_sampling_runs():
    model_ndim = 1
    step = lmc.HamiltonianMC(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim)
    draws = 3
    tune = 1
    chains = 2
    cores = 1

    expected_shape = (2, 3, 1)

    trace, stats = lmc.sample(
        logp_dlogp_func, model_ndim, draws, tune, step=step, chains=chains, cores=cores
    )
    assert trace.shape == expected_shape
    assert all([stats[name].shape == expected_shape for (name, _) in step.stats_dtypes[0].items()])
    assert all(
        [
            stats[name].dtype == expected_dtype
            for (name, expected_dtype) in step.stats_dtypes[0].items()
        ]
    )


def test_nuts_sampling_runs():
    model_ndim = 1
    step = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim)
    draws = 3
    tune = 1
    chains = 2
    cores = 1

    expected_shape = (2, 3, 1)

    trace, stats = lmc.sample(
        logp_dlogp_func, model_ndim, draws, tune, step=step, chains=chains, cores=cores
    )
    assert trace.shape == (2, 3, 1)
    assert all([stats[name].shape == expected_shape for (name, _) in step.stats_dtypes[0].items()])
    assert all(
        [
            stats[name].dtype == expected_dtype
            for (name, expected_dtype) in step.stats_dtypes[0].items()
        ]
    )


def test_multiprocess_sampling_runs():
    model_ndim = 1
    step = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim)
    draws = 1
    tune = 1
    chains = None
    cores = None
    trace, stats = lmc.sample(
        logp_dlogp_func, model_ndim, draws, tune, step=step, chains=chains, cores=cores
    )


def test_hmc_recovers_1d_normal():
    model_ndim = 1
    step = lmc.HamiltonianMC(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim)
    draws = 1000
    tune = 1000
    chains = 1
    cores = 1
    trace, stats = lmc.sample(
        logp_dlogp_func, model_ndim, draws, tune, step=step, chains=chains, cores=cores
    )

    assert np.allclose(np.mean(trace), 0, atol=1)
    assert np.allclose(np.std(trace), 1, atol=1)


def test_nuts_recovers_1d_normal():
    model_ndim = 1
    step = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim)
    draws = 1000
    tune = 1000
    chains = 1
    cores = 1
    trace, stats = lmc.sample(
        logp_dlogp_func, model_ndim, draws, tune, step=step, chains=chains, cores=cores
    )

    assert np.allclose(np.mean(trace), 0, atol=1)
    assert np.allclose(np.std(trace), 1, atol=1)


def test_reset_tuning():
    model_ndim = 1
    draws = 2
    tune = 50
    chains = 2
    start, step = lmc.init_nuts(logp_dlogp_func=logp_dlogp_func, model_ndim=1, chains=chains)
    cores = 1
    lmc.sample(
        logp_dlogp_func, draws=draws, tune=tune, chains=chains, step=step, start=start, cores=cores
    )
    assert step.potential._n_samples == tune
    assert step.step_adapt._count == tune + 1
