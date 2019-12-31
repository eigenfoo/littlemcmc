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
import scipy.sparse

from littlemcmc import quadpotential
import pytest
import numpy.testing as npt


def test_elemwise_posdef():
    scaling = np.array([0, 2, 3])
    with pytest.raises(quadpotential.PositiveDefiniteError):
        quadpotential.quad_potential(scaling, True)


"""
def test_elemwise_velocity():
    scaling = np.array([1, 2, 3])
    x = np.ones_like(scaling)
    pot = quadpotential.quad_potential(scaling, True)
    v = pot.velocity(x)
    npt.assert_allclose(v, scaling)
    assert v.dtype == pot.dtype
"""


def test_elemwise_energy():
    scaling = np.array([1, 2, 3])
    x = np.ones_like(scaling)
    pot = quadpotential.quad_potential(scaling, True)
    energy = pot.energy(x)
    npt.assert_allclose(energy, 0.5 * scaling.sum())


def test_equal_diag():
    np.random.seed(42)
    for _ in range(3):
        diag = np.random.rand(5)
        x = np.random.randn(5)
        pots = [
            quadpotential.quad_potential(diag, False),
            quadpotential.quad_potential(1.0 / diag, True),
            quadpotential.quad_potential(np.diag(diag), False),
            quadpotential.quad_potential(np.diag(1.0 / diag), True),
        ]

        v = np.diag(1.0 / diag).dot(x)
        e = x.dot(np.diag(1.0 / diag).dot(x)) / 2
        for pot in pots:
            v_ = pot.velocity(x)
            e_ = pot.energy(x)
            npt.assert_allclose(v_, v, rtol=1e-6)
            npt.assert_allclose(e_, e, rtol=1e-6)


def test_equal_dense():
    np.random.seed(42)
    for _ in range(3):
        cov = np.random.rand(5, 5)
        cov += cov.T
        cov += 10 * np.eye(5)
        inv = np.linalg.inv(cov)
        npt.assert_allclose(inv.dot(cov), np.eye(5), atol=1e-10)
        x = np.random.randn(5)
        pots = [
            quadpotential.quad_potential(cov, False),
            quadpotential.quad_potential(inv, True),
        ]

        v = np.linalg.solve(cov, x)
        e = 0.5 * x.dot(v)
        for pot in pots:
            v_ = pot.velocity(x)
            e_ = pot.energy(x)
            npt.assert_allclose(v_, v, rtol=1e-4)
            npt.assert_allclose(e_, e, rtol=1e-4)


def test_random_diag():
    d = np.arange(10) + 1
    np.random.seed(42)
    pots = [
        quadpotential.quad_potential(d, True),
        quadpotential.quad_potential(1.0 / d, False),
        quadpotential.quad_potential(np.diag(d), True),
        quadpotential.quad_potential(np.diag(1.0 / d), False),
    ]
    for pot in pots:
        vals = np.array([pot.random() for _ in range(1000)])
        npt.assert_allclose(vals.std(0), np.sqrt(1.0 / d), atol=0.1)


def test_random_dense():
    np.random.seed(42)
    for _ in range(3):
        cov = np.random.rand(5, 5)
        cov += cov.T
        cov += 10 * np.eye(5)
        inv = np.linalg.inv(cov)
        assert np.allclose(inv.dot(cov), np.eye(5))

        pots = [
            quadpotential.QuadPotentialFull(cov),
            quadpotential.QuadPotentialFullInv(inv),
        ]
        for pot in pots:
            cov_ = np.cov(np.array([pot.random() for _ in range(1000)]).T)
            assert np.allclose(cov_, inv, atol=0.1)


def test_weighted_covariance(ndim=10, seed=5432):
    np.random.seed(seed)

    L = np.random.randn(ndim, ndim)
    L[np.triu_indices_from(L, 1)] = 0.0
    L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
    cov = np.dot(L, L.T)
    mean = np.random.randn(ndim)

    samples = np.random.multivariate_normal(mean, cov, size=100)
    mu_est0 = np.mean(samples, axis=0)
    cov_est0 = np.cov(samples, rowvar=0)

    est = quadpotential._WeightedCovariance(ndim)
    for sample in samples:
        est.add_sample(sample, 1)
    mu_est = est.current_mean()
    cov_est = est.current_covariance()

    assert np.allclose(mu_est, mu_est0)
    assert np.allclose(cov_est, cov_est0)

    # Make sure that the weighted estimate also works
    est2 = quadpotential._WeightedCovariance(
        ndim,
        np.mean(samples[:10], axis=0),
        np.cov(samples[:10], rowvar=0, bias=True),
        10,
    )
    for sample in samples[10:]:
        est2.add_sample(sample, 1)
    mu_est2 = est2.current_mean()
    cov_est2 = est2.current_covariance()

    assert np.allclose(mu_est2, mu_est0)
    assert np.allclose(cov_est2, cov_est0)


def test_full_adapt_sample_p(seed=4566):
    # ref: https://github.com/stan-dev/stan/pull/2672
    np.random.seed(seed)
    m = np.array([[3.0, -2.0], [-2.0, 4.0]])
    m_inv = np.linalg.inv(m)

    var = np.array(
        [
            [2 * m[0, 0], m[1, 0] * m[1, 0] + m[1, 1] * m[0, 0]],
            [m[0, 1] * m[0, 1] + m[1, 1] * m[0, 0], 2 * m[1, 1]],
        ]
    )

    n_samples = 1000
    pot = quadpotential.QuadPotentialFullAdapt(2, np.zeros(2), m_inv, 1)
    samples = [pot.random() for n in range(n_samples)]
    sample_cov = np.cov(samples, rowvar=0)

    # Covariance matrix within 5 sigma of expected value
    # (comes from a Wishart distribution)
    assert np.all(np.abs(m - sample_cov) < 5 * np.sqrt(var / n_samples))


def test_full_adapt_update_window(seed=1123):
    np.random.seed(seed)
    init_cov = np.array([[1.0, 0.02], [0.02, 0.8]])
    pot = quadpotential.QuadPotentialFullAdapt(
        2, np.zeros(2), init_cov, 1, update_window=50
    )
    assert np.allclose(pot._cov, init_cov)
    for i in range(49):
        pot.update(np.random.randn(2), None, True)
    assert np.allclose(pot._cov, init_cov)
    pot.update(np.random.randn(2), None, True)
    assert not np.allclose(pot._cov, init_cov)


def test_full_adapt_adaptation_window(seed=8978):
    np.random.seed(seed)
    window = 10
    pot = quadpotential.QuadPotentialFullAdapt(
        2, np.zeros(2), np.eye(2), 1, adaptation_window=window
    )
    for i in range(window + 1):
        pot.update(np.random.randn(2), None, True)
    assert pot._previous_update == window
    assert pot._adaptation_window == window * pot._adaptation_window_multiplier

    pot = quadpotential.QuadPotentialFullAdapt(
        2, np.zeros(2), np.eye(2), 1, adaptation_window=window
    )
    for i in range(window + 1):
        pot.update(np.random.randn(2), None, True)
    assert pot._previous_update == window
    assert pot._adaptation_window == window * pot._adaptation_window_multiplier


def test_full_adapt_not_invertible():
    window = 10
    pot = quadpotential.QuadPotentialFullAdapt(
        2, np.zeros(2), np.eye(2), 0, adaptation_window=window
    )
    for i in range(window + 1):
        pot.update(np.ones(2), None, True)
    with pytest.raises(ValueError):
        pot.raise_ok(None)
