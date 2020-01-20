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

"""Quadpotentials a.k.a. mass matrices or metrics."""

import numpy as np
from numpy.random import normal
import scipy
from scipy.sparse import issparse


__all__ = [
    "quad_potential",
    "QuadPotentialDiag",
    "QuadPotentialFull",
    "QuadPotentialFullInv",
    "QuadPotentialDiagAdapt",
    "QuadPotentialFullAdapt",
]


def quad_potential(C, is_cov):
    """
    Compute a QuadPotential object from a scaling matrix.

    Parameters
    ----------
    C : arraylike, 0 <= ndim <= 2
        scaling matrix for the potential
        vector treated as diagonal matrix.
    is_cov : Boolean
        whether C is provided as a covariance matrix or hessian

    Returns
    -------
    q : Quadpotential
    """
    if issparse(C):
        if is_cov:
            return QuadPotentialSparse(C)
        else:
            raise ValueError("Sparse precision matrices are not supported")

    partial_check_positive_definite(C)
    if C.ndim == 1:
        if is_cov:
            return QuadPotentialDiag(C)
        else:
            return QuadPotentialDiag(1.0 / C)
    else:
        if is_cov:
            return QuadPotentialFull(C)
        else:
            return QuadPotentialFullInv(C)


def partial_check_positive_definite(C):
    """Make a simple but partial check for Positive Definiteness."""
    if C.ndim == 1:
        d = C
    else:
        d = np.diag(C)
    (i,) = np.nonzero(np.logical_or(np.isnan(d), d <= 0))

    if len(i):
        raise PositiveDefiniteError("Simple check failed. Diagonal contains negatives", i)


class PositiveDefiniteError(ValueError):
    def __init__(self, msg, idx):
        super(PositiveDefiniteError, self).__init__(msg)
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return "Scaling is not positive definite: %s. Check indexes %s." % (self.msg, self.idx,)


class QuadPotential(object):
    """Base class for quadpotentials."""

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        raise NotImplementedError("Abstract method")

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        raise NotImplementedError("Abstract method")

    def random(self):
        """Draw random value from QuadPotential."""
        raise NotImplementedError("Abstract method")

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        raise NotImplementedError("Abstract method")

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning.

        This can be used by adaptive potentials to change the
        mass matrix.
        """
        pass

    def raise_ok(self, vmap=None):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Parameters
        ----------
        vmap : blocking.ArrayOrdering.vmap
            List of `VarMap`s, which are namedtuples with var, slc, shp, dtyp

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        return None

    def reset(self):
        pass


def isquadpotential(value):
    """Check whether an object might be a QuadPotential object."""
    return isinstance(value, QuadPotential)


class QuadPotentialDiagAdapt(QuadPotential):
    """Adapt a diagonal mass matrix from the sample variances."""

    def __init__(
        self,
        n,
        initial_mean,
        initial_diag=None,
        initial_weight=0,
        adaptation_window=101,
        adaptation_window_multiplier=1,
        dtype=None,
    ):
        """Set up a diagonal mass matrix."""
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError("Initial diagonal must be one-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError(
                "Wrong shape for initial_diag: expected %s got %s" % (n, len(initial_diag))
            )
        if len(initial_mean) != n:
            raise ValueError(
                "Wrong shape for initial_mean: expected %s got %s" % (n, len(initial_mean))
            )

        if dtype is None:
            dtype = "float32"

        if initial_diag is None:
            initial_diag = np.ones(n, dtype=dtype)
            initial_weight = 1
        else:
            initial_diag = initial_diag.astype(dtype)

        self.dtype = dtype
        self._n = n
        self._var = np.array(initial_diag, dtype=self.dtype, copy=True)
        self._stds = np.sqrt(initial_diag)
        self._inv_stds = 1.0 / self._stds
        self._foreground_var = _WeightedVariance(
            self._n, initial_mean, initial_diag, initial_weight, self.dtype
        )
        self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
        self._n_samples = 0
        self.adaptation_window = adaptation_window
        self.adaptation_window_multiplier = float(adaptation_window_multiplier)

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        return np.multiply(self._var, x, out=out)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * x.dot(velocity)
        return 0.5 * x.dot(self._var * x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    def random(self):
        """Draw random value from QuadPotential."""
        vals = normal(size=self._n).astype(self.dtype)
        return self._inv_stds * vals

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if not tune:
            return

        self._foreground_var.add_sample(sample, weight=1)
        self._background_var.add_sample(sample, weight=1)
        self._update_from_weightvar(self._foreground_var)

        if self._n_samples > 0 and self._n_samples % self.adaptation_window == 0:
            self._foreground_var = self._background_var
            self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
            self.adaptation_window = int(self.adaptation_window * self.adaptation_window_multiplier)

        self._n_samples += 1

    def raise_ok(self, vmap):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Parameters
        ----------
        vmap : blocking.ArrayOrdering.vmap
            List of `VarMap`s, which are namedtuples with var, slc, shp, dtyp

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        if np.any(self._stds == 0):
            name_slc = []
            tmp_hold = list(range(self._stds.size))
            for vmap_ in vmap:
                slclen = len(tmp_hold[vmap_.slc])
                for i in range(slclen):
                    name_slc.append((vmap_.var, i))
            index = np.where(self._stds == 0)[0]
            errmsg = ["Mass matrix contains zeros on the diagonal. "]
            for ii in index:
                errmsg.append(
                    "The derivative of RV `{}`.ravel()[{}]" " is zero.".format(*name_slc[ii])
                )
            raise ValueError("\n".join(errmsg))

        if np.any(~np.isfinite(self._stds)):
            name_slc = []
            tmp_hold = list(range(self._stds.size))
            for vmap_ in vmap:
                slclen = len(tmp_hold[vmap_.slc])
                for i in range(slclen):
                    name_slc.append((vmap_.var, i))
            index = np.where(~np.isfinite(self._stds))[0]
            errmsg = ["Mass matrix contains non-finite values on the diagonal. "]
            for ii in index:
                errmsg.append(
                    "The derivative of RV `{}`.ravel()[{}]" " is non-finite.".format(*name_slc[ii])
                )
            raise ValueError("\n".join(errmsg))


class _WeightedVariance(object):
    """Online algorithm for computing mean of variance."""

    def __init__(
        self, nelem, initial_mean=None, initial_variance=None, initial_weight=0, dtype="d",
    ):
        self._dtype = dtype
        self.w_sum = float(initial_weight)
        self.w_sum2 = float(initial_weight) ** 2
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype="d")
        else:
            self.mean = np.array(initial_mean, dtype="d", copy=True)
        if initial_variance is None:
            self.raw_var = np.zeros(nelem, dtype="d")
        else:
            self.raw_var = np.array(initial_variance, dtype="d", copy=True)

        self.raw_var[:] *= self.w_sum

        if self.raw_var.shape != (nelem,):
            raise ValueError("Invalid shape for initial variance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.w_sum += weight
        self.w_sum2 += weight * weight
        prop = weight / self.w_sum
        old_diff = x - self.mean
        self.mean[:] += prop * old_diff
        new_diff = x - self.mean
        self.raw_var[:] += weight * old_diff * new_diff

    def current_variance(self, out=None):
        if self.w_sum == 0:
            raise ValueError("Can not compute variance without samples.")
        if out is not None:
            return np.divide(self.raw_var, self.w_sum, out=out)
        else:
            return (self.raw_var / self.w_sum).astype(self._dtype)

    def current_mean(self):
        return self.mean.copy(dtype=self._dtype)


class QuadPotentialDiag(QuadPotential):
    """Quad potential using a diagonal covariance matrix."""

    def __init__(self, v, dtype=None):
        """Use a vector to represent a diagonal matrix for a covariance matrix.

        Parameters
        ----------
        v : vector, 0 <= ndim <= 1
           Diagonal of covariance matrix for the potential vector
        """
        if dtype is None:
            dtype = "float32"
        self.dtype = dtype
        v = v.astype(self.dtype)
        s = v ** 0.5

        self.s = s
        self.inv_s = 1.0 / s
        self.v = v

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        if out is not None:
            np.multiply(x, self.v, out=out)
            return
        return self.v * x

    def random(self):
        """Draw random value from QuadPotential."""
        return normal(size=self.s.shape) * self.inv_s

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * np.dot(x, velocity)
        return 0.5 * x.dot(self.v * x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        np.multiply(x, self.v, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadPotentialFullInv(QuadPotential):
    """QuadPotential object for Hamiltonian calculations using inverse of covariance matrix."""

    def __init__(self, A, dtype=None):
        """Compute the lower cholesky decomposition of the potential.

        Parameters
        ----------
        A : matrix, ndim = 2
           Inverse of covariance matrix for the potential vector
        """
        if dtype is None:
            dtype = "float32"
        self.dtype = dtype
        self.L = scipy.linalg.cholesky(A, lower=True)

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        vel = scipy.linalg.cho_solve((self.L, True), x)
        if out is None:
            return vel
        out[:] = vel

    def random(self):
        """Draw random value from QuadPotential."""
        n = normal(size=self.L.shape[0])
        return np.dot(self.L, n)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * x.dot(velocity)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadPotentialFull(QuadPotential):
    """Basic QuadPotential object for Hamiltonian calculations."""

    def __init__(self, cov, dtype=None):
        """Compute the lower cholesky decomposition of the potential.

        Parameters
        ----------
        A : matrix, ndim = 2
            scaling matrix for the potential vector
        """
        if dtype is None:
            dtype = "float32"
        self.dtype = dtype
        self._cov = np.array(cov, dtype=self.dtype, copy=True)
        self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        self._n = len(self._cov)

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        return np.dot(self._cov, x, out=out)

    def random(self):
        """Draw random value from QuadPotential."""
        vals = np.random.normal(size=self._n).astype(self.dtype)
        return scipy.linalg.solve_triangular(self._chol.T, vals, overwrite_b=True)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * np.dot(x, velocity)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return self.energy(x, v_out)

    __call__ = random


class QuadPotentialFullAdapt(QuadPotentialFull):
    """Adapt a dense mass matrix using the sample covariances."""

    def __init__(
        self,
        n,
        initial_mean,
        initial_cov=None,
        initial_weight=0,
        adaptation_window=101,
        adaptation_window_multiplier=2,
        update_window=1,
        dtype=None,
    ):
        if initial_cov is not None and initial_cov.ndim != 2:
            raise ValueError("Initial covariance must be two-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_cov is not None and initial_cov.shape != (n, n):
            raise ValueError(
                "Wrong shape for initial_cov: expected %s got %s" % (n, initial_cov.shape)
            )
        if len(initial_mean) != n:
            raise ValueError(
                "Wrong shape for initial_mean: expected %s got %s" % (n, len(initial_mean))
            )

        if dtype is None:
            dtype = "float32"

        if initial_cov is None:
            initial_cov = np.eye(n, dtype=dtype)
            initial_weight = 1

        self.dtype = dtype
        self._n = n
        self._cov = np.array(initial_cov, dtype=self.dtype, copy=True)
        self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        self._chol_error = None
        self._foreground_cov = _WeightedCovariance(
            self._n, initial_mean, initial_cov, initial_weight, self.dtype
        )
        self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)
        self._n_samples = 0

        self._adaptation_window = int(adaptation_window)
        self._adaptation_window_multiplier = float(adaptation_window_multiplier)
        self._update_window = int(update_window)
        self._previous_update = 0

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)
        try:
            self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        except (scipy.linalg.LinAlgError, ValueError) as error:
            self._chol_error = error

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if not tune:
            return

        # Steps since previous update
        delta = self._n_samples - self._previous_update

        self._foreground_cov.add_sample(sample, weight=1)
        self._background_cov.add_sample(sample, weight=1)

        # Update the covariance matrix and recompute the Cholesky factorization
        # every "update_window" steps
        if (delta + 1) % self._update_window == 0:
            self._update_from_weightvar(self._foreground_cov)

        # Reset the background covariance if we are at the end of the adaptation
        # window.
        if delta >= self._adaptation_window:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)

            self._previous_update = self._n_samples
            self._adaptation_window = int(
                self._adaptation_window * self._adaptation_window_multiplier
            )

        self._n_samples += 1

    def raise_ok(self, vmap):
        """Check if the mass matrix is ok, and raise ValueError if not."""
        if self._chol_error is not None:
            raise ValueError("{0}".format(self._chol_error))


class _WeightedCovariance:
    """Online algorithm for computing mean and covariance.

    This implements the `Welford's algorithm
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_ based
    on the implementation in `the Stan math library
    <https://github.com/stan-dev/math>`_.
    """

    def __init__(
        self, nelem, initial_mean=None, initial_covariance=None, initial_weight=0, dtype="d",
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype="d")
        else:
            self.mean = np.array(initial_mean, dtype="d", copy=True)
        if initial_covariance is None:
            self.raw_cov = np.eye(nelem, dtype="d")
        else:
            self.raw_cov = np.array(initial_covariance, dtype="d", copy=True)

        self.raw_cov[:] *= self.n_samples

        if self.raw_cov.shape != (nelem, nelem):
            raise ValueError("Invalid shape for initial covariance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_cov[:] += weight * new_diff[:, None] * old_diff[None, :]

    def current_covariance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute covariance without samples.")
        if out is not None:
            return np.divide(self.raw_cov, self.n_samples - 1, out=out)
        else:
            return (self.raw_cov / (self.n_samples - 1)).astype(self._dtype)

    def current_mean(self):
        return np.array(self.mean, dtype=self._dtype)
