import numpy as np
import torch
import jax.config

jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import pymc3 as pm
import theano
import littlemcmc as lmc
import pytest

np.random.seed(42)

true_params = np.array([0.5, -2.3, -0.23])

N = 50
t = np.linspace(0, 10, 2)
x = np.random.uniform(0, 10, 50)
y = x * true_params[0] + true_params[1]
y_obs = y + np.exp(true_params[-1]) * np.random.randn(N)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.m = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.logs = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, x, y):
        mean = self.m * x + self.b
        loglike = -0.5 * torch.sum((y - mean) ** 2 * torch.exp(-2 * self.logs) + 2 * self.logs)
        return loglike


torch_model = torch.jit.script(LinearModel())
torch_params = [torch_model.m, torch_model.b, torch_model.logs]
args = [torch.tensor(x, dtype=torch.double), torch.tensor(y_obs, dtype=torch.double)]


def torch_logp_dlogp_func(x):
    for i, p in enumerate(torch_params):
        p.data = torch.tensor(x[i])
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    result = torch_model(*args)
    result.backward()

    return result.detach().numpy(), np.array([p.grad.numpy() for p in torch_params])


def jax_model(params):
    mean = params[0] * x + params[1]
    loglike = -0.5 * jnp.sum((y_obs - mean) ** 2 * jnp.exp(-2 * params[2]) + 2 * params[2])
    return loglike


@jax.jit
def jax_model_and_grad(x):
    return jax_model(x), jax.grad(jax_model)(x)


def jax_logp_dlogp_func(x):
    v, g = jax_model_and_grad(x)
    return np.asarray(v), np.asarray(g)


with pm.Model() as pm_model:
    pm_params = pm.Flat("pm_params", shape=3)
    mean = pm_params[0] * x + pm_params[1]
    pm.Normal("obs", mu=mean, sigma=pm.math.exp(pm_params[2]), observed=y_obs)


pm_model_and_grad = pm_model.fastfn([pm_model.logpt] + theano.grad(pm_model.logpt, pm_model.vars))


def pm_logp_dlogp_func(x):
    return pm_model_and_grad(pm_model.bijection.rmap(x))


@pytest.mark.parametrize(
    "framework", ["pytorch", "jax", "pymc3"],
)
def test_multiprocessing_with_various_frameworks(framework):
    logp_dlogp_funcs = {
        "pytorch": torch_logp_dlogp_func,
        "jax": jax_logp_dlogp_func,
        "pymc3": pm_logp_dlogp_func,
    }

    logp_dlogp_func = logp_dlogp_funcs[framework]

    model_ndim = 3
    tune = 10
    draws = 10
    chains = 4
    cores = 4

    trace, stats = lmc.sample(
        logp_dlogp_func=logp_dlogp_func,
        model_ndim=model_ndim,
        tune=tune,
        draws=draws,
        chains=chains,
        cores=cores,
        progressbar=False,
    )

    assert trace.shape == (chains, draws, model_ndim)
