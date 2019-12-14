import numpy as np
from littlemcmc.base_hmc import metropolis_select


def test_metropolis_select():
    q = "q"
    q0 = "q0"

    # Corresponds to acceptance rate of 1
    selected, accepted = metropolis_select(np.log(1), q, q0)
    assert selected == q
    assert accepted

    # Corresponds to acceptance rate of 0
    selected, accepted = metropolis_select(-np.inf, q, q0)
    assert selected == q0
    assert not accepted
