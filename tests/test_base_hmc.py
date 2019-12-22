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
