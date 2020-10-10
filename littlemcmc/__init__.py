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

"""LittleMCMC."""

__version__ = "0.2.2"

from .sampling import sample, init_nuts
from .hmc import HamiltonianMC
from .nuts import NUTS
from .quadpotential import (
    quad_potential,
    QuadPotentialDiag,
    QuadPotentialFull,
    QuadPotentialFullInv,
    QuadPotentialDiagAdapt,
    QuadPotentialFullAdapt,
)

import multiprocessing as mp

ctx = mp.get_context("spawn")
