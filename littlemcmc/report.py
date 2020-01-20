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

"""Sampler warnings."""

from collections import namedtuple
import enum

SamplerWarning = namedtuple("SamplerWarning", "kind, message, level, step, exec_info, extra")


@enum.unique
class WarningType(enum.Enum):
    """Enumeration of sampler warnings."""

    # For HMC and NUTS
    DIVERGENCE = 1
    TUNING_DIVERGENCE = 2
    DIVERGENCES = 3
    TREEDEPTH = 4
    # Problematic sampler parameters
    BAD_PARAMS = 5
    # Indications that chains did not converge, eg Rhat
    CONVERGENCE = 6
    BAD_ACCEPTANCE = 7
    BAD_ENERGY = 8
