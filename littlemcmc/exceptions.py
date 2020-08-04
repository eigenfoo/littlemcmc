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

__all__ = [
    "SamplingError",
    "IncorrectArgumentsError",
    "TraceDirectoryError",
    "ImputationWarning",
    "ShapeError",
]


class SamplingError(RuntimeError):
    pass


class IncorrectArgumentsError(ValueError):
    pass


class TraceDirectoryError(ValueError):
    """Error from trying to load a trace from an incorrectly-structured directory,"""

    pass


class ImputationWarning(UserWarning):
    """Warning that there are missing values that will be imputed."""

    pass


class ShapeError(Exception):
    """Error that the shape of a variable is incorrect."""

    def __init__(self, message, actual=None, expected=None):
        if actual is not None and expected is not None:
            super().__init__("{} (actual {} != expected {})".format(message, actual, expected))
        elif actual is not None and expected is None:
            super().__init__("{} (actual {})".format(message, actual))
        elif actual is None and expected is not None:
            super().__init__("{} (expected {})".format(message, expected))
        else:
            super().__init__(message)


class DtypeError(TypeError):
    """Error that the dtype of a variable is incorrect."""

    def __init__(self, message, actual=None, expected=None):
        if actual is not None and expected is not None:
            super().__init__("{} (actual {} != expected {})".format(message, actual, expected))
        elif actual is not None and expected is None:
            super().__init__("{} (actual {})".format(message, actual))
        elif actual is None and expected is not None:
            super().__init__("{} (expected {})".format(message, expected))
        else:
            super().__init__(message)
