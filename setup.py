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

import codecs
import re
from pathlib import Path
from setuptools import setup, find_packages


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
README_FILE = PROJECT_ROOT / "README.md"
VERSION_FILE = PROJECT_ROOT / "littlemcmc" / "__init__.py"

NAME = "littlemcmc"
DESCRIPTION = "A lightweight and performant implementation of HMC and NUTS in Python, spun out of the PyMC project."
AUTHOR = "George Ho"
AUTHOR_EMAIL = "mail@eigenfoo.xyz"
URL = "https://github.com/eigenfoo/littlemcmc"
LICENSE = "Apache License, Version 2.0"

CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent",
]


def get_requirements(path):
    with codecs.open(path) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (VERSION_FILE,))


if __name__ == "__main__":
    setup(
        name=NAME,
        version=get_version(),
        description=DESCRIPTION,
        license=LICENSE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        install_requires=get_requirements(REQUIREMENTS_FILE),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        include_package_data=True,
    )
