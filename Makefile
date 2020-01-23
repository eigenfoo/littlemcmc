.DEFAULT_GOAL = help

PYTHON := python3
PIP := pip3
CONDA := conda
SHELL := bash

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

.PHONY: help
help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

.PHONY: conda
conda:  # Set up a conda environment for development.
	@printf "Creating conda environment...\n"
	${CONDA} create --yes --name env-littlemcmc python=3.6
	${CONDA} activate env-littlemcmc
	${PIP} install -U pip
	${PIP} install -r requirements.txt
	${PIP} install -r requirements-dev.txt
	${CONDA} deactivate
	@printf "\n\nConda environment created! \033[1;34mRun \`conda activate env-littlemcmc\` to activate it.\033[0m\n\n\n"

.PHONY: venv
venv:  # Set up a Python virtual environment for development.
	@printf "Creating Python virtual environment...\n"
	rm -rf venv/
	${PYTHON} -m venv venv/
	source venv/bin/activate
	${PIP} install -U pip
	${PIP} install -r requirements.txt
	${PIP} install -r requirements-dev.txt
	deactivate
	@printf "\n\nVirtual environment created! \033[1;34mRun \`source venv/bin/activate\` to activate it.\033[0m\n\n\n"

.PHONY: blackstyle
blackstyle:
	@printf "Checking code style with black...\n"
	black --check --diff littlemcmc/ tests/ docs/
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

.PHONY: pylintstyle
pylintstyle:
	@printf "Checking code style with pylint...\n"
	pylint littlemcmc/ tests/
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

.PHONY: pydocstyle
pydocstyle:
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle --convention=numpy littlemcmc/
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

.PHONY: mypytypes
mypytypes:
	@printf "Checking code type signatures with mypy...\n"
	python -m mypy --ignore-missing-imports littlemcmc/
	@printf "\033[1;34mMypy passes!\033[0m\n\n"

.PHONY: black
black:  # Format code in-place using black.
	black littlemcmc/ tests/ docs/

.PHONY: test
test:  # Test code using pytest.
	pytest -v littlemcmc tests --doctest-modules --html=testing-report.html --self-contained-html --cov=./ --cov-report=xml

.PHONY: lint
lint: blackstyle pylintstyle pydocstyle mypytypes  # Lint code using black, pylint, pydocstyle and mypy.

.PHONY: check
check: lint test  # Both lint and test code. Runs `make lint` followed by `make test`.

.PHONY: clean
clean:  # Clean project directories.
	rm -rf dist/ site/ littlemcmc.egg-info/ pip-wheel-metadata/ __pycache__/ testing-report.html coverage.xml
	find littlemcmc/ tests/ -type d -name "__pycache__" -exec rm -rf {} +
	find littlemcmc/ tests/ -type d -name "__pycache__" -delete
	find littlemcmc/ tests/ -type f -name "*.pyc" -delete
	${MAKE} -C docs/ clean

.PHONY: package
package: clean  # Package glaze in preparation for releasing to PyPI.
	${PYTHON} setup.py sdist bdist_wheel
	twine check dist/*
	@printf "\n\n\033[1;34mTo upload to Test PyPI (recommended!), run:\033[0m\n\n"
	@printf "\t\033[1;34mpython3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*\033[0m\n\n"
	@printf "\033[1;34mTo upload to PyPI, run:\033[0m\n\n"
	@printf "\t\033[1;34mpython3 -m twine upload dist/*\033[0m\n\n"
	@printf "\033[1;34mYou will need PyPI credentials.\033[0m\n\n"
