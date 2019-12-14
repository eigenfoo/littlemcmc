.PHONY: help venv conda docstyle format style types black test lint check
.DEFAULT_GOAL = help

PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

conda:  # Set up a conda environment for development.
	@printf "Creating conda environment...\n"
	${CONDA} create --yes --name env-littlemcmc python=3.6
	( \
	${CONDA} activate env-littlemcmc; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	${PIP} install -r requirements-dev.txt; \
	${CONDA} deactivate; \
	)
	@printf "\n\nConda environment created! \033[1;34mRun \`conda activate env-littlemcmc\` to activate it.\033[0m\n\n\n"

venv:  # Set up a Python virtual environment for development.
	@printf "Creating Python virtual environment...\n"
	rm -rf venv-littlemcmc/
	${PYTHON} -m venv venv-littlemcmc/
	( \
	source venv-littlemcmc/bin/activate; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	${PIP} install -r requirements-dev.txt; \
	deactivate; \
	)
	@printf "\n\nVirtual environment created! \033[1;34mRun \`source venv-littlemcmc/bin/activate\` to activate it.\033[0m\n\n\n"

docstyle:
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle littlemcmc/
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

format:
	@printf "Checking code style with black...\n"
	black --check --diff littlemcmc/ tests/
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style with pylint...\n"
	pylint littlemcmc/
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

types:
	@printf "Checking code type signatures with mypy...\n"
	python -m mypy --ignore-missing-imports littlemcmc/
	@printf "\033[1;34mMypy passes!\033[0m\n\n"

black:  # Format code in-place using black.
	black littlemcmc/ tests/

test:  # Test code using pytest.
	pytest -v littlemcmc tests --doctest-modules --html=testing-report.html --self-contained-html

lint: docstyle format style types  # Lint code using pydocstyle, black, pylint and mypy.

check: lint test  # Both lint and test code. Runs `make lint` followed by `make test`.

clean:  # Clean project directories.
	rm -rf dist/ site/ __pycache__/ testing-report.html
	find littlemcmc/ -type d -name "__pycache__" -exec rm -rf {} +
	find littlemcmc/ -type d -name "__pycache__" -delete
	find littlemcmc/ -type f -name "*.pyc" -delete
