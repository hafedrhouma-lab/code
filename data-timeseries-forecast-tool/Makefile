.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

PEX := pex
VIRTUALENV := python3 -m venv
PIP := venv/bin/pip

VENV := .venv
PYTHON_MODULES := src
PYLINT := pylint --disable=I0011 --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}"

# COLORS
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

## Show help
.PHONY: help
help: ## Show this help.
	@echo $(IMAGE)
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET} ${GREEN}<optional arguments>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "    ${YELLOW}%-25s${GREEN}%s${RESET}\n", $$1, $$2} \
		else if (/^# --\[ [^\]]+ \]-*$$/) {printf "  ${CYAN}%s${RESET}\n", substr($$1,5,75)} \
		}' $(MAKEFILE_LIST)
	@echo ''

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.DS_Store' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache

check-coding-style: ## check code style
	$(PYLINT) -E $(PYTHON_MODULES)

pylint-full: check-coding-style ## run pylint
	$(PYLINT) $(PYTHON_MODULES)

lint: ## check style with flake8
	flake8 src tests
	mypy src

test: ## run unitest
	pytest

docs: ## build the documenation
	mkdocs build

servedocs: docs ## compile the docs watching for changes
	mkdocs gh-deploy

tree: ## Show project structure
	tree -I 'venv|docs|site|test_*'

git:
	git add .
	git commit -m "$m"
	git push