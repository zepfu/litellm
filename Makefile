# LiteLLM Makefile
# Simple Makefile for running tests and basic development tasks

.PHONY: help test test-unit test-unit-llms test-unit-proxy-guardrails test-unit-proxy-core test-unit-proxy-misc \
	test-unit-integrations test-unit-core-utils test-unit-other test-unit-root \
	test-proxy-unit-a test-proxy-unit-b test-integration test-unit-helm \
	info lint lint-dev format \
	install-dev install-proxy-dev install-test-deps \
	install-helm-unittest check-circular-imports check-import-safety

# Default target
help:
	@echo "Available commands:"
	@echo "  make install-dev        - Install development dependencies"
	@echo "  make install-proxy-dev  - Install proxy development dependencies"
	@echo "  make install-dev-ci     - Install dev dependencies (CI-compatible, pins OpenAI)"
	@echo "  make install-proxy-dev-ci - Install proxy dev dependencies (CI-compatible)"
	@echo "  make install-test-deps  - Install test dependencies"
	@echo "  make install-helm-unittest - Install helm unittest plugin"
	@echo "  make format             - Apply Black code formatting"
	@echo "  make format-check       - Check Black code formatting (matches CI)"
	@echo "  make lint               - Run all linting (Ruff, MyPy, Black check, circular imports, import safety)"
	@echo "  make lint-ruff          - Run Ruff linting only"
	@echo "  make lint-mypy          - Run MyPy type checking only"
	@echo "  make lint-black         - Check Black formatting (matches CI)"
	@echo "  make check-circular-imports - Check for circular imports"
	@echo "  make check-import-safety - Check import safety"
	@echo "  make test               - Run all tests"
	@echo "  make test-unit          - Run unit tests (tests/test_litellm)"
	@echo "  make test-unit-llms     - Run LLM provider tests (~225 files)"
	@echo "  make test-unit-proxy-guardrails - Run proxy guardrails+mgmt tests (~51 files)"
	@echo "  make test-unit-proxy-core - Run proxy auth+client+db+hooks tests (~52 files)"
	@echo "  make test-unit-proxy-misc - Run proxy misc tests (~77 files)"
	@echo "  make test-unit-integrations - Run integration tests (~60 files)"
	@echo "  make test-unit-core-utils - Run core utils tests (~32 files)"
	@echo "  make test-unit-other    - Run other tests (caching, responses, etc., ~69 files)"
	@echo "  make test-unit-root     - Run root-level tests (~34 files)"
	@echo "  make test-proxy-unit-a  - Run proxy_unit_tests (a-o, ~20 files)"
	@echo "  make test-proxy-unit-b  - Run proxy_unit_tests (p-z, ~28 files)"
	@echo "  make test-integration   - Run integration tests"
	@echo "  make test-unit-helm     - Run helm unit tests"

UV ?= $(shell command -v uv)
UV_CACHE_DIR ?= /tmp/uv-cache
UV_VENV := UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) venv
VENV_BIN := .venv/bin
PYTHON := .venv/bin/python
UV_PIP := UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) pip install --python $(PYTHON)
PYTEST := $(VENV_BIN)/pytest
RUFF := $(VENV_BIN)/ruff
BLACK := $(VENV_BIN)/black
MYPY := $(VENV_BIN)/mypy
PIP_RUNTIME := $(PYTHON) -m pip

DEV_PACKAGES := \
	"diff-cover>=9.0,<10" \
	"flake8>=6.1.0,<7" \
	"black>=23.12.0,<24" \
	"mypy>=1.0,<2" \
	"pytest>=7.4.3,<8" \
	"pytest-mock>=3.12.0,<4" \
	"pytest-asyncio>=0.21.1,<0.22" \
	"pytest-postgresql>=6.0.0,<7" \
	"pytest-xdist>=3.5.0,<4" \
	"requests-mock>=1.12.1,<2" \
	"responses>=0.25.7,<0.26" \
	"respx>=0.22.0,<0.23" \
	"ruff>=0.2.1,<0.3" \
	types-requests \
	types-setuptools \
	types-redis \
	types-PyYAML \
	"opentelemetry-api>=1.28.0,<2" \
	"opentelemetry-sdk>=1.28.0,<2" \
	"opentelemetry-exporter-otlp>=1.28.0,<2" \
	"langfuse>=2.45.0,<3" \
	"fastapi-offline>=1.7.3,<2" \
	"fakeredis>=2.27.1,<3" \
	"pytest-rerunfailures>=14.0,<15"

PROXY_DEV_PACKAGES := \
	prisma==0.11.0 \
	"hypercorn>=0.15.0,<0.16" \
	prometheus-client==0.20.0 \
	"opentelemetry-api>=1.28.0,<2" \
	"opentelemetry-sdk>=1.28.0,<2" \
	"opentelemetry-exporter-otlp>=1.28.0,<2" \
	"azure-identity>=1.15.0,<2" \
	"a2a-sdk>=0.3.22,<0.4"

# Show info
info:
	@echo "UV: $(UV)"
	@echo "PYTHON: $(PYTHON)"

ensure-venv:
	test -x $(PYTHON) || $(UV_VENV) .venv

# Installation targets
install-dev: ensure-venv
	$(UV_PIP) poetry-core wheel
	$(UV_PIP) -e .
	$(UV_PIP) $(DEV_PACKAGES)

install-proxy-dev: ensure-venv
	$(UV_PIP) poetry-core wheel
	$(UV_PIP) -e ".[proxy]"
	$(UV_PIP) $(DEV_PACKAGES) $(PROXY_DEV_PACKAGES)

# CI-compatible installations (matches GitHub workflows exactly)
install-dev-ci:
	$(MAKE) install-dev
	$(PIP_RUNTIME) install openai==2.8.0

install-proxy-dev-ci:
	$(MAKE) install-proxy-dev
	$(PIP_RUNTIME) install openai==2.8.0

install-test-deps: install-proxy-dev
	$(PIP_RUNTIME) install "pytest-retry==1.6.3"
	$(PIP_RUNTIME) install pytest-xdist
	$(PIP_RUNTIME) install openapi-core
	cd enterprise && $(PIP_RUNTIME) install -e . && cd ..

install-helm-unittest:
	helm plugin install https://github.com/helm-unittest/helm-unittest --version v0.4.4 || echo "ignore error if plugin exists"

# Formatting
format: install-dev
	cd litellm && ../$(BLACK) . && cd ..

format-check: install-dev
	cd litellm && ../$(BLACK) --check . && cd ..

# Linting targets
lint-ruff: install-dev
	cd litellm && ../$(RUFF) check . && cd ..

# faster linter for developing ...
# inspiration from:
# https://github.com/astral-sh/ruff/discussions/10977
# https://github.com/astral-sh/ruff/discussions/4049
lint-format-changed: install-dev
	@git diff origin/main --unified=0 --no-color -- '*.py' | \
	perl -ne '\
		if (/^diff --git a\/(.*) b\//) { $$file = $$1; } \
		if (/^@@ .* \+(\d+)(?:,(\d+))? @@/) { \
			$$start = $$1; $$count = $$2 || 1; $$end = $$start + $$count - 1; \
			print "$$file:$$start:1-$$end:999\n"; \
		}' | \
	while read range; do \
		file="$${range%%:*}"; \
		lines="$${range#*:}"; \
		echo "Formatting $$file (lines $$lines)"; \
		$(RUFF) format --range "$$lines" "$$file"; \
	done

lint-ruff-dev: install-dev
	@tmpfile=$$(mktemp /tmp/ruff-dev.XXXXXX) && \
	cd litellm && \
	(../$(RUFF) check . --output-format=pylint || true) > "$$tmpfile" && \
	../$(VENV_BIN)/diff-quality --violations=pylint "$$tmpfile" --compare-branch=origin/main && \
	cd .. ; \
	rm -f "$$tmpfile"

lint-ruff-FULL-dev: install-dev
	@files=$$(git diff --name-only origin/main -- '*.py'); \
	if [ -n "$$files" ]; then echo "$$files" | xargs $(RUFF) check; \
	else echo "No changed .py files to check."; fi

lint-mypy: install-dev
	$(PIP_RUNTIME) install types-requests types-setuptools types-redis types-PyYAML
	cd litellm && ../$(MYPY) . --ignore-missing-imports && cd ..

lint-black: format-check

check-circular-imports: install-dev
	cd litellm && ../$(PYTHON) ../tests/documentation_tests/test_circular_imports.py && cd ..

check-import-safety: install-dev
	@$(PYTHON) -c "from litellm import *; print('[from litellm import *] OK! no issues!');" || (echo '🚨 import failed, this means you introduced unprotected imports! 🚨'; exit 1)

# Combined linting (matches test-linting.yml workflow)
lint: format-check lint-ruff lint-mypy check-circular-imports check-import-safety

# Faster linting for local development (only checks changed code)
lint-dev: lint-format-changed lint-mypy check-circular-imports check-import-safety

# Testing targets
test:
	$(PYTEST) tests/

test-unit: install-test-deps
	$(PYTEST) tests/test_litellm -x -vv -n 4

# Matrix test targets (matching CI workflow groups)
test-unit-llms: install-test-deps
	$(PYTEST) tests/test_litellm/llms --tb=short -vv -n 4 --durations=20

test-unit-proxy-guardrails: install-test-deps
	$(PYTEST) tests/test_litellm/proxy/guardrails tests/test_litellm/proxy/management_endpoints tests/test_litellm/proxy/management_helpers --tb=short -vv -n 4 --durations=20

test-unit-proxy-core: install-test-deps
	$(PYTEST) tests/test_litellm/proxy/auth tests/test_litellm/proxy/client tests/test_litellm/proxy/db tests/test_litellm/proxy/hooks tests/test_litellm/proxy/policy_engine --tb=short -vv -n 4 --durations=20

test-unit-proxy-misc: install-test-deps
	$(PYTEST) tests/test_litellm/proxy/_experimental tests/test_litellm/proxy/agent_endpoints tests/test_litellm/proxy/anthropic_endpoints tests/test_litellm/proxy/common_utils tests/test_litellm/proxy/discovery_endpoints tests/test_litellm/proxy/experimental tests/test_litellm/proxy/google_endpoints tests/test_litellm/proxy/health_endpoints tests/test_litellm/proxy/image_endpoints tests/test_litellm/proxy/middleware tests/test_litellm/proxy/openai_files_endpoint tests/test_litellm/proxy/pass_through_endpoints tests/test_litellm/proxy/prompts tests/test_litellm/proxy/public_endpoints tests/test_litellm/proxy/response_api_endpoints tests/test_litellm/proxy/spend_tracking tests/test_litellm/proxy/ui_crud_endpoints tests/test_litellm/proxy/vector_store_endpoints tests/test_litellm/proxy/test_*.py --tb=short -vv -n 4 --durations=20

test-unit-integrations: install-test-deps
	$(PYTEST) tests/test_litellm/integrations --tb=short -vv -n 4 --durations=20

test-unit-core-utils: install-test-deps
	$(PYTEST) tests/test_litellm/litellm_core_utils --tb=short -vv -n 2 --durations=20

test-unit-other: install-test-deps
	$(PYTEST) tests/test_litellm/caching tests/test_litellm/responses tests/test_litellm/secret_managers tests/test_litellm/vector_stores tests/test_litellm/a2a_protocol tests/test_litellm/anthropic_interface tests/test_litellm/completion_extras tests/test_litellm/containers tests/test_litellm/enterprise tests/test_litellm/experimental_mcp_client tests/test_litellm/google_genai tests/test_litellm/images tests/test_litellm/interactions tests/test_litellm/passthrough tests/test_litellm/router_strategy tests/test_litellm/router_utils tests/test_litellm/types --tb=short -vv -n 4 --durations=20

test-unit-root: install-test-deps
	$(PYTEST) tests/test_litellm/test_*.py --tb=short -vv -n 4 --durations=20

# Proxy unit tests (tests/proxy_unit_tests split alphabetically)
test-proxy-unit-a: install-test-deps
	$(PYTEST) tests/proxy_unit_tests/test_[a-o]*.py --tb=short -vv -n 2 --durations=20

test-proxy-unit-b: install-test-deps
	$(PYTEST) tests/proxy_unit_tests/test_[p-z]*.py --tb=short -vv -n 2 --durations=20

test-integration:
	$(PYTEST) tests/ -k "not test_litellm"

test-unit-helm: install-helm-unittest
	helm unittest -f 'tests/*.yaml' deploy/charts/litellm-helm

# LLM Translation testing targets
test-llm-translation: install-test-deps
	@echo "Running LLM translation tests..."
	@python .github/workflows/run_llm_translation_tests.py

test-llm-translation-single: install-test-deps
	@echo "Running single LLM translation test file..."
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-llm-translation-single FILE=test_filename.py"; exit 1; fi
	@mkdir -p test-results
	$(PYTEST) tests/llm_translation/$(FILE) \
		--junitxml=test-results/junit.xml \
		-v --tb=short --maxfail=100 --timeout=300
