#!/usr/bin/env bash
set -euo pipefail

python -m pip install \
  "semantic_router==0.1.15" \
  "colorama==0.4.6" \
  --no-deps
python -m pip install "aurelio-sdk==0.0.19"
python - <<'PY'
from importlib.metadata import version

import colorama
from semantic_router.routers import SemanticRouter
from semantic_router.routers.base import Route

assert version("semantic-router") == "0.1.15"
assert version("colorama") == "0.4.6"
PY
