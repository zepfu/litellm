# ---------------------------------------------------------------------------
# AAWM venv artifact image
# Consumed via: COPY --from=ghcr.io/zepfu/litellm:<tag> /opt/litellm-venv /opt/litellm-venv
# Not a runnable image — no CMD, no entrypoint.
# ---------------------------------------------------------------------------

FROM python:3.13-slim AS wheel-builder
WORKDIR /src
COPY . .
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.13-slim AS venv-builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=wheel-builder /dist/*.whl /tmp/
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/litellm-venv \
    && /opt/litellm-venv/bin/pip install --no-cache-dir --no-compile \
       /tmp/*.whl -r /tmp/requirements.txt \
    && /opt/litellm-venv/bin/pip install --no-cache-dir PyJWT==2.9.0 \
    && find /opt/litellm-venv -type d -name __pycache__ -exec rm -rf {} + \
    && rm -rf /opt/litellm-venv/lib/python*/site-packages/pip

FROM scratch
COPY --from=venv-builder /opt/litellm-venv /opt/litellm-venv
