"""HEAD probe tests for the /anthropic base route.

Uses httpx.ASGITransport to exercise full ASGI route dispatch (including
HEAD method registration for both path spellings) without the synchronous
TestClient portal, which deadlocks when the production router's lifespan
or dependency graph stalls the anyio blocking-portal event loop.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import router


def _head(app: FastAPI, path: str) -> httpx.Response:
    """Send a HEAD request through ASGI transport (no portal thread)."""

    async def _send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request("HEAD", path)

    return asyncio.run(_send())


@pytest.mark.parametrize("path", ["/anthropic", "/anthropic/"])
def test_anthropic_base_head_probe_accepts_both_path_spellings_without_redirect(
    path: str,
):
    app = FastAPI()
    app.include_router(router)

    response = _head(app, path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "location" not in response.headers
