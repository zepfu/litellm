from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import router


def test_anthropic_base_head_probe_accepts_both_path_spellings_without_redirect():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    for path in ("/anthropic", "/anthropic/"):
        response = client.head(path, follow_redirects=False)

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert "location" not in response.headers
