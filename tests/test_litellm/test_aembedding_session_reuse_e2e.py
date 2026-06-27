"""
Regression test for commit 819a6b5f18

Ensures shared_session is in all_litellm_params to prevent
"Object of type ClientSession is not JSON serializable" errors.
"""
import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.types.utils import all_litellm_params


def test_shared_session_in_all_litellm_params():
    """
    CRITICAL: shared_session must be in all_litellm_params.
    
    If missing, it gets passed to provider APIs causing JSON serialization errors.
    Regression test for commit 819a6b5f18.
    """
    assert "shared_session" in all_litellm_params


def test_openai_embedding_passes_shared_session():
    """
    Verify shared_session flows through the complete call chain.
    
    Full chain: litellm.embedding() -> OpenAI.embedding() -> _get_openai_client() 
                -> AsyncHTTPHandler -> _create_async_transport() -> _create_aiohttp_transport()
    """
    import litellm
    from litellm.llms.openai.openai import OpenAIChatCompletion
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    # Step 1: litellm.embedding() extracts and passes shared_session
    main_source = inspect.getsource(litellm.embedding)
    assert 'shared_session' in main_source
    
    # Step 2: OpenAI handlers pass it forward
    aembedding_source = inspect.getsource(OpenAIChatCompletion.aembedding)
    embedding_source = inspect.getsource(OpenAIChatCompletion.embedding)
    assert 'shared_session=shared_session' in aembedding_source
    assert 'shared_session=shared_session' in embedding_source
    
    # Step 3: _get_openai_client passes it to AsyncHTTPHandler
    client_source = inspect.getsource(OpenAIChatCompletion._get_openai_client)
    assert 'shared_session' in client_source
    
    # Step 4: AsyncHTTPHandler.create_client passes it to _create_async_transport
    create_client_source = inspect.getsource(AsyncHTTPHandler.create_client)
    assert 'shared_session=shared_session' in create_client_source
    
    # Step 5: _create_async_transport passes it to _create_aiohttp_transport
    async_transport_source = inspect.getsource(AsyncHTTPHandler._create_async_transport)
    assert 'shared_session=shared_session' in async_transport_source
    
    # Step 6: _create_aiohttp_transport uses it
    aiohttp_transport_source = inspect.getsource(AsyncHTTPHandler._create_aiohttp_transport)
    assert 'shared_session' in aiohttp_transport_source


def _extract_call_blocks(source: str, call: str):
    blocks = []
    start_index = 0
    while True:
        start = source.find(call, start_index)
        if start == -1:
            return blocks

        depth = 0
        for index in range(start, len(source)):
            char = source[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(source[start : index + 1])
                    start_index = index + 1
                    break
        else:
            raise AssertionError(f"Unclosed call block for {call!r}")


def test_base_llm_embedding_branches_pass_shared_session():
    """
    Proxy embedding routes inject shared_session before litellm.embedding().
    Every BaseLLMHTTPHandler embedding branch must forward it so local/OpenRouter
    style async embeddings reuse the proxy-owned aiohttp session.
    """
    import litellm

    main_source = inspect.getsource(litellm.embedding)
    call_blocks = _extract_call_blocks(
        main_source, "base_llm_http_handler.embedding("
    )
    assert call_blocks

    missing_blocks = [
        block
        for block in call_blocks
        if "shared_session=shared_session" not in block
    ]
    assert missing_blocks == []
