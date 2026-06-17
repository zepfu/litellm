#### Rerank Endpoints #####

import orjson
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


@router.post(
    "/v2/rerank",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rerank"],
)
@router.post(
    "/v1/rerank",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rerank"],
)
@router.post(
    "/rerank",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rerank"],
)
async def rerank(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    data = {}
    try:
        body = await request.body()
        data = orjson.loads(body)

        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        return await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="arerank",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            user_model=user_model,
            version=version,
        )
    except Exception as e:
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
