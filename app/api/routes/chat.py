"""LLM 转发路由：POST /v1/chat/completions。

流程：鉴权 → 取并发槽位 → 转发上游 → 提取 usage 估算成本 → 落库 → 返回响应体。

落库失败只记日志、不改变已返回的响应：审计数据丢失不应让用户白等一次推理。
客户端取消时记录仍会在 finally 中写入，避免统计漏账。
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.concurrency import ConcurrencyLimiter, ConcurrencyTimeout
from app.config import Settings
from app.cost import estimate_cost_for_settings
from app.db.repository import CallRecordRepository, NewCallRecord
from app.dependencies import (
    get_app_settings,
    get_db_session,
    get_limiter,
    get_upstream_client,
)
from app.errors import ErrorType, GatewayError
from app.logging_config import get_logger
from app.request_context import new_request_id, set_request_id
from app.schemas import ChatCompletionRequest
from app.security import require_client_api_key, summarize_prompt
from app.upstream import ChatCompletionsClient

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat/completions",
    summary="OpenAI 兼容的 Chat Completions（非流式）",
    response_model=None,
)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
    upstream: ChatCompletionsClient = Depends(get_upstream_client),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
    key_fingerprint: str = Depends(require_client_api_key),
) -> JSONResponse:
    """转发一次非流式 Chat Completions 请求。"""
    request_id = new_request_id()
    set_request_id(request_id)
    model = payload.model or settings.upstream_model

    prompt_meta = summarize_prompt([m.model_dump() for m in payload.messages])
    client_ip = request.client.host if request.client else None

    started_at = time.perf_counter()

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_type = ErrorType.NONE
    http_status = 200

    try:
        try:
            async with limiter.slot():
                upstream_result = await upstream.create_chat_completion(
                    payload.upstream_payload(default_model=settings.upstream_model)
                )
        except ConcurrencyTimeout as exc:
            raise GatewayError(
                str(exc),
                error_type=ErrorType.CONCURRENCY_TIMEOUT,
                status_code=503,
            ) from exc

        prompt_tokens = upstream_result.prompt_tokens
        completion_tokens = upstream_result.completion_tokens
        http_status = upstream_result.status_code

        # 上游可能返回请求之外的模型名（别名），以实际使用的模型入库
        response_model = upstream_result.payload.get("model")
        if isinstance(response_model, str) and response_model:
            model = response_model

        return JSONResponse(
            status_code=http_status,
            content=upstream_result.payload,
            headers={"X-Request-Id": request_id},
        )

    except GatewayError as exc:
        error_type = exc.error_type
        http_status = exc.status_code
        raise

    except asyncio.CancelledError:
        # 客户端断开：先留一条记录再抛出，否则这次调用不会计入统计
        error_type = ErrorType.CLIENT_CANCELLED
        http_status = 499  # Nginx 约定的「客户端关闭连接」
        logger.info("客户端取消请求 | request_id=%s", request_id)
        raise

    except Exception as exc:  # noqa: BLE001
        error_type = ErrorType.INTERNAL_ERROR
        http_status = 500
        logger.exception("未预期异常 | request_id=%s | %s", request_id, exc)
        raise

    finally:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        await _persist_call_record(
            session=session,
            record=NewCallRecord(
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                http_status=http_status,
                error_type=error_type.value,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimate_cost_for_settings(
                    settings,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                key_fingerprint=key_fingerprint,
                message_count=prompt_meta["message_count"],
                prompt_chars=prompt_meta["prompt_chars"],
                client_ip=client_ip,
            ),
            request_id=request_id,
        )


async def _persist_call_record(
    *,
    session: AsyncSession,
    record: NewCallRecord,
    request_id: str,
) -> None:
    """写入调用记录。

    数据库异常在这里被吞掉：记录失败不应改变已经返回给客户端的响应。
    代价是审计可能缺失，因此用 error 级日志暴露出来。
    """
    try:
        repository = CallRecordRepository(session)
        await repository.insert(record)
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "调用记录写入失败（不影响本次响应）| request_id=%s | %s",
            request_id,
            exc,
        )
        try:
            await session.rollback()
        except Exception:  # noqa: BLE001
            logger.error("回滚失败 | request_id=%s", request_id)
