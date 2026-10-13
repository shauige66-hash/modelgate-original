"""LLM 转发路由：POST /v1/chat/completions。

流程：鉴权 → 解析模型到服务商 → 取并发槽位 → 转发上游
→ 提取 usage 估算成本 → 落库 → 返回响应体。

模型必须在 model_registry 中登记，未登记直接返回 400；
上游地址与凭证只来自配置，不受请求内容影响。

落库失败只记日志、不改变已返回的响应：审计数据丢失不应让用户白等一次推理。
客户端取消时记录仍会在 finally 中写入，避免统计漏账。
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_tracking import parse_agent_tags
from app.budget import BudgetGuard
from app.concurrency import ConcurrencyLimiter, ConcurrencyTimeout
from app.config import Settings
from app.cost import calculate_cost
from app.db.budget_repository import BudgetExceededError
from app.db.repository import CallRecordRepository, NewCallRecord
from app.dependencies import (
    get_app_settings,
    get_budget_guard,
    get_db_session,
    get_limiter,
    get_provider_pool,
)
from app.errors import ErrorType, GatewayError, error_origin
from app.logging_config import get_logger, log_extra
from app.provider_pool import ProviderPool
from app.request_context import new_request_id, set_request_id
from app.retry import AttemptLog, RetryPolicy
from app.schemas import ChatCompletionRequest
from app.security import require_client_api_key, summarize_prompt
from app.upstream import ChatCompletionsClient, UpstreamResult

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
    pool: ProviderPool = Depends(get_provider_pool),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
    budget: BudgetGuard = Depends(get_budget_guard),
    key_fingerprint: str = Depends(require_client_api_key),
    x_client_app: str | None = Header(default=None, alias="X-Client-App"),
    x_agent_name: str | None = Header(default=None, alias="X-Agent-Name"),
    x_run_id: str | None = Header(default=None, alias="X-Run-Id"),
) -> JSONResponse:
    """转发一次非流式 Chat Completions 请求。"""
    request_id = new_request_id()
    set_request_id(request_id)

    # 标签只用于观测，解析失败直接 400，不静默丢弃
    tags = parse_agent_tags(
        client_app=x_client_app,
        agent_name=x_agent_name,
        run_id=x_run_id,
        max_length=settings.agent_tag_max_length,
    )

    # 未登记的模型与未配置的服务商在这里直接抛错，不会走到任何上游调用
    route, client = pool.require_client(payload.model)

    # 配额预扣：内部用独立事务并立即提交，行锁不会跨上游请求持有。
    # 放在上游调用之前，因此并发请求无法通过同时发起绕过限制。
    # 认证失败与参数校验失败在此之前就已返回，不消耗配额。
    outcome = await budget.reserve(
        key_fingerprint=key_fingerprint,
        run_id=tags.run_id,
    )
    if not outcome.allowed:
        violation = outcome.first_violation
        assert violation is not None
        raise BudgetExceededError(violation)

    policy = RetryPolicy(
        max_attempts=settings.retry_max_attempts,
        backoff_base_seconds=settings.retry_backoff_base_seconds,
        backoff_max_seconds=settings.retry_backoff_max_seconds,
        retry_on_read_timeout=settings.retry_on_read_timeout,
        retry_on_upstream_error=settings.retry_on_upstream_error,
    )

    prompt_meta = summarize_prompt([m.model_dump() for m in payload.messages])
    client_ip = request.client.host if request.client else None

    started_at = time.perf_counter()

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_model: str | None = None
    error_type = ErrorType.NONE
    http_status = 200
    attempt_log = AttemptLog()

    try:
        try:
            async with limiter.slot():
                upstream_result = await _call_with_retry(
                    client=client,
                    payload=payload.upstream_payload(default_model=route.model),
                    policy=policy,
                    context={
                        "request_id": request_id,
                        "client_app": tags.client_app,
                        "run_id": tags.run_id,
                        "agent_name": tags.agent_name,
                        "provider": route.provider,
                        "model": route.model,
                    },
                    attempt_log=attempt_log,
                )
        except ConcurrencyTimeout as exc:
            raise GatewayError(
                str(exc),
                error_type=ErrorType.CONCURRENCY_TIMEOUT,
                status_code=503,
            ) from exc

        prompt_tokens = upstream_result.prompt_tokens
        completion_tokens = upstream_result.completion_tokens
        response_model = upstream_result.response_model
        http_status = upstream_result.status_code

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
        origin = error_origin(error_type)

        logger.info(
            "调用结束",
            **log_extra(
                request_id=request_id,
                client_app=tags.client_app,
                run_id=tags.run_id,
                agent_name=tags.agent_name,
                provider=route.provider,
                model=route.model,
                latency_ms=latency_ms,
                error_type=error_type.value,
                error_origin=origin.value if origin else None,
                http_status=http_status,
                retry_count=attempt_log.retry_count,
            ),
        )

        await _persist_call_record(
            session=session,
            record=NewCallRecord(
                request_id=request_id,
                provider=route.provider,
                # 历史列 model 保留，便于旧查询继续工作
                model=route.model,
                requested_model=route.model,
                response_model=response_model,
                latency_ms=latency_ms,
                retry_count=attempt_log.retry_count,
                http_status=http_status,
                error_type=error_type.value,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=calculate_cost(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    input_price_per_million=route.price_input_per_million_usd,
                    output_price_per_million=route.price_output_per_million_usd,
                ),
                key_fingerprint=key_fingerprint,
                message_count=prompt_meta["message_count"],
                prompt_chars=prompt_meta["prompt_chars"],
                client_ip=client_ip,
                client_app=tags.client_app,
                agent_name=tags.agent_name,
                run_id=tags.run_id,
            ),
            request_id=request_id,
        )


async def _call_with_retry(
    *,
    client: ChatCompletionsClient,
    payload: dict,
    policy: RetryPolicy,
    context: dict,
    attempt_log: AttemptLog,
) -> UpstreamResult:
    """按策略调用上游，把每次尝试写入 attempt_log。

    attempt_log 由调用方持有并原地更新：函数在重试耗尽时会抛异常，
    若在这里新建对象再返回，异常路径上调用方就拿不到尝试次数了。

    只有 policy 判定可重试的错误才会重试；每次失败都写日志，
    便于事后判断某个失败是否伴随过重试 —— 也就是重复推理与重复计费的风险。
    """
    attempt = 1

    while True:
        attempt_started = time.perf_counter()

        try:
            result = await client.create_chat_completion(payload)
        except GatewayError as exc:
            attempt_ms = int((time.perf_counter() - attempt_started) * 1000)
            attempt_log.record(attempt, exc.error_type, attempt_ms)
            origin = error_origin(exc.error_type)

            logger.warning(
                "上游调用失败",
                **log_extra(
                    **context,
                    attempt=attempt,
                    latency_ms=attempt_ms,
                    error_type=exc.error_type.value,
                    error_origin=origin.value if origin else None,
                    http_status=exc.status_code,
                ),
            )

            if not policy.should_retry(exc.error_type, attempt=attempt):
                raise

            delay = policy.backoff_seconds(attempt)
            logger.warning(
                "准备重试",
                **log_extra(
                    **context,
                    attempt=attempt,
                    retry_count=attempt,
                    error_type=exc.error_type.value,
                ),
            )
            await asyncio.sleep(delay)
            attempt += 1
            continue

        attempt_log.record(
            attempt,
            ErrorType.NONE,
            int((time.perf_counter() - attempt_started) * 1000),
        )
        return result


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
