"""错误分类与统一异常响应。

每次调用都会落库一个结构化的 error_type，便于统计失败原因；
返回给客户端的内容保持稳定且不含内部细节。
"""

from __future__ import annotations

from enum import Enum

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.agent_tracking import InvalidTrackingHeaderError
from app.db.budget_repository import QUOTA_SCOPE_CLIENT, BudgetExceededError
from app.logging_config import get_logger
from app.providers import ProviderNotConfiguredError, UnknownModelError
from app.request_context import get_request_id

logger = get_logger(__name__)


class ErrorType(str, Enum):
    """调用记录中 error_type 字段的取值。"""

    NONE = "none"  # 成功
    UPSTREAM_TIMEOUT = "upstream_timeout"  # 上游响应超时
    UPSTREAM_CONNECT_ERROR = "upstream_connect_error"  # 无法连接上游
    UPSTREAM_RATE_LIMITED = "upstream_rate_limited"  # 上游限流（HTTP 429）
    UPSTREAM_HTTP_ERROR = "upstream_http_error"  # 上游返回其他 4xx/5xx
    UPSTREAM_AUTH_ERROR = "upstream_auth_error"  # 上游拒绝我们的密钥
    UPSTREAM_INVALID_RESPONSE = "upstream_invalid_response"  # 响应体不符合预期
    UPSTREAM_NOT_CONFIGURED = "upstream_not_configured"  # 服务端未配置上游密钥
    CONCURRENCY_TIMEOUT = "concurrency_timeout"  # 等待并发槽位超时
    QUOTA_EXCEEDED = "quota_exceeded"  # 超出本网关配额，未调用上游
    CLIENT_CANCELLED = "client_cancelled"  # 客户端主动断开
    INTERNAL_ERROR = "internal_error"  # 未预期的服务端异常


class ErrorOrigin(str, Enum):
    """故障归属。

    统计时用来区分「调用方的问题」「网关的问题」和「上游的问题」，
    避免把所有失败混在一个错误率里看不出责任方。
    """

    CLIENT = "client"  # 请求本身不合法或未被授权，未消耗上游
    GATEWAY = "gateway"  # 网关自身的限制或故障（并发、配额、内部异常）
    UPSTREAM = "upstream"  # 上游服务的故障或拒绝


# 错误类型 -> 归属。新增 ErrorType 时必须在这里登记，
# 否则 _ORIGIN_BY_TYPE 查询会 KeyError，测试也会失败。
_ORIGIN_BY_TYPE: dict[ErrorType, ErrorOrigin] = {
    ErrorType.NONE: ErrorOrigin.GATEWAY,  # 不适用，仅占位；成功不会记录归属
    ErrorType.UPSTREAM_TIMEOUT: ErrorOrigin.UPSTREAM,
    ErrorType.UPSTREAM_CONNECT_ERROR: ErrorOrigin.UPSTREAM,
    ErrorType.UPSTREAM_RATE_LIMITED: ErrorOrigin.UPSTREAM,
    ErrorType.UPSTREAM_HTTP_ERROR: ErrorOrigin.UPSTREAM,
    ErrorType.UPSTREAM_AUTH_ERROR: ErrorOrigin.UPSTREAM,
    ErrorType.UPSTREAM_INVALID_RESPONSE: ErrorOrigin.UPSTREAM,
    ErrorType.UPSTREAM_NOT_CONFIGURED: ErrorOrigin.GATEWAY,
    ErrorType.CONCURRENCY_TIMEOUT: ErrorOrigin.GATEWAY,
    ErrorType.QUOTA_EXCEEDED: ErrorOrigin.GATEWAY,
    ErrorType.CLIENT_CANCELLED: ErrorOrigin.CLIENT,
    ErrorType.INTERNAL_ERROR: ErrorOrigin.GATEWAY,
}


def error_origin(error_type: ErrorType | str) -> ErrorOrigin | None:
    """返回错误归属；成功或未登记的类型返回 None。"""
    if isinstance(error_type, str):
        try:
            error_type = ErrorType(error_type)
        except ValueError:
            return None

    if error_type is ErrorType.NONE:
        return None
    return _ORIGIN_BY_TYPE.get(error_type)


class GatewayError(Exception):
    """网关内部异常，携带对外状态码与错误类型。

    retry_after_seconds 只在「上游明确给出了 Retry-After」时才有值，
    用于把上游的限流建议透传给调用方；网关自己不会据此自动重试。
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: ErrorType,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _request_id_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    request_id = get_request_id()
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def _error_body(message: str, error_code: str, status_code: int) -> dict[str, object]:
    return {
        "error": {"message": message, "type": error_code, "code": status_code}
    }


async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
    """把 GatewayError 转成 OpenAI 风格的错误响应体。

    失败响应同样带 X-Request-Id，便于客户端与数据库记录对账。
    上游给出 Retry-After 时一并透传，让调用方自行决定是否等待。
    """
    logger.warning(
        "网关错误 | type=%s status=%s detail=%s",
        exc.error_type.value,
        exc.status_code,
        exc.message,
    )

    body = _error_body(exc.message, exc.error_type.value, exc.status_code)
    error = body["error"]
    assert isinstance(error, dict)

    headers = _request_id_headers()
    if exc.retry_after_seconds is not None:
        error["retry_after_seconds"] = exc.retry_after_seconds
        headers["Retry-After"] = str(exc.retry_after_seconds)

    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=headers,
    )


async def unknown_model_handler(request: Request, exc: Exception) -> JSONResponse:
    """未登记的模型：返回 400，并列出可用模型。"""
    assert isinstance(exc, UnknownModelError)
    available = getattr(request.app.state, "provider_pool", None)
    models = available.router.registered_models if available else []

    logger.info("拒绝未登记的模型 | model=%s", exc.model)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=_error_body(
            f"模型 {exc.model!r} 未在网关登记；可用模型：{', '.join(models) or '无'}",
            "unknown_model",
            status.HTTP_400_BAD_REQUEST,
        ),
        headers=_request_id_headers(),
    )


async def provider_not_configured_handler(_: Request, exc: Exception) -> JSONResponse:
    """模型所属服务商缺少 API Key：返回 503，不把请求发往任何上游。"""
    assert isinstance(exc, ProviderNotConfiguredError)
    logger.warning("服务商未配置 | provider=%s", exc.provider)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_error_body(
            f"服务商 {exc.provider} 未配置 API Key，暂时不可用",
            ErrorType.UPSTREAM_NOT_CONFIGURED.value,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ),
        headers=_request_id_headers(),
    )


async def invalid_tracking_header_handler(_: Request, exc: Exception) -> JSONResponse:
    """追踪标签不合法：返回 400，而不是静默截断或忽略。"""
    assert isinstance(exc, InvalidTrackingHeaderError)
    logger.info("追踪标签不合法 | %s", exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=_error_body(
            f"请求头 {exc.header} 不合法：{exc.reason}",
            "invalid_tracking_header",
            status.HTTP_400_BAD_REQUEST,
        ),
        headers=_request_id_headers(),
    )


async def quota_exceeded_handler(_: Request, exc: Exception) -> JSONResponse:
    """配额用尽：返回 429，并在响应体里说明维度与上限。

    identity 不返回给客户端：它是服务端内部标识（含 API Key 指纹），
    暴露出去等于泄露调用方自身的密钥指纹。
    """
    assert isinstance(exc, BudgetExceededError)
    decision = exc.decision
    scope_label = (
        "客户端当日调用次数"
        if decision.scope == QUOTA_SCOPE_CLIENT
        else "本轮 Run 调用次数"
    )

    logger.warning(
        "配额超限拒绝 | scope=%s used=%s limit=%s",
        decision.scope,
        decision.used,
        decision.limit,
    )

    body = _error_body(
        f"已达{scope_label}上限（{decision.limit}），请求被拒绝且未调用上游模型",
        ErrorType.QUOTA_EXCEEDED.value,
        status.HTTP_429_TOO_MANY_REQUESTS,
    )
    error = body["error"]
    assert isinstance(error, dict)
    error["scope"] = decision.scope
    error["limit"] = decision.limit
    error["used"] = decision.used

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=body,
        headers=_request_id_headers(),
    )
