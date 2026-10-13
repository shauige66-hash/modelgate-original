"""错误分类与统一异常响应。

设计目标：
1. 每一次调用都能落库一个结构化错误类型，便于统计失败原因。
2. 对客户端返回稳定、不含内部细节的错误体。
"""

from __future__ import annotations

from enum import Enum

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.logging_config import get_logger
from app.request_context import get_request_id

logger = get_logger(__name__)


class ErrorType(str, Enum):
    """调用记录中 error_type 字段的取值。"""

    NONE = "none"  # 成功
    UPSTREAM_TIMEOUT = "upstream_timeout"  # 上游响应超时
    UPSTREAM_CONNECT_ERROR = "upstream_connect_error"  # 无法连接上游
    UPSTREAM_HTTP_ERROR = "upstream_http_error"  # 上游返回 4xx/5xx
    UPSTREAM_AUTH_ERROR = "upstream_auth_error"  # 上游拒绝我们的密钥
    UPSTREAM_INVALID_RESPONSE = "upstream_invalid_response"  # 响应体不符合预期
    UPSTREAM_NOT_CONFIGURED = "upstream_not_configured"  # 服务端未配置上游密钥
    CONCURRENCY_TIMEOUT = "concurrency_timeout"  # 等待并发槽位超时
    CLIENT_CANCELLED = "client_cancelled"  # 客户端主动断开
    INTERNAL_ERROR = "internal_error"  # 未预期的服务端异常


class GatewayError(Exception):
    """网关内部异常，携带对外状态码与错误类型。"""

    def __init__(
        self,
        message: str,
        *,
        error_type: ErrorType,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
    """把 GatewayError 转成 OpenAI 风格的错误响应体。

    失败响应同样带上 X-Request-Id，让客户端可以拿它去对账数据库记录。
    """
    logger.warning(
        "网关错误 | type=%s status=%s detail=%s",
        exc.error_type.value,
        exc.status_code,
        exc.message,
    )

    headers: dict[str, str] = {}
    request_id = get_request_id()
    if request_id:
        headers["X-Request-Id"] = request_id

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type.value,
                "code": exc.status_code,
            }
        },
        headers=headers,
    )
