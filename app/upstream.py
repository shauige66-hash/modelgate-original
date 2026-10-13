"""上游 LLM 客户端（OpenAI Chat Completions 兼容）。

复用单个长生命周期的 httpx.AsyncClient，由 lifespan 创建与释放；
每请求新建客户端会丢失 keep-alive 并快速耗尽本地端口。
httpx 的各类异常统一翻译为 GatewayError，路由层只需处理一种异常类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.errors import ErrorType, GatewayError
from app.logging_config import get_logger
from app.redaction import redact_secrets

logger = get_logger(__name__)

_CHAT_COMPLETIONS_PATH = "/chat/completions"


@dataclass(frozen=True)
class UpstreamResult:
    """一次成功的上游调用结果。"""

    status_code: int
    payload: dict[str, Any]

    @property
    def prompt_tokens(self) -> int | None:
        """输入 Token；usage 缺失时返回 None 表示未知，不按 0 处理。"""
        usage = self.payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return _as_optional_int(usage.get("prompt_tokens"))

    @property
    def completion_tokens(self) -> int | None:
        """输出 Token；usage 缺失时返回 None 表示未知，不按 0 处理。"""
        usage = self.payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return _as_optional_int(usage.get("completion_tokens"))


def _as_optional_int(value: Any) -> int | None:
    """安全转 int；无法解析时返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_http_client(settings: Settings) -> httpx.AsyncClient:
    """创建共享的异步 HTTP 客户端。

    read 超时取完整配置值（LLM 推理耗时集中在这里），
    connect/write 限制为最多 10s，便于区分「连不上」与「算得慢」。
    """
    timeout = httpx.Timeout(
        settings.upstream_timeout_seconds,
        connect=min(10.0, settings.upstream_timeout_seconds),
        read=settings.upstream_timeout_seconds,
        write=min(10.0, settings.upstream_timeout_seconds),
        pool=settings.upstream_timeout_seconds,
    )

    limits = httpx.Limits(
        max_connections=settings.upstream_max_connections,
        max_keepalive_connections=settings.upstream_max_keepalive_connections,
    )

    return httpx.AsyncClient(
        base_url=settings.upstream_base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
        headers={
            "Authorization": f"Bearer {settings.upstream_api_key_value}",
            "Content-Type": "application/json",
            "User-Agent": f"ModelGate/{settings.app_version}",
        },
    )


class ChatCompletionsClient:
    """封装对上游 /chat/completions 的调用。"""

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

        if not settings.is_upstream_configured:
            # 只警告不抛错：服务仍能启动，健康检查与管理接口保持可用，
            # 未配置密钥的错误在真正调用时返回。
            logger.warning(
                "UPSTREAM_API_KEY 未配置，/v1/chat/completions 将返回错误"
            )

    async def create_chat_completion(self, payload: dict[str, Any]) -> UpstreamResult:
        """向上游发起一次非流式请求。

        所有失败路径都会抛出 GatewayError，异常中带有可落库的 error_type。
        """
        if not self._settings.is_upstream_configured:
            raise GatewayError(
                "服务端未配置上游 API Key，无法转发请求",
                error_type=ErrorType.UPSTREAM_NOT_CONFIGURED,
                status_code=503,
            )

        try:
            response = await self._client.post(_CHAT_COMPLETIONS_PATH, json=payload)
        except httpx.TimeoutException as exc:
            # 连接阶段与读取阶段超时都归入上游超时
            logger.warning("上游超时 | %s", type(exc).__name__)
            raise GatewayError(
                f"上游请求超时（{self._settings.upstream_timeout_seconds:g}s）",
                error_type=ErrorType.UPSTREAM_TIMEOUT,
                status_code=504,
            ) from exc
        except httpx.ConnectError as exc:
            # httpx 的连接错误文本可能带完整 URL，先脱敏再落日志
            logger.warning("无法连接上游 | %s", redact_secrets(str(exc)))
            raise GatewayError(
                "无法连接上游模型服务",
                error_type=ErrorType.UPSTREAM_CONNECT_ERROR,
                status_code=502,
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "上游传输层错误 | %s | %s",
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
            raise GatewayError(
                "上游请求失败",
                error_type=ErrorType.UPSTREAM_CONNECT_ERROR,
                status_code=502,
            ) from exc

        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> UpstreamResult:
        """校验上游响应，失败时抛出 GatewayError。"""
        if response.status_code >= 400:
            error_type = (
                ErrorType.UPSTREAM_AUTH_ERROR
                if response.status_code in (401, 403)
                else ErrorType.UPSTREAM_HTTP_ERROR
            )
            # 上游错误文本可能回显收到的 API Key 片段（如 DeepSeek 的
            # "Your api key: ****xxxx is invalid"），因此只脱敏后写日志，
            # 返回给客户端的是固定描述，不拼接任何上游原文。
            detail = redact_secrets(_extract_upstream_detail(response))
            logger.warning(
                "上游返回错误 | status=%s type=%s detail=%s",
                response.status_code,
                error_type.value,
                detail,
            )
            raise GatewayError(
                f"上游模型服务返回 {response.status_code}",
                error_type=error_type,
                # 上游 4xx/5xx 对客户端统一呈现为 502：问题出在网关的上游
                status_code=502,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("上游响应不是合法 JSON")
            raise GatewayError(
                "上游返回了非 JSON 响应",
                error_type=ErrorType.UPSTREAM_INVALID_RESPONSE,
                status_code=502,
            ) from exc

        if not isinstance(payload, dict) or "choices" not in payload:
            logger.warning("上游响应缺少 choices 字段")
            raise GatewayError(
                "上游响应缺少 choices 字段",
                error_type=ErrorType.UPSTREAM_INVALID_RESPONSE,
                status_code=502,
            )

        return UpstreamResult(status_code=response.status_code, payload=payload)


def _extract_upstream_detail(response: httpx.Response, *, limit: int = 200) -> str:
    """提取上游错误信息，仅用于服务端日志。

    调用方需再用 redact_secrets() 脱敏，且不得把结果返回给客户端。
    """
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                message = str(error["message"])
            elif body.get("message"):
                message = str(body["message"])
            else:
                message = str(body)
        else:
            message = str(body)
    except ValueError:
        message = response.text

    message = " ".join(message.split())
    return message[:limit] if message else "<empty>"
