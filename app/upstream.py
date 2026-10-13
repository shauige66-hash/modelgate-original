"""上游 LLM 客户端（OpenAI Chat Completions 兼容）。

要点：
- 使用单个长生命周期的 httpx.AsyncClient，复用连接池；
  绝不为每个请求新建客户端（那样会丢失 keep-alive 并快速耗尽端口）。
- 客户端在应用启动时创建、关闭时释放（见 app/main.py 的 lifespan）。
- 把 httpx 的各种异常翻译成 GatewayError，路由层只需处理一种异常类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.errors import ErrorType, GatewayError
from app.logging_config import get_logger

logger = get_logger(__name__)

_CHAT_COMPLETIONS_PATH = "/chat/completions"


@dataclass(frozen=True)
class UpstreamResult:
    """一次成功的上游调用结果。"""

    status_code: int
    payload: dict[str, Any]

    @property
    def prompt_tokens(self) -> int | None:
        """从 usage 中提取输入 Token；缺失返回 None（未知），而不是 0。"""
        usage = self.payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return _as_optional_int(usage.get("prompt_tokens"))

    @property
    def completion_tokens(self) -> int | None:
        """从 usage 中提取输出 Token；缺失返回 None（未知），而不是 0。"""
        usage = self.payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return _as_optional_int(usage.get("completion_tokens"))


def _as_optional_int(value: Any) -> int | None:
    """把上游返回值安全地转成 int；无法解析时返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_http_client(settings: Settings) -> httpx.AsyncClient:
    """创建共享的异步 HTTP 客户端。

    timeout 拆成三段，便于区分「连不上」和「连上了但算得慢」：
    - connect：建立 TCP/TLS 连接
    - read：等待上游返回（LLM 推理耗时主要在这里，因此设置较长）
    - write / pool：发送与等待空闲连接
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
            # 便于在日志中标识流量来源
            "User-Agent": f"ModelGate/{settings.app_version}",
        },
    )


class ChatCompletionsClient:
    """封装对上游 /chat/completions 的调用。"""

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

        if not settings.is_upstream_configured:
            # 只警告不抛错：服务仍可启动，调用时返回明确错误，
            # 这样健康检查和管理接口在未配置密钥时依然可用。
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
            # 连上但超时、或连接阶段超时，都归为上游超时
            logger.warning("上游超时 | %s", type(exc).__name__)
            raise GatewayError(
                f"上游请求超时（{self._settings.upstream_timeout_seconds:g}s）",
                error_type=ErrorType.UPSTREAM_TIMEOUT,
                status_code=504,
            ) from exc
        except httpx.ConnectError as exc:
            logger.warning("无法连接上游 | %s", exc)
            raise GatewayError(
                "无法连接上游模型服务",
                error_type=ErrorType.UPSTREAM_CONNECT_ERROR,
                status_code=502,
            ) from exc
        except httpx.HTTPError as exc:
            # 兜底：其他 httpx 传输层异常
            logger.warning("上游传输层错误 | %s | %s", type(exc).__name__, exc)
            raise GatewayError(
                "上游请求失败",
                error_type=ErrorType.UPSTREAM_CONNECT_ERROR,
                status_code=502,
            ) from exc

        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> UpstreamResult:
        """校验并解析上游响应。"""
        if response.status_code >= 400:
            error_type = (
                ErrorType.UPSTREAM_AUTH_ERROR
                if response.status_code in (401, 403)
                else ErrorType.UPSTREAM_HTTP_ERROR
            )
            # 只截取少量上游信息用于排查，避免把上游返回的大段内容写进日志
            detail = _safe_upstream_detail(response)
            logger.warning(
                "上游返回错误 | status=%s type=%s detail=%s",
                response.status_code,
                error_type.value,
                detail,
            )
            raise GatewayError(
                f"上游模型服务返回 {response.status_code}: {detail}",
                error_type=error_type,
                # 429 / 5xx 属于上游临时故障，用 502 告知客户端「是网关的上游出问题」
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


def _safe_upstream_detail(response: httpx.Response, *, limit: int = 200) -> str:
    """提取上游错误信息，限制长度并去掉换行。"""
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
