"""上游 LLM 客户端（OpenAI Chat Completions 兼容）。

每个服务商持有自己的 httpx.AsyncClient，由 lifespan 创建与释放；
连接池按服务商隔离，任一服务商的池耗尽不会影响另一个。
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

    @property
    def response_model(self) -> str | None:
        """上游实际返回的模型名，可能与请求的模型不同（别名、快照版本）。"""
        model = self.payload.get("model")
        return model if isinstance(model, str) and model else None


def _as_optional_int(value: Any) -> int | None:
    """安全转 int；无法解析时返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_http_client(
    *,
    base_url: str,
    api_key: str,
    settings: Settings,
) -> httpx.AsyncClient:
    """为一个服务商创建异步 HTTP 客户端。

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
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"ModelGate/{settings.app_version}",
        },
    )


class ChatCompletionsClient:
    """对某个服务商 /chat/completions 的调用封装。"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        provider: str,
        timeout_seconds: float,
        is_configured: bool,
    ) -> None:
        self._client = client
        self._provider = provider
        self._timeout_seconds = timeout_seconds
        self._is_configured = is_configured

        if not is_configured:
            # 只警告不抛错：服务仍能启动，健康检查与管理接口保持可用，
            # 针对该服务商的调用返回明确错误。
            logger.warning("服务商 %s 未配置 API Key，相关模型调用将返回 503", provider)

    @property
    def provider(self) -> str:
        return self._provider

    def reconfigure(self, is_configured: bool) -> None:
        """更新「是否已配置」状态；供测试替换凭证后复用同一客户端。"""
        self._is_configured = is_configured

    async def create_chat_completion(self, payload: dict[str, Any]) -> UpstreamResult:
        """向上游发起一次非流式请求。"""
        if not self._is_configured:
            raise GatewayError(
                f"服务商 {self._provider} 未配置 API Key，无法转发请求",
                error_type=ErrorType.UPSTREAM_NOT_CONFIGURED,
                status_code=503,
            )

        try:
            response = await self._client.post(_CHAT_COMPLETIONS_PATH, json=payload)
        except httpx.TimeoutException as exc:
            # 连接阶段与读取阶段超时都归入上游超时
            logger.warning("上游超时 | provider=%s | %s", self._provider, type(exc).__name__)
            raise GatewayError(
                f"上游请求超时（{self._timeout_seconds:g}s）",
                error_type=ErrorType.UPSTREAM_TIMEOUT,
                status_code=504,
            ) from exc
        except httpx.ConnectError as exc:
            # httpx 的连接错误文本可能带完整 URL，先脱敏再落日志
            logger.warning(
                "无法连接上游 | provider=%s | %s",
                self._provider,
                redact_secrets(str(exc)),
            )
            raise GatewayError(
                "无法连接上游模型服务",
                error_type=ErrorType.UPSTREAM_CONNECT_ERROR,
                status_code=502,
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "上游传输层错误 | provider=%s | %s | %s",
                self._provider,
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
            error_type = self._classify_error(response)
            retry_after = (
                _parse_retry_after(response)
                if error_type is ErrorType.UPSTREAM_RATE_LIMITED
                else None
            )

            # 上游错误文本可能回显收到的 API Key 片段（如 DeepSeek 的
            # "Your api key: ****xxxx is invalid"），因此只脱敏后写日志，
            # 返回给客户端的是固定描述，不拼接任何上游原文。
            detail = redact_secrets(_extract_upstream_detail(response))
            logger.warning(
                "上游返回错误 | provider=%s status=%s type=%s retry_after=%s detail=%s",
                self._provider,
                response.status_code,
                error_type.value,
                retry_after,
                detail,
            )
            raise GatewayError(
                _upstream_error_message(error_type, response.status_code),
                error_type=error_type,
                # 上游 4xx/5xx 对客户端统一呈现为 502：问题出在网关的上游。
                # 网关自身的配额 429 由 quota_exceeded 单独表达（HTTP 429），
                # 因此不会与上游限流混淆。
                status_code=502,
                retry_after_seconds=retry_after,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("上游响应不是合法 JSON | provider=%s", self._provider)
            raise GatewayError(
                "上游返回了非 JSON 响应",
                error_type=ErrorType.UPSTREAM_INVALID_RESPONSE,
                status_code=502,
            ) from exc

        if not isinstance(payload, dict) or "choices" not in payload:
            logger.warning("上游响应缺少 choices 字段 | provider=%s", self._provider)
            raise GatewayError(
                "上游响应缺少 choices 字段",
                error_type=ErrorType.UPSTREAM_INVALID_RESPONSE,
                status_code=502,
            )

        return UpstreamResult(status_code=response.status_code, payload=payload)

    @staticmethod
    def _classify_error(response: httpx.Response) -> ErrorType:
        """把上游状态码映射到错误类型。

        429 单独归类，避免与网关自身的配额拒绝混在一起统计：
        两者都是「限流」，但责任方不同（上游 vs 本网关）。
        """
        status_code = response.status_code
        if status_code in (401, 403):
            return ErrorType.UPSTREAM_AUTH_ERROR
        if status_code == 429:
            return ErrorType.UPSTREAM_RATE_LIMITED
        return ErrorType.UPSTREAM_HTTP_ERROR


def _upstream_error_message(error_type: ErrorType, status_code: int) -> str:
    if error_type is ErrorType.UPSTREAM_RATE_LIMITED:
        return f"上游触发限流（HTTP {status_code}），请稍后重试"
    return f"上游模型服务返回 {status_code}"


def _parse_retry_after(response: httpx.Response, *, cap: int = 3600) -> int | None:
    """解析上游的 Retry-After。

    只接受整数秒的形式；HTTP-date 形式需要解析绝对时间，且上游很少使用，
    这里不做支持以免引入时区与时钟偏移的歧义。
    上限用于防止上游给出异常大的值。
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None

    try:
        seconds = int(str(raw).strip())
    except (TypeError, ValueError):
        return None

    if seconds < 0:
        return None
    return min(seconds, cap)


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
