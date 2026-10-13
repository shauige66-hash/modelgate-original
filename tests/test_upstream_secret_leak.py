"""上游敏感信息不外泄的回归测试。

上游（如 DeepSeek）拒绝请求时会回显收到的 API Key 片段，
因此网关返回给客户端的必须是固定描述，原文只能脱敏后进日志。
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.redaction import redact_secrets
from tests.conftest import (
    SAMPLE_COMPLETION,
    DEEPSEEK_KEY,
    build_settings,
)

CHAT_URL = "/v1/chat/completions"

# 一个「看起来像真密钥」的字符串。
# 用它来验证：即使上游把它回显出来，也绝不会出现在客户端响应里。
FAKE_LEAKED_KEY = "sk-live-9f8e7d6c5b4a39281706f5e4d3c2b1a0"

# 上游错误体里额外塞入敏感字段，验证网关只透出固定描述
SENSITIVE_UPSTREAM_BODY = {
    "error": {
        "message": f"Authentication Fails, Your api key: {FAKE_LEAKED_KEY} is invalid",
        "type": "authentication_error",
        "code": "invalid_api_key",
        # 上游有时还会回显请求上下文
        "request_id": "upstream-req-abc123",
        "authorization": f"Bearer {DEEPSEEK_KEY}",
        "internal_host": "upstream-internal-01.corp.local",
    }
}


def _chat_payload() -> dict:
    return {"messages": [{"role": "user", "content": "hi"}]}


def _leaky_handler(status_code: int = 401):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=SENSITIVE_UPSTREAM_BODY)

    return handler


class TestUpstreamErrorIsSanitized:
    """核心回归：上游错误原文不得返回给客户端。"""

    async def test_client_response_does_not_leak_upstream_key_fragment(
        self, make_harness
    ) -> None:
        h = await make_harness(_leaky_handler(401))

        response = await h.client.post(CHAT_URL, json=_chat_payload())
        body = response.text

        assert response.status_code == 502
        # 上游回显的密钥必须完全不出现
        assert FAKE_LEAKED_KEY not in body
        # 连片段都不行（上游消息里出现过 "api key"）
        assert "api key" not in body.lower()
        assert "invalid_api_key" not in body
        # 我们自己的上游密钥也不能出现
        assert DEEPSEEK_KEY not in body
        # 上游内部信息不能外泄
        assert "upstream-internal-01" not in body
        assert "upstream-req-abc123" not in body

    async def test_error_message_is_a_fixed_safe_description(
        self, make_harness
    ) -> None:
        """返回的 message 必须是固定文案，只包含状态码。"""
        h = await make_harness(_leaky_handler(401))

        response = await h.client.post(CHAT_URL, json=_chat_payload())
        payload = response.json()

        assert payload["error"]["message"] == "上游模型服务返回 401"
        assert payload["error"]["type"] == "upstream_auth_error"
        assert payload["error"]["code"] == 502

    @pytest.mark.parametrize("status_code", [401, 403, 429, 500, 503])
    async def test_all_upstream_error_statuses_are_sanitized(
        self, make_harness, status_code: int
    ) -> None:
        h = await make_harness(_leaky_handler(status_code))

        response = await h.client.post(CHAT_URL, json=_chat_payload())
        body = response.text

        assert response.status_code == 502
        assert FAKE_LEAKED_KEY not in body
        assert DEEPSEEK_KEY not in body

        # 429 使用专门的限流文案，其余仍走固定描述
        expected = (
            f"上游触发限流（HTTP {status_code}），请稍后重试"
            if status_code == 429
            else f"上游模型服务返回 {status_code}"
        )
        assert response.json()["error"]["message"] == expected

    async def test_sanitized_for_non_json_upstream_error_too(
        self, make_harness
    ) -> None:
        """上游返回纯文本错误（非 JSON）时同样不能透传原文。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                text=f"Internal error, token={FAKE_LEAKED_KEY} rejected",
            )

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 502
        assert FAKE_LEAKED_KEY not in response.text
        assert response.json()["error"]["message"] == "上游模型服务返回 500"

    async def test_successful_response_still_passes_through_unchanged(
        self, harness
    ) -> None:
        """修复不能误伤正常路径：成功响应依然原样透传。"""
        response = await harness.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 200
        assert response.json()["id"] == SAMPLE_COMPLETION["id"]


class TestSecurityNoteIsLoggedButRedacted:
    """上游错误原文可以进日志（便于排查），但必须先脱敏。"""

    async def test_log_contains_redaction_not_raw_key(
        self, make_harness, caplog
    ) -> None:
        h = await make_harness(_leaky_handler(401))

        with caplog.at_level(logging.WARNING, logger="app.upstream"):
            await h.client.post(CHAT_URL, json=_chat_payload())

        log_text = "\n".join(record.getMessage() for record in caplog.records)

        # 日志里应该能看出发生了上游错误（便于排查）
        assert "上游返回错误" in log_text
        assert "401" in log_text
        # 但绝不能出现完整密钥
        assert FAKE_LEAKED_KEY not in log_text
        assert DEEPSEEK_KEY not in log_text
        # 应该出现脱敏标记
        assert "REDACTED" in log_text

    async def test_connect_error_log_is_redacted(self, make_harness, caplog) -> None:
        """连接错误的文本可能带完整 URL，也必须脱敏。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(
                f"failed to connect to https://api.example.com?key={FAKE_LEAKED_KEY}",
                request=request,
            )

        h = await make_harness(handler)

        with caplog.at_level(logging.WARNING, logger="app.upstream"):
            response = await h.client.post(CHAT_URL, json=_chat_payload())

        log_text = "\n".join(record.getMessage() for record in caplog.records)

        assert response.status_code == 502
        assert FAKE_LEAKED_KEY not in log_text
        assert FAKE_LEAKED_KEY not in response.text


class TestRedactSecretsFunction:
    """直接验证脱敏函数本身。"""

    @pytest.mark.parametrize(
        "raw",
        [
            f"Your api key: {FAKE_LEAKED_KEY} is invalid",
            f"Bearer {FAKE_LEAKED_KEY}",
            f"api_key={FAKE_LEAKED_KEY}",
            f"api-key: {FAKE_LEAKED_KEY}",
            f"token={FAKE_LEAKED_KEY}",
            f"secret: {FAKE_LEAKED_KEY}",
            f"password={FAKE_LEAKED_KEY}",
            "sk-" + "a" * 40,
            "9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c",  # 40 位十六进制
        ],
    )
    def test_redacts_key_like_strings(self, raw: str) -> None:
        result = redact_secrets(raw)

        assert "REDACTED" in result
        # 原始敏感串（或其后 20 位特征）不应残留
        assert FAKE_LEAKED_KEY not in result

    def test_keeps_ordinary_text_unchanged(self) -> None:
        raw = "上游模型服务返回 500"

        assert redact_secrets(raw) == raw

    def test_handles_empty_string(self) -> None:
        assert redact_secrets("") == ""

    def test_preserves_useful_context(self) -> None:
        """脱敏后仍应保留可排查的上下文（字段名、状态码）。"""
        result = redact_secrets(f"api_key={FAKE_LEAKED_KEY}")

        assert "api_key" in result
        assert "REDACTED" in result


class TestSettingsNeverExposeUpstreamKey:
    async def test_error_body_never_contains_configured_upstream_key(
        self, make_harness
    ) -> None:
        """即使上游把我们的密钥原样回显，也不能出现在响应里。"""

        def handler(request: httpx.Request) -> httpx.Response:
            # 模拟上游把收到的 Authorization 原样回显
            return httpx.Response(
                401,
                json={"error": {"message": f"bad credentials: Bearer {DEEPSEEK_KEY}"}},
            )

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert DEEPSEEK_KEY not in response.text

    def test_build_settings_masks_key_in_repr(self) -> None:
        """配置对象的 repr 不能暴露上游密钥。"""
        settings = build_settings()

        assert DEEPSEEK_KEY not in repr(settings)
