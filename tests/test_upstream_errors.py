"""上游异常与超时处理测试。

Mock 上游返回各种失败：超时、连接错误、401、429、500、非 JSON、缺少 choices。
重点验证：每种失败都被翻译成明确的 HTTP 状态码 + 可落库的 error_type。
"""

from __future__ import annotations

import httpx
import pytest

from app.concurrency import ConcurrencyLimiter
from app.errors import ErrorType
from tests.conftest import SAMPLE_COMPLETION, build_settings

CHAT_URL = "/v1/chat/completions"


def _chat_payload() -> dict:
    return {"messages": [{"role": "user", "content": "hi"}]}


class TestUpstreamTimeout:
    async def test_read_timeout_returns_504(self, make_harness) -> None:
        """上游响应超时 -> 504，error_type=upstream_timeout。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("upstream too slow", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 504
        assert response.json()["error"]["type"] == ErrorType.UPSTREAM_TIMEOUT.value

    async def test_connect_timeout_returns_504(self, make_harness) -> None:
        """连接阶段超时同样归为 upstream_timeout。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("cannot connect", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 504
        assert response.json()["error"]["type"] == ErrorType.UPSTREAM_TIMEOUT.value


class TestUpstreamConnectionErrors:
    async def test_connect_error_returns_502(self, make_harness) -> None:
        """连不上上游 -> 502，error_type=upstream_connect_error。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 502
        assert (
            response.json()["error"]["type"] == ErrorType.UPSTREAM_CONNECT_ERROR.value
        )


class TestUpstreamHttpErrors:
    @pytest.mark.parametrize(
        ("status_code", "expected_error_type"),
        [
            (401, ErrorType.UPSTREAM_AUTH_ERROR),
            (403, ErrorType.UPSTREAM_AUTH_ERROR),
            (429, ErrorType.UPSTREAM_HTTP_ERROR),
            (500, ErrorType.UPSTREAM_HTTP_ERROR),
            (503, ErrorType.UPSTREAM_HTTP_ERROR),
        ],
    )
    async def test_upstream_error_status_is_mapped(
        self,
        make_harness,
        status_code: int,
        expected_error_type: ErrorType,
    ) -> None:
        """上游 4xx/5xx 一律对客户端返回 502，并区分鉴权失败与一般错误。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code, json={"error": {"message": "upstream says no"}}
            )

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 502
        assert response.json()["error"]["type"] == expected_error_type.value


class TestUpstreamInvalidResponse:
    async def test_non_json_body_returns_502(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 502
        assert (
            response.json()["error"]["type"]
            == ErrorType.UPSTREAM_INVALID_RESPONSE.value
        )

    async def test_json_without_choices_returns_502(self, make_harness) -> None:
        """合法 JSON 但结构不对，也必须拒绝，避免把脏数据透传给客户端。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 502
        assert (
            response.json()["error"]["type"]
            == ErrorType.UPSTREAM_INVALID_RESPONSE.value
        )


class TestUpstreamNotConfigured:
    async def test_missing_upstream_key_returns_503(self, make_harness) -> None:
        """服务端没配上游密钥时给出明确提示，而不是抛 500。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=SAMPLE_COMPLETION)

        # 未配置上游密钥
        h = await make_harness(handler, settings=build_settings(upstream_api_key=""))
        response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 503
        assert (
            response.json()["error"]["type"] == ErrorType.UPSTREAM_NOT_CONFIGURED.value
        )


class TestConcurrencyLimiterReleasesSlots:
    async def test_exception_releases_slot(self) -> None:
        """业务异常时槽位必须归还，否则后续请求会永久卡死。"""
        limiter = ConcurrencyLimiter(limit=1, acquire_timeout=1.0)

        with pytest.raises(httpx.ConnectError):
            async with limiter.slot():
                raise httpx.ConnectError("boom")

        assert limiter.available == 1
        assert limiter.in_flight == 0

    async def test_cancellation_releases_slot(self) -> None:
        """客户端取消（CancelledError）同样必须归还槽位。"""
        import asyncio

        limiter = ConcurrencyLimiter(limit=1, acquire_timeout=1.0)
        entered = asyncio.Event()

        async def hold_slot() -> None:
            async with limiter.slot():
                entered.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(hold_slot())
        await entered.wait()
        assert limiter.in_flight == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert limiter.available == 1
        assert limiter.in_flight == 0

        # 之后仍然可以正常获取槽位
        async with limiter.slot():
            assert limiter.in_flight == 1
