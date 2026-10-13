"""可靠性测试：故障注入、重试策略、错误归因与结构化日志。

上游故障全部由 MockTransport 注入，不接触真实 API。

重试相关用例的重点不是「重试能不能工作」，而是**默认不重试**：
LLM 请求不是幂等的，读取超时无法判断上游是否已开始推理，
自动重试会造成重复推理与重复计费。
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.errors import ErrorOrigin, ErrorType, error_origin
from app.retry import (
    NEVER_RETRY,
    RISKY_TO_RETRY,
    SAFE_TO_RETRY,
    AttemptLog,
    RetryPolicy,
)
from tests.conftest import build_settings

CHAT_URL = "/v1/chat/completions"


def _payload() -> dict:
    return {"messages": [{"role": "user", "content": "hi"}]}


def _failing(status_code: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": {"message": "boom"}})

    return handler


# ---------------------------------------------------------------- 默认不重试


class TestNoRetryByDefault:
    @pytest.mark.parametrize(
        ("status_code", "expected_status"),
        [(429, 502), (500, 502), (503, 502)],
    )
    async def test_upstream_error_calls_once(
        self, make_harness, status_code: int, expected_status: int
    ) -> None:
        """429 / 5xx 默认不重试：上游只应收到一次请求。"""
        h = await make_harness(_failing(status_code))

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == expected_status
        assert len(h.upstream_requests) == 1

    async def test_read_timeout_calls_once(self, make_harness) -> None:
        """读取超时默认不重试：无法确认上游是否已在推理。"""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 504
        assert calls == 1

    async def test_connect_error_calls_once_by_default(self, make_harness) -> None:
        """连接失败虽然安全，但默认配置（max_attempts=1）同样不重试。"""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused", request=request)

        h = await make_harness(handler)

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 502
        assert calls == 1

    async def test_retry_count_recorded_as_zero(self, make_harness) -> None:
        h = await make_harness()
        response = await h.client.post(CHAT_URL, json=_payload())
        request_id = response.headers["X-Request-Id"]

        from app.db.repository import CallRecordRepository

        async with h.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.retry_count == 0


# ---------------------------------------------------------------- 策略判定


class TestRetryPolicy:
    def test_disabled_when_max_attempts_is_one(self) -> None:
        policy = RetryPolicy(max_attempts=1)

        assert policy.enabled is False
        assert policy.should_retry(ErrorType.UPSTREAM_CONNECT_ERROR, attempt=1) is False

    def test_connect_error_retried_when_enabled(self) -> None:
        """连接失败是唯一默认安全的可重试错误。"""
        policy = RetryPolicy(max_attempts=3)

        assert policy.should_retry(ErrorType.UPSTREAM_CONNECT_ERROR, attempt=1) is True
        assert policy.should_retry(ErrorType.UPSTREAM_CONNECT_ERROR, attempt=2) is True
        # 达到上限后不再重试
        assert policy.should_retry(ErrorType.UPSTREAM_CONNECT_ERROR, attempt=3) is False

    @pytest.mark.parametrize(
        "error_type",
        [ErrorType.UPSTREAM_TIMEOUT, ErrorType.UPSTREAM_HTTP_ERROR],
    )
    def test_risky_errors_not_retried_by_default(
        self, error_type: ErrorType
    ) -> None:
        """读取超时与上游错误即使开启重试也不算安全，需显式开关。"""
        policy = RetryPolicy(max_attempts=3)

        assert policy.should_retry(error_type, attempt=1) is False

    def test_read_timeout_retried_only_when_explicitly_enabled(self) -> None:
        policy = RetryPolicy(max_attempts=3, retry_on_read_timeout=True)

        assert policy.should_retry(ErrorType.UPSTREAM_TIMEOUT, attempt=1) is True

    def test_upstream_error_retried_only_when_explicitly_enabled(self) -> None:
        policy = RetryPolicy(max_attempts=3, retry_on_upstream_error=True)

        assert policy.should_retry(ErrorType.UPSTREAM_HTTP_ERROR, attempt=1) is True

    @pytest.mark.parametrize("error_type", sorted(NEVER_RETRY, key=lambda e: e.value))
    def test_never_retry_set_is_respected(self, error_type: ErrorType) -> None:
        """鉴权失败、配额、并发超时、客户端取消等一律不重试。"""
        policy = RetryPolicy(
            max_attempts=5,
            retry_on_read_timeout=True,
            retry_on_upstream_error=True,
        )

        assert policy.should_retry(error_type, attempt=1) is False

    def test_backoff_grows_exponentially_and_is_capped(self) -> None:
        policy = RetryPolicy(
            max_attempts=10, backoff_base_seconds=0.5, backoff_max_seconds=4.0
        )

        assert policy.backoff_seconds(1) == 0.5
        assert policy.backoff_seconds(2) == 1.0
        assert policy.backoff_seconds(3) == 2.0
        # 第 4 次理论值 4.0，之后被上限压住
        assert policy.backoff_seconds(4) == 4.0
        assert policy.backoff_seconds(9) == 4.0

    def test_safe_and_risky_sets_do_not_overlap(self) -> None:
        assert not (SAFE_TO_RETRY & RISKY_TO_RETRY)
        assert not (SAFE_TO_RETRY & NEVER_RETRY)
        assert not (RISKY_TO_RETRY & NEVER_RETRY)


class TestAttemptLog:
    def test_retry_count_is_attempts_minus_one(self) -> None:
        log = AttemptLog()
        log.record(1, ErrorType.UPSTREAM_CONNECT_ERROR, 10)
        log.record(2, ErrorType.NONE, 20)

        assert log.count == 2
        assert log.retry_count == 1

    def test_single_attempt_has_zero_retries(self) -> None:
        log = AttemptLog()
        log.record(1, ErrorType.NONE, 10)

        assert log.retry_count == 0

    def test_describe_is_compact(self) -> None:
        log = AttemptLog()
        log.record(1, ErrorType.UPSTREAM_CONNECT_ERROR, 5)

        assert log.describe() == "1:upstream_connect_error:5ms"


# ---------------------------------------------------------------- 显式启用重试


class TestRetryWhenEnabled:
    async def test_connect_error_is_retried_and_can_succeed(
        self, make_harness
    ) -> None:
        """连接失败重试后成功：上游总共收到 2 次请求。"""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=3, retry_backoff_base_seconds=0.01
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 200
        assert attempts == 2

    async def test_retry_count_persisted(self, make_harness) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                },
            )

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=3, retry_backoff_base_seconds=0.01
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())
        request_id = response.headers["X-Request-Id"]

        from app.db.repository import CallRecordRepository

        async with h.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.retry_count == 1

    async def test_attempts_stop_at_max(self, make_harness) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused", request=request)

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=3, retry_backoff_base_seconds=0.01
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 502
        assert calls == 3

    async def test_read_timeout_retried_only_when_enabled(self, make_harness) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=3,
                retry_backoff_base_seconds=0.01,
                retry_on_read_timeout=True,
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 504
        assert calls == 3

    async def test_429_retried_only_when_enabled(self, make_harness) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=2,
                retry_backoff_base_seconds=0.01,
                retry_on_upstream_error=True,
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 502
        assert calls == 2

    async def test_auth_error_never_retried_even_when_enabled(
        self, make_harness
    ) -> None:
        """上游拒绝我们的密钥，重试没有意义。"""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=5,
                retry_backoff_base_seconds=0.01,
                retry_on_upstream_error=True,
                retry_on_read_timeout=True,
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 502
        assert calls == 1


# ---------------------------------------------------------------- 故障注入矩阵


class TestFaultInjectionMatrix:
    @pytest.mark.parametrize(
        ("handler_factory", "expected_status", "expected_error_type"),
        [
            (
                lambda: _failing(429),
                502,
                ErrorType.UPSTREAM_RATE_LIMITED,
            ),
            (
                lambda: _failing(500),
                502,
                ErrorType.UPSTREAM_HTTP_ERROR,
            ),
            (
                lambda: _failing(503),
                502,
                ErrorType.UPSTREAM_HTTP_ERROR,
            ),
            (
                lambda: _failing(401),
                502,
                ErrorType.UPSTREAM_AUTH_ERROR,
            ),
            (
                lambda: _failing(403),
                502,
                ErrorType.UPSTREAM_AUTH_ERROR,
            ),
        ],
    )
    async def test_http_faults(
        self, make_harness, handler_factory, expected_status: int, expected_error_type
    ) -> None:
        h = await make_harness(handler_factory())

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == expected_status
        request_id = response.headers["X-Request-Id"]

        from app.db.repository import CallRecordRepository

        async with h.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.error_type == expected_error_type.value
        assert record.http_status == expected_status

    async def test_connect_timeout(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("no route", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 504

    async def test_pool_timeout(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.PoolTimeout("pool exhausted", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 504

    async def test_malformed_json_body(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>oops</html>")

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 502
        assert (
            response.json()["error"]["type"]
            == ErrorType.UPSTREAM_INVALID_RESPONSE.value
        )


# ---------------------------------------------------------------- 错误归因


class TestErrorAttributionInRecords:
    @pytest.mark.parametrize(
        ("error_type", "origin"),
        [
            (ErrorType.UPSTREAM_TIMEOUT, ErrorOrigin.UPSTREAM),
            (ErrorType.UPSTREAM_HTTP_ERROR, ErrorOrigin.UPSTREAM),
            (ErrorType.UPSTREAM_AUTH_ERROR, ErrorOrigin.UPSTREAM),
            (ErrorType.QUOTA_EXCEEDED, ErrorOrigin.GATEWAY),
            (ErrorType.CONCURRENCY_TIMEOUT, ErrorOrigin.GATEWAY),
            (ErrorType.CLIENT_CANCELLED, ErrorOrigin.CLIENT),
        ],
    )
    def test_origin_is_decidable(self, error_type: ErrorType, origin: ErrorOrigin) -> None:
        assert error_origin(error_type) == origin

    async def test_upstream_failure_logs_origin_upstream(
        self, make_harness, caplog
    ) -> None:
        h = await make_harness(_failing(500))

        with caplog.at_level(logging.WARNING, logger="app.api.routes.chat"):
            await h.client.post(CHAT_URL, json=_payload())

        text = "\n".join(record.getMessage() for record in caplog.records)
        assert "上游调用失败" in text


# ---------------------------------------------------------------- 结构化日志


class TestStructuredLogging:
    async def test_call_log_carries_required_fields(
        self, make_harness, caplog
    ) -> None:
        """调用日志必须能关联 request_id / 标签 / provider / 耗时 / 错误类型。"""
        h = await make_harness()

        with caplog.at_level(logging.INFO, logger="app.api.routes.chat"):
            response = await h.client.post(
                CHAT_URL,
                json=_payload(),
                headers={
                    "X-Client-App": "insightflow",
                    "X-Agent-Name": "planner",
                    "X-Run-Id": "run-log-1",
                },
            )

        request_id = response.headers["X-Request-Id"]
        records = [
            record
            for record in caplog.records
            if record.getMessage() == "调用结束"
        ]
        assert len(records) == 1

        entry = records[0]
        assert entry.request_id == request_id
        assert entry.client_app == "insightflow"
        assert entry.agent_name == "planner"
        assert entry.run_id == "run-log-1"
        assert entry.provider == "deepseek"
        assert entry.model == "deepseek-chat"
        assert entry.latency_ms >= 0
        assert entry.error_type == ErrorType.NONE.value
        assert entry.http_status == 200
        assert entry.retry_count == 0
        # 成功调用没有错误归属
        assert getattr(entry, "error_origin", None) is None

    async def test_failure_log_includes_error_origin(
        self, make_harness, caplog
    ) -> None:
        h = await make_harness(_failing(503))

        with caplog.at_level(logging.INFO, logger="app.api.routes.chat"):
            await h.client.post(CHAT_URL, json=_payload())

        records = [
            record for record in caplog.records if record.getMessage() == "调用结束"
        ]
        assert records[0].error_origin == ErrorOrigin.UPSTREAM.value
        assert records[0].error_type == ErrorType.UPSTREAM_HTTP_ERROR.value

    async def test_quota_rejection_has_no_call_log(self, make_harness, caplog) -> None:
        """被配额拒绝的请求不产生调用记录（它没有占用上游）。"""
        h = await make_harness(settings=build_settings(client_daily_request_quota=1))

        await h.client.post(CHAT_URL, json=_payload())

        with caplog.at_level(logging.INFO, logger="app.api.routes.chat"):
            blocked = await h.client.post(CHAT_URL, json=_payload())

        assert blocked.status_code == 429
        assert not [
            record for record in caplog.records if record.getMessage() == "调用结束"
        ]

    async def test_logs_never_contain_prompt_or_key(
        self, make_harness, caplog
    ) -> None:
        """日志中不得出现完整 Prompt 或 API Key。"""
        from tests.conftest import DEEPSEEK_KEY

        secret_prompt = "我的银行卡号是 6222021234567890123"
        h = await make_harness()

        with caplog.at_level(logging.DEBUG):
            await h.client.post(
                CHAT_URL,
                json={"messages": [{"role": "user", "content": secret_prompt}]},
            )

        text = "\n".join(record.getMessage() for record in caplog.records)
        assert secret_prompt not in text
        assert "6222021234567890123" not in text
        assert DEEPSEEK_KEY not in text


class TestReliabilityDoesNotBreakExistingBehaviour:
    async def test_quota_transaction_boundary_unchanged(self, make_harness) -> None:
        """启用重试逻辑后，配额仍在调用上游之前预扣。"""
        h = await make_harness(
            settings=build_settings(
                client_daily_request_quota=1, retry_max_attempts=3
            )
        )

        first = await h.client.post(CHAT_URL, json=_payload())
        assert first.status_code == 200

        second = await h.client.post(CHAT_URL, json=_payload())
        assert second.status_code == 429
        # 只有第一次真正打到上游
        assert len(h.upstream_requests) == 1

    async def test_call_record_still_written_on_retry_exhaustion(
        self, make_harness
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=2, retry_backoff_base_seconds=0.01
            ),
        )

        response = await h.client.post(CHAT_URL, json=_payload())
        request_id = response.headers["X-Request-Id"]

        from app.db.repository import CallRecordRepository

        async with h.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.retry_count == 1
        assert record.error_type == ErrorType.UPSTREAM_CONNECT_ERROR.value
