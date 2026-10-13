"""调用记录持久化与统计聚合测试（默认走内存 SQLite）。

真实 PostgreSQL 的验证见 tests/test_postgres_integration.py。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import CallRecordRepository, NewCallRecord, UsageFilter
from app.errors import ErrorType
from tests.conftest import SAMPLE_COMPLETION_WITHOUT_USAGE


async def _find_record(harness, request_id: str):
    """按 request_id 读取记录（会话用完即关，避免泄漏连接）。"""
    async with harness.session_factory() as session:
        return await CallRecordRepository(session).get_by_request_id(request_id)


def _success_record(request_id: str = "req_ok", **overrides) -> NewCallRecord:
    values = {
        "request_id": request_id,
        "model": "deepseek-chat",
        "latency_ms": 120,
        "http_status": 200,
        "error_type": ErrorType.NONE.value,
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "estimated_cost_usd": Decimal("0.000028"),
        "key_fingerprint": "abc123",
        "message_count": 1,
        "prompt_chars": 5,
        "client_ip": "127.0.0.1",
    }
    values.update(overrides)
    return NewCallRecord(**values)


def _failure_record(request_id: str = "req_fail", **overrides) -> NewCallRecord:
    values = {
        "request_id": request_id,
        "model": "deepseek-chat",
        "latency_ms": 5000,
        "http_status": 504,
        "error_type": ErrorType.UPSTREAM_TIMEOUT.value,
        "prompt_tokens": None,
        "completion_tokens": None,
        "estimated_cost_usd": None,
        "key_fingerprint": "abc123",
        "message_count": 1,
        "prompt_chars": 5,
        "client_ip": "127.0.0.1",
    }
    values.update(overrides)
    return NewCallRecord(**values)


class TestInsert:
    async def test_insert_and_read_back(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)

        created = await repository.insert(_success_record())
        await db_session.commit()

        assert created.id is not None

        fetched = await repository.get_by_request_id("req_ok")
        assert fetched is not None
        assert fetched.model == "deepseek-chat"
        assert fetched.http_status == 200
        assert fetched.prompt_tokens == 12
        assert fetched.completion_tokens == 8
        assert fetched.estimated_cost_usd == Decimal("0.000028")

    async def test_request_id_must_be_unique(self, db_session: AsyncSession) -> None:
        """request_id 唯一，重复写入必须失败，避免统计重复计数。"""
        from sqlalchemy.exc import IntegrityError

        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_dup"))
        await db_session.commit()

        with pytest.raises(IntegrityError):
            await repository.insert(_success_record("req_dup"))
            await db_session.commit()

        await db_session.rollback()

    async def test_list_recent_returns_newest_first(
        self, db_session: AsyncSession
    ) -> None:
        repository = CallRecordRepository(db_session)
        for index in range(3):
            await repository.insert(_success_record(f"req_{index}"))
        await db_session.commit()

        records = await repository.list_recent(limit=10)

        assert len(records) == 3


class TestUsageTotals:
    async def test_counts_success_and_failure(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_1"))
        await repository.insert(_success_record("req_2"))
        await repository.insert(_failure_record("req_3"))
        await db_session.commit()

        totals = await repository.usage_totals()

        assert totals.total_requests == 3
        assert totals.success_count == 2
        assert totals.failure_count == 1

    async def test_tokens_and_cost_are_summed(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_1"))
        await repository.insert(_success_record("req_2"))
        await db_session.commit()

        totals = await repository.usage_totals()

        assert totals.prompt_tokens == 24
        assert totals.completion_tokens == 16
        assert totals.total_tokens == 40
        assert totals.estimated_cost_usd == Decimal("0.000056")

    async def test_unknown_tokens_are_not_counted_as_zero(
        self, db_session: AsyncSession
    ) -> None:
        """未知 Token 的记录不参与求和，但必须被单独计数暴露出来。"""
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_1"))
        await repository.insert(_failure_record("req_2"))
        await db_session.commit()

        totals = await repository.usage_totals()

        # 只有成功那条贡献了 Token
        assert totals.prompt_tokens == 12
        assert totals.completion_tokens == 8
        # 失败那条的 Token 与成本记为未知
        assert totals.records_with_unknown_tokens == 1
        assert totals.records_without_cost == 1

    async def test_empty_database_returns_zeros(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)

        totals = await repository.usage_totals()

        assert totals.total_requests == 0
        assert totals.prompt_tokens == 0
        assert totals.estimated_cost_usd == Decimal("0")
        assert totals.records_with_unknown_tokens == 0

    async def test_time_range_filter(self, db_session: AsyncSession) -> None:
        """时间范围过滤只统计区间内的记录。"""
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_old"))
        await db_session.commit()

        # 把已有记录的 created_at 手工改成 2 天前
        record = await repository.get_by_request_id("req_old")
        assert record is not None
        record.created_at = datetime.now(timezone.utc) - timedelta(days=2)

        await repository.insert(_success_record("req_new"))
        await db_session.commit()

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        totals = await repository.usage_totals(UsageFilter(since=since))

        assert totals.total_requests == 1

    async def test_model_filter(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_a", model="deepseek-chat"))
        await repository.insert(_success_record("req_b", model="deepseek-reasoner"))
        await db_session.commit()

        totals = await repository.usage_totals(UsageFilter(model="deepseek-reasoner"))

        assert totals.total_requests == 1


class TestUsageByModel:
    async def test_groups_by_model(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_a", model="deepseek-chat"))
        await repository.insert(_success_record("req_b", model="deepseek-chat"))
        await repository.insert(_success_record("req_c", model="deepseek-reasoner"))
        await db_session.commit()

        rows = await repository.usage_by_model()

        by_name = {row.key: row for row in rows}
        assert by_name["deepseek-chat"].total_requests == 2
        assert by_name["deepseek-reasoner"].total_requests == 1


class TestErrorBreakdown:
    async def test_counts_only_failures(self, db_session: AsyncSession) -> None:
        repository = CallRecordRepository(db_session)
        await repository.insert(_success_record("req_1"))
        await repository.insert(_failure_record("req_2"))
        await repository.insert(
            _failure_record("req_3", error_type=ErrorType.UPSTREAM_HTTP_ERROR.value)
        )
        await db_session.commit()

        rows = await repository.error_breakdown()
        counts = {row.error_type: row.count for row in rows}

        # 成功的记录（error_type="none"）不应出现在错误分布里
        assert "none" not in counts
        assert counts[ErrorType.UPSTREAM_TIMEOUT.value] == 1
        assert counts[ErrorType.UPSTREAM_HTTP_ERROR.value] == 1


class TestEndpointPersistence:
    async def test_successful_call_is_persisted(self, harness) -> None:
        """一次成功的代理请求必须落库，且 usage 与成本被正确记录。"""
        response = await harness.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )
        assert response.status_code == 200

        request_id = response.headers["X-Request-Id"]
        record = await _find_record(harness, request_id)

        assert record is not None
        assert record.http_status == 200
        assert record.error_type == ErrorType.NONE.value
        assert record.prompt_tokens == 12
        assert record.completion_tokens == 8
        assert record.estimated_cost_usd is not None
        assert record.latency_ms >= 0
        assert record.message_count == 1

    async def test_prompt_content_is_never_stored(self, harness) -> None:
        """落库的记录里不能出现用户 Prompt 原文（只存长度）。"""
        secret_prompt = "我的身份证号是 1234567890，请记住"
        response = await harness.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": secret_prompt}]},
        )
        request_id = response.headers["X-Request-Id"]

        record = await _find_record(harness, request_id)
        assert record is not None

        # 逐字段确认没有原文
        for value in vars(record).values():
            if isinstance(value, str):
                assert "1234567890" not in value

        # 只记录长度
        assert record.prompt_chars == len(secret_prompt)

    async def test_api_key_is_not_stored_in_plaintext(self, harness) -> None:
        """数据库里只能有密钥指纹，不能有明文密钥。"""
        from tests.conftest import CLIENT_KEY

        response = await harness.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        request_id = response.headers["X-Request-Id"]

        record = await _find_record(harness, request_id)
        assert record is not None

        assert record.key_fingerprint is not None
        assert CLIENT_KEY not in record.key_fingerprint
        assert len(record.key_fingerprint) == 12

    async def test_failed_call_is_also_persisted(self, make_harness) -> None:
        """失败请求同样要留痕，否则失败率统计会失真。"""
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        response = await h.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 504

        request_id = response.headers["X-Request-Id"]
        record = await _find_record(h, request_id)

        assert record is not None
        assert record.http_status == 504
        assert record.error_type == ErrorType.UPSTREAM_TIMEOUT.value
        # 失败时 Token 与成本都应为未知（NULL），而不是 0
        assert record.prompt_tokens is None
        assert record.completion_tokens is None
        assert record.estimated_cost_usd is None

    async def test_missing_usage_is_stored_as_null(self, make_harness) -> None:
        """上游没返回 usage 时，Token 必须是 NULL，不能被当成 0。"""
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=SAMPLE_COMPLETION_WITHOUT_USAGE)

        h = await make_harness(handler)
        response = await h.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200

        request_id = response.headers["X-Request-Id"]
        record = await _find_record(h, request_id)

        assert record is not None
        assert record.prompt_tokens is None
        assert record.completion_tokens is None
        assert record.estimated_cost_usd is None
        assert record.http_status == 200
