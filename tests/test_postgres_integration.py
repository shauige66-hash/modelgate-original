r"""真实 PostgreSQL 集成测试。

未设置 TEST_DATABASE_URL 时整体跳过。

破坏性：会 drop_all + create_all 重建 call_records 表。
TEST_DATABASE_URL 指向的库名必须以 `_test` 结尾，否则导入阶段即报错
（保护实现见 tests/db_safety.py）。使用独立的测试库：

    docker compose exec db psql -U modelgate -d postgres -c "CREATE DATABASE modelgate_test;"
    $env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
    .\.venv\Scripts\python.exe -m pytest tests/test_postgres_integration.py -v

这些用例覆盖生产实际使用的 asyncpg + PostgreSQL 组合，
SQLite 与它的 SQL 方言及类型行为并不完全一致。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db.models import CallRecord
from app.db.repository import CallRecordRepository, NewCallRecord, UsageFilter
from app.errors import ErrorType
from tests.conftest import TEST_DATABASE_URL, build_settings

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="未设置 TEST_DATABASE_URL，跳过 PostgreSQL 集成测试",
)


@pytest.fixture
def pg_settings():
    return build_settings(database_url=TEST_DATABASE_URL)


def _record(request_id: str = "req_pg", **overrides) -> NewCallRecord:
    values = {
        "request_id": request_id,
        "model": "deepseek-chat",
        "latency_ms": 250,
        "http_status": 200,
        "error_type": ErrorType.NONE.value,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "estimated_cost_usd": Decimal("0.000082"),
        "key_fingerprint": "deadbeef1234",
        "message_count": 2,
        "prompt_chars": 42,
        "client_ip": "127.0.0.1",
    }
    values.update(overrides)
    return NewCallRecord(**values)


class TestSchema:
    async def test_tables_are_created(self, pg_engine: AsyncEngine) -> None:
        """create_all 应能在真实 PostgreSQL 上建出 call_records 表。"""
        async with pg_engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_name = 'call_records'"
                )
            )
            assert result.scalar_one() == "call_records"

    async def test_numeric_column_keeps_decimal_precision(
        self, pg_engine: AsyncEngine
    ) -> None:
        """Numeric(18,6) 必须精确保存 6 位小数，不能变成浮点。"""
        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            created = await repository.insert(
                _record("req_pg_precision", estimated_cost_usd=Decimal("0.000123"))
            )
            await session.commit()

            fetched = await repository.get_by_request_id("req_pg_precision")
            assert fetched is not None
            assert fetched.estimated_cost_usd == Decimal("0.000123")
            assert isinstance(fetched.estimated_cost_usd, Decimal)


class TestPersistence:
    async def test_insert_and_query(self, pg_engine: AsyncEngine) -> None:
        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(_record("req_pg_1"))
            await session.commit()

            fetched = await repository.get_by_request_id("req_pg_1")
            assert fetched is not None
            assert fetched.prompt_tokens == 100
            assert fetched.completion_tokens == 50

    async def test_nullable_tokens_stay_null(self, pg_engine: AsyncEngine) -> None:
        """未知 Token 必须以 NULL 落库（而不是 0）。"""
        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(
                _record(
                    "req_pg_null",
                    prompt_tokens=None,
                    completion_tokens=None,
                    estimated_cost_usd=None,
                )
            )
            await session.commit()

            fetched = await repository.get_by_request_id("req_pg_null")
            assert fetched is not None
            assert fetched.prompt_tokens is None
            assert fetched.completion_tokens is None
            assert fetched.estimated_cost_usd is None

    async def test_timezone_aware_created_at(self, pg_engine: AsyncEngine) -> None:
        """timestamptz 读回来必须带时区信息。"""
        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(_record("req_pg_tz"))
            await session.commit()

            fetched = await repository.get_by_request_id("req_pg_tz")
            assert fetched is not None
            assert fetched.created_at is not None
            assert fetched.created_at.tzinfo is not None


class TestAggregationOnPostgres:
    async def test_usage_totals(self, pg_engine: AsyncEngine) -> None:
        """PostgreSQL 上的聚合（SUM/CASE）结果应与预期一致。"""
        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(_record("req_pg_a"))
            await repository.insert(_record("req_pg_b"))
            await repository.insert(
                _record(
                    "req_pg_c",
                    http_status=504,
                    error_type=ErrorType.UPSTREAM_TIMEOUT.value,
                    prompt_tokens=None,
                    completion_tokens=None,
                    estimated_cost_usd=None,
                )
            )
            await session.commit()

            totals = await repository.usage_totals()

            assert totals.total_requests == 3
            assert totals.success_count == 2
            assert totals.failure_count == 1
            assert totals.prompt_tokens == 200
            assert totals.completion_tokens == 100
            assert totals.records_with_unknown_tokens == 1

    async def test_time_range_filter_uses_timestamptz(
        self, pg_engine: AsyncEngine
    ) -> None:
        """带时区的时间过滤必须能在 PostgreSQL 上正确执行。"""
        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(_record("req_pg_range"))
            await session.commit()

            since = datetime.now(timezone.utc) - timedelta(hours=1)
            totals = await repository.usage_totals(UsageFilter(since=since))
            assert totals.total_requests == 1

            future = datetime.now(timezone.utc) + timedelta(hours=1)
            totals_future = await repository.usage_totals(UsageFilter(since=future))
            assert totals_future.total_requests == 0

    async def test_unique_request_id_constraint(self, pg_engine: AsyncEngine) -> None:
        """唯一约束必须由数据库真正执行。"""
        from sqlalchemy.exc import IntegrityError

        from app.db.session import create_session_factory

        factory = create_session_factory(pg_engine)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(_record("req_pg_dup"))
            await session.commit()

            with pytest.raises(IntegrityError):
                await repository.insert(_record("req_pg_dup"))
                await session.commit()

            await session.rollback()


class TestIndexes:
    async def test_expected_indexes_exist(self, pg_engine: AsyncEngine) -> None:
        """统计接口依赖 created_at 与 model 索引。"""
        async with pg_engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'call_records'"
                )
            )
            names = {row[0] for row in result.all()}

        assert "ix_call_records_created_at" in names
        assert "ix_call_records_model" in names
