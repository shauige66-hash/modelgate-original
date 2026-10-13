"""增量迁移测试。

验证迁移只做加列与建索引：可重复执行、不改动已有行、
不会删除或清空数据。用 SQLite 模拟「旧版本建的表」。
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import inspect, select, text

from app.db.migrations import apply_additive_migrations
from app.db.models import CallRecord
from app.db.repository import CallRecordRepository, NewCallRecord
from tests.conftest import _make_shared_memory_engine

# 旧版本的表结构：没有 provider / requested_model / response_model / agent 三列
_LEGACY_SCHEMA = """
CREATE TABLE call_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    model VARCHAR(128) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    latency_ms INTEGER NOT NULL,
    http_status INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    estimated_cost_usd NUMERIC(18, 6),
    error_type VARCHAR(48) NOT NULL,
    key_fingerprint VARCHAR(16),
    message_count INTEGER,
    prompt_chars INTEGER,
    client_ip VARCHAR(64)
)
"""

_ADDED_COLUMNS = (
    "provider",
    "requested_model",
    "response_model",
    "retry_count",
    "client_app",
    "agent_name",
    "run_id",
)


async def _column_names(engine) -> set[str]:
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync: {c["name"] for c in inspect(sync).get_columns("call_records")}
        )


async def _index_names(engine) -> set[str]:
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync: {i["name"] for i in inspect(sync).get_indexes("call_records")}
        )


@pytest.fixture
async def legacy_engine():
    """建出旧版结构并写入两条历史数据。"""
    engine = _make_shared_memory_engine()
    async with engine.begin() as conn:
        await conn.execute(text(_LEGACY_SCHEMA))
        await conn.execute(
            text(
                "INSERT INTO call_records "
                "(request_id, model, latency_ms, http_status, error_type, "
                " prompt_tokens, completion_tokens, estimated_cost_usd) "
                "VALUES ('legacy-1', 'deepseek-chat', 100, 200, 'none', 10, 5, 0.000004)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO call_records "
                "(request_id, model, latency_ms, http_status, error_type) "
                "VALUES ('legacy-2', 'deepseek-chat', 200, 504, 'upstream_timeout')"
            )
        )
    try:
        yield engine
    finally:
        await engine.dispose()


class TestAdditiveMigration:
    async def test_missing_columns_are_added(self, legacy_engine) -> None:
        before = await _column_names(legacy_engine)
        for column in _ADDED_COLUMNS:
            assert column not in before

        applied = await apply_additive_migrations(legacy_engine)

        after = await _column_names(legacy_engine)
        for column in _ADDED_COLUMNS:
            assert column in after

        # applied 用 "表.列" 与索引名两种形式描述本次变更
        added = {item.split(".", 1)[1] for item in applied if "." in item}
        assert added == set(_ADDED_COLUMNS)

    async def test_existing_rows_survive(self, legacy_engine) -> None:
        await apply_additive_migrations(legacy_engine)

        async with legacy_engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT request_id, model, prompt_tokens, estimated_cost_usd "
                        "FROM call_records ORDER BY request_id"
                    )
                )
            ).all()

        assert len(rows) == 2
        assert rows[0][0] == "legacy-1"
        assert rows[0][1] == "deepseek-chat"
        assert rows[0][2] == 10
        assert Decimal(str(rows[0][3])) == Decimal("0.000004")
        assert rows[1][0] == "legacy-2"

    async def test_old_rows_get_null_for_new_columns(self, legacy_engine) -> None:
        await apply_additive_migrations(legacy_engine)

        async with legacy_engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT provider, requested_model, response_model, "
                        "client_app, agent_name, run_id "
                        "FROM call_records WHERE request_id = 'legacy-1'"
                    )
                )
            ).one()

        assert all(value is None for value in row)

    async def test_migration_is_idempotent(self, legacy_engine) -> None:
        first = await apply_additive_migrations(legacy_engine)
        second = await apply_additive_migrations(legacy_engine)

        assert first
        # 第二次不应再产生任何变更
        assert second == []

    async def test_indexes_are_created(self, legacy_engine) -> None:
        await apply_additive_migrations(legacy_engine)

        indexes = await _index_names(legacy_engine)
        assert "ix_call_records_provider" in indexes
        assert "ix_call_records_client_app" in indexes
        assert "ix_call_records_agent_name" in indexes
        assert "ix_call_records_run_id" in indexes

    async def test_missing_call_records_is_skipped_not_created(self) -> None:
        """call_records 由 create_all 负责；迁移只补列，不建这张业务表。"""
        engine = _make_shared_memory_engine()
        try:
            await apply_additive_migrations(engine)

            async with engine.connect() as conn:
                tables = await conn.run_sync(
                    lambda sync: set(inspect(sync).get_table_names())
                )
            assert "call_records" not in tables
        finally:
            await engine.dispose()


class TestBootstrapNewTables:
    """新表由迁移兜底创建，不依赖 create_all。

    迁移可能被单独调用（脚本、运维手工执行），
    因此它必须能自己把缺失的表建出来。
    """

    async def test_migration_creates_budget_usage_on_empty_database(self) -> None:
        engine = _make_shared_memory_engine()
        try:
            applied = await apply_additive_migrations(engine)

            assert "table:budget_usage" in applied

            async with engine.connect() as conn:
                tables = await conn.run_sync(
                    lambda sync: set(inspect(sync).get_table_names())
                )
            assert "budget_usage" in tables
        finally:
            await engine.dispose()

    async def test_budget_usage_has_quota_unique_constraint(self) -> None:
        """唯一约束是原子扣减的前提，缺了它会插入多行、配额直接失效。"""
        engine = _make_shared_memory_engine()
        try:
            await apply_additive_migrations(engine)

            async with engine.connect() as conn:
                names = await conn.run_sync(
                    lambda sync: {
                        c["name"]
                        for c in inspect(sync).get_unique_constraints("budget_usage")
                    }
                )
            assert "uq_budget_usage_identity_window" in names
        finally:
            await engine.dispose()

    async def test_bootstrap_is_idempotent(self) -> None:
        engine = _make_shared_memory_engine()
        try:
            first = await apply_additive_migrations(engine)
            second = await apply_additive_migrations(engine)

            assert first
            assert second == []
        finally:
            await engine.dispose()

    async def test_bootstrap_does_not_touch_call_records(self) -> None:
        """兜底建表不得影响已有业务表。"""
        engine = _make_shared_memory_engine()
        try:
            async with engine.begin() as conn:
                await conn.execute(text(_LEGACY_SCHEMA))
                await conn.execute(
                    text(
                        "INSERT INTO call_records "
                        "(request_id, model, latency_ms, http_status, error_type) "
                        "VALUES ('keep-me', 'deepseek-chat', 10, 200, 'none')"
                    )
                )

            await apply_additive_migrations(engine)

            async with engine.connect() as conn:
                count = (
                    await conn.execute(text("SELECT count(*) FROM call_records"))
                ).scalar_one()
            assert count == 1
        finally:
            await engine.dispose()


class TestMigrationThenWrite:
    async def test_new_columns_usable_after_migration(self, legacy_engine) -> None:
        """迁移后新代码可以正常写入带 provider 与 agent 标签的记录。"""
        await apply_additive_migrations(legacy_engine)

        async with legacy_engine.begin() as conn:
            await conn.run_sync(CallRecord.__table__.create, checkfirst=True)

        from sqlalchemy.ext.asyncio import async_sessionmaker

        factory = async_sessionmaker(bind=legacy_engine, expire_on_commit=False)
        async with factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(
                NewCallRecord(
                    request_id="new-1",
                    provider="openai",
                    model="gpt-4o-mini",
                    requested_model="gpt-4o-mini",
                    response_model="gpt-4o-mini-2024-07-18",
                    latency_ms=50,
                    http_status=200,
                    error_type="none",
                    prompt_tokens=3,
                    completion_tokens=1,
                    client_app="insightflow",
                    agent_name="planner",
                    run_id="run-x",
                )
            )
            await session.commit()

            record = await repository.get_by_request_id("new-1")

        assert record is not None
        assert record.provider == "openai"
        assert record.run_id == "run-x"

        # 历史行依然在，且能被新字段查询空值
        async with factory() as session:
            legacy = await CallRecordRepository(session).get_by_request_id("legacy-1")
        assert legacy is not None
        assert legacy.provider is None

    async def test_history_and_new_rows_coexist(self, legacy_engine) -> None:
        await apply_additive_migrations(legacy_engine)

        from sqlalchemy.ext.asyncio import async_sessionmaker

        factory = async_sessionmaker(bind=legacy_engine, expire_on_commit=False)
        async with factory() as session:
            result = await session.execute(select(CallRecord))
            assert len(result.scalars().all()) == 2
