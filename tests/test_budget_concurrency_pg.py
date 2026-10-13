r"""配额并发与跨窗口行为测试。

分两部分：

1. PostgreSQL 并发竞争 —— 原子扣减最终跑在生产库上，
   只在 SQLite 上验证不足以证明 PostgreSQL 的行锁行为一致，
   因此这里用 TEST_DATABASE_URL 指向的真实 PostgreSQL 重跑同样的竞争场景。

2. 窗口语义 —— run 配额覆盖整个 Run 生命周期，跨 UTC 零点不得重置。

未设置 TEST_DATABASE_URL 时，第一部分整体跳过。
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.budget import BudgetGuard, BudgetLimits
from app.db.budget_repository import (
    QUOTA_SCOPE_CLIENT,
    QUOTA_SCOPE_RUN,
    BudgetRepository,
    run_identity,
    window_for,
)
from app.db.models import QUOTA_WINDOW_DAILY, QUOTA_WINDOW_RUN_TOTAL
from tests.conftest import TEST_DATABASE_URL, build_settings

TODAY = date(2026, 5, 1)

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="未设置 TEST_DATABASE_URL，跳过 PostgreSQL 并发验证",
)


@pytest.fixture
def pg_factory(pg_engine):
    return async_sessionmaker(bind=pg_engine, expire_on_commit=False)


class TestPostgresConcurrentQuota:
    async def test_40_concurrent_for_limit_10(self, pg_factory) -> None:
        """40 并发抢 10 个额度，恰好 10 次成功。"""
        limit = 10

        async def consume_once() -> int | None:
            async with pg_factory() as session:
                result = await BudgetRepository(session).try_consume(
                    scope=QUOTA_SCOPE_CLIENT,
                    identity="pg-40-10",
                    limit=limit,
                    window_start=TODAY,
                )
                await session.commit()
                return result

        results = await asyncio.gather(*(consume_once() for _ in range(40)))
        granted = [value for value in results if value is not None]

        assert len(granted) == limit
        assert sorted(granted) == list(range(1, limit + 1))

        async with pg_factory() as session:
            snapshot = await BudgetRepository(session).read_usage(
                scope=QUOTA_SCOPE_CLIENT,
                identity="pg-40-10",
                limit=limit,
                window_start=TODAY,
            )
        assert snapshot.used == limit

    async def test_20_concurrent_for_limit_1(self, pg_factory) -> None:
        """20 并发抢 1 个额度，只能有 1 次成功。"""

        async def consume_once() -> int | None:
            async with pg_factory() as session:
                result = await BudgetRepository(session).try_consume(
                    scope=QUOTA_SCOPE_CLIENT,
                    identity="pg-20-1",
                    limit=1,
                    window_start=TODAY,
                )
                await session.commit()
                return result

        results = await asyncio.gather(*(consume_once() for _ in range(20)))

        assert [value for value in results if value is not None] == [1]

    async def test_concurrent_first_insert_conflict(self, pg_factory) -> None:
        """并发首次插入不得产生重复行。

        没有唯一约束时，ON CONFLICT 无法生效，会插入多行并让配额失效。
        """
        limit = 5

        async def consume_once() -> int | None:
            async with pg_factory() as session:
                result = await BudgetRepository(session).try_consume(
                    scope=QUOTA_SCOPE_RUN,
                    identity=run_identity("fp", "concurrent-run"),
                    limit=limit,
                    window_start=TODAY,
                )
                await session.commit()
                return result

        await asyncio.gather(*(consume_once() for _ in range(30)))

        async with pg_factory() as session:
            rows = await BudgetRepository(session).list_usage(
                window_start=TODAY, scope=QUOTA_SCOPE_RUN
            )

        matching = [
            row for row in rows if row.identity == run_identity("fp", "concurrent-run")
        ]
        assert len(matching) == 1, "同一身份出现了多行配额记录"
        assert matching[0].used == limit


class TestDualQuotaRollbackOnPostgres:
    async def test_run_violation_rolls_back_client_consumption(
        self, pg_factory
    ) -> None:
        """run 超限时，同一次请求已扣的 client 额度必须回滚。

        否则会出现「run 被拒绝，但客户端当天的额度已经白扣一次」的错账。
        """
        settings = build_settings(
            client_daily_request_quota=100, run_max_llm_calls=1
        )
        guard = BudgetGuard(
            session_factory=pg_factory, limits=BudgetLimits(settings)
        )

        first = await guard.reserve(key_fingerprint="pg-rollback", run_id="run-x")
        assert first.allowed

        blocked = await guard.reserve(key_fingerprint="pg-rollback", run_id="run-x")
        assert blocked.allowed is False
        assert blocked.first_violation.scope == QUOTA_SCOPE_RUN

        window_kind, window_start = window_for(
            QUOTA_SCOPE_CLIENT, datetime.now(timezone.utc).date()
        )
        async with pg_factory() as session:
            client_usage = await BudgetRepository(session).read_usage(
                scope=QUOTA_SCOPE_CLIENT,
                identity="pg-rollback",
                limit=100,
                window_start=window_start,
                window_kind=window_kind,
            )

        assert client_usage.used == 1, "被拒绝的请求不应计入 client 额度"

    async def test_client_violation_leaves_run_untouched(self, pg_factory) -> None:
        """client 超限时，run 额度也不应被扣。"""
        settings = build_settings(
            client_daily_request_quota=1, run_max_llm_calls=100
        )
        guard = BudgetGuard(
            session_factory=pg_factory, limits=BudgetLimits(settings)
        )

        assert (await guard.reserve(key_fingerprint="pg-c1", run_id="run-y")).allowed
        blocked = await guard.reserve(key_fingerprint="pg-c1", run_id="run-y")
        assert blocked.allowed is False
        assert blocked.first_violation.scope == QUOTA_SCOPE_CLIENT

        window_kind, window_start = window_for(
            QUOTA_SCOPE_RUN, datetime.now(timezone.utc).date()
        )
        async with pg_factory() as session:
            run_usage = await BudgetRepository(session).read_usage(
                scope=QUOTA_SCOPE_RUN,
                identity=run_identity("pg-c1", "run-y"),
                limit=100,
                window_start=window_start,
                window_kind=window_kind,
            )

        assert run_usage.used == 1


class TestWindowSemantics:
    """窗口定义：client 按日、run 覆盖整个生命周期。"""

    def test_client_window_is_the_current_day(self) -> None:
        kind, start = window_for(QUOTA_SCOPE_CLIENT, TODAY)

        assert kind == QUOTA_WINDOW_DAILY
        assert start == TODAY

    def test_run_window_ignores_the_date(self) -> None:
        """同一天与另一天必须得到同一个窗口键，run 额度才不会重置。"""
        first = window_for(QUOTA_SCOPE_RUN, date(2026, 5, 1))
        second = window_for(QUOTA_SCOPE_RUN, date(2026, 5, 2))

        assert first == second
        assert first[0] == QUOTA_WINDOW_RUN_TOTAL

    async def test_run_quota_does_not_reset_across_days(self, pg_factory) -> None:
        """跨 UTC 零点后，同一个 run 仍然用尽额度。"""
        settings = build_settings(run_max_llm_calls=2, client_daily_request_quota=0)
        guard = BudgetGuard(
            session_factory=pg_factory, limits=BudgetLimits(settings)
        )

        day_one = datetime(2026, 5, 1, 23, 0, tzinfo=timezone.utc)
        day_two = datetime(2026, 5, 2, 1, 0, tzinfo=timezone.utc)

        assert (
            await guard.reserve(key_fingerprint="pg-day", run_id="long-run", now=day_one)
        ).allowed
        assert (
            await guard.reserve(key_fingerprint="pg-day", run_id="long-run", now=day_one)
        ).allowed

        blocked = await guard.reserve(
            key_fingerprint="pg-day", run_id="long-run", now=day_two
        )

        assert blocked.allowed is False, "run 配额不应因跨日而重置"
        assert blocked.first_violation.scope == QUOTA_SCOPE_RUN

    async def test_client_quota_does_reset_across_days(self, pg_factory) -> None:
        """client 维度按日重置，跨日后恢复额度。"""
        settings = build_settings(client_daily_request_quota=1)
        guard = BudgetGuard(
            session_factory=pg_factory, limits=BudgetLimits(settings)
        )

        day_one = datetime(2026, 5, 1, 23, 0, tzinfo=timezone.utc)
        day_two = datetime(2026, 5, 2, 1, 0, tzinfo=timezone.utc)

        assert (
            await guard.reserve(key_fingerprint="pg-reset", run_id=None, now=day_one)
        ).allowed
        assert (
            await guard.reserve(key_fingerprint="pg-reset", run_id=None, now=day_one)
        ).allowed is False

        assert (
            await guard.reserve(key_fingerprint="pg-reset", run_id=None, now=day_two)
        ).allowed

    async def test_same_run_id_different_clients_are_isolated(
        self, pg_factory
    ) -> None:
        """不同客户端用同一个 run_id 时，配额互不影响。"""
        settings = build_settings(run_max_llm_calls=1)
        guard = BudgetGuard(
            session_factory=pg_factory, limits=BudgetLimits(settings)
        )

        assert (
            await guard.reserve(key_fingerprint="client-a", run_id="shared-run")
        ).allowed
        # client-a 用尽
        assert (
            await guard.reserve(key_fingerprint="client-a", run_id="shared-run")
        ).allowed is False

        # client-b 的同名 run 仍有完整额度
        assert (
            await guard.reserve(key_fingerprint="client-b", run_id="shared-run")
        ).allowed
