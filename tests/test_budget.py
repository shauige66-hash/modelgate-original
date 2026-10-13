"""配额治理测试。

重点验证三件事：
1. 并发请求不会突破配额（这是配额功能存在的意义）。
2. 被拒绝的请求不会调用上游。
3. 认证失败与参数校验失败不消耗已认证客户端的配额。

并发用例使用文件型 SQLite 与独立 session，确保是真正的多连接竞争；
内存库的 StaticPool 会把连接串行化，测不出行级竞态。
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
)
from app.errors import ErrorType, ErrorOrigin, error_origin
from tests.conftest import CLIENT_KEY, build_settings

CHAT_URL = "/v1/chat/completions"
ADMIN_URL = "/admin/usage"
BUDGET_URL = "/admin/budget"
ADMIN_HEADERS = {"X-Admin-Key": "sk-admin-test-key"}

TODAY = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _payload() -> dict:
    return {"messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------- 仓储层原子性


class TestAtomicConsume:
    async def test_first_consume_inserts_with_used_one(self, db_session) -> None:
        repository = BudgetRepository(db_session)

        used = await repository.try_consume(
            scope=QUOTA_SCOPE_CLIENT,
            identity="fp1",
            limit=5,
            window_start=TODAY.date(),
        )

        assert used == 1

    async def test_consume_returns_none_when_exhausted(self, db_session) -> None:
        repository = BudgetRepository(db_session)
        for expected in range(1, 4):
            used = await repository.try_consume(
                scope=QUOTA_SCOPE_CLIENT,
                identity="fp1",
                limit=3,
                window_start=TODAY.date(),
            )
            assert used == expected

        # 第 4 次应被拒绝，且不再改变计数
        assert (
            await repository.try_consume(
                scope=QUOTA_SCOPE_CLIENT,
                identity="fp1",
                limit=3,
                window_start=TODAY.date(),
            )
            is None
        )

        snapshot = await repository.read_usage(
            scope=QUOTA_SCOPE_CLIENT,
            identity="fp1",
            limit=3,
            window_start=TODAY.date(),
        )
        assert snapshot.used == 3
        assert snapshot.exceeded is True
        assert snapshot.remaining == 0

    async def test_identities_are_isolated(self, db_session) -> None:
        repository = BudgetRepository(db_session)

        await repository.try_consume(
            scope=QUOTA_SCOPE_CLIENT, identity="fp1", limit=1, window_start=TODAY.date()
        )

        # 另一个身份仍有额度
        assert (
            await repository.try_consume(
                scope=QUOTA_SCOPE_CLIENT,
                identity="fp2",
                limit=1,
                window_start=TODAY.date(),
            )
            == 1
        )

    async def test_scopes_are_isolated(self, db_session) -> None:
        repository = BudgetRepository(db_session)

        await repository.try_consume(
            scope=QUOTA_SCOPE_CLIENT, identity="same", limit=1, window_start=TODAY.date()
        )

        # 同名的 run 维度是另一行，不受 client 维度影响
        assert (
            await repository.try_consume(
                scope=QUOTA_SCOPE_RUN,
                identity="same",
                limit=1,
                window_start=TODAY.date(),
            )
            == 1
        )

    async def test_windows_are_isolated(self, db_session) -> None:
        repository = BudgetRepository(db_session)

        await repository.try_consume(
            scope=QUOTA_SCOPE_CLIENT,
            identity="fp1",
            limit=1,
            window_start=date(2026, 5, 1),
        )

        # 第二天额度重置
        assert (
            await repository.try_consume(
                scope=QUOTA_SCOPE_CLIENT,
                identity="fp1",
                limit=1,
                window_start=date(2026, 5, 2),
            )
            == 1
        )

    async def test_limit_below_one_is_rejected(self, db_session) -> None:
        with pytest.raises(ValueError):
            await BudgetRepository(db_session).try_consume(
                scope=QUOTA_SCOPE_CLIENT,
                identity="fp1",
                limit=0,
                window_start=TODAY.date(),
            )

    async def test_run_identity_isolated_by_client(self, db_session) -> None:
        """不同客户端复用同一个 run_id 时必须各自计数。"""
        repository = BudgetRepository(db_session)
        same_run = "run-shared"

        identity_a = run_identity("fingerprint-a", same_run)
        identity_b = run_identity("fingerprint-b", same_run)

        assert identity_a != identity_b

        await repository.try_consume(
            scope=QUOTA_SCOPE_RUN,
            identity=identity_a,
            limit=1,
            window_start=TODAY.date(),
        )

        # A 用尽不影响 B
        assert (
            await repository.try_consume(
                scope=QUOTA_SCOPE_RUN,
                identity=identity_b,
                limit=1,
                window_start=TODAY.date(),
            )
            == 1
        )


class TestConcurrentConsume:
    """并发竞争：这是配额能否真正生效的关键。"""

    async def test_parallel_consumption_never_exceeds_limit(self, file_engine) -> None:
        """40 个并发请求抢 10 个额度，只能成功 10 次。"""
        factory = async_sessionmaker(bind=file_engine, expire_on_commit=False)
        limit = 10
        attempts = 40

        async def consume_once() -> int | None:
            async with factory() as session:
                repository = BudgetRepository(session)
                result = await repository.try_consume(
                    scope=QUOTA_SCOPE_CLIENT,
                    identity="concurrent-fp",
                    limit=limit,
                    window_start=TODAY.date(),
                )
                await session.commit()
                return result

        results = await asyncio.gather(*(consume_once() for _ in range(attempts)))

        granted = [value for value in results if value is not None]
        assert len(granted) == limit, (
            f"期望恰好 {limit} 次成功，实际 {len(granted)} 次；"
            f"配额被并发绕过了"
        )
        # 成功的返回值必须正好覆盖 1..limit，没有跳号或重复
        assert sorted(granted) == list(range(1, limit + 1))

        async with factory() as session:
            snapshot = await BudgetRepository(session).read_usage(
                scope=QUOTA_SCOPE_CLIENT,
                identity="concurrent-fp",
                limit=limit,
                window_start=TODAY.date(),
            )
        assert snapshot.used == limit

    async def test_parallel_consumption_with_limit_one(self, file_engine) -> None:
        """限额 1 时，N 个并发只能有 1 个拿到额度。"""
        factory = async_sessionmaker(bind=file_engine, expire_on_commit=False)

        async def consume_once() -> int | None:
            async with factory() as session:
                repository = BudgetRepository(session)
                result = await repository.try_consume(
                    scope=QUOTA_SCOPE_CLIENT,
                    identity="single",
                    limit=1,
                    window_start=TODAY.date(),
                )
                await session.commit()
                return result

        results = await asyncio.gather(*(consume_once() for _ in range(20)))

        assert [value for value in results if value is not None] == [1]


# ---------------------------------------------------------------- 组合判定


class TestBudgetGuard:
    async def test_disabled_by_default_performs_no_writes(self, session_factory) -> None:
        """未启用时不应产生任何配额记录。"""
        guard = BudgetGuard(
            session_factory=session_factory, limits=BudgetLimits(build_settings())
        )
        assert guard.enabled is False

        outcome = await guard.reserve(key_fingerprint="fp", run_id="run-1")

        assert outcome.allowed is True
        assert outcome.violations == ()

        async with session_factory() as session:
            rows = await BudgetRepository(session).list_usage(
                window_start=datetime.now(timezone.utc).date()
            )
        assert rows == []

    async def test_client_quota_rejects_after_limit(self, session_factory) -> None:
        settings = build_settings(client_daily_request_quota=2)
        guard = BudgetGuard(
            session_factory=session_factory, limits=BudgetLimits(settings)
        )

        assert (await guard.reserve(key_fingerprint="fp", run_id=None)).allowed
        assert (await guard.reserve(key_fingerprint="fp", run_id=None)).allowed

        third = await guard.reserve(key_fingerprint="fp", run_id=None)
        assert third.allowed is False
        violation = third.first_violation
        assert violation is not None
        assert violation.scope == QUOTA_SCOPE_CLIENT
        assert violation.limit == 2

    async def test_run_quota_rejects_after_limit(self, session_factory) -> None:
        settings = build_settings(run_max_llm_calls=2)
        guard = BudgetGuard(
            session_factory=session_factory, limits=BudgetLimits(settings)
        )

        for _ in range(2):
            assert (await guard.reserve(key_fingerprint="fp", run_id="run-a")).allowed

        blocked = await guard.reserve(key_fingerprint="fp", run_id="run-a")
        assert blocked.allowed is False
        assert blocked.first_violation.scope == QUOTA_SCOPE_RUN

        # 另一个 run 不受影响
        assert (await guard.reserve(key_fingerprint="fp", run_id="run-b")).allowed

    async def test_run_quota_skipped_when_no_run_id(self, session_factory) -> None:
        """没有 run_id 时不做 run 维度限制。"""
        settings = build_settings(run_max_llm_calls=1)
        guard = BudgetGuard(
            session_factory=session_factory, limits=BudgetLimits(settings)
        )

        for _ in range(3):
            assert (await guard.reserve(key_fingerprint="fp", run_id=None)).allowed

    async def test_rejected_request_rolls_back_other_dimensions(
        self, session_factory
    ) -> None:
        """run 超限时，本次已扣的 client 额度必须回滚，避免错账。"""
        settings = build_settings(client_daily_request_quota=100, run_max_llm_calls=1)
        guard = BudgetGuard(
            session_factory=session_factory, limits=BudgetLimits(settings)
        )

        assert (await guard.reserve(key_fingerprint="fp", run_id="run-a")).allowed

        blocked = await guard.reserve(key_fingerprint="fp", run_id="run-a")
        assert blocked.allowed is False

        async with session_factory() as session:
            client_usage = await BudgetRepository(session).read_usage(
                scope=QUOTA_SCOPE_CLIENT,
                identity="fp",
                limit=100,
                window_start=datetime.now(timezone.utc).date(),
            )

        # 第二次请求被拒，不应计入 client 维度
        assert client_usage.used == 1

    async def test_snapshot_is_read_only(self, session_factory) -> None:
        settings = build_settings(client_daily_request_quota=5)
        guard = BudgetGuard(
            session_factory=session_factory, limits=BudgetLimits(settings)
        )

        await guard.reserve(key_fingerprint="fp", run_id=None)
        decisions = await guard.snapshot(key_fingerprint="fp", run_id=None)

        assert len(decisions) == 1
        assert decisions[0].used == 1

        again = await guard.snapshot(key_fingerprint="fp", run_id=None)
        assert again[0].used == 1


# ---------------------------------------------------------------- 端到端


class TestQuotaEndpoint:
    async def test_request_rejected_with_429_and_stable_error_type(
        self, make_harness
    ) -> None:
        h = await make_harness(settings=build_settings(client_daily_request_quota=1))

        first = await h.client.post(CHAT_URL, json=_payload())
        assert first.status_code == 200

        second = await h.client.post(CHAT_URL, json=_payload())
        assert second.status_code == 429

        body = second.json()["error"]
        assert body["type"] == ErrorType.QUOTA_EXCEEDED.value
        assert body["code"] == 429
        assert body["scope"] == QUOTA_SCOPE_CLIENT
        assert body["limit"] == 1

    async def test_rejected_request_does_not_call_upstream(self, make_harness) -> None:
        """被配额拒绝的请求绝不能碰到上游。"""
        h = await make_harness(settings=build_settings(client_daily_request_quota=1))

        await h.client.post(CHAT_URL, json=_payload())
        calls_after_first = len(h.upstream_requests)

        response = await h.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 429
        assert len(h.upstream_requests) == calls_after_first

    async def test_authentication_failure_does_not_consume_quota(
        self, make_harness
    ) -> None:
        """认证失败发生在配额之前，不消耗额度。"""
        h = await make_harness(
            settings=build_settings(client_daily_request_quota=1), authenticated=False
        )

        for _ in range(3):
            assert (await h.client.post(CHAT_URL, json=_payload())).status_code == 401

        # 配额未被消耗：带正确凭据仍可成功
        ok = await h.client.post(
            CHAT_URL,
            json=_payload(),
            headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        )
        assert ok.status_code == 200

    async def test_validation_failure_does_not_consume_quota(
        self, make_harness
    ) -> None:
        """请求体校验失败返回 422，不消耗配额。"""
        h = await make_harness(settings=build_settings(client_daily_request_quota=1))

        assert (await h.client.post(CHAT_URL, json={"messages": []})).status_code == 422
        assert (await h.client.post(CHAT_URL, json=_payload())).status_code == 200

    async def test_unknown_model_does_not_consume_quota(self, make_harness) -> None:
        """未登记的模型在配额之前被拒。"""
        h = await make_harness(settings=build_settings(client_daily_request_quota=1))

        bad = await h.client.post(CHAT_URL, json={"model": "nope", **_payload()})
        assert bad.status_code == 400

        assert (await h.client.post(CHAT_URL, json=_payload())).status_code == 200

    async def test_invalid_agent_header_does_not_consume_quota(
        self, make_harness
    ) -> None:
        h = await make_harness(settings=build_settings(client_daily_request_quota=1))

        bad = await h.client.post(
            CHAT_URL, json=_payload(), headers={"X-Agent-Name": "bad name"}
        )
        assert bad.status_code == 400

        assert (await h.client.post(CHAT_URL, json=_payload())).status_code == 200

    async def test_run_quota_rejects_within_same_run(self, make_harness) -> None:
        h = await make_harness(settings=build_settings(run_max_llm_calls=1))
        headers = {"X-Run-Id": "run-quota-1"}

        assert (
            await h.client.post(CHAT_URL, json=_payload(), headers=headers)
        ).status_code == 200

        blocked = await h.client.post(CHAT_URL, json=_payload(), headers=headers)
        assert blocked.status_code == 429
        assert blocked.json()["error"]["scope"] == QUOTA_SCOPE_RUN

        # 换一个 run 仍然可用
        other = await h.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": "run-quota-2"}
        )
        assert other.status_code == 200

    async def test_failed_upstream_call_still_consumes_quota(
        self, make_harness
    ) -> None:
        """配额在调用前预扣，因此上游失败也占用额度。

        这是刻意的：请求确实发生并消耗了上游资源，
        事后补偿会让「预留」退化成「事后统计」，失去并发保护。
        """
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(
            handler, settings=build_settings(client_daily_request_quota=1)
        )

        first = await h.client.post(CHAT_URL, json=_payload())
        assert first.status_code == 504

        second = await h.client.post(CHAT_URL, json=_payload())
        assert second.status_code == 429

    async def test_quota_disabled_by_default(self, harness) -> None:
        """默认配置不限制，多个请求都应成功。"""
        for _ in range(5):
            assert (await harness.client.post(CHAT_URL, json=_payload())).status_code == 200

    async def test_concurrent_requests_cannot_bypass_quota(
        self, make_harness
    ) -> None:
        """并发打同一配额：成功数不得超过限额。

        必须用 file_backed：内存库的 StaticPool 只有一条连接，
        并发请求会共用同一连接而互相干扰，测不出真实的配额竞争。
        """
        h = await make_harness(
            settings=build_settings(client_daily_request_quota=5),
            file_backed=True,
        )

        responses = await asyncio.gather(
            *(h.client.post(CHAT_URL, json=_payload()) for _ in range(25))
        )

        ok = [r for r in responses if r.status_code == 200]
        blocked = [r for r in responses if r.status_code == 429]

        assert len(ok) == 5, f"期望 5 次成功，实际 {len(ok)}"
        assert len(blocked) == 20
        assert len(h.upstream_requests) == 5


class TestBudgetAdminEndpoint:
    async def test_budget_endpoint_requires_admin_key(self, harness) -> None:
        assert (await harness.client.get(BUDGET_URL)).status_code == 401

    async def test_budget_endpoint_lists_usage(self, make_harness) -> None:
        h = await make_harness(settings=build_settings(client_daily_request_quota=3))
        await h.client.post(CHAT_URL, json=_payload())

        response = await h.client.get(BUDGET_URL, headers=ADMIN_HEADERS)

        assert response.status_code == 200
        body = response.json()
        assert body["limits"]["client_daily_request_quota"] == 3
        assert body["count"] == 1
        assert body["entries"][0]["scope"] == QUOTA_SCOPE_CLIENT
        assert body["entries"][0]["used"] == 1
        assert body["entries"][0]["remaining"] == 2

    async def test_identity_is_masked(self, make_harness) -> None:
        """内部身份键（含指纹）不能完整外泄。"""
        h = await make_harness(settings=build_settings(client_daily_request_quota=3))
        await h.client.post(CHAT_URL, json=_payload())

        response = await h.client.get(BUDGET_URL, headers=ADMIN_HEADERS)
        identity = response.json()["entries"][0]["identity"]

        assert identity.startswith("***")
        assert CLIENT_KEY not in response.text

    async def test_run_scope_shows_plain_run_id(self, make_harness) -> None:
        h = await make_harness(settings=build_settings(run_max_llm_calls=3))
        await h.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": "run-visible"}
        )

        response = await h.client.get(
            f"{BUDGET_URL}?scope={QUOTA_SCOPE_RUN}", headers=ADMIN_HEADERS
        )
        entries = response.json()["entries"]

        assert len(entries) == 1
        # run 维度的 identity 是「指纹:run_id」，对外只展示 run_id
        assert entries[0]["identity"] == "run-visible"

    async def test_scope_filter(self, make_harness) -> None:
        h = await make_harness(
            settings=build_settings(
                client_daily_request_quota=5, run_max_llm_calls=5
            )
        )
        await h.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": "filtered-run"}
        )

        only_client = await h.client.get(
            f"{BUDGET_URL}?scope={QUOTA_SCOPE_CLIENT}", headers=ADMIN_HEADERS
        )
        assert all(
            entry["scope"] == QUOTA_SCOPE_CLIENT
            for entry in only_client.json()["entries"]
        )


class TestErrorAttribution:
    """错误归因：客户端 / 网关 / 上游三类必须能区分。"""

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            (ErrorType.QUOTA_EXCEEDED, ErrorOrigin.GATEWAY),
            (ErrorType.CONCURRENCY_TIMEOUT, ErrorOrigin.GATEWAY),
            (ErrorType.UPSTREAM_NOT_CONFIGURED, ErrorOrigin.GATEWAY),
            (ErrorType.INTERNAL_ERROR, ErrorOrigin.GATEWAY),
            (ErrorType.UPSTREAM_TIMEOUT, ErrorOrigin.UPSTREAM),
            (ErrorType.UPSTREAM_HTTP_ERROR, ErrorOrigin.UPSTREAM),
            (ErrorType.UPSTREAM_AUTH_ERROR, ErrorOrigin.UPSTREAM),
            (ErrorType.CLIENT_CANCELLED, ErrorOrigin.CLIENT),
        ],
    )
    def test_origin_mapping(self, error_type: ErrorType, expected: ErrorOrigin) -> None:
        assert error_origin(error_type) == expected

    def test_success_has_no_origin(self) -> None:
        assert error_origin(ErrorType.NONE) is None

    def test_every_error_type_is_mapped(self) -> None:
        """新增 ErrorType 却忘记登记归因时，这条会失败。"""
        from app.errors import _ORIGIN_BY_TYPE

        for error_type in ErrorType:
            assert error_type in _ORIGIN_BY_TYPE, f"{error_type} 缺少归因登记"
