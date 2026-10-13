"""数据一致性验收测试。

覆盖四类容易出错的地方：

1. 配额账本与调用记录必须对得上 —— 预扣是一次，落库也是一条
2. 同名 run 不能跨客户端合并
3. 未知 Token / 成本必须与已知部分分开呈现，不能被当成 0
4. 重试与失败落库之后，统计口径仍然自洽
"""

from __future__ import annotations

from decimal import Decimal

import httpx

from app.db.models import QUOTA_SCOPE_CLIENT, QUOTA_SCOPE_RUN
from app.db.repository import CallRecordRepository, NewCallRecord, UsageFilter
from app.errors import ErrorType
from app.security import key_fingerprint
from tests.conftest import CLIENT_KEY, SAMPLE_COMPLETION_WITHOUT_USAGE, build_settings

CHAT_URL = "/v1/chat/completions"
ADMIN_HEADERS = {"X-Admin-Key": "sk-admin-test-key"}


def _payload(model: str = "deepseek-chat") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


class TestQuotaLedgerMatchesCallRecords:
    """配额是预扣的：成功与失败都算一次，被拒绝的不算。

    这个等式是配额统计可信的基础，任何一侧错位都会让「用了多少」无法对账。
    """

    async def test_successful_requests_reconcile(self, make_harness) -> None:
        h = await make_harness(
            settings=build_settings(client_daily_request_quota=100)
        )
        for _ in range(5):
            response = await h.client.post(CHAT_URL, json=_payload())
            assert response.status_code == 200

        budget = (
            await h.client.get("/admin/budget", headers=ADMIN_HEADERS)
        ).json()
        client_rows = [
            row for row in budget["entries"] if row["scope"] == QUOTA_SCOPE_CLIENT
        ]
        assert len(client_rows) == 1
        assert client_rows[0]["used"] == 5

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals(
                UsageFilter(key_fingerprint=key_fingerprint(CLIENT_KEY))
            )
        assert totals.total_requests == 5

    async def test_failed_upstream_still_consumes_and_is_recorded(
        self, make_harness
    ) -> None:
        """上游失败也占用了配额，并且要留下一条失败记录。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        h = await make_harness(
            handler, settings=build_settings(client_daily_request_quota=10)
        )
        response = await h.client.post(CHAT_URL, json=_payload())
        assert response.status_code == 502

        budget = (
            await h.client.get("/admin/budget", headers=ADMIN_HEADERS)
        ).json()
        used = budget["entries"][0]["used"]
        assert used == 1, "失败的调用同样预扣了配额"

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals(
                UsageFilter(key_fingerprint=key_fingerprint(CLIENT_KEY))
            )
        assert totals.total_requests == 1
        assert totals.failure_count == 1

    async def test_rejected_requests_consume_nothing(self, make_harness) -> None:
        """被配额拒绝的请求既不扣额度也不产生调用记录。"""
        h = await make_harness(
            settings=build_settings(client_daily_request_quota=2)
        )
        await h.client.post(CHAT_URL, json=_payload())
        await h.client.post(CHAT_URL, json=_payload())

        blocked = await h.client.post(CHAT_URL, json=_payload())
        assert blocked.status_code == 429

        budget = (
            await h.client.get("/admin/budget", headers=ADMIN_HEADERS)
        ).json()
        assert budget["entries"][0]["used"] == 2

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()
        assert totals.total_requests == 2, "被拒绝的请求不应落库"

        # 上游只被调用两次
        assert len(h.upstream_requests) == 2

    async def test_rejected_request_does_not_consume_run_quota(
        self, make_harness
    ) -> None:
        """client 维度超限时，run 维度的额度不能被白扣。

        两个维度在同一次预扣里，任一回滚必须一致。
        """
        h = await make_harness(
            settings=build_settings(
                client_daily_request_quota=1, run_max_llm_calls=10
            )
        )
        headers = {"X-Run-Id": "consistency-run"}

        first = await h.client.post(CHAT_URL, json=_payload(), headers=headers)
        assert first.status_code == 200

        blocked = await h.client.post(CHAT_URL, json=_payload(), headers=headers)
        assert blocked.status_code == 429

        budget = (
            await h.client.get("/admin/budget", headers=ADMIN_HEADERS)
        ).json()
        run_rows = [r for r in budget["entries"] if r["scope"] == QUOTA_SCOPE_RUN]
        assert len(run_rows) == 1
        assert run_rows[0]["used"] == 1, "被拒绝的请求不应计入 run 额度"

    async def test_retry_counts_once_in_quota(self, make_harness) -> None:
        """一次逻辑请求即使多次上游尝试，也只占一个配额单位。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        h = await make_harness(
            handler,
            settings=build_settings(
                client_daily_request_quota=10,
                retry_max_attempts=3,
                retry_backoff_base_seconds=0.01,
            ),
        )
        response = await h.client.post(CHAT_URL, json=_payload())
        assert response.status_code == 502
        assert len(h.upstream_requests) == 3, "确实重试了 3 次"

        budget = (
            await h.client.get("/admin/budget", headers=ADMIN_HEADERS)
        ).json()
        assert budget["entries"][0]["used"] == 1, "配额按逻辑请求计数，不按尝试次数"

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()
        assert totals.total_requests == 1
        assert totals.records_with_retries == 1


class TestRunIsolationConsistency:
    async def test_same_run_id_across_clients_stays_separate(self, harness) -> None:
        run_id = "iso-run"
        headers = {"X-Run-Id": run_id}

        await harness.client.post(CHAT_URL, json=_payload(), headers=headers)

        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(
                NewCallRecord(
                    request_id="other-client-row",
                    provider="deepseek",
                    model="deepseek-chat",
                    latency_ms=20,
                    http_status=200,
                    error_type="none",
                    prompt_tokens=3,
                    completion_tokens=3,
                    key_fingerprint="fingerprint-of-another-client",
                    run_id=run_id,
                )
            )
            await session.commit()

            mine = await repository.usage_totals(
                UsageFilter(
                    run_id=run_id, key_fingerprint=key_fingerprint(CLIENT_KEY)
                )
            )
            theirs = await repository.usage_totals(
                UsageFilter(
                    run_id=run_id, key_fingerprint="fingerprint-of-another-client"
                )
            )

        assert mine.total_requests == 1
        assert theirs.total_requests == 1
        assert mine.prompt_tokens != theirs.prompt_tokens

    async def test_quota_isolates_same_run_id_across_clients(
        self, make_harness
    ) -> None:
        """配额层面同样隔离：一个客户端用尽不影响另一个。"""
        h = await make_harness(settings=build_settings(run_max_llm_calls=1))
        run_id = "shared-run"

        first = await h.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": run_id}
        )
        assert first.status_code == 200

        second = await h.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": run_id}
        )
        assert second.status_code == 429

        async with h.session_factory() as session:
            from app.db.budget_repository import BudgetRepository

            rows = await BudgetRepository(session).list_usage(
                window_start=__import__("datetime").date(1970, 1, 1),
                scope=QUOTA_SCOPE_RUN,
            )

        # 身份键包含客户端指纹，因此不会与其他客户端撞车
        assert len(rows) == 1
        assert rows[0].identity.startswith(key_fingerprint(CLIENT_KEY))


class TestUnknownValuesStayUnknown:
    async def test_unknown_usage_excluded_from_totals(self, make_harness) -> None:
        """usage 缺失时，Token 与成本记为未知而不是 0。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=SAMPLE_COMPLETION_WITHOUT_USAGE)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload())
        assert response.status_code == 200

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()

        assert totals.total_requests == 1
        assert totals.total_tokens == 0
        assert totals.records_with_unknown_tokens == 1
        assert totals.records_without_cost == 1
        assert totals.estimated_cost_usd == Decimal("0")

    async def test_known_and_unknown_are_reported_separately(
        self, make_harness
    ) -> None:
        """混合场景：已知部分照常累加，未知部分单独计数。"""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=SAMPLE_COMPLETION_WITHOUT_USAGE)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "deepseek-chat",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        h = await make_harness(handler)
        await h.client.post(CHAT_URL, json=_payload())
        await h.client.post(CHAT_URL, json=_payload())

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()

        assert totals.total_requests == 2
        assert totals.total_tokens == 15, "只累加已知的 10 + 5"
        assert totals.records_with_unknown_tokens == 1

    async def test_unpriced_model_reports_unknown_cost(self, make_harness) -> None:
        """未配置价格的模型，其成本必须是未知而不是 0 元。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            )

        h = await make_harness(
            handler,
            settings=build_settings(
                openai_price_input_per_million_usd=0,
                openai_price_output_per_million_usd=0,
            ),
        )
        await h.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()

        assert totals.total_tokens == 150
        assert totals.records_without_cost == 1
        assert totals.estimated_cost_usd == Decimal("0")


class TestFailureBookkeeping:
    async def test_every_error_type_has_an_origin(self) -> None:
        """每个错误类型都要有归属；只有「成功」没有归属。"""
        from app.errors import ErrorType as ET
        from app.errors import error_origin

        for error_type in ET:
            origin = error_origin(error_type)
            if error_type is ET.NONE:
                assert origin is None, "成功不是故障，不应有归属"
            else:
                assert origin is not None, f"{error_type.value} 缺少归属登记"

    async def test_error_breakdown_matches_status_counts(self, make_harness) -> None:
        """错误分布之和应等于失败请求数。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        for _ in range(3):
            await h.client.post(CHAT_URL, json=_payload())

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()
            errors = await CallRecordRepository(session).error_breakdown()

        assert totals.failure_count == 3
        assert sum(item.count for item in errors) == totals.failure_count
        assert errors[0].error_type == ErrorType.UPSTREAM_TIMEOUT.value

    async def test_latency_includes_all_retry_attempts(self, make_harness) -> None:
        """重试的总耗时记在一条记录里，不拆成多条。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        h = await make_harness(
            handler,
            settings=build_settings(
                retry_max_attempts=3, retry_backoff_base_seconds=0.05
            ),
        )
        await h.client.post(CHAT_URL, json=_payload())

        async with h.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals()

        assert totals.total_requests == 1
        # 两次退避至少 0.05 + 0.1 = 150ms
        assert totals.avg_latency_ms is not None
        assert totals.avg_latency_ms >= 150


class TestRunAggregationConsistency:
    async def test_agent_totals_sum_to_run_total(self, harness) -> None:
        """按节点聚合的各分量之和必须等于 run 总量。"""
        run_id = "sum-run"
        for agent, times in (("planner", 1), ("executor", 3), ("critic", 2)):
            for _ in range(times):
                await harness.client.post(
                    CHAT_URL,
                    json=_payload(),
                    headers={"X-Run-Id": run_id, "X-Agent-Name": agent},
                )

        body = (
            await harness.client.get(
                f"/admin/runs/{run_id}", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 6
        assert (
            sum(item["total_requests"] for item in body["by_agent"]) == 6
        )
        assert sum(item["total_tokens"] for item in body["by_agent"]) == body[
            "totals"
        ]["total_tokens"]
        assert Decimal(
            sum(Decimal(item["estimated_cost_usd"]) for item in body["by_agent"])
        ) == Decimal(body["totals"]["estimated_cost_usd"])

    async def test_run_total_matches_provider_breakdown(self, harness) -> None:
        run_id = "provider-sum"
        await harness.client.post(
            CHAT_URL, json=_payload("deepseek-chat"), headers={"X-Run-Id": run_id}
        )
        await harness.client.post(
            CHAT_URL, json=_payload("gpt-4o-mini"), headers={"X-Run-Id": run_id}
        )

        body = (
            await harness.client.get(
                f"/admin/runs/{run_id}", headers=ADMIN_HEADERS
            )
        ).json()

        assert sum(
            item["total_requests"] for item in body["by_provider"]
        ) == body["totals"]["total_requests"]
        assert sum(
            item["total_requests"] for item in body["by_model"]
        ) == body["totals"]["total_requests"]

    async def test_runs_list_totals_match_detail(self, harness) -> None:
        """列表页的汇总必须与详情页一致，否则分页会误导。"""
        run_id = "list-detail-run"
        for _ in range(3):
            await harness.client.post(
                CHAT_URL, json=_payload(), headers={"X-Run-Id": run_id}
            )

        listing = (await harness.client.get("/admin/runs", headers=ADMIN_HEADERS)).json()
        entry = next(r for r in listing["runs"] if r["run_id"] == run_id)
        detail = (
            await harness.client.get(
                f"/admin/runs/{run_id}", headers=ADMIN_HEADERS
            )
        ).json()

        assert entry["request_count"] == detail["totals"]["total_requests"]
        assert entry["total_tokens"] == detail["totals"]["total_tokens"]
        assert Decimal(entry["estimated_cost_usd"]) == Decimal(
            detail["totals"]["estimated_cost_usd"]
        )
