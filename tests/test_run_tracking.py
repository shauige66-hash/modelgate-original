"""Run 级可观测性测试。

一次 Agent 工作流由多个节点共享 run_id 组成，这里验证聚合视图能回答：
谁在跑（client_app）、各节点申请了多少次调用（by_agent）、
花了多少（Token/成本）、失败在哪（errors）。

关键约束：run_id 由调用方自报，不同客户端可能同名，
因此统计必须能用客户端身份隔离，不能把两者合并。
"""

from __future__ import annotations

from decimal import Decimal

import httpx

from app.db.repository import CallRecordRepository, NewCallRecord, UsageFilter
from app.errors import ErrorType
from tests.conftest import CLIENT_KEY

CHAT_URL = "/v1/chat/completions"
RUNS_URL = "/admin/runs"
ADMIN_HEADERS = {"X-Admin-Key": "sk-admin-test-key"}


def _payload(model: str = "deepseek-chat") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _tags(run_id: str, agent: str, app: str = "insightflow") -> dict:
    return {
        "X-Client-App": app,
        "X-Agent-Name": agent,
        "X-Run-Id": run_id,
    }


class TestRunAggregation:
    async def test_agent_distribution_within_one_run(self, harness) -> None:
        """同一 run 下三个节点的调用分布。"""
        run_id = "run-dist-1"
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags(run_id, "planner")
        )
        for _ in range(2):
            await harness.client.post(
                CHAT_URL, json=_payload(), headers=_tags(run_id, "executor")
            )
        for _ in range(3):
            await harness.client.post(
                CHAT_URL, json=_payload(), headers=_tags(run_id, "critic")
            )

        response = await harness.client.get(
            f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS
        )

        assert response.status_code == 200
        body = response.json()
        assert body["totals"]["total_requests"] == 6

        by_agent = {item["key"]: item["total_requests"] for item in body["by_agent"]}
        assert by_agent == {"planner": 1, "executor": 2, "critic": 3}

    async def test_unlabeled_calls_are_grouped_not_dropped(self, harness) -> None:
        """没有 agent_name 的调用归入 unlabeled，否则各节点之和对不上总数。"""
        run_id = "run-unlabeled"
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags(run_id, "planner")
        )
        # 同一 run 但不带 agent 标签
        await harness.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": run_id}
        )

        body = (
            await harness.client.get(f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS)
        ).json()

        by_agent = {item["key"]: item["total_requests"] for item in body["by_agent"]}
        assert by_agent == {"planner": 1, "unlabeled": 1}
        assert sum(by_agent.values()) == body["totals"]["total_requests"]

    async def test_provider_and_model_breakdown(self, harness) -> None:
        run_id = "run-providers"
        await harness.client.post(
            CHAT_URL, json=_payload("deepseek-chat"), headers=_tags(run_id, "planner")
        )
        await harness.client.post(
            CHAT_URL, json=_payload("gpt-4o-mini"), headers=_tags(run_id, "planner")
        )

        body = (
            await harness.client.get(f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS)
        ).json()

        providers = {item["key"] for item in body["by_provider"]}
        models = {item["key"] for item in body["by_model"]}
        assert providers == {"deepseek", "openai"}
        assert models == {"deepseek-chat", "gpt-4o-mini"}

    async def test_costs_and_tokens_summed(self, harness) -> None:
        run_id = "run-cost"
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags(run_id, "planner")
        )
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags(run_id, "executor")
        )

        body = (
            await harness.client.get(f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS)
        ).json()

        # 每次 12 输入 + 8 输出
        assert body["totals"]["prompt_tokens"] == 24
        assert body["totals"]["completion_tokens"] == 16
        # deepseek 0.20 / 0.80 -> 每次 0.000009，两次 0.000018
        assert Decimal(body["totals"]["estimated_cost_usd"]) == Decimal("0.000018")

    async def test_error_distribution(self, make_harness) -> None:
        """失败调用也要能按 run 聚合出来。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        run_id = "run-errors"
        await h.client.post(CHAT_URL, json=_payload(), headers=_tags(run_id, "planner"))

        body = (
            await h.client.get(f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS)
        ).json()

        assert body["totals"]["failure_count"] == 1
        assert body["errors"] == [
            {"error_type": ErrorType.UPSTREAM_TIMEOUT.value, "count": 1}
        ]

    async def test_unknown_tokens_stay_unknown(self, make_harness) -> None:
        """usage 缺失时不得按 0 计入总量。"""
        from tests.conftest import SAMPLE_COMPLETION_WITHOUT_USAGE

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=SAMPLE_COMPLETION_WITHOUT_USAGE)

        h = await make_harness(handler)
        run_id = "run-unknown"
        await h.client.post(CHAT_URL, json=_payload(), headers=_tags(run_id, "planner"))

        body = (
            await h.client.get(f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS)
        ).json()

        assert body["totals"]["total_tokens"] == 0
        assert body["data_quality"]["records_with_unknown_tokens"] == 1
        assert body["data_quality"]["records_without_cost"] == 1

    async def test_empty_run_returns_zero_totals(self, harness) -> None:
        body = (
            await harness.client.get(
                f"{RUNS_URL}/run-does-not-exist", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 0
        assert body["by_agent"] == []
        assert Decimal(body["totals"]["estimated_cost_usd"]) == Decimal("0")


class TestRunIsolation:
    async def test_same_run_id_different_clients_not_merged(self, harness) -> None:
        """不同客户端使用同一个 run_id 时，必须能按指纹区分。"""
        run_id = "run-shared"

        # 用当前客户端产生一条
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags(run_id, "planner")
        )

        # 直接写入一条属于另一个客户端指纹的记录，模拟第二个调用方
        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(
                NewCallRecord(
                    request_id="other-client-1",
                    provider="deepseek",
                    model="deepseek-chat",
                    requested_model="deepseek-chat",
                    latency_ms=50,
                    http_status=200,
                    error_type="none",
                    prompt_tokens=5,
                    completion_tokens=5,
                    key_fingerprint="other-client-fingerprint",
                    run_id=run_id,
                    agent_name="intruder",
                )
            )
            await session.commit()

        # 不带指纹：两个客户端的同名 run 会被合并
        merged = (
            await harness.client.get(f"{RUNS_URL}/{run_id}", headers=ADMIN_HEADERS)
        ).json()
        assert merged["totals"]["total_requests"] == 2
        assert merged["client_scope"] == "all_clients"

        # 带指纹：只统计该客户端
        from app.security import key_fingerprint

        scoped = (
            await harness.client.get(
                f"{RUNS_URL}/{run_id}?key_fingerprint={key_fingerprint(CLIENT_KEY)}",
                headers=ADMIN_HEADERS,
            )
        ).json()
        assert scoped["totals"]["total_requests"] == 1
        agents = {item["key"] for item in scoped["by_agent"]}
        assert agents == {"planner"}

    async def test_repository_filter_by_fingerprint(self, harness) -> None:
        run_id = "run-repo-filter"
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags(run_id, "planner")
        )
        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            await repository.insert(
                NewCallRecord(
                    request_id="other-2",
                    provider="deepseek",
                    model="deepseek-chat",
                    latency_ms=10,
                    http_status=200,
                    error_type="none",
                    key_fingerprint="someone-else",
                    run_id=run_id,
                )
            )
            await session.commit()

            scoped = await repository.usage_totals(
                UsageFilter(run_id=run_id, key_fingerprint="someone-else")
            )

        assert scoped.total_requests == 1


class TestRunListPagination:
    async def test_list_returns_runs_newest_first(self, harness) -> None:
        for index in range(3):
            await harness.client.post(
                CHAT_URL, json=_payload(), headers=_tags(f"run-list-{index}", "planner")
            )

        body = (await harness.client.get(RUNS_URL, headers=ADMIN_HEADERS)).json()

        assert body["total"] == 3
        assert len(body["runs"]) == 3
        assert body["has_more"] is False

    async def test_pagination_slices_results(self, harness) -> None:
        for index in range(5):
            await harness.client.post(
                CHAT_URL,
                json=_payload(),
                headers=_tags(f"run-page-{index}", "planner"),
            )

        first = (
            await harness.client.get(
                f"{RUNS_URL}?limit=2&offset=0", headers=ADMIN_HEADERS
            )
        ).json()
        second = (
            await harness.client.get(
                f"{RUNS_URL}?limit=2&offset=2", headers=ADMIN_HEADERS
            )
        ).json()

        assert first["total"] == 5
        assert first["has_more"] is True
        assert len(first["runs"]) == 2
        assert len(second["runs"]) == 2

        # 分页之间不重叠
        ids_first = {item["run_id"] for item in first["runs"]}
        ids_second = {item["run_id"] for item in second["runs"]}
        assert not (ids_first & ids_second)

    async def test_runs_without_run_id_are_excluded(self, harness) -> None:
        """没有 run_id 的调用不属于任何工作流。"""
        await harness.client.post(CHAT_URL, json=_payload())

        body = (await harness.client.get(RUNS_URL, headers=ADMIN_HEADERS)).json()

        assert body["total"] == 0

    async def test_client_app_filter(self, harness) -> None:
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags("run-app-a", "planner", "app-a")
        )
        await harness.client.post(
            CHAT_URL, json=_payload(), headers=_tags("run-app-b", "planner", "app-b")
        )

        body = (
            await harness.client.get(
                f"{RUNS_URL}?client_app=app-a", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["total"] == 1
        assert body["runs"][0]["run_id"] == "run-app-a"


class TestRunEndpointsAuth:
    async def test_runs_list_requires_admin_key(self, harness) -> None:
        assert (await harness.client.get(RUNS_URL)).status_code == 401

    async def test_run_detail_requires_admin_key(self, harness) -> None:
        assert (await harness.client.get(f"{RUNS_URL}/any")).status_code == 401

    async def test_client_key_does_not_grant_admin_access(self, harness) -> None:
        response = await harness.client.get(
            RUNS_URL, headers={"Authorization": f"Bearer {CLIENT_KEY}"}
        )

        assert response.status_code == 401
