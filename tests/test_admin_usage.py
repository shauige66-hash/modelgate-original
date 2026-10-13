"""GET /admin/usage 统计接口测试。"""

from __future__ import annotations

from decimal import Decimal

import httpx

from app.db.repository import CallRecordRepository, NewCallRecord
from app.errors import ErrorType

ADMIN_URL = "/admin/usage"
CHAT_URL = "/v1/chat/completions"

ADMIN_HEADERS = {"X-Admin-Key": "sk-admin-test-key"}


def _payload(content: str = "hi") -> dict:
    return {"messages": [{"role": "user", "content": content}]}


async def _seed(harness, count: int = 1) -> None:
    """跑 count 次成功的代理请求，产生真实记录。"""
    for index in range(count):
        response = await harness.client.post(CHAT_URL, json=_payload(f"hi {index}"))
        assert response.status_code == 200


class TestUsageEndpointBasics:
    async def test_empty_database_returns_zero_totals(self, harness) -> None:
        response = await harness.client.get(ADMIN_URL, headers=ADMIN_HEADERS)

        assert response.status_code == 200
        body = response.json()
        assert body["totals"]["total_requests"] == 0
        assert body["totals"]["success_count"] == 0
        assert body["totals"]["failure_count"] == 0
        assert body["totals"]["total_tokens"] == 0
        assert body["by_model"] == []
        assert body["errors"] == []

    async def test_counts_requests_and_tokens(self, harness) -> None:
        await _seed(harness, 2)

        response = await harness.client.get(ADMIN_URL, headers=ADMIN_HEADERS)
        body = response.json()

        assert body["totals"]["total_requests"] == 2
        assert body["totals"]["success_count"] == 2
        assert body["totals"]["failure_count"] == 0
        # 每次 12 输入 + 8 输出
        assert body["totals"]["prompt_tokens"] == 24
        assert body["totals"]["completion_tokens"] == 16
        assert body["totals"]["total_tokens"] == 40

    async def test_cost_is_a_decimal_string(self, harness) -> None:
        """金额以字符串返回，避免被 JSON 解析成不精确的浮点。"""
        await _seed(harness, 1)

        body = (await harness.client.get(ADMIN_URL, headers=ADMIN_HEADERS)).json()

        cost = body["totals"]["estimated_cost_usd"]
        assert isinstance(cost, str)
        # 12 * 0.27/1e6 + 8 * 1.10/1e6 = 0.00000324 + 0.0000088
        # -> 量化到 6 位小数：0.000012
        assert Decimal(cost) == Decimal("0.000012")
        assert body["totals"]["price_unit"] == "USD per 1,000,000 tokens"

    async def test_failure_count_and_error_breakdown(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        await h.client.post(CHAT_URL, json=_payload())

        body = (await h.client.get(ADMIN_URL, headers=ADMIN_HEADERS)).json()

        assert body["totals"]["total_requests"] == 1
        assert body["totals"]["failure_count"] == 1
        assert body["totals"]["success_count"] == 0
        assert body["errors"] == [
            {"error_type": ErrorType.UPSTREAM_TIMEOUT.value, "count": 1}
        ]

    async def test_data_quality_flags_unknown_tokens(self, make_harness) -> None:
        """上游不返回 usage 时，必须通过 data_quality 提示统计不完整。"""
        from tests.conftest import SAMPLE_COMPLETION_WITHOUT_USAGE

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=SAMPLE_COMPLETION_WITHOUT_USAGE)

        h = await make_harness(handler)
        await h.client.post(CHAT_URL, json=_payload())

        body = (await h.client.get(ADMIN_URL, headers=ADMIN_HEADERS)).json()

        assert body["totals"]["total_requests"] == 1
        assert body["totals"]["total_tokens"] == 0
        assert body["data_quality"]["records_with_unknown_tokens"] == 1
        assert body["data_quality"]["records_without_cost"] == 1


class TestModelFilterIsConsistent:
    """model 筛选必须同时作用于 totals / by_model / errors。

    修复前：只有 totals 被筛选，by_model 与 errors 返回的是全量数据，
    导致同一份响应里三组数字口径不一致。

    注意：不能用 HTTP 层造数据 —— 上游 mock 固定返回 model=deepseek-chat，
    请求里指定的 model 会被上游返回值覆盖。因此这里直接写仓储层，
    精确控制每条记录的 model 与 error_type。
    """

    async def _seed_multi_model(self, harness) -> None:
        """写入两个模型的数据，各自带成功与失败记录。

        deepseek-chat     : 2 条（1 成功 1 失败，timeout）
        deepseek-reasoner : 3 条（2 成功 1 失败，http_error）
        """
        from decimal import Decimal

        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)

            def record(
                request_id: str,
                model: str,
                *,
                ok: bool,
                error_type: str = "none",
                prompt_tokens: int | None = 10,
            ) -> NewCallRecord:
                return NewCallRecord(
                    request_id=request_id,
                    model=model,
                    latency_ms=100,
                    http_status=200 if ok else 502,
                    error_type=error_type,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=5 if prompt_tokens is not None else None,
                    estimated_cost_usd=(
                        Decimal("0.000010") if prompt_tokens is not None else None
                    ),
                    message_count=1,
                    prompt_chars=10,
                )

            await repository.insert(
                record("req_chat_ok", "deepseek-chat", ok=True)
            )
            await repository.insert(
                record(
                    "req_chat_fail",
                    "deepseek-chat",
                    ok=False,
                    error_type="upstream_timeout",
                )
            )
            await repository.insert(
                record("req_reason_ok1", "deepseek-reasoner", ok=True)
            )
            await repository.insert(
                record("req_reason_ok2", "deepseek-reasoner", ok=True)
            )
            await repository.insert(
                record(
                    "req_reason_fail",
                    "deepseek-reasoner",
                    ok=False,
                    error_type="upstream_http_error",
                )
            )
            await session.commit()

    async def test_filter_narrows_totals(self, harness) -> None:
        await self._seed_multi_model(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?model=deepseek-chat", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 2
        assert body["totals"]["success_count"] == 1
        assert body["totals"]["failure_count"] == 1
        assert body["range"]["model_filter"] == "deepseek-chat"

    async def test_filter_narrows_by_model(self, harness) -> None:
        """by_model 必须只包含被筛选的模型（修复前会返回全部模型）。"""
        await self._seed_multi_model(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?model=deepseek-chat", headers=ADMIN_HEADERS
            )
        ).json()

        assert len(body["by_model"]) == 1
        assert body["by_model"][0]["model"] == "deepseek-chat"
        assert body["by_model"][0]["total_requests"] == 2
        # 不允许出现另一个模型
        assert all(item["model"] != "deepseek-reasoner" for item in body["by_model"])

    async def test_filter_narrows_errors(self, harness) -> None:
        """errors 必须只包含被筛选模型的错误（修复前会混入其他模型的错误）。"""
        await self._seed_multi_model(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?model=deepseek-chat", headers=ADMIN_HEADERS
            )
        ).json()

        error_types = {item["error_type"] for item in body["errors"]}
        assert error_types == {"upstream_timeout"}
        assert "upstream_http_error" not in error_types

    async def test_all_three_sections_agree(self, harness) -> None:
        """三处口径必须一致：totals 的失败数 = errors 总数 = by_model 失败数。"""
        await self._seed_multi_model(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?model=deepseek-reasoner", headers=ADMIN_HEADERS
            )
        ).json()

        errors_total = sum(item["count"] for item in body["errors"])
        by_model = body["by_model"]

        assert len(by_model) == 1
        assert body["totals"]["total_requests"] == 3
        assert body["totals"]["failure_count"] == 1
        assert errors_total == body["totals"]["failure_count"]
        assert by_model[0]["total_requests"] == body["totals"]["total_requests"]
        assert by_model[0]["failure_count"] == body["totals"]["failure_count"]

    async def test_unfiltered_request_returns_all_models(self, harness) -> None:
        """不加筛选时，by_model 应包含全部模型，errors 汇总全部错误。"""
        await self._seed_multi_model(harness)

        body = (await harness.client.get(ADMIN_URL, headers=ADMIN_HEADERS)).json()

        models = {item["model"] for item in body["by_model"]}
        assert models == {"deepseek-chat", "deepseek-reasoner"}
        assert body["totals"]["total_requests"] == 5

        error_types = {item["error_type"] for item in body["errors"]}
        assert error_types == {"upstream_timeout", "upstream_http_error"}

    async def test_filter_matching_nothing_returns_empty_sections(
        self, harness
    ) -> None:
        """筛选一个不存在的模型：三处都应清空，而不是回落到全量。"""
        await self._seed_multi_model(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?model=no-such-model", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 0
        assert body["by_model"] == []
        assert body["errors"] == []


class TestUsageEndpointGrouping:
    async def test_by_model_splits_records(self, harness) -> None:
        """按模型分组的记录数之和应等于总调用数。"""
        # 直接经 HTTP 层生成两条记录（其中一条显式指定模型）
        await harness.client.post(
            CHAT_URL, json={"model": "deepseek-reasoner", **_payload()}
        )
        await harness.client.post(CHAT_URL, json=_payload())

        body = (await harness.client.get(ADMIN_URL, headers=ADMIN_HEADERS)).json()

        # 上游 mock 固定返回 model=deepseek-chat，
        # 因此网关会以「上游实际返回的模型」为准归类
        assert len(body["by_model"]) >= 1
        assert sum(item["total_requests"] for item in body["by_model"]) == 2

    async def test_group_by_model_false_returns_empty_list(self, harness) -> None:
        await _seed(harness, 1)

        response = await harness.client.get(
            f"{ADMIN_URL}?group_by_model=false", headers=ADMIN_HEADERS
        )

        assert response.json()["by_model"] == []

    async def test_top_errors_zero_returns_empty_list(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        await h.client.post(CHAT_URL, json=_payload())

        response = await h.client.get(f"{ADMIN_URL}?top_errors=0", headers=ADMIN_HEADERS)

        assert response.json()["errors"] == []


class TestUsageEndpointTimeRange:
    async def test_default_range_is_last_24_hours(self, harness) -> None:
        response = await harness.client.get(ADMIN_URL, headers=ADMIN_HEADERS)

        body = response.json()
        assert body["range"]["since"] is not None
        assert body["range"]["until"] is None

    async def test_future_since_excludes_everything(self, harness) -> None:
        await _seed(harness, 1)

        response = await harness.client.get(
            f"{ADMIN_URL}?since=2999-01-01T00:00:00Z", headers=ADMIN_HEADERS
        )

        assert response.json()["totals"]["total_requests"] == 0

    async def test_past_until_includes_everything(self, harness) -> None:
        await _seed(harness, 1)

        response = await harness.client.get(
            f"{ADMIN_URL}?since=2000-01-01T00:00:00Z&until=2999-01-01T00:00:00Z",
            headers=ADMIN_HEADERS,
        )

        assert response.json()["totals"]["total_requests"] == 1

    async def test_naive_datetime_is_treated_as_utc(self, harness) -> None:
        """不带时区的时间参数不应导致 500（按 UTC 处理）。"""
        await _seed(harness, 1)

        response = await harness.client.get(
            f"{ADMIN_URL}?since=2000-01-01T00:00:00", headers=ADMIN_HEADERS
        )

        assert response.status_code == 200
        assert response.json()["totals"]["total_requests"] == 1
