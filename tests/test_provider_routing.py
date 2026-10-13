"""多服务商模型路由测试。

覆盖：按 model 字段选择服务商、两个服务商的凭证完全隔离、
未登记模型返回 400、服务商未配置时返回 503、
以及价格按服务商/模型取值。所有上游均为 Mock，不产生真实调用。
"""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from app.db.repository import CallRecordRepository
from app.providers import (
    ModelRouter,
    ProviderNotConfiguredError,
    UnknownModelError,
    parse_model_prices,
    parse_model_registry,
)
from tests.conftest import DEEPSEEK_KEY, OPENAI_KEY, SAMPLE_COMPLETION, build_settings

CHAT_URL = "/v1/chat/completions"


def _payload(model: str | None = None) -> dict:
    body: dict = {"messages": [{"role": "user", "content": "hi"}]}
    if model is not None:
        body["model"] = model
    return body


class TestModelToProviderRouting:
    async def test_deepseek_model_reaches_deepseek(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))

        assert response.status_code == 200
        assert len(harness.requests_for("deepseek")) == 1
        assert harness.requests_for("openai") == []

    async def test_openai_model_reaches_openai(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        assert response.status_code == 200
        assert len(harness.requests_for("openai")) == 1
        assert harness.requests_for("deepseek") == []

    async def test_second_deepseek_model_also_routes_to_deepseek(self, harness) -> None:
        response = await harness.client.post(
            CHAT_URL, json=_payload("deepseek-reasoner")
        )

        assert response.status_code == 200
        assert len(harness.requests_for("deepseek")) == 1
        assert harness.requests_for("openai") == []

    async def test_omitting_model_uses_configured_default(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload())

        assert response.status_code == 200
        assert len(harness.requests_for("deepseek")) == 1

        import json

        sent = json.loads(harness.requests_for("deepseek")[0].content)
        assert sent["model"] == "deepseek-chat"

    async def test_both_providers_recorded_with_correct_provider(self, harness) -> None:
        await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))
        await harness.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            records = await repository.list_recent(limit=10)

        by_model = {record.model: record for record in records}
        assert by_model["deepseek-chat"].provider == "deepseek"
        assert by_model["gpt-4o-mini"].provider == "openai"


class TestCredentialIsolation:
    async def test_each_provider_uses_its_own_api_key(self, harness) -> None:
        await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))
        await harness.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        deepseek_request = harness.requests_for("deepseek")[0]
        openai_request = harness.requests_for("openai")[0]

        assert deepseek_request.headers["Authorization"] == f"Bearer {DEEPSEEK_KEY}"
        assert openai_request.headers["Authorization"] == f"Bearer {OPENAI_KEY}"

    async def test_keys_are_not_interchanged(self, harness) -> None:
        """任一方向的错配都要被发现。"""
        await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))
        await harness.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        deepseek_auth = harness.requests_for("deepseek")[0].headers["Authorization"]
        openai_auth = harness.requests_for("openai")[0].headers["Authorization"]

        assert OPENAI_KEY not in deepseek_auth
        assert DEEPSEEK_KEY not in openai_auth

    async def test_each_provider_uses_its_own_base_url(self, harness) -> None:
        await harness.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        request = harness.requests_for("openai")[0]
        assert request.url.host == "openai.test"
        assert request.url.scheme == "https"
        assert request.url.path == "/chat/completions"

    async def test_client_key_never_forwarded_upstream(self, harness) -> None:
        """调用方的网关 Key 不得出现在发往上游的请求里。"""
        from tests.conftest import CLIENT_KEY

        await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))

        for request in harness.upstream_requests:
            assert CLIENT_KEY not in request.headers.get("Authorization", "")


class TestUnknownModelRejected:
    async def test_unknown_model_returns_400(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload("gpt-5-turbo"))

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "unknown_model"

    async def test_unknown_model_never_reaches_any_provider(self, harness) -> None:
        """不得静默回落到默认服务商。"""
        await harness.client.post(CHAT_URL, json=_payload("llama-3-70b"))

        assert harness.upstream_requests == []

    async def test_error_lists_available_models(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload("nope"))

        message = response.json()["error"]["message"]
        assert "deepseek-chat" in message
        assert "gpt-4o-mini" in message

    async def test_unknown_model_failure_is_persisted(self, harness) -> None:
        """被拒绝的请求不应写入调用记录，因为它从未占用上游资源。

        这里锁定当前行为：拒绝发生在路由阶段，记录为空。
        """
        response = await harness.client.post(CHAT_URL, json=_payload("unknown-x"))

        assert response.status_code == 400

        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            records = await repository.list_recent(limit=10)

        assert records == []

    async def test_model_name_cannot_inject_url(self, harness) -> None:
        """模型名不参与 URL 构造，带路径的模型名同样被拒绝。"""
        response = await harness.client.post(
            CHAT_URL, json=_payload("../../admin/usage")
        )

        assert response.status_code == 400
        assert harness.upstream_requests == []


class TestProviderNotConfigured:
    async def test_unconfigured_provider_returns_503(self, make_harness) -> None:
        h = await make_harness(settings=build_settings(openai_api_key=""))

        response = await h.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        assert response.status_code == 503
        assert (
            response.json()["error"]["type"] == "upstream_not_configured"
        )
        assert h.upstream_requests == []

    async def test_other_provider_still_works(self, make_harness) -> None:
        h = await make_harness(settings=build_settings(openai_api_key=""))

        response = await h.client.post(CHAT_URL, json=_payload("deepseek-chat"))

        assert response.status_code == 200


class TestPerModelPricing:
    async def test_price_comes_from_owning_provider(self, harness) -> None:
        """deepseek 与 openai 的单价不同，成本必须按各自价格算。"""
        await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))
        await harness.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            records = await repository.list_recent(limit=10)

        by_model = {record.model: record for record in records}
        # SAMPLE_COMPLETION: 12 输入 + 8 输出
        # deepseek: 12*0.20/1e6 + 8*0.80/1e6 = 0.0000088 -> 0.000009
        # openai  : 12*1.50/1e6 + 8*6.00/1e6 = 0.000066  -> 0.000066
        assert by_model["deepseek-chat"].estimated_cost_usd == Decimal("0.000009")
        assert by_model["gpt-4o-mini"].estimated_cost_usd == Decimal("0.000066")

    async def test_model_price_overrides_provider_price(self) -> None:
        settings = build_settings(
            model_prices=["deepseek-chat:1:2"],
            deepseek_price_input_per_million_usd=Decimal("0.20"),
            deepseek_price_output_per_million_usd=Decimal("0.80"),
        )
        router = ModelRouter(settings)

        route = router.resolve("deepseek-chat")
        assert route.price_input_per_million_usd == Decimal("1")
        assert route.price_output_per_million_usd == Decimal("2")

        # 同一服务商的另一个模型仍用服务商默认价
        other = router.resolve("deepseek-reasoner")
        assert other.price_input_per_million_usd == Decimal("0.20")

    async def test_unpriced_model_stores_null_cost(self, make_harness) -> None:
        """价格未配置时成本必须是 NULL，不能伪造为 0。"""
        settings = build_settings(
            deepseek_price_input_per_million_usd=Decimal("0"),
            deepseek_price_output_per_million_usd=Decimal("0"),
        )
        h = await make_harness(settings=settings)

        response = await h.client.post(CHAT_URL, json=_payload("deepseek-chat"))
        assert response.status_code == 200

        request_id = response.headers["X-Request-Id"]
        async with h.session_factory() as session:
            repository = CallRecordRepository(session)
            record = await repository.get_by_request_id(request_id)

        assert record is not None
        # Token 已知，但价格未知 -> 成本未知
        assert record.prompt_tokens == 12
        assert record.estimated_cost_usd is None

    async def test_unknown_priced_model_reports_unknown_cost(
        self, make_harness
    ) -> None:
        settings = build_settings(
            model_registry=["custom-model:openai"],
            default_model_name="custom-model",
            openai_price_input_per_million_usd=Decimal("0"),
            openai_price_output_per_million_usd=Decimal("0"),
        )
        h = await make_harness(settings=settings)

        response = await h.client.post(CHAT_URL, json=_payload("custom-model"))
        assert response.status_code == 200

        request_id = response.headers["X-Request-Id"]
        async with h.session_factory() as session:
            repository = CallRecordRepository(session)
            record = await repository.get_by_request_id(request_id)

        assert record is not None
        assert record.estimated_cost_usd is None
        assert record.provider == "openai"


class TestRegistryConfiguration:
    def test_custom_model_can_be_registered(self) -> None:
        settings = build_settings(
            model_registry=["my-model:openai"],
            default_model_name="my-model",
        )
        router = ModelRouter(settings)

        route = router.resolve("my-model")
        assert route.provider == "openai"
        assert router.default_model == "my-model"

    def test_registry_parser_rejects_malformed_entry(self) -> None:
        with pytest.raises(ValueError):
            parse_model_registry(["deepseek-chat"])
        with pytest.raises(ValueError):
            parse_model_registry([":deepseek"])

    def test_price_parser_rejects_malformed_entry(self) -> None:
        with pytest.raises(ValueError):
            parse_model_prices(["deepseek-chat:1"])
        with pytest.raises(ValueError):
            parse_model_prices(["deepseek-chat:abc:2"])

    def test_unknown_provider_in_registry_is_rejected(self) -> None:
        settings = build_settings(model_registry=["mystery-model:bedrock"])

        with pytest.raises(ValueError):
            ModelRouter(settings)

    def test_default_model_must_be_registered(self) -> None:
        settings = build_settings(default_model_name="not-registered")

        with pytest.raises(ValueError):
            ModelRouter(settings)

    def test_router_reports_registered_models(self) -> None:
        router = ModelRouter(build_settings())

        assert router.registered_models == [
            "deepseek-chat",
            "deepseek-reasoner",
            "gpt-4o-mini",
        ]
        assert router.providers == ["deepseek", "openai"]

    def test_resolve_raises_for_unknown_model(self) -> None:
        router = ModelRouter(build_settings())

        with pytest.raises(UnknownModelError):
            router.resolve("not-a-model")

    def test_client_for_raises_when_provider_unconfigured(self, harness) -> None:
        route = harness.pool.resolve("deepseek-chat")
        harness.pool.router._credentials["deepseek"] = ("https://x", "")

        with pytest.raises(ProviderNotConfiguredError):
            harness.pool.client_for(route)


class TestResponseModelTracking:
    async def test_requested_and_response_model_recorded(self, harness) -> None:
        """上游可能返回与请求不同的模型名，两者都要留存。"""
        response = await harness.client.post(CHAT_URL, json=_payload("deepseek-chat"))
        request_id = response.headers["X-Request-Id"]

        async with harness.session_factory() as session:
            repository = CallRecordRepository(session)
            record = await repository.get_by_request_id(request_id)

        assert record is not None
        assert record.requested_model == "deepseek-chat"
        # Mock 返回的 payload 里 model 固定是 deepseek-chat
        assert record.response_model == SAMPLE_COMPLETION["model"]
        # 历史列保持可用
        assert record.model == "deepseek-chat"


class TestProviderErrorsStillTranslated:
    @pytest.mark.parametrize("status_code", [401, 429, 500])
    async def test_upstream_error_from_openai_is_translated(
        self, make_harness, status_code: int
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, json={"error": {"message": "nope"}})

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        assert response.status_code == 502

    async def test_timeout_from_openai_returns_504(self, make_harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))

        assert response.status_code == 504

    async def test_failed_provider_call_records_provider(self, make_harness) -> None:
        """失败记录同样要带上服务商，否则按服务商统计失败率会失真。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload("gpt-4o-mini"))
        request_id = response.headers["X-Request-Id"]

        async with h.session_factory() as session:
            repository = CallRecordRepository(session)
            record = await repository.get_by_request_id(request_id)

        assert record is not None
        assert record.provider == "openai"
        assert record.http_status == 504
