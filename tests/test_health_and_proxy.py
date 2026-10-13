"""健康检查与基础代理请求测试。"""

from __future__ import annotations

import httpx

from tests.conftest import SAMPLE_COMPLETION, DEEPSEEK_KEY


class TestHealth:
    async def test_health_returns_ok(self, client: httpx.AsyncClient) -> None:
        """存活探针不需要鉴权，且不依赖数据库。"""
        response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_readiness_reports_database_and_concurrency(
        self, client: httpx.AsyncClient
    ) -> None:
        """/health/ready 报告数据库状态、服务商配置与并发水位。"""
        response = await client.get("/health/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database"]["ok"] is True
        assert body["providers"]["registered"] == ["deepseek", "openai"]
        assert body["providers"]["configured"] == ["deepseek", "openai"]
        assert body["concurrency"]["limit"] == 4
        assert body["concurrency"]["in_flight"] == 0

    async def test_providers_endpoint_lists_registered_models(
        self, client: httpx.AsyncClient
    ) -> None:
        """/health/providers 只暴露模型名与服务商，不含凭证。"""
        response = await client.get("/health/providers")

        assert response.status_code == 200
        body = response.json()
        assert body["default_model"] == "deepseek-chat"

        by_model = {item["model"]: item for item in body["models"]}
        assert by_model["deepseek-chat"]["provider"] == "deepseek"
        assert by_model["gpt-4o-mini"]["provider"] == "openai"
        assert by_model["deepseek-chat"]["pricing_configured"] is True

        text = response.text
        assert DEEPSEEK_KEY not in text
        assert "api.openai.com" not in text
        assert "api.deepseek.com" not in text


class TestChatCompletionsProxy:
    async def test_basic_proxy_request(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        chat_payload: dict,
    ) -> None:
        """最基本的转发：上游响应原样返回，并带上 request id 头。"""
        response = await client.post(
            "/v1/chat/completions", json=chat_payload, headers=auth_headers
        )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == SAMPLE_COMPLETION["id"]
        assert body["choices"][0]["message"]["content"] == "你好，我是 ModelGate。"
        # 网关自己生成的调用 ID 必须回传，便于与数据库记录对账
        assert response.headers["X-Request-Id"].startswith("req_")

    async def test_forwards_payload_and_uses_configured_key(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        chat_payload: dict,
        upstream_requests: list[httpx.Request],
    ) -> None:
        """转发时补上 stream=False，并使用服务端配置的上游密钥。"""
        await client.post(
            "/v1/chat/completions", json=chat_payload, headers=auth_headers
        )

        assert len(upstream_requests) == 1
        request = upstream_requests[0]

        assert request.url.path == "/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {DEEPSEEK_KEY}"

        import json

        sent = json.loads(request.content)
        assert sent["stream"] is False
        assert sent["model"] == "deepseek-chat"
        assert sent["messages"] == [{"role": "user", "content": "你好"}]

    async def test_uses_server_default_model_when_client_omits_it(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        upstream_requests: list[httpx.Request],
    ) -> None:
        """客户端不传 model 时，使用服务端配置的 DEFAULT_MODEL_NAME。"""
        await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )

        import json

        sent = json.loads(upstream_requests[0].content)
        assert sent["model"] == "deepseek-chat"

    async def test_extra_openai_params_are_passed_through(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        upstream_requests: list[httpx.Request],
    ) -> None:
        """OpenAI 兼容参数（如 temperature）应原样透传。"""
        await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.2,
                "max_tokens": 128,
            },
            headers=auth_headers,
        )

        import json

        sent = json.loads(upstream_requests[0].content)
        assert sent["temperature"] == 0.2
        assert sent["max_tokens"] == 128

    async def test_stream_true_is_rejected(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        upstream_requests: list[httpx.Request],
    ) -> None:
        """本阶段不支持流式：必须显式拒绝，而不是静默返回整包。"""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers=auth_headers,
        )

        assert response.status_code == 422
        # 被拒绝的请求不应打到上游
        assert upstream_requests == []

    async def test_empty_messages_is_rejected(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """messages 为空属于无效请求。"""
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers=auth_headers,
        )

        assert response.status_code == 422
