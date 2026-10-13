"""API Key 鉴权测试。"""

from __future__ import annotations

import httpx

from tests.conftest import ADMIN_KEY, CLIENT_KEY


class TestClientAuth:
    async def test_no_credential_returns_401(
        self, client: httpx.AsyncClient, chat_payload: dict
    ) -> None:
        response = await client.post("/v1/chat/completions", json=chat_payload)

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    async def test_wrong_key_returns_403(
        self, client: httpx.AsyncClient, chat_payload: dict
    ) -> None:
        """提供了凭据但不正确 -> 403（区别于「没提供凭据」的 401）。"""
        response = await client.post(
            "/v1/chat/completions",
            json=chat_payload,
            headers={"Authorization": "Bearer sk-wrong-key"},
        )

        assert response.status_code == 403

    async def test_wrong_key_never_reaches_upstream(
        self,
        client: httpx.AsyncClient,
        chat_payload: dict,
        upstream_requests: list[httpx.Request],
    ) -> None:
        """鉴权失败必须在调用上游之前就拦住，避免白白消耗额度。"""
        await client.post(
            "/v1/chat/completions",
            json=chat_payload,
            headers={"Authorization": "Bearer sk-wrong-key"},
        )

        assert upstream_requests == []

    async def test_bearer_token_is_accepted(
        self, client: httpx.AsyncClient, auth_headers: dict[str, str], chat_payload: dict
    ) -> None:
        response = await client.post(
            "/v1/chat/completions", json=chat_payload, headers=auth_headers
        )

        assert response.status_code == 200

    async def test_x_api_key_header_is_accepted(
        self, client: httpx.AsyncClient, chat_payload: dict
    ) -> None:
        """同时支持 X-API-Key 写法（部分 OpenAI 客户端使用）。"""
        response = await client.post(
            "/v1/chat/completions",
            json=chat_payload,
            headers={"X-API-Key": CLIENT_KEY},
        )

        assert response.status_code == 200

    async def test_malformed_authorization_header_returns_401(
        self, client: httpx.AsyncClient, chat_payload: dict
    ) -> None:
        """缺少 Bearer 前缀时按「未提供凭据」处理。"""
        response = await client.post(
            "/v1/chat/completions",
            json=chat_payload,
            headers={"Authorization": CLIENT_KEY},
        )

        assert response.status_code == 401


class TestAdminAuth:
    async def test_admin_needs_its_own_key(
        self, client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """客户端 Key 不能访问管理接口。"""
        response = await client.get("/admin/usage", headers=auth_headers)

        assert response.status_code == 401

    async def test_admin_wrong_key_returns_403(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(
            "/admin/usage", headers={"X-Admin-Key": "sk-not-the-admin-key"}
        )

        assert response.status_code == 403

    async def test_admin_correct_key_is_accepted(
        self, client: httpx.AsyncClient, admin_headers: dict[str, str]
    ) -> None:
        response = await client.get("/admin/usage", headers=admin_headers)

        assert response.status_code == 200


class TestSecretsNeverLeak:
    async def test_response_never_contains_upstream_key(
        self,
        client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        chat_payload: dict,
    ) -> None:
        """上游密钥不得出现在返回给客户端的内容里。"""
        from tests.conftest import DEEPSEEK_KEY

        response = await client.post(
            "/v1/chat/completions", json=chat_payload, headers=auth_headers
        )

        assert DEEPSEEK_KEY not in response.text

    async def test_admin_key_never_echoed(
        self, client: httpx.AsyncClient, admin_headers: dict[str, str]
    ) -> None:
        response = await client.get("/admin/usage", headers=admin_headers)

        assert ADMIN_KEY not in response.text

    async def test_settings_repr_hides_secrets(self) -> None:
        """Settings 的 repr 不打印密钥，避免日志/异常里意外泄露。"""
        from tests.conftest import build_settings

        text = repr(build_settings())

        assert CLIENT_KEY not in text
        assert ADMIN_KEY not in text
