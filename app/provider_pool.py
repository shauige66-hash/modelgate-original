"""服务商连接池：把路由表与各服务商的 HTTP 客户端装配在一起。

每个服务商一个 httpx.AsyncClient，连接池互相隔离。
生命周期由 app lifespan 管理；通过注入客户端构造时，
aclose 只关闭自建的连接，避免测试里重复关闭 Mock 客户端。
"""

from __future__ import annotations

import httpx

from app.config import Settings
from app.providers import ModelRoute, ModelRouter, ProviderNotConfiguredError
from app.upstream import ChatCompletionsClient, create_http_client


class ProviderPool:
    """按模型解析服务商，并分派到对应的上游客户端。

    模型未登记抛 UnknownModelError，服务商未配置 Key 抛 ProviderNotConfiguredError，
    两者都由 app.errors 中的处理器转成对外的 4xx/5xx。
    """

    def __init__(
        self,
        *,
        router: ModelRouter,
        clients: dict[str, ChatCompletionsClient],
        owned_http_clients: list[httpx.AsyncClient],
    ) -> None:
        self._router = router
        self._clients = clients
        self._owned_http_clients = owned_http_clients

    @property
    def router(self) -> ModelRouter:
        return self._router

    @property
    def providers(self) -> list[str]:
        return sorted(self._clients)

    def resolve(self, model: str | None) -> ModelRoute:
        return self._router.resolve(model)

    def client_for(self, route: ModelRoute) -> ChatCompletionsClient:
        client = self._clients.get(route.provider)
        if client is None or not self._router.is_provider_configured(route.provider):
            raise ProviderNotConfiguredError(route.provider)
        return client

    def require_client(
        self, model: str | None
    ) -> tuple[ModelRoute, ChatCompletionsClient]:
        """一次拿到路由与客户端；任一步失败即抛异常，不会调用上游。

        把两步合成一个入口，避免调用方拿到路由却忘了检查服务商是否可用。
        """
        route = self._router.resolve(model)
        return route, self.client_for(route)

    async def aclose(self) -> None:
        for client in self._owned_http_clients:
            await client.aclose()


def build_provider_pool(
    settings: Settings,
    *,
    router: ModelRouter | None = None,
    http_clients: dict[str, httpx.AsyncClient] | None = None,
) -> ProviderPool:
    """为每个服务商建立一个独立客户端。

    http_clients 中已给出的服务商直接复用（测试注入 MockTransport），
    其余按配置新建；两者都在 aclose 时统一关闭。
    """
    router = router or ModelRouter(settings)
    provided = http_clients or {}

    clients: dict[str, ChatCompletionsClient] = {}
    resolved_http: list[httpx.AsyncClient] = []

    for provider in router.providers:
        base_url, api_key = settings.provider_credentials[provider]

        http_client = provided.get(provider)
        if http_client is None:
            http_client = create_http_client(
                base_url=base_url,
                api_key=api_key,
                settings=settings,
            )

        resolved_http.append(http_client)
        clients[provider] = ChatCompletionsClient(
            http_client,
            provider=provider,
            timeout_seconds=settings.upstream_timeout_seconds,
            is_configured=bool(api_key),
        )

    # 注入的客户端归测试所有，不在这里关闭
    owned = [client for client in resolved_http if client not in provided.values()]
    return ProviderPool(router=router, clients=clients, owned_http_clients=owned)
