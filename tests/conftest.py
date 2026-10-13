"""pytest 共享 fixtures。

关键约定：
- 测试**不依赖**真实付费模型 API。上游服务由 httpx.MockTransport 模拟。
- 测试默认用内存 SQLite；设置了 TEST_DATABASE_URL 时改用真实 PostgreSQL。
- 每个用例得到全新的表结构，避免用例之间互相污染。

两套上游 Mock 的分工：
- `app` / `client`：上游行为固定为「返回成功响应」，用于大多数正向用例。
- `make_harness`：可自定义上游行为与配置，用于失败路径与并发用例。
两者共用同一个 `upstream_requests` 记录列表，不会出现两套互不相通的记录。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from app.concurrency import ConcurrencyLimiter
from app.db.models import Base
from app.db.session import create_engine, create_session_factory
from app.main import create_app
from app.upstream import ChatCompletionsClient

# ---------------------------------------------------------------- 常量

CLIENT_KEY = "sk-client-test-key"
ADMIN_KEY = "sk-admin-test-key"
UPSTREAM_KEY = "sk-upstream-test-key"

# 真实 PostgreSQL 集成测试用；未设置则跳过相关用例
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")

# ---------------------------------------------------------------- 上游响应样本

SAMPLE_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-test-1",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "你好，我是 ModelGate。"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    },
}

# 没有 usage 字段的响应：用于验证「未知 Token」不会被当成 0
SAMPLE_COMPLETION_WITHOUT_USAGE: dict[str, Any] = {
    "id": "chatcmpl-test-2",
    "object": "chat.completion",
    "created": 1_700_000_001,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "no usage here"},
            "finish_reason": "stop",
        }
    ],
}


# ---------------------------------------------------------------- 配置

def build_settings(**overrides: Any) -> Settings:
    """构造测试用配置。

    显式传入 _env_file=None，避免测试机器上的 .env 影响结果。
    """
    values: dict[str, Any] = {
        "upstream_base_url": "https://upstream.test",
        "upstream_api_key": UPSTREAM_KEY,
        "upstream_model": "deepseek-chat",
        "upstream_timeout_seconds": 5.0,
        "max_concurrent_requests": 4,
        "concurrency_acquire_timeout_seconds": 1.0,
        "client_api_keys": CLIENT_KEY,
        "admin_api_key": ADMIN_KEY,
        "database_url": "sqlite+aiosqlite:///:memory:",
        "log_level": "WARNING",
    }
    values.update(overrides)

    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


# ---------------------------------------------------------------- 数据库


async def _prepare_database(engine: AsyncEngine) -> None:
    """先删后建，保证每个用例从空表开始。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


def _make_shared_memory_engine() -> AsyncEngine:
    """构造共享连接的内存 SQLite 引擎。

    必须用 StaticPool：内存 SQLite 在连接关闭后数据即丢失，
    所有会话需要共用同一个连接，否则每个连接会看到一个独立的空库。
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """SQLite 内存数据库引擎（单元测试默认使用）。"""
    engine = _make_shared_memory_engine()
    await _prepare_database(engine)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def pg_engine() -> AsyncIterator[AsyncEngine]:
    """真实 PostgreSQL 引擎；未配置 TEST_DATABASE_URL 时跳过。"""
    if not TEST_DATABASE_URL:
        pytest.skip("未设置 TEST_DATABASE_URL，跳过 PostgreSQL 集成测试")

    engine = create_engine(build_settings(database_url=TEST_DATABASE_URL))
    await _prepare_database(engine)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest_asyncio.fixture
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """单个数据库会话，供仓储层测试使用。"""
    async with session_factory() as session:
        yield session


# ---------------------------------------------------------------- 上游 Mock


@pytest.fixture
def upstream_requests() -> list[httpx.Request]:
    """记录发往上游的请求，便于断言转发内容。

    作用域是 function：每个用例都拿到一个干净列表。
    """
    return []


def _make_mock_upstream_client(
    *,
    handler: Callable[[httpx.Request], httpx.Response] | None,
    settings: Settings,
    recorded: list[httpx.Request],
) -> httpx.AsyncClient:
    """构造带 MockTransport 的上游客户端。

    带上与生产一致的 Authorization 头，
    这样「是否用了配置的上游密钥」才是一个有意义的断言。
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if handler is None:
            return httpx.Response(200, json=SAMPLE_COMPLETION)
        return handler(request)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        base_url=settings.upstream_base_url,
        timeout=settings.upstream_timeout_seconds,
        headers={"Authorization": f"Bearer {settings.upstream_api_key_value}"},
    )


# ---------------------------------------------------------------- 测试夹具


class TestHarness:
    """把「测试应用 + Mock 上游 + 数据库 + 客户端」打包在一起。

    用例只需面对一个对象，不用手工管理多份需要关闭的资源。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        engine: AsyncEngine,
        upstream_http_client: httpx.AsyncClient,
        client: httpx.AsyncClient,
        upstream_requests: list[httpx.Request],
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.upstream_http_client = upstream_http_client
        self.client = client
        self.upstream_requests = upstream_requests
        self.app = client._transport.app  # type: ignore[attr-defined]

    @property
    def limiter(self) -> ConcurrencyLimiter:
        return self.app.state.limiter

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self.app.state.session_factory

    async def aclose(self) -> None:
        """释放全部资源。"""
        await self.client.aclose()
        await self.upstream_http_client.aclose()
        await self.engine.dispose()


def _build_test_app(
    *,
    settings: Settings,
    upstream_http_client: httpx.AsyncClient,
    engine: AsyncEngine,
) -> Any:
    """构造已注入测试依赖的 app，不运行 lifespan。

    lifespan 会连接真实数据库并创建真实的 HTTP 客户端，
    测试里改用内存 SQLite 与 MockTransport，因此手动组装 app.state。
    """
    app = create_app(settings)
    app.state.settings = settings
    app.state.db_engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.http_client = upstream_http_client
    app.state.upstream_client = ChatCompletionsClient(upstream_http_client, settings)
    app.state.limiter = ConcurrencyLimiter(
        limit=settings.max_concurrent_requests,
        acquire_timeout=settings.concurrency_acquire_timeout_seconds,
    )
    return app


@pytest_asyncio.fixture
async def make_harness(
    upstream_requests: list[httpx.Request],
) -> AsyncIterator[Callable[..., Awaitable[TestHarness]]]:
    """按需构造测试夹具，支持自定义上游行为与配置。

    用法::

        h = await make_harness(handler)                    # 自定义上游响应
        h = await make_harness(settings=my_settings)       # 自定义配置
        h = await make_harness(authenticated=False)        # 不带客户端密钥
    """
    created: list[TestHarness] = []

    async def _factory(
        handler: Callable[[httpx.Request], httpx.Response] | None = None,
        *,
        settings: Settings | None = None,
        authenticated: bool = True,
    ) -> TestHarness:
        settings = settings or build_settings()

        engine = _make_shared_memory_engine()
        await _prepare_database(engine)

        upstream_http_client = _make_mock_upstream_client(
            handler=handler, settings=settings, recorded=upstream_requests
        )

        app = _build_test_app(
            settings=settings,
            upstream_http_client=upstream_http_client,
            engine=engine,
        )

        # 默认带上合法客户端密钥，让用例专注于被测行为本身。
        # authenticated=False 时不带任何凭据，用于测试鉴权拒绝路径。
        default_headers = (
            {"Authorization": f"Bearer {CLIENT_KEY}"} if authenticated else {}
        )

        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            timeout=30.0,
            headers=default_headers,
        )

        harness = TestHarness(
            settings=settings,
            engine=engine,
            upstream_http_client=upstream_http_client,
            client=client,
            upstream_requests=upstream_requests,
        )
        created.append(harness)
        return harness

    try:
        yield _factory
    finally:
        for harness in created:
            await harness.aclose()


@pytest_asyncio.fixture
async def harness(
    make_harness: Callable[..., Awaitable[TestHarness]],
) -> TestHarness:
    """默认夹具：上游固定返回 SAMPLE_COMPLETION，客户端带合法密钥。"""
    return await make_harness()


@pytest_asyncio.fixture
async def app(engine: AsyncEngine, upstream_requests: list[httpx.Request]) -> Any:
    """默认测试应用（内存 SQLite + Mock 上游），供 client 夹具使用。"""
    upstream_http_client = _make_mock_upstream_client(
        handler=None, settings=build_settings(), recorded=upstream_requests
    )
    app = _build_test_app(
        settings=build_settings(),
        upstream_http_client=upstream_http_client,
        engine=engine,
    )
    try:
        yield app
    finally:
        await upstream_http_client.aclose()


@pytest_asyncio.fixture
async def client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """直接调用 ASGI 应用的异步客户端（不经过网络）。

    刻意**不**带默认鉴权头：让每个用例显式声明自己用的凭据，
    这样「缺少凭据返回 401」这类用例才不会被夹具悄悄放行。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        timeout=30.0,
    ) as http_client:
        yield http_client


# ---------------------------------------------------------------- 辅助 fixtures


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """合法的客户端鉴权头。"""
    return {"Authorization": f"Bearer {CLIENT_KEY}"}


@pytest.fixture
def admin_headers() -> dict[str, str]:
    """合法的管理鉴权头。"""
    return {"X-Admin-Key": ADMIN_KEY}


@pytest.fixture
def chat_payload() -> dict[str, Any]:
    """一个最小的合法请求体。"""
    return {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "你好"}],
    }


@pytest.fixture
def run_concurrently() -> Callable[..., Awaitable[list[Any]]]:
    """并发执行多个协程，返回结果列表。"""

    async def _run(*coros: Awaitable[Any]) -> list[Any]:
        return await asyncio.gather(*coros)

    return _run


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """每个用例前后清掉配置缓存，避免用例之间互相影响。"""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
