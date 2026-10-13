"""pytest 共享 fixtures。

约定：
- 不依赖真实付费模型 API，上游由 httpx.MockTransport 模拟。
- 默认使用内存 SQLite；设置 TEST_DATABASE_URL 后改用真实 PostgreSQL。
- 每个用例拿到全新的表结构，避免用例之间互相污染。
- PostgreSQL 集成测试会 drop_all + create_all，TEST_DATABASE_URL 必须指向
  库名以 `_test` 结尾的专用测试库，否则导入阶段即报错（见 tests/db_safety.py）。

上游 Mock 有两套：`app`/`client` 固定返回成功响应，
`make_harness` 可自定义上游行为与配置；两者共用同一个 upstream_requests 记录列表。
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
from tests.db_safety import (
    UnsafeTestDatabaseError,
    assert_connected_database_is_test,
    assert_safe_test_database_url,
)

CLIENT_KEY = "sk-client-test-key"
ADMIN_KEY = "sk-admin-test-key"
UPSTREAM_KEY = "sk-upstream-test-key"

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")

# 在导入阶段校验目标库名：配置错误时用例还没开始跑就报错，
# 不会执行到 drop_all。
if TEST_DATABASE_URL:
    try:
        TEST_DATABASE_NAME = assert_safe_test_database_url(TEST_DATABASE_URL)
    except UnsafeTestDatabaseError as exc:
        raise UnsafeTestDatabaseError(
            f"\n\n[ModelGate 测试安全保护] 已拒绝启动 PostgreSQL 集成测试。\n{exc}\n"
        ) from exc
else:
    TEST_DATABASE_NAME = ""

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
    """构造测试配置。

    传入 _env_file=None，避免测试机器上的 .env 影响结果。
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
    """删表重建，让每个用例从空表开始。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


def _make_shared_memory_engine() -> AsyncEngine:
    """构造共享连接的内存 SQLite 引擎。

    必须用 StaticPool：内存 SQLite 的数据随连接关闭而消失，
    所有会话需要共用同一个连接，否则各自看到一个独立的空库。
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
    """真实 PostgreSQL 引擎；未配置 TEST_DATABASE_URL 时跳过。

    drop_all/create_all 之前有两道保护：导入阶段已校验连接串中的库名以 `_test`
    结尾，这里再用 current_database() 确认实际连到的库。
    """
    if not TEST_DATABASE_URL:
        pytest.skip("未设置 TEST_DATABASE_URL，跳过 PostgreSQL 集成测试")

    # 再校验一次，防止该变量在运行期被改写
    database_name = assert_safe_test_database_url(TEST_DATABASE_URL)

    engine = create_engine(build_settings(database_url=TEST_DATABASE_URL))

    try:
        actual_name = await assert_connected_database_is_test(engine)
    except Exception:
        await engine.dispose()
        raise

    assert actual_name == database_name, (
        f"连接串声明的库是 {database_name!r}，实际连到 {actual_name!r}，已中止测试。"
    )

    # 把破坏性操作写进输出，便于在 CI 日志中审计
    print(
        f"\n[ModelGate 测试安全保护] 即将重建 PostgreSQL 数据库 "
        f"{actual_name!r} 的全部表（drop_all + create_all）。"
        f"\n                         该库以 '_test' 结尾，已确认是专用测试库。"
    )

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
    """记录发往上游的请求，供断言转发内容。每个用例拿到独立列表。"""
    return []


def _make_mock_upstream_client(
    *,
    handler: Callable[[httpx.Request], httpx.Response] | None,
    settings: Settings,
    recorded: list[httpx.Request],
) -> httpx.AsyncClient:
    """构造带 MockTransport 的上游客户端。

    带上与生产一致的 Authorization 头，这样「是否使用了配置的上游密钥」
    才是一个有意义的断言。
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
    """把测试应用、Mock 上游、数据库与客户端打包在一起，便于统一关闭。"""

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

    lifespan 会连接真实数据库、创建真实 HTTP 客户端，
    测试改用内存 SQLite 与 MockTransport，因此手动组装 app.state。
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
    """按需构造测试夹具，可自定义上游行为与配置。

    用法::

        h = await make_harness(handler)               # 自定义上游响应
        h = await make_harness(settings=my_settings)  # 自定义配置
        h = await make_harness(authenticated=False)   # 不带客户端凭据
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

        # 默认带合法客户端密钥；authenticated=False 时不带凭据，用于鉴权拒绝路径
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
    """直接调用 ASGI 应用的异步客户端。

    刻意不带默认鉴权头：由用例显式声明凭据，
    否则「缺少凭据返回 401」这类用例会被夹具悄悄放行。
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