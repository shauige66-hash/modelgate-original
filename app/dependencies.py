"""依赖注入。

应用级单例（HTTP 客户端、并发限制器、数据库引擎与会话工厂）挂在 app.state 上，
通过 Depends 取用。这样测试可直接替换 app.state 上的对象，
也避免模块级全局变量带来的导入顺序耦合。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.budget import BudgetGuard
from app.concurrency import ConcurrencyLimiter
from app.config import Settings
from app.provider_pool import ProviderPool


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_provider_pool(request: Request) -> ProviderPool:
    return request.app.state.provider_pool


def get_budget_guard(request: Request) -> BudgetGuard:
    return request.app.state.budget_guard


def get_limiter(request: Request) -> ConcurrencyLimiter:
    return request.app.state.limiter


def get_db_engine(request: Request) -> AsyncEngine:
    return request.app.state.db_engine


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """为单次请求提供数据库会话。

    不在这里 commit：调用记录落库失败不应影响已经成功返回的模型响应，
    因此提交时机与失败处理交给路由层决定。异常时回滚，会话由 finally 关闭。
    """
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
