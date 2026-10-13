"""FastAPI 依赖注入。

把「应用级单例」（HTTP 客户端、并发限制器、数据库引擎/会话工厂）
统一挂在 app.state 上，再通过 Depends 取用。

好处：
- 测试时可以直接替换 app.state 上的对象，无需 monkeypatch 全局变量。
- 不需要模块级全局变量，避免导入顺序带来的隐性耦合。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.concurrency import ConcurrencyLimiter
from app.config import Settings
from app.upstream import ChatCompletionsClient


def get_app_settings(request: Request) -> Settings:
    """取应用配置。"""
    return request.app.state.settings


def get_http_client(request: Request) -> httpx.AsyncClient:
    """取共享的上游 HTTP 客户端。"""
    return request.app.state.http_client


def get_upstream_client(request: Request) -> ChatCompletionsClient:
    """取上游调用客户端。"""
    return request.app.state.upstream_client


def get_limiter(request: Request) -> ConcurrencyLimiter:
    """取并发限制器。"""
    return request.app.state.limiter


def get_db_engine(request: Request) -> AsyncEngine:
    """取数据库引擎（用于健康检查）。"""
    return request.app.state.db_engine


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """为单次请求提供一个数据库会话。

    事务策略：
    - 会话由 FastAPI 负责关闭，但**不在这里自动 commit**。
    - 路由层在成功写入后显式 `commit()`。
    - 出现异常时这里回滚，避免把半成品事务留在连接上。

    为什么不在依赖里直接 commit：
    「调用记录落库失败」不应该影响已经成功返回给客户端的模型响应，
    因此提交时机与失败处理必须由路由层决定。
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
