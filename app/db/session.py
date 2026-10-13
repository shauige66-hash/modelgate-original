"""数据库连接与会话管理。

使用 SQLAlchemy 2.x 的异步引擎。
- 生产环境：PostgreSQL + asyncpg
- 单元测试：SQLite + aiosqlite（见 tests/conftest.py）

两种驱动共用同一套 ORM 与仓储代码，避免测试与生产出现两套实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.db.models import Base
from app.logging_config import get_logger

logger = get_logger(__name__)

def create_engine(settings: Settings) -> AsyncEngine:
    """按配置创建异步引擎。

    SQLite 不支持连接池参数，测试场景需要跳过，否则会抛 TypeError。
    """
    url = settings.database_url

    if url.startswith("sqlite"):
        # SQLite 是文件/内存数据库，pool_size 等参数不适用
        return create_async_engine(url, echo=settings.db_echo, future=True)

    return create_async_engine(
        url,
        echo=settings.db_echo,
        future=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # 取连接前先 ping，避免拿到已被数据库单方面关闭的死连接
        pool_pre_ping=True,
        # 连接回收时间要小于 PostgreSQL / 云负载均衡的空闲断开时间
        pool_recycle=1800,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """创建会话工厂。

    expire_on_commit=False：commit 之后仍可读取对象属性，避免意外的懒加载 IO。
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_models(engine: AsyncEngine) -> None:
    """建表。

    本项目规模小，直接用 create_all 而不是 Alembic 迁移，
    降低上手成本（局限见 README「已知限制」）。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表已就绪（create_all）")


async def dispose_engine(engine: AsyncEngine) -> None:
    """关闭连接池。应用退出时调用，避免残留连接。"""
    await engine.dispose()
    logger.info("数据库连接池已关闭")
