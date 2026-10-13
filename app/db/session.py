"""数据库引擎与会话工厂。

生产使用 PostgreSQL + asyncpg，单元测试使用 SQLite + aiosqlite，
两者共用同一套 ORM 与仓储代码，避免测试与生产出现两套实现。
"""

from __future__ import annotations

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
    """按配置创建异步引擎。SQLite 不支持连接池参数，需要单独分支。"""
    url = settings.database_url

    if url.startswith("sqlite"):
        return create_async_engine(url, echo=settings.db_echo, future=True)

    return create_async_engine(
        url,
        echo=settings.db_echo,
        future=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # 取连接前先 ping，避免拿到已被数据库单方面关闭的死连接
        pool_pre_ping=True,
        # 回收时间需小于 PostgreSQL / 云负载均衡的空闲断开时间
        pool_recycle=1800,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """创建会话工厂。

    expire_on_commit=False 让 commit 后仍可读取对象属性，避免意外的懒加载 IO。
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_models(engine: AsyncEngine) -> None:
    """建表。项目规模小，用 create_all 而非 Alembic 迁移（局限见 README）。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表已就绪（create_all）")


async def dispose_engine(engine: AsyncEngine) -> None:
    """关闭连接池，应用退出时调用。"""
    await engine.dispose()
    logger.info("数据库连接池已关闭")
