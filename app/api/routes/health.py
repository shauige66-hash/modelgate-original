"""健康检查。

/health 是存活探针，不依赖外部组件，供 Docker healthcheck 使用。
/health/ready 额外检查数据库连通性，并报告上游配置与并发水位。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.concurrency import ConcurrencyLimiter
from app.config import Settings
from app.dependencies import get_app_settings, get_db_engine, get_limiter
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="存活探针")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="就绪探针")
async def readiness(
    response: Response,
    engine: AsyncEngine = Depends(get_db_engine),
    settings: Settings = Depends(get_app_settings),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
) -> dict[str, object]:
    """检查依赖项是否就绪。

    数据库不可用时返回 503，便于编排系统在数据库就绪前不转发流量；
    上游密钥未配置不算致命（服务本身可响应），只在结果中标记。
    """
    database_ok = True
    database_error: str | None = None

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - 探针捕获一切失败并降级
        database_ok = False
        database_error = type(exc).__name__
        logger.warning("就绪探针：数据库不可用 | %s", exc)

    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if database_ok else "degraded",
        "database": {
            "ok": database_ok,
            "error": database_error,
        },
        "upstream_configured": settings.is_upstream_configured,
        "concurrency": {
            "limit": limiter.limit,
            "in_flight": limiter.in_flight,
            "available": limiter.available,
        },
    }
