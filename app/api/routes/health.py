"""健康检查。

/health 是存活探针，不依赖外部组件，供 Docker healthcheck 使用。
/health/ready 额外检查数据库连通性，并报告各服务商的配置状态。
/health/providers 列出已登记的模型与服务商（不含任何凭证）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.concurrency import ConcurrencyLimiter
from app.dependencies import get_db_engine, get_limiter, get_provider_pool
from app.logging_config import get_logger
from app.provider_pool import ProviderPool

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="存活探针")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="就绪探针")
async def readiness(
    response: Response,
    engine: AsyncEngine = Depends(get_db_engine),
    pool: ProviderPool = Depends(get_provider_pool),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
) -> dict[str, object]:
    """检查依赖项是否就绪。

    数据库不可用时返回 503，便于编排系统在数据库就绪前不转发流量；
    某个服务商未配置 Key 不算致命（其他服务商仍可用），只在结果中标记。
    """
    database_ok = True
    database_error: str | None = None
    database_name: str | None = None

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            # 报告实际连接到的库名：测试需要确认被测容器用的是测试库，
            # 而不是依赖宿主机的环境变量去猜测。库名不属于敏感信息。
            if conn.dialect.name == "sqlite":
                # SQLite 没有 current_database()，用 PRAGMA 取附加库名
                database_name = (
                    await conn.exec_driver_sql("PRAGMA database_list")
                ).first()[1]
            else:
                database_name = (
                    await conn.execute(text("SELECT current_database()"))
                ).scalar_one()
    except Exception as exc:  # noqa: BLE001 - 探针捕获一切失败并降级
        database_ok = False
        database_error = type(exc).__name__
        logger.warning("就绪探针：数据库不可用 | %s", exc)

    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    configured = pool.router.configured_providers()
    return {
        "status": "ok" if database_ok else "degraded",
        "database": {
            "ok": database_ok,
            "error": database_error,
            "name": database_name,
        },
        "providers": {
            "registered": pool.router.providers,
            "configured": configured,
        },
        "concurrency": {
            "limit": limiter.limit,
            "in_flight": limiter.in_flight,
            "available": limiter.available,
        },
    }


@router.get("/health/providers", summary="已登记的模型与服务商")
async def providers(pool: ProviderPool = Depends(get_provider_pool)) -> dict[str, object]:
    """返回模型注册表，便于调用方发现可用模型名。

    只暴露模型名与所属服务商，不包含 Base URL 与任何凭证。
    """
    return {
        "default_model": pool.router.default_model,
        "models": [
            {
                "model": model,
                "provider": pool.resolve(model).provider,
                "pricing_configured": pool.resolve(model).has_pricing,
                "provider_configured": pool.router.is_provider_configured(
                    pool.resolve(model).provider
                ),
            }
            for model in pool.router.registered_models
        ],
    }
