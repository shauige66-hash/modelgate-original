"""应用入口：用 lifespan 装配共享资源，并注册路由与异常处理。

启动：uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import admin, chat, health
from app.concurrency import ConcurrencyLimiter
from app.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory, dispose_engine, init_models
from app.errors import GatewayError, gateway_error_handler
from app.logging_config import configure_logging, get_logger
from app.request_context import get_request_id
from app.upstream import ChatCompletionsClient, create_http_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """创建并释放进程级共享资源。

    数据库引擎与 HTTP 客户端在这里各创建一次、全程复用，
    退出时统一关闭，避免连接泄漏。
    """
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)

    logger.info(
        "启动 %s v%s | 并发上限=%s | 上游=%s | 模型=%s",
        settings.app_name,
        settings.app_version,
        settings.max_concurrent_requests,
        settings.upstream_base_url,
        settings.upstream_model,
    )

    engine = create_engine(settings)
    app.state.db_engine = engine
    app.state.session_factory = create_session_factory(engine)

    try:
        await init_models(engine)
    except Exception as exc:  # noqa: BLE001
        # 建表失败不阻断启动：让 /health 可用、/health/ready 返回 503，
        # 便于区分「应用没起来」和「数据库没就绪」。
        logger.error("初始化数据库失败，服务将以降级状态启动 | %s", exc)

    http_client = create_http_client(settings)
    app.state.http_client = http_client
    app.state.upstream_client = ChatCompletionsClient(http_client, settings)

    app.state.limiter = ConcurrencyLimiter(
        limit=settings.max_concurrent_requests,
        acquire_timeout=settings.concurrency_acquire_timeout_seconds,
    )

    try:
        yield
    finally:
        logger.info("正在关闭 %s ...", settings.app_name)
        await http_client.aclose()
        await dispose_engine(engine)


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """记录完整堆栈，但只向客户端返回泛化错误。"""
    logger.exception("未处理异常 | %s %s", request.method, request.url.path)

    headers: dict[str, str] = {}
    request_id = get_request_id()
    if request_id:
        headers["X-Request-Id"] = request_id

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "服务内部错误",
                "type": "internal_error",
                "code": 500,
            }
        },
        headers=headers,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造应用。传入 settings 可注入测试配置，绕过环境变量与 lru_cache。"""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="轻量级 LLM API 网关：异步转发、并发保护、用量与成本统计",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # lifespan 从这里取配置
    app.state.settings = settings

    app.add_exception_handler(GatewayError, gateway_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    return app


app = create_app()
