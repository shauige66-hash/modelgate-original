"""用量统计接口：GET /admin/usage。

使用独立的管理密钥（X-Admin-Key），与客户端 API Key 分离。

Token 与成本只累加已知值，未知值不按 0 计入，
因此响应附带 records_with_unknown_tokens / records_without_cost，
避免把不完整的数据误读成准确账单。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import CallRecordRepository
from app.dependencies import get_db_session
from app.logging_config import get_logger
from app.security import require_admin_api_key

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _decimal_to_str(value: Decimal) -> str:
    """金额以字符串返回，避免 JSON 数字在客户端被解析成浮点。"""
    return format(value, "f")


@router.get("/usage", summary="调用量与成本统计")
async def get_usage(
    session: AsyncSession = Depends(get_db_session),
    _: str = Depends(require_admin_api_key),
    since: datetime | None = Query(
        default=None,
        description="起始时间（ISO 8601，含时区），例如 2026-01-01T00:00:00Z",
    ),
    until: datetime | None = Query(
        default=None,
        description="结束时间（ISO 8601，含时区）",
    ),
    model: str | None = Query(default=None, description="只看指定模型"),
    group_by_model: bool = Query(
        default=True,
        description="是否返回按模型拆分的明细",
    ),
    top_errors: int = Query(
        default=5,
        ge=0,
        le=50,
        description="返回前 N 个错误类型统计",
    ),
) -> dict[str, object]:
    """返回调用次数、Token 总量、估算成本与失败数量。

    model 筛选对 totals / by_model / errors 三者**统一生效**，
    避免出现「totals 是筛选后的数字，by_model 与 errors 却是全量」的不一致结果。
    """
    since, until = _normalize_range(since, until)
    repository = CallRecordRepository(session)

    totals = await repository.usage_totals(since=since, until=until, model=model)
    errors = (
        await repository.error_breakdown(since=since, until=until, model=model)
        if top_errors > 0
        else []
    )
    by_model = (
        await repository.usage_by_model(since=since, until=until, model=model)
        if group_by_model
        else []
    )

    return {
        "range": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "model_filter": model,
        },
        "totals": {
            "total_requests": totals.total_requests,
            "success_count": totals.success_count,
            "failure_count": totals.failure_count,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "total_tokens": totals.total_tokens,
            "estimated_cost_usd": _decimal_to_str(totals.estimated_cost_usd),
            "price_unit": "USD per 1,000,000 tokens",
        },
        "data_quality": {
            "records_with_unknown_tokens": totals.records_with_unknown_tokens,
            "records_without_cost": totals.records_without_cost,
            "note": (
                "上游未返回 usage 的记录 Token 记为未知，不会按 0 计入总量；"
                "上述两个字段用于提示统计的完整性"
            ),
        },
        "by_model": [
            {
                "model": item.model,
                "total_requests": item.total_requests,
                "success_count": item.success_count,
                "failure_count": item.failure_count,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "estimated_cost_usd": _decimal_to_str(item.estimated_cost_usd),
            }
            for item in by_model
        ],
        "errors": [
            {"error_type": item.error_type, "count": item.count}
            for item in errors[:top_errors]
        ],
    }


def _normalize_range(
    since: datetime | None,
    until: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    """规范化时间范围。

    - 两者都没给：默认看最近 24 小时。
    - 传入不带时区的时间：按 UTC 处理，避免与 timestamptz 比较时报错。
    """
    if since is None and until is None:
        since = datetime.now(timezone.utc) - timedelta(hours=24)

    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    if until is not None and until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)

    return since, until
