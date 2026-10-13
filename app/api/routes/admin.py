"""管理接口。

- GET /admin/usage      调用量与成本统计
- GET /admin/budget     配额用量
- GET /admin/runs       某次 Agent 工作流（run_id）的聚合视图

均使用独立的管理密钥（X-Admin-Key），与客户端 API Key 分离。

Token 与成本只累加已知值，未知值不按 0 计入，
因此统计响应附带 records_with_unknown_tokens / records_without_cost，
避免把不完整的数据误读成准确账单。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.budget_repository import BudgetRepository, QUOTA_SCOPE_RUN
from app.db.repository import CallRecordRepository, GroupUsage, UsageFilter
from app.dependencies import get_app_settings, get_db_session
from app.logging_config import get_logger
from app.security import require_admin_api_key

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _decimal_to_str(value: Decimal) -> str:
    """金额以字符串返回，避免 JSON 数字在客户端被解析成浮点。"""
    return format(value, "f")


def _serialize_group(item: GroupUsage) -> dict[str, object]:
    return {
        "key": item.key,
        "total_requests": item.total_requests,
        "success_count": item.success_count,
        "failure_count": item.failure_count,
        "prompt_tokens": item.prompt_tokens,
        "completion_tokens": item.completion_tokens,
        "total_tokens": item.prompt_tokens + item.completion_tokens,
        "estimated_cost_usd": _decimal_to_str(item.estimated_cost_usd),
        "avg_latency_ms": item.avg_latency_ms,
    }


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
    provider: str | None = Query(
        default=None, description="只看指定服务商，例如 deepseek / openai"
    ),
    client_app: str | None = Query(default=None, description="按 X-Client-App 筛选"),
    agent_name: str | None = Query(default=None, description="按 X-Agent-Name 筛选"),
    run_id: str | None = Query(default=None, description="按 X-Run-Id 筛选"),
    group_by_model: bool = Query(default=True, description="是否按模型拆分明细"),
    group_by_provider: bool = Query(default=True, description="是否按服务商拆分明细"),
    top_errors: int = Query(
        default=5, ge=0, le=50, description="返回前 N 个错误类型统计"
    ),
) -> dict[str, object]:
    """返回调用次数、成功/失败数、Token、估算成本与平均耗时。"""
    since, until = _normalize_range(since, until)

    filters = UsageFilter(
        since=since,
        until=until,
        model=model,
        provider=provider,
        client_app=client_app,
        agent_name=agent_name,
        run_id=run_id,
    )
    repository = CallRecordRepository(session)

    totals = await repository.usage_totals(filters)
    errors = await repository.error_breakdown(filters) if top_errors > 0 else []
    by_model = await repository.usage_by_model(filters) if group_by_model else []
    by_provider = (
        await repository.usage_by_provider(filters) if group_by_provider else []
    )

    return {
        "range": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "filters": {
            "model": model,
            "provider": provider,
            "client_app": client_app,
            "agent_name": agent_name,
            "run_id": run_id,
        },
        "totals": {
            "total_requests": totals.total_requests,
            "success_count": totals.success_count,
            "failure_count": totals.failure_count,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "total_tokens": totals.total_tokens,
            "estimated_cost_usd": _decimal_to_str(totals.estimated_cost_usd),
            "avg_latency_ms": totals.avg_latency_ms,
            "price_unit": "USD per 1,000,000 tokens",
        },
        "data_quality": {
            "records_with_unknown_tokens": totals.records_with_unknown_tokens,
            "records_without_cost": totals.records_without_cost,
            "records_with_retries": totals.records_with_retries,
            "note": (
                "上游未返回 usage 或模型价格未配置时，Token 与成本记为未知，"
                "不会按 0 计入总量。发生过重试的请求（records_with_retries），"
                "其 usage 仅来自最后一次尝试，中间尝试可能已在上游计费，"
                "因此成本是下限而非完整账单。"
            ),
        },
        "by_provider": [_serialize_group(item) for item in by_provider],
        "by_model": [_serialize_group(item) for item in by_model],
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


def _serialize_run_summary(item) -> dict[str, Any]:
    return {
        "run_id": item.run_id,
        "client_app": item.client_app,
        "provider_count": item.provider_count,
        "request_count": item.request_count,
        "success_count": item.success_count,
        "failure_count": item.failure_count,
        "prompt_tokens": item.prompt_tokens,
        "completion_tokens": item.completion_tokens,
        "total_tokens": item.prompt_tokens + item.completion_tokens,
        "estimated_cost_usd": _decimal_to_str(item.estimated_cost_usd),
        "records_without_cost": item.records_without_cost,
        "avg_latency_ms": item.avg_latency_ms,
        "first_seen": item.first_seen.isoformat() if item.first_seen else None,
        "last_seen": item.last_seen.isoformat() if item.last_seen else None,
    }


@router.get("/runs", summary="Agent 工作流列表")
async def list_runs(
    session: AsyncSession = Depends(get_db_session),
    _: str = Depends(require_admin_api_key),
    since: datetime | None = Query(default=None, description="起始时间（ISO 8601）"),
    until: datetime | None = Query(default=None, description="结束时间（ISO 8601）"),
    client_app: str | None = Query(default=None, description="按 X-Client-App 筛选"),
    key_fingerprint: str | None = Query(
        default=None,
        description="按客户端指纹筛选；同名 run_id 需要它才能区分归属",
    ),
    limit: int = Query(default=50, ge=1, le=200, description="每页条数"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
) -> dict[str, Any]:
    """按 run_id 聚合的工作流列表，按最近活动时间倒序，支持分页。

    run_id 由调用方自报，不同客户端可能使用同一个 run_id；
    带上 key_fingerprint 才会只统计该客户端的调用。
    """
    since, until = _normalize_range_for_filter(since, until)

    filters = UsageFilter(
        since=since,
        until=until,
        client_app=client_app,
        key_fingerprint=key_fingerprint,
    )
    runs, total = await CallRecordRepository(session).list_runs(
        filters, limit=limit, offset=offset
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(runs) < total,
        "runs": [_serialize_run_summary(run) for run in runs],
    }


@router.get("/runs/{run_id}", summary="单个 Agent 工作流的聚合视图")
async def get_run(
    run_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: str = Depends(require_admin_api_key),
    key_fingerprint: str | None = Query(
        default=None,
        description="限定到某个客户端；不传则汇总所有客户端的同名 run",
    ),
) -> dict[str, Any]:
    """某次 Agent 工作流的调用明细聚合。

    包含每个 Agent 节点的调用分布、按服务商与模型拆分的 Token/成本/耗时，
    以及错误类型分布。未返回 usage 的调用其 Token 与成本保持未知（NULL），
    不会按 0 计入。
    """
    filters = UsageFilter(run_id=run_id, key_fingerprint=key_fingerprint)
    repository = CallRecordRepository(session)

    totals = await repository.usage_totals(filters)
    by_agent = await repository.usage_by_agent(filters)
    by_provider = await repository.usage_by_provider(filters)
    by_model = await repository.usage_by_model(filters)
    errors = await repository.error_breakdown(filters)

    return {
        "run_id": run_id,
        "client_scope": key_fingerprint or "all_clients",
        "isolation_note": (
            "run_id 由调用方自报。未指定 client 指纹时，"
            "不同客户端的同名 run 会被合并统计"
        ),
        "totals": {
            "total_requests": totals.total_requests,
            "success_count": totals.success_count,
            "failure_count": totals.failure_count,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "total_tokens": totals.total_tokens,
            "estimated_cost_usd": _decimal_to_str(totals.estimated_cost_usd),
            "avg_latency_ms": totals.avg_latency_ms,
        },
        "data_quality": {
            "records_with_unknown_tokens": totals.records_with_unknown_tokens,
            "records_without_cost": totals.records_without_cost,
            "records_with_retries": totals.records_with_retries,
            "note": (
                "Token 与成本只累加已知值。发生过重试的请求，其 usage 仅来自"
                "最后一次尝试，中间尝试可能已在上游计费，因此成本是不完整的下限"
            ),
        },
        "by_agent": [_serialize_group(item) for item in by_agent],
        "by_provider": [_serialize_group(item) for item in by_provider],
        "by_model": [_serialize_group(item) for item in by_model],
        "errors": [
            {"error_type": item.error_type, "count": item.count} for item in errors
        ],
    }


def _normalize_range_for_filter(
    since: datetime | None, until: datetime | None
) -> tuple[datetime | None, datetime | None]:
    """为列表接口规范化时间范围：不设默认窗口，由调用方显式指定。"""
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until is not None and until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return since, until


def _mask_identity(scope: str, identity: str) -> str:
    """对外展示的身份标识。

    返回的 identity 是服务端内部键，其中 client 维度是网关 API Key 的指纹，
    run 维度是「指纹:run_id」。指纹本身不可逆，但仍不应完整外泄，
    因此 client 只留后 8 位；run 只展示客户端自己提供的 run_id。
    """
    if scope == QUOTA_SCOPE_RUN and ":" in identity:
        return identity.split(":", 1)[1]
    return f"***{identity[-8:]}" if len(identity) > 8 else "***"


@router.get("/budget", summary="配额用量与限额")
async def get_budget(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
    _: str = Depends(require_admin_api_key),
    window_start: date | None = Query(
        default=None,
        description="只看某个 UTC 日期的 client 用量（YYYY-MM-DD）；默认今天",
    ),
    scope: str | None = Query(
        default=None, description="只看某个维度：client 或 run"
    ),
    limit: int = Query(default=100, ge=1, le=1000, description="最多返回多少行"),
) -> dict[str, Any]:
    """列出各身份的配额用量。

    client 维度按 UTC 自然日重置，因此会显示窗口日期；
    run 维度覆盖整个 Run 生命周期、不随日期重置，窗口显示为 null。
    本接口只统计**请求次数**，不涉及 Token 或金额预算。
    """
    target_window = window_start or datetime.now(timezone.utc).date()

    rows = await BudgetRepository(session).list_usage(
        window_start=target_window, scope=scope
    )

    entries = []
    for row in rows[:limit]:
        is_run = row.scope == QUOTA_SCOPE_RUN
        row_limit = (
            settings.run_max_llm_calls if is_run else settings.client_daily_request_quota
        )
        entries.append(
            {
                "scope": row.scope,
                "identity": _mask_identity(row.scope, row.identity),
                "window_kind": row.window_kind,
                "window_start": None if is_run else row.window_start.isoformat(),
                "used": row.used,
                "limit": row_limit or None,
                "remaining": max(row_limit - row.used, 0) if row_limit else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
        )

    return {
        "client_window_date": target_window.isoformat(),
        "reset_policy": {
            "client": "每个 UTC 自然日 00:00 重置",
            "run": "不重置，覆盖整个 Run 生命周期",
        },
        "limits": {
            "client_daily_request_quota": settings.client_daily_request_quota or None,
            "run_max_llm_calls": settings.run_max_llm_calls or None,
        },
        "note": (
            "配额仅限制请求次数，不包含 Token 或金额预算；"
            "0 或 null 表示该维度未启用限制"
        ),
        "count": len(entries),
        "entries": entries,
    }
