"""call_records 的数据访问层。

SQL 集中在这里，路由层只调用语义化方法。

聚合口径：SUM() 会跳过 NULL，因此 Token 总量只累加已知值，
有未知 Token 的记录时 total 会偏小。这是刻意选择 —— 把未知记成 0
会让统计静默失真，所以额外返回 records_with_unknown_tokens 供调用方判断完整性。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CallRecord


@dataclass(frozen=True)
class UsageTotals:
    """一段时间内的汇总结果。"""

    total_requests: int
    success_count: int
    failure_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal
    records_with_unknown_tokens: int
    records_without_cost: int


@dataclass(frozen=True)
class ModelUsage:
    """按模型拆分的汇总结果。"""

    model: str
    total_requests: int
    success_count: int
    failure_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: Decimal


@dataclass(frozen=True)
class ErrorBreakdown:
    """按错误类型拆分的失败计数。"""

    error_type: str
    count: int


@dataclass
class NewCallRecord:
    """写入一条调用记录所需的数据。"""

    request_id: str
    model: str
    latency_ms: int
    http_status: int
    error_type: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    key_fingerprint: str | None = None
    message_count: int | None = None
    prompt_chars: int | None = None
    client_ip: str | None = None


class CallRecordRepository:
    """call_records 表的读写封装。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, record: NewCallRecord) -> CallRecord:
        """插入一条记录并 flush 以取得主键，commit 由调用方决定。"""
        orm = CallRecord(
            request_id=record.request_id,
            model=record.model,
            latency_ms=record.latency_ms,
            http_status=record.http_status,
            error_type=record.error_type,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            estimated_cost_usd=record.estimated_cost_usd,
            key_fingerprint=record.key_fingerprint,
            message_count=record.message_count,
            prompt_chars=record.prompt_chars,
            client_ip=record.client_ip,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm

    async def get_by_request_id(self, request_id: str) -> CallRecord | None:
        result = await self._session.execute(
            select(CallRecord).where(CallRecord.request_id == request_id)
        )
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 50) -> list[CallRecord]:
        result = await self._session.execute(
            select(CallRecord).order_by(CallRecord.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def usage_totals(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> UsageTotals:
        """汇总总调用次数、Token 总量、估算成本与失败数量。"""
        conditions = _time_conditions(since, until, model)

        # 用 SUM(CASE WHEN ...) 一次扫描同时算出成功与失败数，避免多条查询
        success_expr = func.sum(case((CallRecord.http_status < 400, 1), else_=0))
        failure_expr = func.sum(case((CallRecord.http_status >= 400, 1), else_=0))
        unknown_tokens_expr = func.sum(
            case(
                (
                    and_(
                        CallRecord.prompt_tokens.is_(None),
                        CallRecord.completion_tokens.is_(None),
                    ),
                    1,
                ),
                else_=0,
            )
        )
        unknown_cost_expr = func.sum(
            case((CallRecord.estimated_cost_usd.is_(None), 1), else_=0)
        )

        stmt = select(
            func.count(CallRecord.id),
            success_expr,
            failure_expr,
            func.sum(CallRecord.prompt_tokens),
            func.sum(CallRecord.completion_tokens),
            func.sum(CallRecord.estimated_cost_usd),
            unknown_tokens_expr,
            unknown_cost_expr,
        )
        if conditions:
            stmt = stmt.where(*conditions)

        row = (await self._session.execute(stmt)).one()

        prompt_tokens = int(row[3] or 0)
        completion_tokens = int(row[4] or 0)

        return UsageTotals(
            total_requests=int(row[0] or 0),
            success_count=int(row[1] or 0),
            failure_count=int(row[2] or 0),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            # 数据库 Numeric 可能返回 Decimal 以外的类型，统一转换
            estimated_cost_usd=Decimal(row[5] or 0),
            records_with_unknown_tokens=int(row[6] or 0),
            records_without_cost=int(row[7] or 0),
        )

    async def usage_by_model(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
        limit: int = 20,
    ) -> list[ModelUsage]:
        """按模型拆分统计。

        model 不为空时只统计该模型，保证与 usage_totals 的筛选口径一致
        （否则 totals 是筛选后的数字，by_model 却是全量，两边对不上）。
        """
        conditions = _time_conditions(since, until, model)

        stmt = (
            select(
                CallRecord.model,
                func.count(CallRecord.id),
                func.sum(case((CallRecord.http_status < 400, 1), else_=0)),
                func.sum(case((CallRecord.http_status >= 400, 1), else_=0)),
                func.sum(CallRecord.prompt_tokens),
                func.sum(CallRecord.completion_tokens),
                func.sum(CallRecord.estimated_cost_usd),
            )
            .group_by(CallRecord.model)
            .order_by(func.count(CallRecord.id).desc())
            .limit(limit)
        )
        if conditions:
            stmt = stmt.where(*conditions)

        rows = (await self._session.execute(stmt)).all()
        return [
            ModelUsage(
                model=row[0],
                total_requests=int(row[1] or 0),
                success_count=int(row[2] or 0),
                failure_count=int(row[3] or 0),
                prompt_tokens=int(row[4] or 0),
                completion_tokens=int(row[5] or 0),
                estimated_cost_usd=Decimal(row[6] or 0),
            )
            for row in rows
        ]

    async def error_breakdown(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> list[ErrorBreakdown]:
        """按错误类型统计失败次数，成功记录（none）不计入。

        model 不为空时只统计该模型的失败，保持与 totals / by_model 一致的口径。
        """
        conditions = _time_conditions(since, until, model)
        conditions.append(CallRecord.error_type != "none")

        stmt = (
            select(CallRecord.error_type, func.count(CallRecord.id))
            .where(*conditions)
            .group_by(CallRecord.error_type)
            .order_by(func.count(CallRecord.id).desc())
        )
        rows = (await self._session.execute(stmt)).all()
        return [ErrorBreakdown(error_type=row[0], count=int(row[1])) for row in rows]


def _time_conditions(
    since: datetime | None,
    until: datetime | None,
    model: str | None,
) -> list:
    """构造通用的过滤条件列表。"""
    conditions: list = []
    if since is not None:
        conditions.append(CallRecord.created_at >= since)
    if until is not None:
        conditions.append(CallRecord.created_at <= until)
    if model:
        conditions.append(CallRecord.model == model)
    return conditions
