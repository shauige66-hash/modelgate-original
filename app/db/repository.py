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
    records_with_retries: int
    avg_latency_ms: int | None


@dataclass(frozen=True)
class RunSummary:
    """一次 Agent 工作流的聚合概览（列表用）。"""

    run_id: str
    client_app: str | None
    provider_count: int
    request_count: int
    success_count: int
    failure_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: Decimal
    records_without_cost: int
    avg_latency_ms: int | None
    first_seen: datetime
    last_seen: datetime


@dataclass(frozen=True)
class GroupUsage:
    """按某个维度（模型或服务商）拆分的汇总结果。"""

    key: str
    total_requests: int
    success_count: int
    failure_count: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: Decimal
    avg_latency_ms: int | None


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
    provider: str | None = None
    requested_model: str | None = None
    response_model: str | None = None
    retry_count: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    key_fingerprint: str | None = None
    message_count: int | None = None
    prompt_chars: int | None = None
    client_ip: str | None = None
    client_app: str | None = None
    agent_name: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class UsageFilter:
    """统计接口的筛选条件。

    key_fingerprint 是已认证客户端的指纹。Run 级查询必须带上它：
    run_id 由调用方自报，不同客户端可能用同一个 run_id，
    只用 run_id 过滤会把它们的调用混在一起。
    """

    since: datetime | None = None
    until: datetime | None = None
    model: str | None = None
    provider: str | None = None
    key_fingerprint: str | None = None
    client_app: str | None = None
    agent_name: str | None = None
    run_id: str | None = None


def _build_conditions(filters: UsageFilter) -> list:
    """把筛选条件翻成 SQL 条件列表。"""
    conditions: list = []

    if filters.since is not None:
        conditions.append(CallRecord.created_at >= filters.since)
    if filters.until is not None:
        conditions.append(CallRecord.created_at <= filters.until)
    if filters.model:
        # 模型筛选同时匹配请求模型与实际响应模型，
        # 否则别名或快照版本（如 gpt-4o-mini-2024-07-18）会被漏掉
        conditions.append(
            (CallRecord.model == filters.model)
            | (CallRecord.requested_model == filters.model)
            | (CallRecord.response_model == filters.model)
        )
    if filters.provider:
        conditions.append(CallRecord.provider == filters.provider)
    if filters.key_fingerprint:
        conditions.append(CallRecord.key_fingerprint == filters.key_fingerprint)
    if filters.client_app:
        conditions.append(CallRecord.client_app == filters.client_app)
    if filters.agent_name:
        conditions.append(CallRecord.agent_name == filters.agent_name)
    if filters.run_id:
        conditions.append(CallRecord.run_id == filters.run_id)

    return conditions


def _as_optional_int(value: object) -> int | None:
    """AVG() 在无数据时返回 NULL，统一转成 None 表示「无样本」。"""
    if value is None:
        return None
    return int(value)


class CallRecordRepository:
    """call_records 表的读写封装。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, record: NewCallRecord) -> CallRecord:
        """插入一条记录并 flush 以取得主键，commit 由调用方决定。

        不再调用 refresh()：flush() 已经用 INSERT ... RETURNING 取回
        id 与 created_at（服务端默认值），再 refresh 会多发一次 SELECT，
        在每请求都写一条记录的路径上是纯浪费。
        """
        orm = CallRecord(
            request_id=record.request_id,
            provider=record.provider,
            model=record.model,
            requested_model=record.requested_model,
            response_model=record.response_model,
            latency_ms=record.latency_ms,
            retry_count=record.retry_count,
            http_status=record.http_status,
            error_type=record.error_type,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            estimated_cost_usd=record.estimated_cost_usd,
            key_fingerprint=record.key_fingerprint,
            message_count=record.message_count,
            prompt_chars=record.prompt_chars,
            client_ip=record.client_ip,
            client_app=record.client_app,
            agent_name=record.agent_name,
            run_id=record.run_id,
        )
        self._session.add(orm)
        await self._session.flush()
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

    async def usage_totals(self, filters: UsageFilter | None = None) -> UsageTotals:
        """汇总调用次数、Token、估算成本、失败数量与平均耗时。"""
        filters = filters or UsageFilter()
        conditions = _build_conditions(filters)

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
        # 发生过重试的请求：这条记录的 usage 只来自最后一次尝试，
        # 中间尝试可能已在上游计费，因此成本是不完整的
        retried_expr = func.sum(
            case((CallRecord.retry_count > 0, 1), else_=0)
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
            retried_expr,
            func.avg(CallRecord.latency_ms),
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
            records_with_retries=int(row[8] or 0),
            avg_latency_ms=_as_optional_int(row[9]),
        )

    async def usage_by_provider(
        self, filters: UsageFilter | None = None, limit: int = 20
    ) -> list[GroupUsage]:
        """按服务商拆分统计。"""
        return await self._grouped_usage(CallRecord.provider, filters, limit)

    async def usage_by_model(
        self, filters: UsageFilter | None = None, limit: int = 20
    ) -> list[GroupUsage]:
        """按请求模型拆分统计。

        分组键用 requested_model 优先、回落到历史列 model，
        这样旧数据（只有 model）与新数据都能正确归类。
        """
        key_expr = func.coalesce(CallRecord.requested_model, CallRecord.model)
        return await self._grouped_usage(key_expr, filters, limit)

    async def usage_by_agent(
        self, filters: UsageFilter | None = None, limit: int = 50
    ) -> list[GroupUsage]:
        """按 Agent 节点拆分统计。

        未提供 agent_name 的调用归入 "unlabeled"，而不是被丢弃 ——
        否则各节点之和会对不上总数，看起来像丢数据。
        """
        key_expr = func.coalesce(CallRecord.agent_name, "unlabeled")
        return await self._grouped_usage(key_expr, filters, limit)

    async def list_runs(
        self,
        filters: UsageFilter | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RunSummary], int]:
        """按 run_id 聚合的工作流列表，返回 (当页数据, 总数)。

        filters 里的 key_fingerprint 用于限制到某个客户端：
        run_id 由调用方自报，不同客户端可能同名，不加指纹就会互相混入。
        """
        filters = filters or UsageFilter()
        conditions = _build_conditions(filters)
        # run_id 为空的调用不属于任何工作流，不列入 run 列表
        conditions.append(CallRecord.run_id.is_not(None))

        grouped = (
            select(
                CallRecord.run_id.label("run_id"),
                func.max(CallRecord.client_app).label("client_app"),
                func.count(func.distinct(CallRecord.provider)).label("provider_count"),
                func.count(CallRecord.id).label("request_count"),
                func.sum(case((CallRecord.http_status < 400, 1), else_=0)).label(
                    "success_count"
                ),
                func.sum(case((CallRecord.http_status >= 400, 1), else_=0)).label(
                    "failure_count"
                ),
                func.sum(CallRecord.prompt_tokens).label("prompt_tokens"),
                func.sum(CallRecord.completion_tokens).label("completion_tokens"),
                func.sum(CallRecord.estimated_cost_usd).label("estimated_cost_usd"),
                func.sum(
                    case((CallRecord.estimated_cost_usd.is_(None), 1), else_=0)
                ).label("records_without_cost"),
                func.avg(CallRecord.latency_ms).label("avg_latency_ms"),
                func.min(CallRecord.created_at).label("first_seen"),
                func.max(CallRecord.created_at).label("last_seen"),
            )
            .where(*conditions)
            .group_by(CallRecord.run_id)
            .subquery()
        )

        total = (
            await self._session.execute(
                select(func.count()).select_from(grouped)
            )
        ).scalar_one()

        rows = (
            await self._session.execute(
                select(grouped)
                .order_by(grouped.c.last_seen.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()

        summaries = [
            RunSummary(
                run_id=row.run_id,
                client_app=row.client_app,
                provider_count=int(row.provider_count or 0),
                request_count=int(row.request_count or 0),
                success_count=int(row.success_count or 0),
                failure_count=int(row.failure_count or 0),
                prompt_tokens=int(row.prompt_tokens or 0),
                completion_tokens=int(row.completion_tokens or 0),
                estimated_cost_usd=Decimal(row.estimated_cost_usd or 0),
                records_without_cost=int(row.records_without_cost or 0),
                avg_latency_ms=_as_optional_int(row.avg_latency_ms),
                first_seen=row.first_seen,
                last_seen=row.last_seen,
            )
            for row in rows
        ]

        return summaries, int(total)

    async def _grouped_usage(
        self, key_expr, filters: UsageFilter | None, limit: int
    ) -> list[GroupUsage]:
        filters = filters or UsageFilter()
        conditions = _build_conditions(filters)

        stmt = (
            select(
                key_expr,
                func.count(CallRecord.id),
                func.sum(case((CallRecord.http_status < 400, 1), else_=0)),
                func.sum(case((CallRecord.http_status >= 400, 1), else_=0)),
                func.sum(CallRecord.prompt_tokens),
                func.sum(CallRecord.completion_tokens),
                func.sum(CallRecord.estimated_cost_usd),
                func.avg(CallRecord.latency_ms),
            )
            .group_by(key_expr)
            .order_by(func.count(CallRecord.id).desc())
            .limit(limit)
        )
        if conditions:
            stmt = stmt.where(*conditions)

        rows = (await self._session.execute(stmt)).all()
        return [
            GroupUsage(
                # provider 在旧数据上为 NULL，用 "unknown" 归类而不是丢弃
                key=row[0] if row[0] is not None else "unknown",
                total_requests=int(row[1] or 0),
                success_count=int(row[2] or 0),
                failure_count=int(row[3] or 0),
                prompt_tokens=int(row[4] or 0),
                completion_tokens=int(row[5] or 0),
                estimated_cost_usd=Decimal(row[6] or 0),
                avg_latency_ms=_as_optional_int(row[7]),
            )
            for row in rows
        ]

    async def error_breakdown(
        self, filters: UsageFilter | None = None
    ) -> list[ErrorBreakdown]:
        """按错误类型统计失败次数，成功记录（none）不计入。"""
        filters = filters or UsageFilter()
        conditions = _build_conditions(filters)
        conditions.append(CallRecord.error_type != "none")

        stmt = (
            select(CallRecord.error_type, func.count(CallRecord.id))
            .where(*conditions)
            .group_by(CallRecord.error_type)
            .order_by(func.count(CallRecord.id).desc())
        )
        rows = (await self._session.execute(stmt)).all()
        return [ErrorBreakdown(error_type=row[0], count=int(row[1])) for row in rows]
