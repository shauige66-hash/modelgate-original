"""配额用量的数据访问层。

扣减必须原子，否则并发请求会同时读到「未超限」然后全部放行。
因此这里用单条 INSERT ... ON CONFLICT DO UPDATE ... WHERE used < limit ... RETURNING，
由数据库行锁串行化，应用层不做「先查再写」。

返回值语义很关键：
- 返回行  -> 扣减成功，used 为扣减后的值
- 返回 None -> 行已存在但 used 已达 limit，即超限（新行不可能返回 None，
  因为插入时 used 恒为 1，而 limit < 1 的配额在配置层就被跳过了）

调用方必须在**独立且已提交**的短事务里调用 try_consume，
绝不能把持有配额行锁的事务带进上游 HTTP 等待，否则同一配额下的请求会串行阻塞。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    QUOTA_SCOPE_CLIENT,
    QUOTA_SCOPE_RUN,
    QUOTA_WINDOW_DAILY,
    QUOTA_WINDOW_RUN_TOTAL,
    BudgetUsage,
)

# run 维度不使用时间窗口：一次 Agent 工作流的调用上限应在整个生命周期内有效，
# 不应因为跨过 UTC 零点而重置。用一个固定哨兵日期占位，让唯一键仍然成立。
RUN_WINDOW_SENTINEL = date(1970, 1, 1)


@dataclass(frozen=True)
class QuotaSnapshot:
    """某个身份在某个窗口的用量。"""

    scope: str
    identity: str
    window_start: date
    used: int
    limit: int
    remaining: int

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit


@dataclass(frozen=True)
class QuotaDecision:
    """一次配额判定的结果。"""

    allowed: bool
    scope: str
    identity: str
    used: int
    limit: int

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)


class BudgetExceededError(RuntimeError):
    """配额已用尽。由 app.errors 的处理器转成 429。

    定义在这一层是为了让异常处理器不必导入 app.budget，
    从而避免 errors <-> budget 的循环导入。
    """

    def __init__(self, decision: QuotaDecision) -> None:
        super().__init__(decision.scope)
        self.decision = decision


def run_identity(key_fingerprint: str, run_id: str) -> str:
    """Run 维度的身份键。

    把客户端指纹和 run_id 拼在一起，避免两个客户端用了同一个 run_id
    时被合并成同一条配额与统计。
    """
    return f"{key_fingerprint}:{run_id}"


def window_for(scope: str, now: date) -> tuple[str, date]:
    """返回该维度使用的 (window_kind, window_start)。

    client 维度按 UTC 自然日重置；run 维度不重置（用哨兵日期）。
    """
    if scope == QUOTA_SCOPE_RUN:
        return QUOTA_WINDOW_RUN_TOTAL, RUN_WINDOW_SENTINEL
    return QUOTA_WINDOW_DAILY, now


class BudgetRepository:
    """budget_usage 表的读写封装。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _insert_for_dialect(self, values: dict):
        """按方言取 INSERT 构造器。

        SQLite 的 on_conflict_do_update 与 PostgreSQL 语义一致，
        因此两者共用同一段扣减逻辑，测试与生产行为不会分叉。
        """
        dialect = self._session.get_bind().dialect.name
        if dialect == "postgresql":
            return pg_insert(BudgetUsage).values(**values)
        if dialect == "sqlite":
            return sqlite_insert(BudgetUsage).values(**values)

        raise RuntimeError(
            f"配额扣减依赖 ON CONFLICT，当前方言 {dialect!r} 不支持；"
            f"请使用 PostgreSQL 或 SQLite。"
        )

    async def try_consume(
        self,
        *,
        scope: str,
        identity: str,
        limit: int,
        window_start: date,
        window_kind: str = QUOTA_WINDOW_DAILY,
    ) -> int | None:
        """尝试占用一次配额，返回扣减后的用量；超限返回 None。

        不 commit：调用方负责在发起上游请求前提交这个短事务。
        """
        if limit < 1:
            raise ValueError("limit 必须 >= 1；无限额时不应调用 try_consume")

        statement = self._insert_for_dialect(
            {
                "scope": scope,
                "identity": identity,
                "window_kind": window_kind,
                "window_start": window_start,
                "used": 1,
            }
        )
        statement = statement.on_conflict_do_update(
            index_elements=["scope", "identity", "window_kind", "window_start"],
            set_={
                "used": BudgetUsage.used + 1,
                "updated_at": text("CURRENT_TIMESTAMP"),
            },
            where=BudgetUsage.used < limit,
        ).returning(BudgetUsage.used)

        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def read_usage(
        self,
        *,
        scope: str,
        identity: str,
        limit: int,
        window_start: date,
        window_kind: str = QUOTA_WINDOW_DAILY,
    ) -> QuotaSnapshot:
        """读取当前用量，不修改计数。"""
        result = await self._session.execute(
            select(BudgetUsage.used).where(
                BudgetUsage.scope == scope,
                BudgetUsage.identity == identity,
                BudgetUsage.window_kind == window_kind,
                BudgetUsage.window_start == window_start,
            )
        )
        used = result.scalar_one_or_none() or 0

        return QuotaSnapshot(
            scope=scope,
            identity=identity,
            window_start=window_start,
            used=used,
            limit=limit,
            remaining=max(limit - used, 0),
        )

    async def list_usage(
        self, *, window_start: date, scope: str | None = None
    ) -> list[BudgetUsage]:
        """列出用量行，供管理接口展示。

        run 维度用哨兵日期、不随时间重置，按日期等于 window_start 查询会漏掉它们，
        因此不带 scope 时同时返回 run 维度的全部行。
        """
        statement = select(BudgetUsage)

        if scope:
            statement = statement.where(BudgetUsage.scope == scope)
        else:
            statement = statement.where(
                (BudgetUsage.scope == QUOTA_SCOPE_RUN)
                | (BudgetUsage.window_start == window_start)
            )

        result = await self._session.execute(
            statement.order_by(BudgetUsage.used.desc(), BudgetUsage.scope)
        )
        return list(result.scalars().all())


__all__ = [
    "BudgetExceededError",
    "BudgetRepository",
    "QuotaDecision",
    "QuotaSnapshot",
    "QUOTA_SCOPE_CLIENT",
    "QUOTA_SCOPE_RUN",
    "QUOTA_WINDOW_DAILY",
    "run_identity",
]
