"""配额治理。

两类配额，均以「请求次数」为单位：

1. client —— 按已认证客户端（API Key 指纹）限制每天（UTC 自然日）的调用次数。
2. run    —— 按「客户端指纹 + X-Run-Id」限制**整个 Run 生命周期**的调用次数，
   不随时间重置：一次 Agent 工作流跨过 UTC 零点不应重新获得额度。

run 身份带上指纹，避免不同客户端复用同名 run 时互相吃掉配额。

身份只来自服务端可验证的信息：
- client 维度用网关 API Key 的指纹（已认证结果），不使用客户端自填的 X-Client-App；
- run 维度额外要求 run_id，而 run_id 只是标签，不参与身份认证。

配额不是 Token / 美元预算。本模块只做请求计数，
没有实现金额预留与结算，因此不能宣称具备费用硬上限。

并发语义：扣减在**独立短事务**中完成并立刻提交，
提交之后才发起上游 HTTP 请求。因此持有配额行锁的时间与上游耗时无关，
同一配额下的请求不会串行阻塞。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.budget_repository import (
    QUOTA_SCOPE_CLIENT,
    QUOTA_SCOPE_RUN,
    BudgetRepository,
    QuotaDecision,
    run_identity,
    window_for,
)
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BudgetOutcome:
    """整体配额判定。"""

    allowed: bool
    violations: tuple[QuotaDecision, ...]

    @property
    def first_violation(self) -> QuotaDecision | None:
        return self.violations[0] if self.violations else None


class BudgetLimits:
    """从配置解析出的生效限额。0 表示该维度不限制。"""

    def __init__(self, settings: Settings) -> None:
        self.client_daily_requests = settings.client_daily_request_quota
        self.run_max_requests = settings.run_max_llm_calls

    @property
    def client_enabled(self) -> bool:
        return self.client_daily_requests > 0

    @property
    def run_enabled(self) -> bool:
        return self.run_max_requests > 0


class BudgetGuard:
    """在调用上游之前完成配额预扣。"""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        limits: BudgetLimits,
    ) -> None:
        self._session_factory = session_factory
        self._limits = limits

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @property
    def enabled(self) -> bool:
        return self._limits.client_enabled or self._limits.run_enabled

    async def reserve(
        self,
        *,
        key_fingerprint: str,
        run_id: str | None,
        now: datetime | None = None,
    ) -> BudgetOutcome:
        """预扣配额。任一维度超限则整体拒绝，并回滚已扣的部分。

        使用独立 session 且立即 commit：行锁在发起上游请求前就已释放。
        未启用的维度不参与，因此默认配置下不产生任何额外写入。
        """
        if not self.enabled:
            return BudgetOutcome(allowed=True, violations=())

        today = _utc_today(now)

        # (scope, identity, limit, window_kind, window_start)
        checks: list[tuple[str, str, int, str, date]] = []
        if self._limits.client_enabled:
            checks.append(
                (
                    QUOTA_SCOPE_CLIENT,
                    key_fingerprint,
                    self._limits.client_daily_requests,
                    *window_for(QUOTA_SCOPE_CLIENT, today),
                )
            )
        if self._limits.run_enabled and run_id:
            checks.append(
                (
                    QUOTA_SCOPE_RUN,
                    run_identity(key_fingerprint, run_id),
                    self._limits.run_max_requests,
                    *window_for(QUOTA_SCOPE_RUN, today),
                )
            )

        async with self._session_factory() as session:
            repository = BudgetRepository(session)
            consumed: list[QuotaDecision] = []
            violation: QuotaDecision | None = None

            for scope, identity, limit, window_kind, window_start in checks:
                used = await repository.try_consume(
                    scope=scope,
                    identity=identity,
                    limit=limit,
                    window_start=window_start,
                    window_kind=window_kind,
                )
                if used is None:
                    current = await repository.read_usage(
                        scope=scope,
                        identity=identity,
                        limit=limit,
                        window_start=window_start,
                        window_kind=window_kind,
                    )
                    violation = QuotaDecision(
                        allowed=False,
                        scope=scope,
                        identity=identity,
                        used=current.used,
                        limit=limit,
                    )
                    break

                consumed.append(
                    QuotaDecision(
                        allowed=True,
                        scope=scope,
                        identity=identity,
                        used=used,
                        limit=limit,
                    )
                )

            if violation is not None:
                # 回滚本次已扣的其它维度，避免「run 超限但 client 已被扣」的错账
                await session.rollback()
                logger.warning(
                    "配额拒绝 | scope=%s used=%s limit=%s",
                    violation.scope,
                    violation.used,
                    violation.limit,
                )
                return BudgetOutcome(allowed=False, violations=(violation,))

            await session.commit()

        return BudgetOutcome(allowed=True, violations=tuple(consumed))

    async def snapshot(
        self, *, key_fingerprint: str, run_id: str | None, now: datetime | None = None
    ) -> list[QuotaDecision]:
        """只读查询当前用量，不扣减。供管理接口使用。"""
        if not self.enabled:
            return []

        today = _utc_today(now)

        async with self._session_factory() as session:
            repository = BudgetRepository(session)
            results: list[QuotaDecision] = []

            if self._limits.client_enabled:
                window_kind, window_start = window_for(QUOTA_SCOPE_CLIENT, today)
                snapshot = await repository.read_usage(
                    scope=QUOTA_SCOPE_CLIENT,
                    identity=key_fingerprint,
                    limit=self._limits.client_daily_requests,
                    window_start=window_start,
                    window_kind=window_kind,
                )
                results.append(_to_decision(snapshot))

            if self._limits.run_enabled and run_id:
                window_kind, window_start = window_for(QUOTA_SCOPE_RUN, today)
                snapshot = await repository.read_usage(
                    scope=QUOTA_SCOPE_RUN,
                    identity=run_identity(key_fingerprint, run_id),
                    limit=self._limits.run_max_requests,
                    window_start=window_start,
                    window_kind=window_kind,
                )
                results.append(_to_decision(snapshot))

            return results


def _to_decision(snapshot) -> QuotaDecision:
    return QuotaDecision(
        allowed=not snapshot.exceeded,
        scope=snapshot.scope,
        identity=snapshot.identity,
        used=snapshot.used,
        limit=snapshot.limit,
    )


def _utc_today(now: datetime | None) -> date:
    """client 维度的窗口起点：UTC 自然日。

    固定窗口而非滑动窗口，是因为它可以用「日期」做唯一键的一部分，
    一次 UPSERT 就能判断并计数；滑动窗口需要额外的时间桶与清理逻辑。
    代价是边界处（UTC 零点前后）可能出现短时间双倍额度，已在 README 说明。
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date()
