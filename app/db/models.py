"""SQLAlchemy 2.x 异步 ORM 模型。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


# 配额维度。client 按已认证客户端统计，run 按「客户端 + run_id」统计。
QUOTA_SCOPE_CLIENT = "client"
QUOTA_SCOPE_RUN = "run"

# 时间窗口。client 按 UTC 自然日重置；run 不重置，覆盖整个 Run 生命周期。
QUOTA_WINDOW_DAILY = "daily"
QUOTA_WINDOW_RUN_TOTAL = "run_total"


class BudgetUsage(Base):
    """配额用量计数。

    每行代表「某个身份在某个时间窗口内已用掉多少次调用」。
    扣减通过 INSERT ... ON CONFLICT DO UPDATE ... WHERE used < limit 完成，
    依赖数据库行锁保证并发下不会超发，因此不需要额外的分布式锁。

    身份由服务端确定：client 维度用网关 API Key 的指纹，
    run 维度用「指纹 + run_id」组合，避免不同客户端复用同名 run 时互相干扰。
    客户端自填的 X-Client-App 不参与身份计算。
    """

    __tablename__ = "budget_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    scope: Mapped[str] = mapped_column(String(16), nullable=False)

    # client 维度为 key 指纹；run 维度为 "指纹:run_id"
    identity: Mapped[str] = mapped_column(String(160), nullable=False)

    window_kind: Mapped[str] = mapped_column(String(16), nullable=False)

    # client 维度为当天日期；run 维度为固定哨兵日期（不随时间重置）
    window_start: Mapped[date] = mapped_column(Date, nullable=False)

    used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "scope",
            "identity",
            "window_kind",
            "window_start",
            name="uq_budget_usage_identity_window",
        ),
        Index("ix_budget_usage_window_start", "window_start"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BudgetUsage scope={self.scope!r} identity={self.identity!r} "
            f"window={self.window_start} used={self.used}>"
        )


class CallRecord(Base):
    """一次 LLM 调用的记录。

    prompt_tokens / completion_tokens / estimated_cost_usd 允许为 NULL，
    表示「未知」，用于区分「上游未返回 usage」与「确实用了 0 个 Token」。
    不保存用户 Prompt 原文，只保留 message_count / prompt_chars。

    新增的 provider / requested_model / response_model 与三个 agent 标签列
    由 app/db/migrations.py 以加列方式补丁到已有表，均为 NULLABLE。
    """

    __tablename__ = "call_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 网关生成的调用 ID，同时回传客户端用于对账
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # 实际处理该请求的服务商
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # 历史列：为了兼容既有数据保留，含义等同于 requested_model
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    # 客户端请求的模型名，以及上游响应里返回的模型名（可能是别名或快照版本）
    requested_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    response_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # 端到端耗时（毫秒），含等待并发槽位与所有重试
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # 额外发起的上游请求次数（0 表示一次成功）。默认不重试，因此通常为 0。
    retry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    http_status: Mapped[int] = mapped_column(Integer, nullable=False)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Numeric 而非 Float：金额需精确保存，并与 Decimal 对应
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
    )

    # 成功为 "none"，失败为 ErrorType 的取值
    error_type: Mapped[str] = mapped_column(String(48), nullable=False, default="none")

    # 排查用上下文，均不含敏感内容
    key_fingerprint: Mapped[str | None] = mapped_column(String(16), nullable=True)
    message_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 调用方提供的观测标签。仅用于统计与检索，不参与鉴权或权限判断。
    client_app: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        # 统计接口按时间倒序过滤，避免全表扫描
        Index("ix_call_records_created_at", "created_at"),
        Index("ix_call_records_model", "model"),
        Index("ix_call_records_provider", "provider"),
        Index("ix_call_records_client_app", "client_app"),
        Index("ix_call_records_agent_name", "agent_name"),
        Index("ix_call_records_run_id", "run_id"),
        # Run 级查询按「指纹 + run_id」过滤并排序，复合索引才能覆盖
        Index("ix_call_records_client_run_created", "key_fingerprint", "run_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<CallRecord id={self.id} request_id={self.request_id!r} "
            f"provider={self.provider!r} model={self.model!r} "
            f"status={self.http_status} error={self.error_type!r}>"
        )
