"""SQLAlchemy 2.x 异步 ORM 模型。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class CallRecord(Base):
    """一次 LLM 调用的记录。

    prompt_tokens / completion_tokens / estimated_cost_usd 允许为 NULL，
    表示「未知」，用于区分「上游未返回 usage」与「确实用了 0 个 Token」。
    不保存用户 Prompt 原文，只保留 message_count / prompt_chars。
    """

    __tablename__ = "call_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 网关生成的调用 ID，同时回传客户端用于对账
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    model: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # 端到端耗时（毫秒），含等待并发槽位的时间
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

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

    __table_args__ = (
        # 统计接口按时间倒序过滤，避免全表扫描
        Index("ix_call_records_created_at", "created_at"),
        Index("ix_call_records_model", "model"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<CallRecord id={self.id} request_id={self.request_id!r} "
            f"model={self.model!r} status={self.http_status} "
            f"error={self.error_type!r}>"
        )
