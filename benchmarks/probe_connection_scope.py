"""用 SQLAlchemy 连接池事件观测「一次请求里数据库连接被持有多久」。

要回答的问题：
- 配额预扣是否在请求生命周期内不必要地占着连接或事务？
- 连接是在上游调用之前就归还，还是被一路持有到响应结束？

做法：给引擎挂上 checkout / checkin 事件，打上时间戳，
再看一次请求的时间线。
"""

from __future__ import annotations

import asyncio
import os
import time

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

EVENTS: list[tuple[str, float]] = []


async def main() -> None:
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-mock")
    settings = get_settings()
    engine = create_async_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=False,
    )

    started = time.perf_counter()

    @event.listens_for(engine.sync_engine, "checkout")
    def _on_checkout(dbapi_conn, record, proxy) -> None:  # noqa: ANN001
        EVENTS.append(("checkout", time.perf_counter() - started))

    @event.listens_for(engine.sync_engine, "checkin")
    def _on_checkin(dbapi_conn, record) -> None:  # noqa: ANN001
        EVENTS.append(("checkin ", time.perf_counter() - started))

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.budget import BudgetGuard, BudgetLimits
    from app.db.repository import CallRecordRepository, NewCallRecord
    from tests.conftest import build_settings

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    limits = BudgetLimits(build_settings(client_daily_request_quota=100))
    guard = BudgetGuard(session_factory=factory, limits=limits)

    # 模拟一次请求：预扣配额 -> （上游调用占位）-> 写调用记录
    await guard.reserve(key_fingerprint="probe-fp", run_id=None)
    quota_done = time.perf_counter() - started

    # 上游调用耗时（这里用 sleep 代替，模拟 LLM 推理）
    await asyncio.sleep(0.30)
    upstream_done = time.perf_counter() - started

    async with factory() as session:
        repository = CallRecordRepository(session)
        await repository.insert(
            NewCallRecord(
                request_id=f"probe-{time.time()}",
                provider="deepseek",
                model="deepseek-chat",
                latency_ms=300,
                http_status=200,
                error_type="none",
                prompt_tokens=12,
                completion_tokens=8,
            )
        )
        await session.commit()
    record_done = time.perf_counter() - started

    await engine.dispose()

    print(f"预扣完成      t={quota_done*1000:7.1f} ms")
    print(f"上游调用完成  t={upstream_done*1000:7.1f} ms  (模拟占了 300ms)")
    print(f"记录落库完成  t={record_done*1000:7.1f} ms")
    print()
    print("连接池事件时间线：")
    for name, ts in EVENTS:
        print(f"  {name}  t={ts*1000:7.1f} ms")

    # 计算上游调用期间是否有连接被持有
    holds: list[tuple[float, float]] = []
    open_at: float | None = None
    for name, ts in EVENTS:
        if name == "checkout":
            open_at = ts
        elif open_at is not None:
            holds.append((open_at, ts))
            open_at = None
    if open_at is not None:
        holds.append((open_at, record_done))

    print()
    print("每次连接持有区间：")
    for begin, end in holds:
        during_upstream = begin < upstream_done and end > quota_done
        flag = "  <-- 与上游调用重叠" if during_upstream else ""
        print(f"  {begin*1000:7.1f} -> {end*1000:7.1f} ms  ({(end-begin)*1000:6.1f} ms){flag}")


if __name__ == "__main__":
    asyncio.run(main())
