"""单进程并发限制。

用 asyncio.Semaphore 限制同时进行的上游请求数，避免单个实例耗尽本地连接池
或把上游打爆。等待槽位有上限，超时抛 ConcurrencyTimeout，由路由层转成 503。

注意：信号量只在当前进程内生效。多副本部署时每个副本各有一份配额，
实际总并发是「副本数 × limit」，因此它不是分布式限流。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from app.logging_config import get_logger

logger = get_logger(__name__)


class ConcurrencyTimeout(RuntimeError):
    """在限定时间内未能取得并发槽位。"""


class ConcurrencyLimiter:
    """基于 asyncio.Semaphore 的并发限制器。

    用法::

        async with limiter.slot():
            ...  # 受并发上限保护的上游请求
    """

    def __init__(self, limit: int, acquire_timeout: float) -> None:
        if limit < 1:
            raise ValueError("并发上限至少为 1")
        if acquire_timeout <= 0:
            raise ValueError("等待超时必须大于 0")

        self._limit = limit
        self._acquire_timeout = acquire_timeout
        self._semaphore = asyncio.Semaphore(limit)

        # 仅用于观测，不参与限流决策
        self._in_flight = 0
        self._peak_in_flight = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def available(self) -> int:
        """当前可用槽位数（近似值，仅供监控展示）。"""
        return self._semaphore._value  # noqa: SLF001 - 只读观测

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def peak_in_flight(self) -> int:
        """历史峰值，用于确认限流确实生效。"""
        return self._peak_in_flight

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """占用一个槽位，退出时释放。

        释放放在 finally，因此业务异常与客户端取消（CancelledError）
        都会归还槽位，不会因异常路径导致信号量泄漏。

        wait_for 超时会取消内部的 acquire()，此时信号量未被占用，无需补偿。
        """
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._acquire_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise ConcurrencyTimeout(
                f"等待并发槽位超过 {self._acquire_timeout:g}s（上限 {self._limit}）"
            ) from exc

        self._in_flight += 1
        self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
        try:
            yield
        finally:
            self._in_flight -= 1
            self._semaphore.release()
