"""单进程并发控制。

用 asyncio.Semaphore 限制同时进行的上游请求数。

必要的说明（避免夸大能力）：
- 它只在「当前进程」内生效。多副本 / 多容器部署时，
  每个副本各自持有一份配额，因此它**不是**分布式限流，也不是生产级高并发方案。
- 它的作用是保护上游与自身资源，防止单个实例把上游打爆或本地连接池耗尽。

超时语义：
- 等待槽位有上限（CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS）。
- 超时抛 ConcurrencyTimeout，由调用方翻译成 HTTP 503。
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
            ...  # 这里的上游请求受并发上限保护
    """

    def __init__(self, limit: int, acquire_timeout: float) -> None:
        if limit < 1:
            raise ValueError("并发上限至少为 1")
        if acquire_timeout <= 0:
            raise ValueError("等待超时必须大于 0")

        self._limit = limit
        self._acquire_timeout = acquire_timeout
        self._semaphore = asyncio.Semaphore(limit)

        # 仅用于可观测性，不参与限流决策
        self._in_flight = 0
        self._peak_in_flight = 0

    @property
    def limit(self) -> int:
        """并发上限。"""
        return self._limit

    @property
    def available(self) -> int:
        """当前可用槽位数量（近似值，仅供监控展示）。"""
        return self._semaphore._value  # noqa: SLF001 - 只读观测，无副作用

    @property
    def in_flight(self) -> int:
        """当前正在执行的请求数。"""
        return self._in_flight

    @property
    def peak_in_flight(self) -> int:
        """历史峰值并发数，用于验证限流是否真的生效。"""
        return self._peak_in_flight

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """获取一个并发槽位。

        在 finally 中释放信号量，因此：
        - 业务代码抛异常会释放
        - 客户端取消（asyncio.CancelledError）也会释放
        不会因为异常路径导致信号量泄漏和后续死锁。
        """
        try:
            # wait_for 在超时后会取消内部的 acquire()，
            # 此时信号量不会被占用，无需额外补偿。
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
