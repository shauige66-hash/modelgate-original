"""并发控制测试：单进程并发上限与等待超时。

测的是 ConcurrencyLimiter 与 HTTP 层的协作，不代表多副本部署的全局并发上限。
"""

from __future__ import annotations

import asyncio

import pytest

from app.concurrency import ConcurrencyLimiter, ConcurrencyTimeout
from tests.conftest import build_settings

CHAT_URL = "/v1/chat/completions"


def _chat_payload() -> dict:
    return {"messages": [{"role": "user", "content": "hi"}]}


class TestSemaphoreEnforcesLimit:
    async def test_peak_never_exceeds_limit(self) -> None:
        """limit=2 时，最多只有 2 个协程同时处于临界区。"""
        limiter = ConcurrencyLimiter(limit=2, acquire_timeout=5.0)

        current = 0
        peak = 0
        lock = asyncio.Lock()
        release = asyncio.Event()
        two_entered = asyncio.Event()

        async def worker() -> None:
            nonlocal current, peak
            async with limiter.slot():
                async with lock:
                    current += 1
                    peak = max(peak, current)
                    if current == 2:
                        two_entered.set()

                await release.wait()

                async with lock:
                    current -= 1

        tasks = [asyncio.create_task(worker()) for _ in range(6)]

        await asyncio.wait_for(two_entered.wait(), timeout=5.0)
        # 给其余协程时间去「尝试」进入；它们应该被信号量挡住
        await asyncio.sleep(0.1)

        async with lock:
            assert current == 2, "超过上限的请求不应进入被保护的代码段"

        release.set()
        await asyncio.gather(*tasks)

        assert peak == 2
        assert limiter.peak_in_flight == 2
        assert limiter.in_flight == 0
        assert limiter.available == 2

    async def test_limit_of_one_serializes_requests(self) -> None:
        """limit=1 时必须完全串行，峰值只能是 1。"""
        limiter = ConcurrencyLimiter(limit=1, acquire_timeout=5.0)

        current = 0
        peak = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal current, peak
            async with limiter.slot():
                async with lock:
                    current += 1
                    peak = max(peak, current)
                await asyncio.sleep(0.01)
                async with lock:
                    current -= 1

        await asyncio.gather(*(worker() for _ in range(5)))

        assert peak == 1
        assert limiter.in_flight == 0

    async def test_through_http_layer_peak_respects_limit(self, make_harness) -> None:
        """通过 HTTP 层并发调用，限制器记录的峰值不得超过配置上限。"""
        settings = build_settings(max_concurrent_requests=2)
        h = await make_harness(settings=settings)

        responses = await asyncio.gather(
            *(h.client.post(CHAT_URL, json=_chat_payload()) for _ in range(6))
        )

        assert all(r.status_code == 200 for r in responses)
        assert h.limiter.peak_in_flight <= 2
        assert h.limiter.in_flight == 0
        assert h.limiter.available == 2


class TestAcquireTimeout:
    async def test_waiting_too_long_raises_concurrency_timeout(self) -> None:
        """槽位被占满且等待超时后，必须抛出 ConcurrencyTimeout。"""
        limiter = ConcurrencyLimiter(limit=1, acquire_timeout=0.1)

        async with limiter.slot():
            with pytest.raises(ConcurrencyTimeout):
                async with limiter.slot():
                    pass  # pragma: no cover - 不应执行到这里

        # 超时不应破坏信号量状态
        assert limiter.available == 1

    async def test_concurrency_timeout_becomes_503(self, make_harness) -> None:
        """等待槽位超时 -> 对外返回 503。"""
        settings = build_settings(
            max_concurrent_requests=1,
            concurrency_acquire_timeout_seconds=0.15,
        )
        h = await make_harness(settings=settings)

        # 手工占住唯一的槽位，模拟「上游很慢，仍在处理中」
        async with h.limiter.slot():
            response = await h.client.post(CHAT_URL, json=_chat_payload())

        assert response.status_code == 503
        assert response.json()["error"]["type"] == "concurrency_timeout"

    async def test_slot_is_reused_after_timeout(self, make_harness) -> None:
        """一次超时之后服务必须能继续处理正常请求（不能死锁）。"""
        settings = build_settings(
            max_concurrent_requests=1,
            concurrency_acquire_timeout_seconds=0.15,
        )
        h = await make_harness(settings=settings)

        async with h.limiter.slot():
            timed_out = await h.client.post(CHAT_URL, json=_chat_payload())
        assert timed_out.status_code == 503

        # 槽位已释放，这次应该成功
        ok = await h.client.post(CHAT_URL, json=_chat_payload())
        assert ok.status_code == 200


class TestLimiterConfigurationValidation:
    def test_limit_cannot_be_below_one(self) -> None:
        with pytest.raises(ValueError):
            ConcurrencyLimiter(limit=0, acquire_timeout=1.0)

    def test_acquire_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ConcurrencyLimiter(limit=1, acquire_timeout=0)
