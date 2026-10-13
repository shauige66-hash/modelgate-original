"""并发压测与指标计算。

指标口径说明（避免把两个不同的东西混为一谈）：

- **端到端延迟**：从客户端发出请求到收到响应的完整耗时，
  包含 Mock 上游注入的固定延迟 + 网关处理 + 网络往返。
- **网关自身开销**：端到端延迟减去 Mock 上游注入的固定延迟。
  这是近似值 —— 它同时包含了网关处理与本地网络往返，
  在单机压测下两者无法严格分离，因此报告中会一并展示原始数据，
  不单独宣称「网关延迟 = X」。

并发模型：用固定数量的并发 worker 循环发请求，直到总请求数达标。
worker 数即并发度；请求按完成顺序记录延迟。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    """单次请求的观测结果。"""

    latency_ms: float
    status_code: int
    ok: bool


@dataclass
class LoadResult:
    """一轮压测的汇总。"""

    concurrency: int
    total_requests: int
    wall_seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    status_counts: dict[int, int] = field(default_factory=dict)
    # 失败明细：状态码 -> 出现次数；客户端异常 -> 出现次数
    error_kinds: dict[str, int] = field(default_factory=dict)
    failures: int = 0
    # 失败响应体的 error.type（网关侧错误分类），用于定位失败位置
    gateway_error_types: dict[str, int] = field(default_factory=dict)
    failure_samples: list[dict] = field(default_factory=list)

    @property
    def rps(self) -> float:
        return self.total_requests / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def error_rate(self) -> float:
        return self.failures / self.total_requests if self.total_requests else 0.0

    def percentile(self, pct: float) -> float:
        """最近秩百分位；样本为空时返回 0。"""
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(int(len(ordered) * pct / 100), len(ordered) - 1)
        return ordered[index]

    def summary(self, *, injected_latency_ms: float = 0.0) -> dict:
        """可序列化的指标快照。"""
        p50 = self.percentile(50)
        p95 = self.percentile(95)
        p99 = self.percentile(99)

        return {
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "wall_seconds": round(self.wall_seconds, 3),
            "rps": round(self.rps, 2),
            "error_rate": round(self.error_rate, 6),
            "failures": self.failures,
            "latency_ms": {
                "min": round(min(self.latencies_ms), 3) if self.latencies_ms else 0,
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3),
                "max": round(max(self.latencies_ms), 3) if self.latencies_ms else 0,
                "mean": round(
                    sum(self.latencies_ms) / len(self.latencies_ms), 3
                )
                if self.latencies_ms
                else 0,
            },
            "gateway_overhead_ms": {
                "note": (
                    "端到端延迟减去 Mock 上游注入延迟的近似值；"
                    "同时包含网关处理与本地网络往返，无法严格分离"
                ),
                "injected_upstream_latency_ms": injected_latency_ms,
                "p50_approx": round(max(p50 - injected_latency_ms, 0), 3),
                "p95_approx": round(max(p95 - injected_latency_ms, 0), 3),
                "p99_approx": round(max(p99 - injected_latency_ms, 0), 3),
            },
            "status_counts": {
                str(code): count for code, count in sorted(self.status_counts.items())
            },
            "error_kinds": dict(sorted(self.error_kinds.items())),
            "gateway_error_types": dict(sorted(self.gateway_error_types.items())),
            "failure_samples": self.failure_samples,
        }


async def run_load(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict,
    concurrency: int,
    total_requests: int,
    timeout_seconds: float = 60.0,
) -> LoadResult:
    """按指定并发跑固定总请求数。

    每个 worker 复用同一个连接池，避免把「建连接」的耗时算进被测路径。
    """
    started = time.perf_counter()
    result = LoadResult(concurrency=concurrency, total_requests=total_requests)

    # worker 之间用共享计数器分配请求，保证总数精确
    next_index = 0
    lock = asyncio.Lock()
    collected: list[RequestResult] = []

    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2
    )

    async with httpx.AsyncClient(
        timeout=timeout_seconds, limits=limits, headers=headers
    ) as client:

        async def worker() -> None:
            nonlocal next_index
            while True:
                async with lock:
                    if next_index >= total_requests:
                        return
                    next_index += 1

                request_started = time.perf_counter()
                try:
                    response = await client.post(url, json=payload)
                    latency = (time.perf_counter() - request_started) * 1000
                    ok = response.status_code == 200
                    collected.append(
                        RequestResult(
                            latency_ms=latency,
                            status_code=response.status_code,
                            ok=ok,
                        )
                    )
                    if not ok:
                        _capture_failure(result, response, latency)
                except httpx.HTTPError as exc:
                    latency = (time.perf_counter() - request_started) * 1000
                    collected.append(
                        RequestResult(
                            latency_ms=latency, status_code=0, ok=False
                        )
                    )
                    _capture_failure(result, None, latency, exc)

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    result.wall_seconds = time.perf_counter() - started

    for item in collected:
        result.latencies_ms.append(item.latency_ms)
        result.status_counts[item.status_code] = (
            result.status_counts.get(item.status_code, 0) + 1
        )
        if not item.ok:
            result.failures += 1

    return result


def _capture_failure(
    result: LoadResult,
    response: httpx.Response | None,
    latency_ms: float,
    exc: httpx.HTTPError | None = None,
) -> None:
    """记录失败的结构化信息，最多留 5 个样本。

    关键是把「客户端连不上网关」与「网关返回错误」区分开：
    前者 status_code=0 且带异常类型，后者能读到网关的 error.type。
    """
    sample: dict = {"latency_ms": round(latency_ms, 1)}

    if exc is not None:
        sample["layer"] = "client_to_gateway"
        sample["exception"] = type(exc).__name__
        sample["message"] = str(exc)[:160]
    elif response is not None:
        sample["layer"] = "gateway_response"
        sample["status"] = response.status_code
        sample["request_id"] = response.headers.get("X-Request-Id")
        # 记录原始响应特征：用于判断 502 是网关自己生成还是从上游透传
        sample["content_type"] = response.headers.get("Content-Type")
        sample["server"] = response.headers.get("Server")
        sample["body_prefix"] = response.text[:200]
        try:
            body = response.json().get("error", {})
            sample["gateway_error_type"] = body.get("type")
            sample["gateway_message"] = body.get("message")
            if body.get("type"):
                key = str(body["type"])
                result.gateway_error_types[key] = (
                    result.gateway_error_types.get(key, 0) + 1
                )
        except ValueError:
            pass

    if len(result.failure_samples) < 5:
        result.failure_samples.append(sample)
