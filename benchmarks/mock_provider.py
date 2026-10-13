"""Mock 上游模型服务。

基准测试与故障注入用的假上游：延迟与故障比例都可控，
因此测试结果可复现，也不消耗真实模型额度。

它是独立的 ASGI 应用，用 uvicorn 起在单独端口，
ModelGate 通过 DEEPSEEK_BASE_URL 指过来即可。

启动：
    python -m benchmarks.mock_provider --port 9099 --latency-ms 30
参数：
    --latency-ms      每次响应前固定等待的毫秒数
    --jitter-ms       在固定延迟上叠加的随机抖动上限
    --error-rate      返回 500 的比例（0~1）
    --rate-limit-rate 返回 429 的比例（0~1）
    --timeout-rate    直接挂起不返回的比例（0~1），用于触发读取超时
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 计数器用于校验「网关发出的请求数」与「上游收到的请求数」是否一致
_STATS: dict[str, int] = {"total": 0, "ok": 0, "error": 0, "rate_limited": 0, "timeout": 0}


def create_mock_app(
    *,
    latency_ms: float = 30.0,
    jitter_ms: float = 0.0,
    error_rate: float = 0.0,
    rate_limit_rate: float = 0.0,
    timeout_rate: float = 0.0,
    hang_seconds: float = 120.0,
    access_log: bool = False,
) -> FastAPI:
    app = FastAPI(title="MockUpstream")

    def _note(request: Request, outcome: str) -> None:
        """逐请求记录来源与结果，用于确认网关是否真的把请求发到了这里。"""
        if not access_log:
            return
        peer = request.client.host if request.client else "?"
        print(f"MOCK peer={peer} {request.method} {request.url.path} -> {outcome}", flush=True)

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        _STATS["total"] += 1

        if latency_ms > 0 or jitter_ms > 0:
            await asyncio.sleep((latency_ms + random.uniform(0, jitter_ms)) / 1000)

        roll = random.random()

        if roll < timeout_rate:
            _STATS["timeout"] += 1
            _note(request, "hang")
            # 挂起直到网关超时或客户端断开
            await asyncio.sleep(hang_seconds)

        if roll < timeout_rate + rate_limit_rate:
            _STATS["rate_limited"] += 1
            _note(request, "429")
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "mock rate limited"}},
                headers={"Retry-After": "1"},
            )

        if roll < timeout_rate + rate_limit_rate + error_rate:
            _STATS["error"] += 1
            _note(request, "500")
            return JSONResponse(
                status_code=500, content={"error": {"message": "mock failure"}}
            )

        _STATS["ok"] += 1
        _note(request, "200")
        body = await request.json()
        model = body.get("model", "mock-model")
        prompt_tokens = 12
        completion_tokens = 8

        return JSONResponse(
            status_code=200,
            content={
                "id": f"chatcmpl-mock-{_STATS['total']}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        """上游侧计数，用于核对网关实际发出的调用次数。"""
        return dict(_STATS)

    @app.post("/stats/reset")
    async def reset_stats() -> dict[str, Any]:
        for key in _STATS:
            _STATS[key] = 0
        return dict(_STATS)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModelGate mock upstream")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--latency-ms", type=float, default=30.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--rate-limit-rate", type=float, default=0.0)
    parser.add_argument("--timeout-rate", type=float, default=0.0)
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="逐请求打印来源与结果，便于确认网关是否真的发出了请求",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn

    app = create_mock_app(
        latency_ms=args.latency_ms,
        jitter_ms=args.jitter_ms,
        error_rate=args.error_rate,
        rate_limit_rate=args.rate_limit_rate,
        timeout_rate=args.timeout_rate,
        access_log=args.access_log,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
