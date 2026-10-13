"""对照诊断：确认基准里的 RPS 瓶颈来自哪里。

分别测量：
1. Mock 上游直连 —— 压测工具与 Mock 自身的上限
2. 网关鉴权拒绝 —— 不碰数据库时的网关吞吐
3. 网关完整链路（配额关）—— 加上 PostgreSQL 写入
4. 网关完整链路（配额开）—— 再加上配额预扣的多一次事务与会话

四组用同一套压测代码与同一台机器，差异才有意义。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from benchmarks.load import run_load

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "x"}]}
MOCK_PORT = 9099
GATEWAY_PORT = 9097
CLIENT_KEY = "sk-diag-client-key"
ADMIN_KEY = "sk-diag-admin-key"


def wait(url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False


def gateway_env(database_url: str, *, client_quota: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": database_url,
            "DEEPSEEK_BASE_URL": f"http://127.0.0.1:{MOCK_PORT}",
            "DEEPSEEK_API_KEY": "sk-mock",
            "MODEL_REGISTRY": "deepseek-chat:deepseek",
            "DEFAULT_MODEL_NAME": "deepseek-chat",
            "CLIENT_API_KEYS": CLIENT_KEY,
            "ADMIN_API_KEY": ADMIN_KEY,
            "MAX_CONCURRENT_REQUESTS": "64",
            "CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS": "30",
            "UPSTREAM_TIMEOUT_SECONDS": "30",
            "CLIENT_DAILY_REQUEST_QUOTA": str(client_quota),
            "RUN_MAX_LLM_CALLS": "0",
            "LOG_LEVEL": "WARNING",
        }
    )
    return env


async def measure(label: str, url: str, headers: dict[str, str]) -> None:
    for concurrency in (10, 20, 50):
        result = await run_load(
            url=url,
            headers=headers,
            payload=PAYLOAD,
            concurrency=concurrency,
            total_requests=400,
            timeout_seconds=40,
        )
        s = result.summary()
        print(
            f"  {label:<22} c={concurrency:<3} rps={s['rps']:<8} "
            f"p50={s['latency_ms']['p50']:<9} p95={s['latency_ms']['p95']:<9} "
            f"p99={s['latency_ms']['p99']:<9} err={s['error_rate']}",
            flush=True,
        )


async def main() -> int:
    database_url = os.environ["TEST_DATABASE_URL"]

    mock = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.mock_provider",
            "--port",
            str(MOCK_PORT),
            "--latency-ms",
            "30",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gateway = None

    def spawn_gateway(quota: int) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(GATEWAY_PORT),
                "--log-level", "warning",
            ],
            cwd=str(ROOT),
            env=gateway_env(database_url, client_quota=quota),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        if not wait(f"http://127.0.0.1:{MOCK_PORT}/stats"):
            print("mock 启动失败", file=sys.stderr)
            return 1

        print("1) Mock 上游直连（压测工具上限）")
        await measure("mock-only", f"http://127.0.0.1:{MOCK_PORT}/chat/completions", {})

        gateway = spawn_gateway(0)
        if not wait(f"http://127.0.0.1:{GATEWAY_PORT}/health"):
            print("网关启动失败", file=sys.stderr)
            return 1

        print("\n2) 网关鉴权拒绝（不碰数据库、不打上游）")
        await measure(
            "gateway-auth-reject",
            f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
            {"Authorization": "Bearer wrong-key"},
        )

        print("\n3) 网关完整链路，配额关闭")
        await measure(
            "gateway-quota-off",
            f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
            {"Authorization": f"Bearer {CLIENT_KEY}"},
        )

        gateway.terminate()
        gateway.wait(timeout=15)
        await asyncio.sleep(1)

        gateway = spawn_gateway(10_000_000)
        if not wait(f"http://127.0.0.1:{GATEWAY_PORT}/health"):
            print("网关启动失败（配额）", file=sys.stderr)
            return 1

        print("\n4) 网关完整链路，配额开启")
        await measure(
            "gateway-quota-on",
            f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
            {"Authorization": f"Bearer {CLIENT_KEY}"},
        )
    finally:
        for process in (gateway, mock):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
