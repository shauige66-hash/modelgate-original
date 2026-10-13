"""阶段 E 演示编排：拉起 Mock 上游 + 使用独立测试库的网关，跑多 Agent 工作流。

只使用 TEST_DATABASE_URL（库名必须 _test 结尾），不触碰业务库，也不调用真实模型。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from examples.modelgate_client import ModelGateConfig
from examples.run_workflow import run_workflow, _print_summary

ROOT = Path(__file__).resolve().parent.parent
MOCK_PORT = 9099
GATEWAY_PORT = 8000
CLIENT_KEY = "sk-demo-client-key"
ADMIN_KEY = "sk-demo-admin-key"


def wait(url: str, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False


def gateway_env(database_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": database_url,
            "DEEPSEEK_BASE_URL": f"http://127.0.0.1:{MOCK_PORT}",
            "DEEPSEEK_API_KEY": "sk-mock-upstream",
            "MODEL_REGISTRY": "deepseek-chat:deepseek",
            "DEFAULT_MODEL_NAME": "deepseek-chat",
            "DEEPSEEK_PRICE_INPUT_PER_MILLION_USD": "0.20",
            "DEEPSEEK_PRICE_OUTPUT_PER_MILLION_USD": "0.80",
            "CLIENT_API_KEYS": CLIENT_KEY,
            "ADMIN_API_KEY": ADMIN_KEY,
            "CLIENT_DAILY_REQUEST_QUOTA": "0",
            "RUN_MAX_LLM_CALLS": "0",
            "LOG_LEVEL": "WARNING",
        }
    )
    return env


async def main() -> int:
    database_url = os.environ.get("TEST_DATABASE_URL", "")
    if not database_url.endswith("_test"):
        print("需要 TEST_DATABASE_URL 且库名以 _test 结尾", file=sys.stderr)
        return 2

    mock = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.mock_provider",
            "--port",
            str(MOCK_PORT),
            "--latency-ms",
            "40",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gateway = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(GATEWAY_PORT),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=gateway_env(database_url),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait(f"http://127.0.0.1:{MOCK_PORT}/stats"):
            print("Mock 上游启动失败", file=sys.stderr)
            return 1
        if not wait(f"http://127.0.0.1:{GATEWAY_PORT}/health"):
            print("网关启动失败（端口 8000 是否被占用？）", file=sys.stderr)
            return 1

        print("Mock 上游与网关已就绪，开始工作流\n")

        config = ModelGateConfig(
            base_url=f"http://127.0.0.1:{GATEWAY_PORT}",
            client_key=CLIENT_KEY,
            admin_key=ADMIN_KEY,
        )
        summary = await run_workflow(
            config=config, model="deepseek-chat", executor_steps=3
        )
        _print_summary(summary)

        # 同时演示列表接口
        async with httpx.AsyncClient(base_url=config.base_url, timeout=30) as client:
            listing = await client.get(
                "/admin/runs?limit=5", headers={"X-Admin-Key": ADMIN_KEY}
            )
            body = listing.json()
            print()
            print(f"GET /admin/runs 共 {body['total']} 个 run（分页 limit=5）")
            for item in body["runs"]:
                print(
                    f"  {item['run_id']:<28} 调用 {item['request_count']:<3} "
                    f"Token {item['total_tokens']:<6} 成本 {item['estimated_cost_usd']}"
                )
        return 0
    finally:
        for process in (gateway, mock):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
