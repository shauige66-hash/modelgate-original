"""性能稳定性验收基准。

在**同一台机器、同一批次**内跑完所有场景，使配额开关的对照尽量接近同条件。
所有数据来自 Mock 上游与独立测试库，不涉及真实模型服务与业务库。

关键测量纪律（本轮踩过的坑，写下来避免重犯）：
- 压测客户端复用单一连接池，避免每轮重建连接池引入额外连接抖动
- 每一轮都单独记录，不只给平均值
- 记录失败明细（客户端异常类型 / 网关 error.type / 裸 502），
  以便区分「被测系统返回错误」与「本机环回传输失败」
- 同时记录 PostgreSQL 连接峰值与锁等待，避免把数据库问题误判为网络问题
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
MOCK_PORT = 9099
GATEWAY_PORT = 9092
CLIENT_KEY = "sk-acc-client-key"
ADMIN_KEY = "sk-acc-admin-key"
PAYLOAD = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "x"}]}


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


def gateway_env(
    database_url: str,
    *,
    client_quota: int,
    run_quota: int,
    pool: str,
    overflow: str,
    maxconc: str,
) -> dict:
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
            # 固定不改：这是被测配置，调大它就等于把问题藏起来
            "MAX_CONCURRENT_REQUESTS": maxconc,
            "CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS": "30",
            "UPSTREAM_TIMEOUT_SECONDS": "30",
            "DB_POOL_SIZE": pool,
            "DB_MAX_OVERFLOW": overflow,
            "CLIENT_DAILY_REQUEST_QUOTA": str(client_quota),
            "RUN_MAX_LLM_CALLS": str(run_quota),
            "LOG_LEVEL": "warning",
        }
    )
    return env


async def reset_database(database_url: str) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    name = database_url.rsplit("/", 1)[-1].split("?")[0]
    if not name.endswith("_test"):
        raise RuntimeError(f"拒绝清理非测试库：{database_url}")

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM budget_usage"))
    finally:
        await engine.dispose()


async def database_counts(database_url: str) -> dict:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            total = (
                await conn.execute(text("SELECT count(*) FROM call_records"))
            ).scalar_one()
            with_tokens = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM call_records "
                        "WHERE prompt_tokens IS NOT NULL"
                    )
                )
            ).scalar_one()
            retried = (
                await conn.execute(
                    text("SELECT count(*) FROM call_records WHERE retry_count > 0")
                )
            ).scalar_one()
    finally:
        await engine.dispose()
    return {
        "call_records": int(total),
        "with_tokens": int(with_tokens),
        "with_retries": int(retried),
    }


class PgSampler:
    """采样 PostgreSQL 连接数峰值与锁等待。"""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self.peak_total = 0
        self.peak_active = 0
        self.peak_lock_wait = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        import asyncpg

        conn = await asyncpg.connect(self._dsn)
        try:
            while not self._stop.is_set():
                row = await conn.fetchrow(
                    "SELECT count(*) AS total, "
                    "count(*) FILTER (WHERE state = 'active') AS active, "
                    "count(*) FILTER (WHERE wait_event_type = 'Lock') AS lock_wait "
                    "FROM pg_stat_activity WHERE datname = current_database()"
                )
                self.peak_total = max(self.peak_total, row["total"])
                self.peak_active = max(self.peak_active, row["active"])
                self.peak_lock_wait = max(self.peak_lock_wait, row["lock_wait"])
                await asyncio.sleep(0.2)
        finally:
            await conn.close()

    async def __aenter__(self) -> "PgSampler":
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task:
            await self._task


async def run_round(
    client: httpx.AsyncClient,
    *,
    concurrency: int,
    total: int,
    pg_dsn: str,
) -> dict:
    counts: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()
    latencies: list[float] = []
    next_index = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= total:
                    return
                next_index += 1

            started = time.perf_counter()
            try:
                response = await client.post(
                    f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                    json=PAYLOAD,
                )
                latencies.append((time.perf_counter() - started) * 1000)
                counts[response.status_code] += 1
                if response.status_code != 200:
                    body = response.text
                    if body.strip().startswith("{"):
                        try:
                            etype = response.json().get("error", {}).get("type")
                        except ValueError:
                            etype = "unparsable"
                        kinds[f"gw_{response.status_code}:{etype}"] += 1
                    else:
                        # 没有 JSON 体：不是网关生成的错误响应
                        kinds[f"nongw_{response.status_code}_raw"] += 1
            except httpx.HTTPError as exc:
                latencies.append((time.perf_counter() - started) * 1000)
                counts[0] += 1
                kinds[f"client_{type(exc).__name__}"] += 1

    async with PgSampler(pg_dsn) as sampler:
        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started

    ordered = sorted(latencies)
    ok = counts.get(200, 0)

    def pct(value: float) -> float:
        if not ordered:
            return 0.0
        index = min(int(len(ordered) * value / 100), len(ordered) - 1)
        return round(ordered[index], 1)

    return {
        "concurrency": concurrency,
        "requests": total,
        "wall_s": round(wall, 2),
        "rps": round(total / wall, 1),
        "ok": ok,
        "error_rate": round((total - ok) / total, 4),
        "latency_ms": {
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "max": round(ordered[-1], 1) if ordered else 0.0,
        },
        "status_counts": {str(k): v for k, v in sorted(counts.items())},
        "failure_kinds": dict(sorted(kinds.items())),
        "pg_peak": {
            "total": sampler.peak_total,
            "active": sampler.peak_active,
            "lock_wait": sampler.peak_lock_wait,
        },
    }


async def measure_scenario(
    *,
    database_url: str,
    pg_dsn: str,
    label: str,
    client_quota: int,
    run_quota: int,
    levels: list[int],
    requests: int,
    repeats: int,
    pool: str,
    overflow: str,
    maxconc: str,
    mock_latency_ms: float,
) -> dict:
    gateway = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(GATEWAY_PORT), "--log-level", "warning"],
        cwd=str(ROOT),
        env=gateway_env(
            database_url,
            client_quota=client_quota,
            run_quota=run_quota,
            pool=pool,
            overflow=overflow,
            maxconc=maxconc,
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    result: dict = {"label": label, "levels": {}}

    try:
        if not wait(f"http://127.0.0.1:{GATEWAY_PORT}/health"):
            raise RuntimeError("网关启动失败")

        limits = httpx.Limits(
            max_connections=max(levels) * 2, max_keepalive_connections=max(levels) * 2
        )
        # 整个场景复用同一个客户端，保证连接池条件在配额开关两组之间一致
        async with httpx.AsyncClient(
            timeout=60.0,
            limits=limits,
            headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        ) as client:
            for concurrency in levels:
                rounds = []
                for _ in range(repeats):
                    await reset_database(database_url)
                    rounds.append(
                        await run_round(
                            client,
                            concurrency=concurrency,
                            total=requests,
                            pg_dsn=pg_dsn,
                        )
                    )
                    counts = await database_counts(database_url)

                rps_values = sorted(item["rps"] for item in rounds)
                median_rps = rps_values[len(rps_values) // 2]
                total_ok = sum(item["ok"] for item in rounds)
                total_requests = sum(item["requests"] for item in rounds)

                merged_kinds: collections.Counter = collections.Counter()
                for item in rounds:
                    merged_kinds.update(item["failure_kinds"])

                result["levels"][f"c{concurrency}"] = {
                    "concurrency": concurrency,
                    "rounds": rounds,
                    "median_rps": median_rps,
                    "rps_all": rps_values,
                    "total_requests": total_requests,
                    "total_ok": total_ok,
                    "error_rate": round(
                        (total_requests - total_ok) / total_requests, 4
                    ),
                    "failure_kinds": dict(sorted(merged_kinds.items())),
                    "db_after_last_round": counts,
                    "pg_peak": rounds[-1]["pg_peak"],
                }
                entry = result["levels"][f"c{concurrency}"]
                print(
                    f"  [{label}] c={concurrency:<3} "
                    f"rps={median_rps:<7} (all={rps_values}) "
                    f"p50={rounds[-1]['latency_ms']['p50']:<7} "
                    f"p99={rounds[-1]['latency_ms']['p99']:<9} "
                    f"err={entry['error_rate']:<7} "
                    f"ok={total_ok}/{total_requests}"
                )
                if merged_kinds:
                    print(f"       失败明细: {dict(merged_kinds)}")
    finally:
        if gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=15)
            except subprocess.TimeoutExpired:
                gateway.kill()

    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description="性能稳定性验收基准")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mock-latency-ms", type=float, default=30.0)
    parser.add_argument("--db-pool-size", default="10")
    parser.add_argument("--db-max-overflow", default="20")
    parser.add_argument("--max-concurrent", default="64")
    parser.add_argument("--label", default="acceptance")
    args = parser.parse_args()

    database_url = os.getenv("TEST_DATABASE_URL", "")
    if not database_url.endswith("_test"):
        print("需要 TEST_DATABASE_URL 且库名以 _test 结尾", file=sys.stderr)
        return 2
    pg_dsn = database_url.replace("postgresql+asyncpg://", "postgresql://")

    mock = subprocess.Popen(
        [sys.executable, "-m", "benchmarks.mock_provider", "--port", str(MOCK_PORT),
         "--latency-ms", str(args.mock_latency_ms)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    report: dict = {
        "label": args.label,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "mock_latency_ms": args.mock_latency_ms,
            "gateway": {
                "max_concurrent_requests": args.max_concurrent,
                "db_pool_size": args.db_pool_size,
                "db_max_overflow": args.db_max_overflow,
                "upstream_timeout_s": 30,
                "processes": 1,
            },
            "load_generator": {
                "client_reused_across_rounds": True,
                "max_connections": max(args.concurrency) * 2,
            },
            "caveat": (
                "本机 Mock 压测。负载生成器、网关、Mock 上游与 PostgreSQL 同机运行，"
                "不代表真实模型服务或线上生产环境的表现。"
            ),
        },
        "scenarios": {},
    }

    try:
        if not wait(f"http://127.0.0.1:{MOCK_PORT}/stats"):
            print("mock 启动失败", file=sys.stderr)
            return 1

        print(f"c={args.concurrency} requests={args.requests} repeats={args.repeats}")
        print(f"mock 延迟={args.mock_latency_ms}ms\n")

        report["scenarios"]["quota_off"] = await measure_scenario(
            database_url=database_url,
            pg_dsn=pg_dsn,
            label="quota_off",
            client_quota=0,
            run_quota=0,
            levels=args.concurrency,
            requests=args.requests,
            repeats=args.repeats,
            pool=args.db_pool_size,
            overflow=args.db_max_overflow,
            maxconc=args.max_concurrent,
            mock_latency_ms=args.mock_latency_ms,
        )

        print()
        # 配额上限设得足够高：测的是「启用配额检查的代价」，不是拒绝路径
        report["scenarios"]["quota_on"] = await measure_scenario(
            database_url=database_url,
            pg_dsn=pg_dsn,
            label="quota_on",
            client_quota=args.requests * args.repeats * 2 + 10_000,
            run_quota=0,
            levels=args.concurrency,
            requests=args.requests,
            repeats=args.repeats,
            pool=args.db_pool_size,
            overflow=args.db_max_overflow,
            maxconc=args.max_concurrent,
            mock_latency_ms=args.mock_latency_ms,
        )

        print("\n=== 配额开关对比（中位 RPS）===")
        for level in args.concurrency:
            key = f"c{level}"
            off = report["scenarios"]["quota_off"]["levels"][key]
            on = report["scenarios"]["quota_on"]["levels"][key]
            delta = (on["median_rps"] - off["median_rps"]) / off["median_rps"] * 100
            print(
                f"  c={level:<3} off={off['median_rps']:<7} on={on['median_rps']:<7} "
                f"变化={delta:+.1f}%"
            )
    finally:
        if mock.poll() is None:
            mock.terminate()
            try:
                mock.wait(timeout=15)
            except subprocess.TimeoutExpired:
                mock.kill()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{args.label}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
