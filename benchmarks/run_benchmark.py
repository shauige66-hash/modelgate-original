"""基准测试编排。

自己拉起 Mock 上游与 ModelGate（两个 uvicorn 子进程），
在真实 PostgreSQL 上跑 10/20/50 并发各若干请求，落盘原始 JSON。

设计要点：
- 只使用独立测试库（TEST_DATABASE_URL，库名必须 _test 结尾），
  跑完即清空该库的调用记录重新开始，不触碰业务库。
- 配额对照在同一环境、同一参数下先后执行，只切换配额配置。
- 数据库记录完整率 = 实际落库条数 / 上游收到的请求数，
  两者对不上说明有记录丢失。
- 报告里所有延迟都标注口径：端到端包含 Mock 上游注入延迟。

用法：
    $env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
    python -m benchmarks.run_benchmark --requests 1000 --concurrency 10 20 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from benchmarks.load import LoadResult, run_load

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"

MOCK_PORT = 9099
GATEWAY_PORT = 9098

CLIENT_KEY = "sk-bench-client-key"
ADMIN_KEY = "sk-bench-admin-key"


def _wait_for(url: str, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code < 500:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.4)
    return False


def _spawn(module: str, *, env: dict[str, str], port: int) -> subprocess.Popen:
    """用当前解释器拉起一个 uvicorn 子进程。"""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _gateway_env(
    *,
    database_url: str,
    client_quota: int,
    run_quota: int,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": database_url,
            "DEEPSEEK_BASE_URL": f"http://127.0.0.1:{MOCK_PORT}",
            "DEEPSEEK_API_KEY": "sk-mock-upstream-key",
            # 基准只路由到一个 mock 上游，注册表按需收敛
            "MODEL_REGISTRY": "deepseek-chat:deepseek",
            "DEFAULT_MODEL_NAME": "deepseek-chat",
            "DEEPSEEK_PRICE_INPUT_PER_MILLION_USD": "0.20",
            "DEEPSEEK_PRICE_OUTPUT_PER_MILLION_USD": "0.80",
            "CLIENT_API_KEYS": CLIENT_KEY,
            "ADMIN_API_KEY": ADMIN_KEY,
            "MAX_CONCURRENT_REQUESTS": "64",
            "CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS": "30",
            "UPSTREAM_TIMEOUT_SECONDS": "30",
            "CLIENT_DAILY_REQUEST_QUOTA": str(client_quota),
            "RUN_MAX_LLM_CALLS": str(run_quota),
            "LOG_LEVEL": "WARNING",
        }
    )
    return env


async def _reset_database(database_url: str) -> None:
    """清空测试库的调用记录与配额，保证每轮从同一起点开始。

    仅对库名以 _test 结尾的库执行，业务库不会被碰到。
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    if not database_url.rsplit("/", 1)[-1].split("?")[0].endswith("_test"):
        raise RuntimeError(
            f"拒绝清空非测试库：{database_url}；库名必须以 _test 结尾"
        )

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM call_records"))
            await conn.execute(text("DELETE FROM budget_usage"))
    finally:
        await engine.dispose()


async def _database_counts(database_url: str) -> dict[str, int]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            calls = (
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
            with_cost = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM call_records "
                        "WHERE estimated_cost_usd IS NOT NULL"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    return {
        "call_records": int(calls),
        "with_tokens": int(with_tokens),
        "with_cost": int(with_cost),
    }


async def _mock_stats() -> dict[str, int]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"http://127.0.0.1:{MOCK_PORT}/stats")
        return response.json()


async def _reset_mock_stats() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(f"http://127.0.0.1:{MOCK_PORT}/stats/reset")


async def _count_quota_rows(database_url: str) -> dict[str, int]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            total = (
                await conn.execute(text("SELECT count(*) FROM budget_usage"))
            ).scalar_one()
            max_used = (
                await conn.execute(text("SELECT coalesce(max(used), 0) FROM budget_usage"))
            ).scalar_one()
    finally:
        await engine.dispose()

    return {"quota_rows": int(total), "max_used": int(max_used)}


def _merge_repeats(runs: list[dict]) -> dict:
    """把多次重复合并成一条结果：指标取中位数，完整性取最小值。

    单次绝对值受机器负载影响很大，取中位数比取平均更能抵抗偶发尖峰；
    完整性取最小值是保守做法 —— 只要有一轮丢记录就应当暴露出来。
    """
    import statistics

    merged = dict(runs[0])
    merged["repeats"] = len(runs)

    injected = merged["gateway_overhead_ms"]["injected_upstream_latency_ms"]
    merged["rps"] = round(statistics.median(r["rps"] for r in runs), 3)
    merged["error_rate"] = round(
        statistics.median(r["error_rate"] for r in runs), 6
    )

    for key in ("p50", "p95", "p99", "mean"):
        merged["latency_ms"][key] = round(
            statistics.median(r["latency_ms"][key] for r in runs), 3
        )

    # 开销近似值必须跟着中位数一起重算，否则会与上面的延迟对不上
    for key in ("p50", "p95", "p99"):
        merged["gateway_overhead_ms"][f"{key}_approx"] = round(
            max(merged["latency_ms"][key] - injected, 0), 3
        )

    merged["per_repeat"] = [
        {
            "rps": run["rps"],
            "p50": run["latency_ms"]["p50"],
            "p95": run["latency_ms"]["p95"],
            "p99": run["latency_ms"]["p99"],
            "db_call_records": run["integrity"]["db_call_records"],
            "coverage": run["integrity"]["record_coverage_of_upstream"],
        }
        for run in runs
    ]
    # 覆盖率取最小值：只要有一轮漏记录就应当暴露出来
    merged["integrity"]["record_coverage_of_upstream"] = min(
        run["integrity"]["record_coverage_of_upstream"] for run in runs
    )
    merged["integrity"]["all_successes_persisted"] = all(
        run["integrity"]["all_successes_persisted"] for run in runs
    )
    merged["integrity"]["note"] += "；重复多次时覆盖率取各轮最小值"
    return merged


def _environment_snapshot(
    *,
    database_url: str,
    mock_latency_ms: float,
    mock_error_rate: float,
    mock_rate_limit_rate: float,
    upstream_timeout_seconds: float,
    settings: argparse.Namespace,
) -> dict:
    """记录可复现所需的机器与环境信息。"""
    import sqlalchemy

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "packages": {
            "sqlalchemy": sqlalchemy.__version__,
            "httpx": httpx.__version__,
        },
        "database": {
            "url_database": database_url.rsplit("/", 1)[-1],
            "engine": "postgresql+asyncpg",
        },
        "gateway": {
            "max_concurrent_requests": 64,
            "concurrency_acquire_timeout_seconds": 30,
            "upstream_timeout_seconds": upstream_timeout_seconds,
            "processes": 1,
            "note": "单 uvicorn 进程；并发上限是进程内的，不是分布式限流",
        },
        "mock_upstream": {
            "latency_ms": mock_latency_ms,
            "error_rate": mock_error_rate,
            "rate_limit_rate": mock_rate_limit_rate,
            "note": "延迟为响应前固定等待；数值不含真实模型推理时间",
        },
        "load_params": {
            "concurrency_levels": settings.concurrency,
            "requests_per_level": settings.requests,
        },
        "measurement_caveats": {
            "single_process_gateway": (
                "网关是单个 uvicorn 进程，并发上限是进程内的；"
                "不代表多副本部署的实际总并发"
            ),
            "colocated_processes": (
                "压测客户端、网关、Mock 上游、PostgreSQL 全部跑在同一台机器上，"
                "互相争抢 CPU，因此绝对吞吐低于独立部署时的水平"
            ),
            "noise_floor": (
                "本机同时运行 IDE / 浏览器等程序，实测同一配置重复运行的"
                "吞吐波动可达约 1.5 倍；因此横向比较同一批次内的相对变化才有意义，"
                "不应把单次绝对值当作容量规划依据"
            ),
            "latency_scope": (
                "端到端延迟包含 Mock 上游注入的固定延迟；"
                "gateway_overhead_ms 是减去注入延迟后的近似值，"
                "其中仍含本地网络往返，无法与网关处理耗时严格分离"
            ),
            "not_real_providers": (
                "全部数据来自 Mock 上游，不代表 DeepSeek / OpenAI 等真实服务的表现"
            ),
        },
        "caveat": (
            "全部数据来自本机 Mock 上游，不代表 DeepSeek/OpenAI 等真实服务的表现。"
            "端到端延迟包含 Mock 注入的固定延迟。"
        ),
    }


def _scenario_result(
    result: LoadResult,
    *,
    injected_latency_ms: float,
    mock_delta: dict[str, int],
    db_counts: dict[str, int],
    quota_counts: dict[str, int],
) -> dict:
    summary = result.summary(injected_latency_ms=injected_latency_ms)

    upstream_received = mock_delta.get("total", 0)
    persisted = db_counts["call_records"]
    success = result.status_counts.get(200, 0)

    # 每条到达上游的请求都应留下一条记录，因此落库数不应少于上游收到数。
    # 反过来不成立：上游未能建立连接（connection refused 等）时，
    # 请求没到上游但仍是一条需要留痕的失败调用。
    # 所以这里用两个方向分别表达，不合成一个含混的比率。
    coverage = round(persisted / upstream_received, 6) if upstream_received else 1.0
    success_persisted = persisted >= success

    summary["integrity"] = {
        "upstream_requests_received": upstream_received,
        "upstream_ok": mock_delta.get("ok", 0),
        "upstream_rate_limited": mock_delta.get("rate_limited", 0),
        "upstream_error": mock_delta.get("error", 0),
        "gateway_2xx": success,
        "gateway_total": result.total_requests,
        "db_call_records": persisted,
        "db_records_with_tokens": db_counts["with_tokens"],
        "db_records_with_cost": db_counts["with_cost"],
        "record_coverage_of_upstream": coverage,
        "all_successes_persisted": success_persisted,
        "note": (
            "record_coverage_of_upstream = 落库记录数 / 上游收到请求数，"
            "应 >= 1：每条到达上游的请求都要留痕，"
            "另有未连上上游的失败调用也会落库。"
            "all_successes_persisted 校验所有 2xx 都有对应记录。"
            "被网关在调用前拒绝的请求（配额 429、400、503）"
            "既不到上游也不产生调用记录。"
        ),
    }
    summary["quota"] = quota_counts
    return summary


async def main() -> int:
    parser = argparse.ArgumentParser(description="ModelGate benchmark")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--mock-latency-ms", type=float, default=30.0)
    parser.add_argument("--mock-error-rate", type=float, default=0.0)
    parser.add_argument("--mock-rate-limit-rate", type=float, default=0.0)
    parser.add_argument("--upstream-timeout", type=float, default=30.0)
    parser.add_argument(
        "--quota-comparison",
        action="store_true",
        help="额外跑一轮开启配额的对照（同一环境同一参数）",
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每个场景重复次数；>1 时额外输出中位数，用于抵消机器负载波动",
    )
    args = parser.parse_args()

    database_url = os.getenv("TEST_DATABASE_URL", "")
    if not database_url:
        print("需要设置 TEST_DATABASE_URL（库名必须以 _test 结尾）", file=sys.stderr)
        return 2

    results: dict = {
        "label": args.label,
        "environment": _environment_snapshot(
            database_url=database_url,
            mock_latency_ms=args.mock_latency_ms,
            mock_error_rate=args.mock_error_rate,
            mock_rate_limit_rate=args.mock_rate_limit_rate,
            upstream_timeout_seconds=args.upstream_timeout,
            settings=args,
        ),
        "scenarios": {},
    }

    mock_env = dict(os.environ)
    mock = _spawn("benchmarks.mock_provider:create_mock_app", env=mock_env, port=MOCK_PORT)
    gateway = None

    try:
        if not _wait_for(f"http://127.0.0.1:{MOCK_PORT}/stats"):
            print("Mock 上游启动失败", file=sys.stderr)
            return 1

        scenario_configs = [
            ("quota_off", {"client_quota": 0, "run_quota": 0}),
        ]
        if args.quota_comparison:
            # 配额上限设得足够高：这轮测量的是「检查配额的代价」，
            # 而不是拒绝路径本身
            scenario_configs.append(
                (
                    "quota_on",
                    {"client_quota": args.requests * 3 + 1000, "run_quota": 0},
                )
            )

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "benchmark"}],
        }
        headers = {"Authorization": f"Bearer {CLIENT_KEY}"}

        for scenario_name, quota in scenario_configs:
            if gateway is not None:
                gateway.terminate()
                gateway.wait(timeout=15)
                await asyncio.sleep(1.0)

            # 迁移由网关启动时自动执行
            gateway = _spawn(
                "app.main:app",
                env=_gateway_env(
                    database_url=database_url,
                    client_quota=quota["client_quota"],
                    run_quota=quota["run_quota"],
                ),
                port=GATEWAY_PORT,
            )

            if not _wait_for(f"http://127.0.0.1:{GATEWAY_PORT}/health"):
                print(f"网关启动失败（{scenario_name}）", file=sys.stderr)
                return 1

            scenario_results = {}
            for concurrency in args.concurrency:
                runs = []
                for _ in range(max(args.repeats, 1)):
                    await _reset_database(database_url)
                    await _reset_mock_stats()

                    result = await run_load(
                        url=f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                        headers=headers,
                        payload=payload,
                        concurrency=concurrency,
                        total_requests=args.requests,
                        timeout_seconds=args.upstream_timeout + 10,
                    )

                    mock_stats = await _mock_stats()
                    db_counts = await _database_counts(database_url)
                    quota_counts = await _count_quota_rows(database_url)

                    runs.append(
                        _scenario_result(
                            result,
                            injected_latency_ms=args.mock_latency_ms,
                            mock_delta=mock_stats,
                            db_counts=db_counts,
                            quota_counts=quota_counts,
                        )
                    )

                entry = runs[0] if len(runs) == 1 else _merge_repeats(runs)
                scenario_results[f"concurrency_{concurrency}"] = entry

                print(
                    f"[{scenario_name}] c={concurrency:<3} "
                    f"rps={entry['rps']:<8} p50={entry['latency_ms']['p50']:<7} "
                    f"p95={entry['latency_ms']['p95']:<7} p99={entry['latency_ms']['p99']:<7} "
                    f"err={entry['error_rate']:<8} "
                    f"records={entry['integrity']['db_call_records']}"
                )

            results["scenarios"][scenario_name] = scenario_results

    finally:
        for process in (gateway, mock):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{args.label}-{stamp}.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
