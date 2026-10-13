"""多 Agent 工作流演示。

模拟 Planner → Executor × N → Critic 的典型编排：三个节点共享同一个 run_id，
各自以自己的 agent_name 调用 ModelGate，最后通过管理接口按 run 聚合核对。

这里**不实现** Agent 循环、检查点、工具执行与上下文管理 —— 那些是上层运行时的职责。
本脚本只演示「Agent 如何通过网关调用模型并被正确计量」。

用法：
    # 1) 先起网关（可用 Mock 上游，不需要真实 API Key）
    python -m examples.run_workflow

    # 指定 Mock 上游与模型
    python -m examples.run_workflow --model deepseek-chat --executor-steps 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from examples.modelgate_client import ModelGateClient, ModelGateConfig

PLANNER_PROMPT = "把任务拆成若干可执行步骤，只输出步骤列表。"
EXECUTOR_PROMPT = "执行当前步骤，输出结果要点。"
CRITIC_PROMPT = "审查前面所有输出，指出问题并给出结论。"


def _messages(system_hint: str, user_content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_hint},
        {"role": "user", "content": user_content},
    ]


async def run_workflow(
    *,
    config: ModelGateConfig,
    model: str,
    executor_steps: int,
    run_id: str | None = None,
) -> dict:
    """跑一次完整工作流，返回管理接口给出的聚合结果。"""
    run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    client = ModelGateClient(config, run_id=run_id)

    try:
        print(f"Run ID: {run_id}")
        print(f"模型  : {model}")
        print()

        # Planner
        plan = await client.complete(
            agent_name="planner",
            messages=_messages(PLANNER_PROMPT, "为「实现一个 LLM 网关」制定计划"),
            model=model,
        )
        print(f"  planner  -> {plan['choices'][0]['message']['content'][:60]}")

        # Executor：多个步骤，每个步骤一次调用
        for step in range(1, executor_steps + 1):
            result = await client.complete(
                agent_name="executor",
                messages=_messages(EXECUTOR_PROMPT, f"执行第 {step} 步"),
                model=model,
            )
            print(f"  executor -> step {step}: {result['choices'][0]['message']['content'][:40]}")

        # Critic
        critique = await client.complete(
            agent_name="critic",
            messages=_messages(CRITIC_PROMPT, "审查上面的执行结果"),
            model=model,
        )
        print(f"  critic   -> {critique['choices'][0]['message']['content'][:60]}")

        summary = await client.run_summary()
        return summary
    finally:
        await client.aclose()


def _print_summary(summary: dict) -> None:
    totals = summary["totals"]
    quality = summary["data_quality"]

    print()
    print("=" * 62)
    print(f"Run {summary['run_id']} 聚合结果（来自 GET /admin/runs/{summary['run_id']}）")
    print("=" * 62)
    print(
        f"  调用次数 : {totals['total_requests']}"
        f"  (成功 {totals['success_count']} / 失败 {totals['failure_count']})"
    )
    print(
        f"  Token    : 输入 {totals['prompt_tokens']} + 输出 {totals['completion_tokens']}"
        f" = {totals['total_tokens']}"
    )
    print(f"  估算成本 : {totals['estimated_cost_usd']} USD")
    print(f"  平均耗时 : {totals['avg_latency_ms']} ms")

    print()
    print("  各 Agent 节点分布：")
    for item in summary["by_agent"]:
        print(
            f"    {item['key']:<12} 调用 {item['total_requests']:<3} "
            f"Token {item['total_tokens']:<6} 成本 {item['estimated_cost_usd']:<12} "
            f"平均 {item['avg_latency_ms']} ms"
        )

    print()
    print("  按服务商 / 模型：")
    for item in summary["by_provider"]:
        print(
            f"    provider {item['key']:<12} 调用 {item['total_requests']:<3} "
            f"成本 {item['estimated_cost_usd']}"
        )
    for item in summary["by_model"]:
        print(
            f"    model    {item['key']:<12} 调用 {item['total_requests']:<3} "
            f"成本 {item['estimated_cost_usd']}"
        )

    if summary["errors"]:
        print()
        print("  错误分布：")
        for item in summary["errors"]:
            print(f"    {item['error_type']}: {item['count']}")

    print()
    print(
        f"  数据完整性：未知 Token {quality['records_with_unknown_tokens']} 条，"
        f"未知成本 {quality['records_without_cost']} 条，"
        f"含重试 {quality['records_with_retries']} 条"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="多 Agent 工作流演示")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--executor-steps", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = ModelGateConfig.from_env()
    if not config.client_key:
        print("缺少 CLIENT_API_KEYS / MODELGATE_CLIENT_KEYS", file=sys.stderr)
        return 2

    summary = await run_workflow(
        config=config,
        model=args.model,
        executor_steps=args.executor_steps,
        run_id=args.run_id,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
