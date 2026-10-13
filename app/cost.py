"""Token 成本估算。

金额全程使用 decimal.Decimal，避免浮点累加误差；单价单位为美元 / 百万 Token。
上游未返回 usage 时 Token 记为 None（未知），不按 0 处理，否则统计会静默偏低。

这里是估算而非计费：单价来自配置，同一模型在不同服务商或不同时间段
价格都可能不同，结果仅供预算参考。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from app.config import MILLION

# 保留 6 位小数：单次调用成本通常远小于 1 美分，
# 位数太少会让大量小请求累加时产生可见截断误差。
_COST_QUANTUM = Decimal("0.000001")


def quantize_cost(value: Decimal) -> Decimal:
    """规范化金额的小数位。"""
    return value.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)


def calculate_cost(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_price_per_million: Decimal | None,
    output_price_per_million: Decimal | None,
) -> Decimal | None:
    """估算单次调用费用（美元），价格为美元 / 百万 Token。

    返回 None 表示无法估算：两侧 Token 都未知，或对应方向的单价未配置。
    只要已知的那一侧有价格，就按已知部分估算，不会把缺失的一侧当作免费。
    """
    if prompt_tokens is None and completion_tokens is None:
        return None

    total = Decimal("0")
    priced = False

    if prompt_tokens is not None and input_price_per_million is not None:
        total += Decimal(prompt_tokens) * input_price_per_million / MILLION
        priced = True

    if completion_tokens is not None and output_price_per_million is not None:
        total += Decimal(completion_tokens) * output_price_per_million / MILLION
        priced = True

    # 有 Token 但没有任何可用单价时保持未知，不返回 0
    return quantize_cost(total) if priced else None
