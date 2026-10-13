"""Token 成本估算。

金额全程使用 decimal.Decimal，避免浮点累加误差；单价单位为美元 / 百万 Token。
上游未返回 usage 时 Token 记为 None（未知），不按 0 处理，否则统计会静默偏低。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from app.config import MILLION, Settings

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
    input_price_per_million: Decimal,
    output_price_per_million: Decimal,
) -> Decimal | None:
    """估算单次调用费用（美元），价格为美元 / 百万 Token。

    两侧 Token 都未知时返回 None；只有一侧已知时仅按已知部分估算。
    """
    if prompt_tokens is None and completion_tokens is None:
        return None

    total = Decimal("0")

    if prompt_tokens is not None:
        total += Decimal(prompt_tokens) * input_price_per_million / MILLION

    if completion_tokens is not None:
        total += Decimal(completion_tokens) * output_price_per_million / MILLION

    return quantize_cost(total)


def estimate_cost_for_settings(
    settings: Settings,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> Decimal | None:
    """按当前配置单价估算成本。

    仅有一套全局单价，未按模型区分，属简化实现；多模型场景需要价格表。
    """
    return calculate_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_price_per_million=settings.price_input_per_million_usd,
        output_price_per_million=settings.price_output_per_million_usd,
    )
