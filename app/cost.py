"""Token 成本估算。

关键约定：
1. 全程使用 decimal.Decimal，不用 float，避免金额累加误差。
2. 单价单位统一为「美元 / 百万 Token」，在配置与代码中显式标注。
3. 上游未返回 usage 时，Token 记为 None（未知），
   绝不能当成 0，否则统计会静默偏低。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from app.config import MILLION, Settings

# 金额保留 6 位小数：单次调用成本通常远小于 1 美分，
# 保留更多位可以减少大量小请求累加时的截断误差。
_COST_QUANTUM = Decimal("0.000001")


def quantize_cost(value: Decimal) -> Decimal:
    """把金额规范化到固定小数位，便于入库与展示。"""
    return value.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)


def calculate_cost(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_price_per_million: Decimal,
    output_price_per_million: Decimal,
) -> Decimal | None:
    """估算单次调用费用，单位：美元。

    返回 None 表示「无法估算」——即输入与输出 Token 都未知。
    只要有一侧 Token 已知，就按已知部分估算，未知部分不参与计算。

    参数中的价格单位为 美元 / 百万 Token。
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
    """使用当前配置的单价估算成本。

    说明：本项目只有一套全局单价，属于简化实现。
    真实多模型场景应按模型维护价格表（见 README「未来改进方向」）。
    """
    return calculate_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_price_per_million=settings.price_input_per_million_usd,
        output_price_per_million=settings.price_output_per_million_usd,
    )
