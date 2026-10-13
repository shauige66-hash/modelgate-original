"""Token 成本计算测试。

覆盖 Decimal 精度、百万 Token 换算，以及 usage 缺失时记为未知而非 0。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.cost import calculate_cost, estimate_cost_for_settings, quantize_cost
from tests.conftest import build_settings

# 测试用单价：输入 1 美元 / 百万，输出 2 美元 / 百万，
# 便于心算核对结果。
INPUT_PRICE = Decimal("1")
OUTPUT_PRICE = Decimal("2")


class TestCalculateCost:
    def test_returns_decimal_type(self) -> None:
        """返回类型必须是 Decimal，不能是 float。"""
        result = calculate_cost(
            prompt_tokens=10,
            completion_tokens=10,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result is not None
        assert isinstance(result, Decimal)
        assert not isinstance(result, float)

    def test_million_tokens_of_input_costs_exactly_one_dollar(self) -> None:
        """100 万输入 Token × 1 美元/百万 = 1.000000 美元。"""
        result = calculate_cost(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result == Decimal("1.000000")

    def test_input_and_output_are_summed(self) -> None:
        """12 输入 + 8 输出：
        12/1e6*1 = 0.000012；8/1e6*2 = 0.000016；合计 0.000028。
        """
        result = calculate_cost(
            prompt_tokens=12,
            completion_tokens=8,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result == Decimal("0.000028")

    def test_unknown_tokens_returns_none_not_zero(self) -> None:
        """两侧 Token 都未知时必须返回 None。

        这是核心安全约定：如果返回 0，统计会把「未知」显示成「免费」。
        """
        result = calculate_cost(
            prompt_tokens=None,
            completion_tokens=None,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result is None

    def test_partial_unknown_still_estimates_known_side(self) -> None:
        """只有输入 Token 已知时，仅按输入侧估算，不虚构输出侧。"""
        result = calculate_cost(
            prompt_tokens=1_000_000,
            completion_tokens=None,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result == Decimal("1.000000")

    def test_zero_tokens_is_a_real_zero(self) -> None:
        """明确返回 0 表示「确实没用 Token」，与未知（None）区分开。"""
        result = calculate_cost(
            prompt_tokens=0,
            completion_tokens=0,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result == Decimal("0.000000")
        assert result is not None

    def test_binary_float_would_drift_but_decimal_does_not(self) -> None:
        """用 0.1 这类无法被二进制浮点精确表示的价格做校验。

        若内部使用 float，累加 100 次会出现可见偏差；
        Decimal 必须给出精确结果。
        """
        price = Decimal("0.1")
        result = calculate_cost(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            input_price_per_million=price,
            output_price_per_million=price,
        )

        # (1e6 + 1e6) / 1e6 * 0.1 = 0.2 精确值
        assert result == Decimal("0.200000")

    def test_repeated_addition_stays_exact(self) -> None:
        """多次小请求累加不应产生浮点漂移。"""
        per_call = calculate_cost(
            prompt_tokens=3,
            completion_tokens=0,
            input_price_per_million=Decimal("0.27"),
            output_price_per_million=Decimal("1.10"),
        )
        assert per_call is not None

        total = sum((per_call for _ in range(1000)), Decimal("0"))

        # 3 * 0.27 / 1e6 = 0.00000081 -> 量化后 0.000001
        assert total == Decimal("0.001000")

    @pytest.mark.parametrize(
        ("prompt_tokens", "completion_tokens", "expected"),
        [
            (0, 0, "0.000000"),
            (1, 0, "0.000001"),  # 1e-6 美元，四舍五入后正好是最小刻度
            (0, 1, "0.000002"),
            (12, 8, "0.000028"),
            (1_000_000, 1_000_000, "3.000000"),  # 1 + 2
        ],
    )
    def test_parametrized_values(
        self, prompt_tokens: int, completion_tokens: int, expected: str
    ) -> None:
        result = calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
        )

        assert result == Decimal(expected)


class TestQuantizeCost:
    def test_rounds_half_up(self) -> None:
        assert quantize_cost(Decimal("0.0000005")) == Decimal("0.000001")

    def test_keeps_six_decimal_places(self) -> None:
        assert quantize_cost(Decimal("1")) == Decimal("1.000000")


class TestEstimateCostForSettings:
    def test_uses_settings_prices(self) -> None:
        """按配置单价计算：输入 0.27、输出 1.10（美元/百万 Token）。"""
        settings = build_settings(
            price_input_per_million_usd=Decimal("0.27"),
            price_output_per_million_usd=Decimal("1.10"),
        )

        result = estimate_cost_for_settings(
            settings, prompt_tokens=1_000_000, completion_tokens=1_000_000
        )

        assert result == Decimal("1.370000")

    def test_unknown_usage_gives_none(self) -> None:
        settings = build_settings()

        result = estimate_cost_for_settings(
            settings, prompt_tokens=None, completion_tokens=None
        )

        assert result is None
