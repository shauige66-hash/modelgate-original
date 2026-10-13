"""模型注册表与服务商路由。

客户端只用 `model` 字段选择模型，上游地址与凭证只能来自配置：
模型名经过字典查表得到服务商，绝不参与 URL 构造或凭证选择，
因此调用方无法把请求指向任意目标。

未登记的模型会抛 UnknownModelError，由路由层转成 400，
不会静默回落到默认服务商。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.config import Settings


class UnknownModelError(LookupError):
    """请求的模型未登记在 model_registry 中。"""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


class ProviderNotConfiguredError(RuntimeError):
    """模型所属服务商缺少 API Key。"""

    def __init__(self, provider: str) -> None:
        super().__init__(provider)
        self.provider = provider


@dataclass(frozen=True)
class ModelRoute:
    """一个已登记模型的路由与计价信息。"""

    model: str
    provider: str
    price_input_per_million_usd: Decimal | None
    price_output_per_million_usd: Decimal | None

    @property
    def has_pricing(self) -> bool:
        return (
            self.price_input_per_million_usd is not None
            and self.price_output_per_million_usd is not None
        )


def parse_model_registry(entries: list[str]) -> dict[str, str]:
    """解析 `模型名:服务商` 列表。"""
    registry: dict[str, str] = {}
    for entry in entries:
        name, separator, provider = entry.partition(":")
        if not separator or not name.strip() or not provider.strip():
            raise ValueError(
                f"MODEL_REGISTRY 条目格式应为 模型名:服务商，收到 {entry!r}"
            )
        registry[name.strip()] = provider.strip()
    return registry


def parse_model_prices(entries: list[str]) -> dict[str, tuple[Decimal, Decimal]]:
    """解析 `模型名:输入价:输出价` 列表，单位为美元 / 百万 Token。"""
    prices: dict[str, tuple[Decimal, Decimal]] = {}
    for entry in entries:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 3 or not parts[0]:
            raise ValueError(
                f"MODEL_PRICES 条目格式应为 模型名:输入价:输出价，收到 {entry!r}"
            )
        name, raw_input, raw_output = parts
        try:
            prices[name] = (Decimal(raw_input), Decimal(raw_output))
        except InvalidOperation as exc:
            raise ValueError(f"MODEL_PRICES 中的价格不是合法数字：{entry!r}") from exc
    return prices


class ModelRouter:
    """按模型名解析服务商与价格。"""

    def __init__(self, settings: Settings) -> None:
        self._default_model = settings.default_model_name
        self._registry = parse_model_registry(settings.model_registry)
        self._model_prices = parse_model_prices(settings.model_prices)
        self._provider_prices = settings.provider_prices
        self._credentials = settings.provider_credentials
        self._base_urls = {name: url for name, (url, _) in self._credentials.items()}

        if self._default_model not in self._registry:
            raise ValueError(
                f"DEFAULT_MODEL_NAME={self._default_model!r} 未登记在 MODEL_REGISTRY 中"
            )

        unknown = set(self._registry.values()) - set(self._base_urls)
        if unknown:
            raise ValueError(
                f"MODEL_REGISTRY 引用了未知服务商：{sorted(unknown)}；"
                f"当前支持 {sorted(self._base_urls)}"
            )

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def registered_models(self) -> list[str]:
        return sorted(self._registry)

    @property
    def providers(self) -> list[str]:
        return sorted(self._base_urls)

    def is_provider_configured(self, provider: str) -> bool:
        """该服务商是否已配置 API Key。"""
        credentials = self._credentials.get(provider)
        return bool(credentials and credentials[1])

    def configured_providers(self) -> list[str]:
        return [name for name in self.providers if self.is_provider_configured(name)]

    def resolve(self, model: str | None) -> ModelRoute:
        """解析模型；未登记时抛 UnknownModelError。"""
        name = (model or self._default_model).strip()
        provider = self._registry.get(name)
        if provider is None:
            raise UnknownModelError(name)

        price_input, price_output = self._resolve_prices(name, provider)
        return ModelRoute(
            model=name,
            provider=provider,
            price_input_per_million_usd=price_input,
            price_output_per_million_usd=price_output,
        )

    def _resolve_prices(
        self, model: str, provider: str
    ) -> tuple[Decimal | None, Decimal | None]:
        """单模型价格优先，其次服务商默认价；价格为 0 视为「未配置」。

        把 0 当作未配置，是为了避免把「还没填价格」误当成「免费」。
        """
        override = self._model_prices.get(model)
        if override is not None:
            price_input, price_output = override
            return (
                price_input if price_input > 0 else None,
                price_output if price_output > 0 else None,
            )

        fallback = self._provider_prices.get(provider)
        if fallback is None:
            return None, None
        price_input, price_output = fallback
        return (
            price_input if price_input > 0 else None,
            price_output if price_output > 0 else None,
        )
