"""应用配置。

所有可变参数集中于此，通过环境变量或 .env 覆盖，源码中不含密钥。
密钥字段使用 SecretStr，避免 repr()、日志或异常回溯意外打印明文。
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MILLION = Decimal("1000000")

# 默认单价，单位：美元 / 百万 Token。仅为估算口径，不是官方报价，
# 部署前请按各服务商实际价格覆盖对应的环境变量。
DEFAULT_PRICE_INPUT = Decimal("0")
DEFAULT_PRICE_OUTPUT = Decimal("0")

# NoDecode 关闭 pydantic-settings 对复杂类型的 JSON 解析，
# 否则 CLIENT_API_KEYS=a,b 会因不是合法 JSON 而解析失败。
CommaSeparated = Annotated[list[str], NoDecode]
CommaSeparatedSecrets = Annotated[list[SecretStr], NoDecode]


def _split_comma(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """从环境变量 / .env 读取的配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ModelGate"
    app_version: str = "0.2.0"
    app_debug: bool = False
    log_level: str = "INFO"

    # ---- 服务商凭证 ----
    # 每个服务商使用独立的 Base URL、API Key 与连接池。
    # Key 留空表示该服务商不可用，调用时返回明确错误而不是发到错误的目标。
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com"
    openai_api_key: SecretStr = SecretStr("")

    # 每服务商的单价（美元 / 百万 Token），可作为该服务商下所有模型的默认价。
    deepseek_price_input_per_million_usd: Decimal = DEFAULT_PRICE_INPUT
    deepseek_price_output_per_million_usd: Decimal = DEFAULT_PRICE_OUTPUT
    openai_price_input_per_million_usd: Decimal = DEFAULT_PRICE_INPUT
    openai_price_output_per_million_usd: Decimal = DEFAULT_PRICE_OUTPUT

    # 模型注册表。格式：模型名:服务商，多个用逗号分隔。
    # 只有登记在此的模型才会被转发；模型名不参与 URL 构造，
    # 上游地址只能来自上面的 *_base_url，避免调用方左右请求目标。
    model_registry: CommaSeparated = Field(
        default_factory=lambda: [
            "deepseek-chat:deepseek",
            "deepseek-reasoner:deepseek",
        ]
    )

    # 单模型价格覆盖。格式：模型名:输入价:输出价，多个用逗号分隔。
    # 未覆盖的模型回落到所属服务商的默认价；两者都缺失时成本记为未知。
    model_prices: CommaSeparated = Field(default_factory=list)

    # 未指定 model 时使用的默认模型，必须已登记在 model_registry 中。
    default_model_name: str = "deepseek-chat"

    upstream_timeout_seconds: float = 60.0
    upstream_max_connections: int = 100
    upstream_max_keepalive_connections: int = 20

    # asyncio.Semaphore 只在单进程内生效；多副本部署时各副本各有一份配额，
    # 因此它不构成分布式限流。
    max_concurrent_requests: int = 8
    concurrency_acquire_timeout_seconds: float = 10.0

    # 环境变量示例：CLIENT_API_KEYS=sk-key-a,sk-key-b
    client_api_keys: CommaSeparatedSecrets = Field(default_factory=list)
    admin_api_key: SecretStr = SecretStr("")

    database_url: str = "postgresql+asyncpg://modelgate:modelgate@db:5432/modelgate"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # Agent 追踪标签的长度上限，超长直接拒绝而不是截断，
    # 避免把被截断的标签当作完整标识入库。
    agent_tag_max_length: int = 64

    # ---- 配额（按请求次数，UTC 自然日重置；0 表示不限制）----
    # 按已认证客户端（网关 API Key 指纹）统计，不使用客户端自填的 X-Client-App。
    client_daily_request_quota: int = 0
    # 单次 Agent 工作流（客户端指纹 + X-Run-Id）的最大 LLM 调用次数。
    # 覆盖整个 Run 生命周期，不随 UTC 日切重置。
    run_max_llm_calls: int = 0

    # ---- 重试（默认不重试）----
    # LLM 请求不是幂等的：读取超时无法判断上游是否已开始推理，
    # 重试可能导致重复推理与重复计费，因此默认关闭。
    retry_max_attempts: int = 1
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_max_seconds: float = 8.0
    # 读取超时默认按「可能已送达」处理，不重试
    retry_on_read_timeout: bool = False
    # 上游 429 / 5xx 默认不重试
    retry_on_upstream_error: bool = False

    @field_validator("client_api_keys", "model_registry", "model_prices", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        """把逗号分隔的字符串拆成列表。NoDecode 之后这里拿到原始字符串。"""
        if isinstance(value, str):
            return _split_comma(value)
        return value

    @field_validator(
        "deepseek_price_input_per_million_usd",
        "deepseek_price_output_per_million_usd",
        "openai_price_input_per_million_usd",
        "openai_price_output_per_million_usd",
        mode="before",
    )
    @classmethod
    def _blank_price_is_zero(cls, value: object) -> object:
        """价格留空视为未配置（0），避免 .env 里的空值导致解析失败。

        0 在 ModelRouter 中按「未配置」处理，因此成本会记为未知而不是免费。
        """
        if isinstance(value, str) and not value.strip():
            return Decimal("0")
        return value

    @field_validator(
        "upstream_timeout_seconds",
        "concurrency_acquire_timeout_seconds",
        mode="after",
    )
    @classmethod
    def _must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("超时时间必须大于 0")
        return value

    @field_validator("max_concurrent_requests", mode="after")
    @classmethod
    def _concurrency_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MAX_CONCURRENT_REQUESTS 至少为 1")
        return value

    @field_validator("client_daily_request_quota", "run_max_llm_calls", mode="after")
    @classmethod
    def _quota_must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("配额必须 >= 0（0 表示不限制）")
        return value

    @field_validator("retry_max_attempts", mode="after")
    @classmethod
    def _retry_attempts_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("RETRY_MAX_ATTEMPTS 至少为 1（1 表示不重试）")
        return value

    @field_validator(
        "retry_backoff_base_seconds", "retry_backoff_max_seconds", mode="after"
    )
    @classmethod
    def _backoff_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("退避时间必须大于 0")
        return value

    @field_validator("agent_tag_max_length", mode="after")
    @classmethod
    def _tag_length_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("AGENT_TAG_MAX_LENGTH 至少为 1")
        return value

    def _secret(self, field: str) -> str:
        value: SecretStr = getattr(self, field)
        return value.get_secret_value()

    @property
    def provider_credentials(self) -> dict[str, tuple[str, str]]:
        """服务商 -> (base_url, api_key)。"""
        return {
            "deepseek": (self.deepseek_base_url, self._secret("deepseek_api_key")),
            "openai": (self.openai_base_url, self._secret("openai_api_key")),
        }

    @property
    def provider_prices(self) -> dict[str, tuple[Decimal, Decimal]]:
        """服务商 -> (输入价, 输出价)，单位美元 / 百万 Token。"""
        return {
            "deepseek": (
                self.deepseek_price_input_per_million_usd,
                self.deepseek_price_output_per_million_usd,
            ),
            "openai": (
                self.openai_price_input_per_million_usd,
                self.openai_price_output_per_million_usd,
            ),
        }

    @property
    def admin_api_key_value(self) -> str:
        """管理密钥明文，仅用于比较。"""
        return self._secret("admin_api_key")

    @property
    def client_api_key_values(self) -> list[str]:
        """客户端密钥明文列表，仅用于比较。"""
        return [item.get_secret_value() for item in self.client_api_keys]


@lru_cache
def get_settings() -> Settings:
    """进程级单例配置。测试可用 get_settings.cache_clear() 重置。"""
    return Settings()
