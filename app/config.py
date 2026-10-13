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

# 默认单价，单位：美元 / 百万 Token（估算口径，非账单金额）。
# DeepSeek 实际按缓存命中与否分别计价，这里简化为输入/输出两档。
PRICE_INPUT_PER_MILLION = Decimal("0.27")
PRICE_OUTPUT_PER_MILLION = Decimal("1.10")

MILLION = Decimal("1000000")

# NoDecode 关闭 pydantic-settings 对复杂类型的 JSON 解析，
# 否则 CLIENT_API_KEYS=a,b 会因不是合法 JSON 而解析失败。
CommaSeparatedSecrets = Annotated[list[SecretStr], NoDecode]


class Settings(BaseSettings):
    """从环境变量 / .env 读取的配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ModelGate"
    app_version: str = "0.1.0"
    app_debug: bool = False
    log_level: str = "INFO"

    upstream_base_url: str = "https://api.deepseek.com"
    upstream_api_key: SecretStr = SecretStr("")
    upstream_model: str = "deepseek-chat"
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

    price_input_per_million_usd: Decimal = PRICE_INPUT_PER_MILLION
    price_output_per_million_usd: Decimal = PRICE_OUTPUT_PER_MILLION
    default_model_name: str = "deepseek-chat"

    @field_validator("client_api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, value: object) -> object:
        """把逗号分隔的字符串拆成列表。NoDecode 之后这里拿到的是原始字符串。"""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
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

    @property
    def upstream_api_key_value(self) -> str:
        """上游密钥明文，仅用于构造请求头。"""
        return self.upstream_api_key.get_secret_value()

    @property
    def admin_api_key_value(self) -> str:
        """管理密钥明文，仅用于比较。"""
        return self.admin_api_key.get_secret_value()

    @property
    def client_api_key_values(self) -> list[str]:
        """客户端密钥明文列表，仅用于比较。"""
        return [item.get_secret_value() for item in self.client_api_keys]

    @property
    def is_upstream_configured(self) -> bool:
        """上游密钥是否已配置；未配置时调用会直接失败。"""
        return bool(self.upstream_api_key_value)


@lru_cache
def get_settings() -> Settings:
    """进程级单例配置。测试可用 get_settings.cache_clear() 重置。"""
    return Settings()
