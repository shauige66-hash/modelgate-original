"""应用配置。

所有可变参数集中在此处，通过环境变量或 .env 文件覆盖，源码中不写死任何密钥。

安全设计：密钥类字段一律使用 pydantic 的 SecretStr，
这样 repr()、日志、异常回溯里都只会显示 `**********`，
不会因为一次意外的 print(settings) 就泄露密钥。
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# 每百万 Token 的价格（美元）。
# 说明：DeepSeek 的定价按缓存命中 / 未命中区分，本项目简化为「输入 / 输出」两档，
# 属于估算口径，不等同于账单金额。
PRICE_INPUT_PER_MILLION = Decimal("0.27")
PRICE_OUTPUT_PER_MILLION = Decimal("1.10")

# 一个标准「百万」常量，用于 Decimal 计算，避免引入浮点误差。
MILLION = Decimal("1000000")

# NoDecode: 关闭 pydantic-settings 对复杂类型的 JSON 解析。
# 否则 CLIENT_API_KEYS=a,b 会因为不是合法 JSON 而直接报错。
CommaSeparatedSecrets = Annotated[list[SecretStr], NoDecode]


class Settings(BaseSettings):
    """从环境变量 / .env 读取的配置对象。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 应用 ----
    app_name: str = "ModelGate"
    app_version: str = "0.1.0"
    app_debug: bool = False
    log_level: str = "INFO"

    # ---- 上游 LLM 服务 ----
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_api_key: SecretStr = SecretStr("")
    upstream_model: str = "deepseek-chat"
    upstream_timeout_seconds: float = 60.0
    upstream_max_connections: int = 100
    upstream_max_keepalive_connections: int = 20

    # ---- 单进程并发控制 ----
    # 注意：asyncio.Semaphore 只在单个进程内生效。
    # 多副本部署时每个副本各有一份配额，它不构成分布式限流。
    max_concurrent_requests: int = 8
    concurrency_acquire_timeout_seconds: float = 10.0

    # ---- 鉴权 ----
    # 环境变量示例：CLIENT_API_KEYS=sk-key-a,sk-key-b
    client_api_keys: CommaSeparatedSecrets = Field(default_factory=list)
    admin_api_key: SecretStr = SecretStr("")

    # ---- 数据库 ----
    database_url: str = "postgresql+asyncpg://modelgate:modelgate@db:5432/modelgate"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # ---- 成本单价（美元 / 百万 Token） ----
    price_input_per_million_usd: Decimal = PRICE_INPUT_PER_MILLION
    price_output_per_million_usd: Decimal = PRICE_OUTPUT_PER_MILLION
    default_model_name: str = "deepseek-chat"

    @field_validator("client_api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, value: object) -> object:
        """把逗号分隔的字符串拆成列表。

        使用 NoDecode 关闭了 JSON 解析，因此这里收到的就是原始字符串。
        """
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
        """取出上游密钥明文。仅在构造请求头时使用。"""
        return self.upstream_api_key.get_secret_value()

    @property
    def admin_api_key_value(self) -> str:
        """取出管理密钥明文。仅在比较时使用。"""
        return self.admin_api_key.get_secret_value()

    @property
    def client_api_key_values(self) -> list[str]:
        """取出客户端密钥明文列表。仅在比较时使用。"""
        return [item.get_secret_value() for item in self.client_api_keys]

    @property
    def is_upstream_configured(self) -> bool:
        """上游密钥是否已配置。未配置时上游调用会直接失败。"""
        return bool(self.upstream_api_key_value)


@lru_cache
def get_settings() -> Settings:
    """返回进程级单例配置。

    使用 lru_cache 让配置只解析一次；测试中可通过 get_settings.cache_clear() 重置。
    """
    return Settings()
