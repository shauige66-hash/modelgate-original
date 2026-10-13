"""上游调用的重试策略。

默认不重试。LLM 请求不是幂等的：读取超时意味着请求已经发出，
上游可能已经在推理、甚至已经计费，重试会造成**重复推理与重复计费**。
因此：

- 只有「请求确定未送达」的故障（连接尚未建立）才默认安全；
- 读取超时、429、5xx 都必须显式打开开关才会重试，默认按不可重试处理；
- 不做跨服务商自动切换：同一模型名在不同服务商上的行为与价格都不同，
  静默切换会让调用方拿到意料之外的模型输出。

每次尝试都会记录，供结构化日志与调用记录使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.errors import ErrorType

# 这些错误表示请求未能建立连接，重试不会产生第二次推理
SAFE_TO_RETRY = frozenset({ErrorType.UPSTREAM_CONNECT_ERROR})

# 这些错误可能已经产生推理，默认不重试
RISKY_TO_RETRY = frozenset(
    {
        ErrorType.UPSTREAM_TIMEOUT,
        ErrorType.UPSTREAM_HTTP_ERROR,
        ErrorType.UPSTREAM_RATE_LIMITED,
    }
)

# 无论怎么配置都不重试：鉴权失败、响应格式错误、服务端配置问题、
# 客户端主动断开、以及任何非上游原因
NEVER_RETRY = frozenset(
    {
        ErrorType.UPSTREAM_AUTH_ERROR,
        ErrorType.UPSTREAM_INVALID_RESPONSE,
        ErrorType.UPSTREAM_NOT_CONFIGURED,
        ErrorType.CONCURRENCY_TIMEOUT,
        ErrorType.QUOTA_EXCEEDED,
        ErrorType.CLIENT_CANCELLED,
        ErrorType.INTERNAL_ERROR,
        ErrorType.NONE,
    }
)


@dataclass
class RetryPolicy:
    """一次调用允许的重试行为。"""

    max_attempts: int = 1
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    retry_on_read_timeout: bool = False
    retry_on_upstream_error: bool = False

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 1

    def should_retry(self, error_type: ErrorType, *, attempt: int) -> bool:
        """第 attempt 次（从 1 开始）失败后是否还要再试。"""
        if not self.enabled or attempt >= self.max_attempts:
            return False

        if error_type in NEVER_RETRY:
            return False

        if error_type in SAFE_TO_RETRY:
            return True

        if error_type is ErrorType.UPSTREAM_TIMEOUT:
            return self.retry_on_read_timeout

        # 上游限流与普通上游错误共用一个开关：两者都表示「请求已送达上游」，
        # 重试同样有重复推理风险。上游给出的 Retry-After 只透传给调用方，
        # 不用于自动等待，避免网关替调用方做长时间阻塞。
        if error_type in (
            ErrorType.UPSTREAM_HTTP_ERROR,
            ErrorType.UPSTREAM_RATE_LIMITED,
        ):
            return self.retry_on_upstream_error

        return False

    def backoff_seconds(self, attempt: int) -> float:
        """第 attempt 次失败后的等待时间（指数退避，有上限）。"""
        delay = self.backoff_base_seconds * (2 ** (attempt - 1))
        return min(delay, self.backoff_max_seconds)


@dataclass
class Attempt:
    """一次上游尝试的结果摘要。"""

    attempt: int
    error_type: ErrorType
    latency_ms: int


@dataclass
class AttemptLog:
    """一次请求内所有尝试的流水，供日志与调用记录使用。"""

    attempts: list[Attempt] = field(default_factory=list)

    @property
    def count(self) -> int:
        """实际执行的上游请求次数。"""
        return len(self.attempts)

    @property
    def retry_count(self) -> int:
        """重试次数 = 总次数 - 1。"""
        return max(self.count - 1, 0)

    def record(self, attempt: int, error_type: ErrorType, latency_ms: int) -> None:
        self.attempts.append(
            Attempt(attempt=attempt, error_type=error_type, latency_ms=latency_ms)
        )

    def describe(self) -> str:
        """紧凑的尝试流水，用于日志。"""
        return ",".join(
            f"{item.attempt}:{item.error_type.value}:{item.latency_ms}ms"
            for item in self.attempts
        )
