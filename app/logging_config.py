"""统一日志配置与结构化字段。

日志中不得出现明文密钥、完整用户 Prompt 或模型回复内容；
需要记录请求规模时用 app.security.summarize_prompt() 的条数与字符数。

调用相关的日志通过 extra 携带固定字段（见 LOG_FIELDS），
由本模块的 formatter 渲染成 key=value 后缀，便于检索与聚合。
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_BASE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# 需要出现在每条调用日志里的字段。顺序即渲染顺序。
LOG_FIELDS: tuple[str, ...] = (
    "request_id",
    "client_app",
    "run_id",
    "agent_name",
    "provider",
    "model",
    "latency_ms",
    "error_type",
    "error_origin",
    "http_status",
    "attempt",
    "retry_count",
)

# uvicorn 的 access log 只记录方法与路径，不含请求体与 Authorization 头。
_CONFIGURED = False


class _ContextFormatter(logging.Formatter):
    """在消息后追加携带的结构化字段。

    没有携带任何字段时（如启动日志）输出保持不变，
    避免为了统一格式而给所有日志都加一串空值。
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)

        parts = [
            f"{name}={getattr(record, name)}"
            for name in LOG_FIELDS
            if getattr(record, name, None) is not None
        ]
        if not parts:
            return base
        return f"{base} | {' '.join(parts)}"


def configure_logging(level: str = "INFO") -> None:
    """配置根 logger。重复调用不会叠加 handler。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ContextFormatter(_BASE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)

    # httpx 会在 INFO 级别打印每个请求的完整 URL，连接池复用下噪声很大，
    # 且与业务日志重复，因此压到 WARNING。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_extra(**fields: Any) -> dict[str, Any]:
    """构造 logging 的 extra。

    只保留登记在 LOG_FIELDS 中的键，避免把任意对象塞进日志，
    也顺带挡掉误传 prompt 内容之类的敏感字段。
    """
    return {"extra": {key: value for key, value in fields.items() if key in LOG_FIELDS}}
