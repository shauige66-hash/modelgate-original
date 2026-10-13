"""统一日志配置。

日志中不得出现明文密钥与完整用户 Prompt；
需要记录请求规模时用 app.security.summarize_prompt()。
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# uvicorn 的 access log 只记录方法与路径，不含请求体与 Authorization 头。
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """配置根 logger。重复调用不会叠加 handler。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

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
