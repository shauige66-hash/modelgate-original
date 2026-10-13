"""统一日志配置。

安全约定：日志中不得出现明文密钥与完整用户 Prompt。
需要排查问题时，使用 app.security.summarize_prompt() 记录长度与消息条数。
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# uvicorn 自带的 access log 会记录完整 URL 与客户端 IP，
# 但不记录请求体，因此不会泄露 Prompt 或 Authorization 头。
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

    # httpx 默认会在 INFO 级别打印每个上游请求的完整 URL。
    # 链接池复用会产生大量噪声，且与我们的业务日志重复，因此压到 WARNING。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(name)
