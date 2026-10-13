"""请求上下文。

用途：把「当前请求 ID」暴露给无法直接拿到路由局部变量的地方，
主要是异常处理器 —— 失败响应同样需要带 X-Request-Id，
否则客户端无法用返回的 ID 与数据库记录对账。

用 ContextVar 而不是全局变量：ContextVar 在 asyncio 中按任务隔离，
并发请求之间不会互相覆盖。
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("modelgate_request_id", default=None)


def new_request_id() -> str:
    """生成一个请求 ID。"""
    return f"req_{uuid.uuid4().hex[:24]}"


def set_request_id(request_id: str) -> None:
    """记录当前请求 ID。"""
    _request_id.set(request_id)


def get_request_id() -> str | None:
    """读取当前请求 ID；不在请求上下文中时返回 None。"""
    return _request_id.get()
