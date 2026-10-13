"""鉴权与敏感信息处理。

安全约定：
- 密钥只从配置（环境变量 / .env）读取，不写死在源码里。
- 比较密钥使用 secrets.compare_digest，避免时序侧信道。
- 明文密钥不写入日志、不写入数据库。
- 用户 Prompt 内容不写入数据库，只记录长度等元信息（见 summarize_prompt）。
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings
from app.dependencies import get_app_settings


def _unauthorized(detail: str, *, authenticate: bool) -> HTTPException:
    """构造 401 / 403 响应。

    authenticate=True  -> 401，表示未提供凭据（附带 WWW-Authenticate）。
    authenticate=False -> 403，表示提供了凭据但不正确。
    """
    headers = {"WWW-Authenticate": "Bearer"} if authenticate else None
    return HTTPException(
        status_code=(
            status.HTTP_401_UNAUTHORIZED if authenticate else status.HTTP_403_FORBIDDEN
        ),
        detail=detail,
        headers=headers,
    )


def _extract_bearer(authorization: str | None) -> str | None:
    """从 Authorization: Bearer <token> 中取出 token。"""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _constant_time_contains(candidate: str, allowed: list[str]) -> bool:
    """在候选值与白名单之间做常数时间比较。

    这里对所有条目都执行一次比较（不做提前 return），
    因此比较耗时与命中的是第几个 Key 无关。
    """
    matched = False
    for item in allowed:
        if secrets.compare_digest(candidate, item):
            matched = True
    return matched


async def require_client_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_app_settings),
) -> str:
    """校验客户端 API Key，返回该 Key 的指纹（用于日志，不含明文）。

    同时支持 `Authorization: Bearer <key>` 与 `X-API-Key: <key>`。
    """
    if not settings.client_api_key_values:
        # 未配置任何客户端 Key 时拒绝全部请求，避免「默认放行」的安全隐患。
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务端未配置 CLIENT_API_KEYS，拒绝所有请求",
        )

    token = _extract_bearer(authorization) or (x_api_key or "").strip() or None
    if token is None:
        raise _unauthorized("缺少 API Key", authenticate=True)

    if not _constant_time_contains(token, settings.client_api_key_values):
        raise _unauthorized("API Key 无效", authenticate=False)

    return key_fingerprint(token)


async def require_admin_api_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    settings: Settings = Depends(get_app_settings),
) -> str:
    """校验管理接口密钥。

    管理接口使用独立密钥，不复用客户端 Key，避免普通调用方读取全局统计。
    """
    if not settings.admin_api_key_value:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务端未配置 ADMIN_API_KEY，管理接口不可用",
        )

    if not x_admin_key:
        raise _unauthorized("缺少管理密钥", authenticate=True)

    if not secrets.compare_digest(x_admin_key, settings.admin_api_key_value):
        raise _unauthorized("管理密钥无效", authenticate=False)

    return key_fingerprint(x_admin_key)


def key_fingerprint(value: str) -> str:
    """返回密钥的短指纹，可安全写入日志。

    使用不可逆的 SHA-256 前缀，泄露日志也无法还原原始密钥。
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def summarize_prompt(messages: list[dict]) -> dict[str, int]:
    """把消息列表压缩成可安全落库的元信息。

    只保留条数与字符数，不保留任何原文，避免 Prompt 内容进入数据库。
    """
    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            # 多模态 content 是数组，按字符串近似统计长度即可。
            total_chars += len(str(content))
    return {"message_count": len(messages), "prompt_chars": total_chars}


def mask_secret(value: str, *, keep: int = 4) -> str:
    """把密钥变成 `sk-****abcd` 形式，仅用于必须展示的调试信息。"""
    if not value:
        return "<empty>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:3]}****{value[-keep:]}"
