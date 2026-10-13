"""上游文本脱敏。

上游返回的错误信息可能回显它收到的 API Key 片段
（例如 DeepSeek 的 "Your api key: ****xxxx is invalid"），
这类文本只能进日志，不得原样返回给客户端，
因此落日志前必须先经过 redact_secrets()。
"""

from __future__ import annotations

import re

# 脱敏规则：(正则, 替换文本)。宁可多脱敏，避免遗漏新的密钥格式。
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # sk-xxxx（含 sk-proj-、sk-ant- 等变体）
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{4,}", re.IGNORECASE), "sk-***REDACTED***"),
    (re.compile(r"\b(Bearer)\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE), r"\1 ***REDACTED***"),
    # key/token/secret/password 赋值形式，保留字段名只脱敏值
    (
        re.compile(
            r"\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password)"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9._\-]{8,})[\"']?",
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
    # 40 位以上的十六进制串（疑似密钥）
    (re.compile(r"\b[0-9a-fA-F]{40,}\b"), "***REDACTED***"),
)


def redact_secrets(text: str) -> str:
    """把文本中疑似密钥的部分替换为 ***REDACTED***。"""
    if not text:
        return text

    redacted = text
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
