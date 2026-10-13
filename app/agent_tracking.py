"""Agent 调用追踪标签。

从可选的请求头读取调用方应用、Agent 节点与工作流运行 ID，
用于把网关的调用记录关联到多智能体工作流。

这些字段是**客户端自报的观测标签**：只用于统计与检索，
不参与鉴权或权限判断 —— 任何人只要能调用网关就能随意填写。

校验失败直接返回 400，而不是静默截断或忽略：
被截断的标识会让不同运行 ID 混在一起，比拒绝更危险。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 允许字母、数字、点、下划线、连字符；覆盖 insightflow / planner / run-2026.01 这类常见命名
_ALLOWED_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

HEADER_CLIENT_APP = "X-Client-App"
HEADER_AGENT_NAME = "X-Agent-Name"
HEADER_RUN_ID = "X-Run-Id"


class InvalidTrackingHeaderError(ValueError):
    """追踪标签不符合长度或字符集约束。"""

    def __init__(self, header: str, reason: str) -> None:
        super().__init__(f"{header}: {reason}")
        self.header = header
        self.reason = reason


@dataclass(frozen=True)
class AgentTags:
    """一次调用的追踪标签，全部可选。"""

    client_app: str | None = None
    agent_name: str | None = None
    run_id: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.client_app or self.agent_name or self.run_id)


def _validate(header: str, value: str | None, max_length: int) -> str | None:
    if value is None:
        return None

    candidate = value.strip()
    if not candidate:
        return None

    if len(candidate) > max_length:
        raise InvalidTrackingHeaderError(
            header, f"长度 {len(candidate)} 超过上限 {max_length}"
        )

    if not _ALLOWED_PATTERN.match(candidate):
        raise InvalidTrackingHeaderError(
            header,
            "只允许字母、数字、点、下划线、连字符，且需以字母或数字开头",
        )

    return candidate


def parse_agent_tags(
    *,
    client_app: str | None,
    agent_name: str | None,
    run_id: str | None,
    max_length: int,
) -> AgentTags:
    """解析并校验三个追踪头；任一不合法即抛 InvalidTrackingHeaderError。"""
    return AgentTags(
        client_app=_validate(HEADER_CLIENT_APP, client_app, max_length),
        agent_name=_validate(HEADER_AGENT_NAME, agent_name, max_length),
        run_id=_validate(HEADER_RUN_ID, run_id, max_length),
    )
