"""把 ModelGate 接入 Agent 运行时的最小适配层。

Agent Harness 只需要关心三件事：用哪个模型、属于哪次运行、当前是哪个节点。
其余（路由、鉴权、并发、配额、计量）都交给 ModelGate，这里不重复实现。

这一层刻意保持很薄：它不是 Agent 框架，不含循环、检查点、工具执行或上下文管理 ——
那些属于上层运行时的职责。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ModelGateConfig:
    """连接 ModelGate 所需的配置。"""

    base_url: str = "http://127.0.0.1:8000"
    client_key: str = ""
    admin_key: str = ""
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "ModelGateConfig":
        """从环境变量读取。

        Client Key 取 CLIENT_API_KEYS 的第一个值；它是网关自己的密钥，
        与真实上游（DeepSeek / OpenAI）的密钥完全无关 —— Agent 侧永远拿不到上游密钥。
        """
        raw_client = os.getenv("MODELGATE_CLIENT_KEYS") or os.getenv("CLIENT_API_KEYS", "")
        client_key = next(
            (item.strip() for item in raw_client.split(",") if item.strip()), ""
        )
        return cls(
            base_url=os.getenv("MODELGATE_BASE_URL", "http://127.0.0.1:8000"),
            client_key=client_key,
            admin_key=os.getenv("MODELGATE_ADMIN_KEY") or os.getenv("ADMIN_API_KEY", ""),
        )


class ModelGateClient:
    """带 Agent 追踪标签的 ModelGate 客户端。

    一次工作流内所有节点共享同一个 run_id，各自带上自己的 agent_name，
    这样管理接口就能按 run 聚合出「哪个节点花了多少」。
    """

    def __init__(
        self,
        config: ModelGateConfig,
        *,
        run_id: str,
        client_app: str = "insightflow",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._run_id = run_id
        self._client_app = client_app
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url, timeout=config.timeout_seconds
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    def _headers(self, agent_name: str) -> dict[str, str]:
        """网关只认 Client Key；其余三个是观测标签，不参与鉴权。"""
        return {
            "Authorization": f"Bearer {self._config.client_key}",
            "Content-Type": "application/json",
            "X-Client-App": self._client_app,
            "X-Agent-Name": agent_name,
            "X-Run-Id": self._run_id,
        }

    async def complete(
        self,
        *,
        agent_name: str,
        messages: list[dict],
        model: str = "deepseek-chat",
    ) -> dict:
        """以某个 Agent 节点的身份发起一次非流式调用。

        只支持非流式：网关当前不实现流式代理，传 stream 会被拒绝。
        """
        response = await self._client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages},
            headers=self._headers(agent_name),
        )
        response.raise_for_status()
        return response.json()

    async def run_summary(self) -> dict:
        """从管理接口取本次 Run 的聚合数据。"""
        response = await self._client.get(
            f"/admin/runs/{self._run_id}",
            headers={"X-Admin-Key": self._config.admin_key},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
