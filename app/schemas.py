"""Chat Completions 的请求 / 响应数据模型。

只实现非流式（stream=false）场景。显式拒绝 stream=true，
避免出现「看似支持流式、实际返回整包」的误导行为。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    """单条消息。content 允许字符串或多模态数组，直接透传给上游。"""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """客户端请求体，字段与 OpenAI Chat Completions 保持兼容。"""

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(
        default=None,
        description="留空则使用服务端配置的 UPSTREAM_MODEL",
    )
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False

    @field_validator("stream")
    @classmethod
    def _reject_stream(cls, value: bool) -> bool:
        """本阶段只支持非流式请求。"""
        if value:
            raise ValueError("暂不支持 stream=true，请使用非流式请求")
        return value

    def upstream_payload(self, *, default_model: str) -> dict[str, Any]:
        """构造转发给上游的请求体。

        显式带上 stream=False，并保留客户端传入的其他兼容参数
        （如 temperature、max_tokens 等，由 extra="allow" 收集）。
        """
        payload = self.model_dump(exclude_none=True)
        payload["model"] = self.model or default_model
        payload["stream"] = False
        return payload


class Usage(BaseModel):
    """上游返回的 Token 用量。字段缺失时为 None，代表未知。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    """透传给客户端的响应。

    使用 extra="allow"，保留上游返回的全部字段（id、choices、finish_reason 等），
    只把 usage 提升为强类型对象以便计费。
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    model: str | None = None
    usage: Usage | None = None
