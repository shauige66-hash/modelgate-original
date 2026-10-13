"""Chat Completions 的请求与响应模型。

只支持非流式；stream=true 会被显式拒绝，避免「声称支持流式但返回整包」的误导。
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
    """客户端请求体，字段与 OpenAI Chat Completions 兼容。

    extra="allow" 保留 temperature、max_tokens 等附加参数并原样转发。
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(
        default=None,
        description="留空则使用服务端配置的 DEFAULT_MODEL_NAME",
    )
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False

    @field_validator("stream")
    @classmethod
    def _reject_stream(cls, value: bool) -> bool:
        if value:
            raise ValueError("暂不支持 stream=true，请使用非流式请求")
        return value

    def upstream_payload(self, *, default_model: str) -> dict[str, Any]:
        """构造转发给上游的请求体，显式带上 stream=False。"""
        payload = self.model_dump(exclude_none=True)
        payload["model"] = self.model or default_model
        payload["stream"] = False
        return payload


class Usage(BaseModel):
    """上游返回的 Token 用量；字段缺失为 None，代表未知。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    """透传给客户端的响应。

    保留上游返回的全部字段，只把 usage 提升为强类型以便计费。
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    model: str | None = None
    usage: Usage | None = None
