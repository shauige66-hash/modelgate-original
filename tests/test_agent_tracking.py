"""Agent 调用追踪测试。

覆盖请求头解析与校验、字段入库、按应用/Agent/Run 查询，
以及这些标签不会被当作权限依据。
"""

from __future__ import annotations

import pytest

from app.agent_tracking import (
    InvalidTrackingHeaderError,
    parse_agent_tags,
)
from app.db.repository import CallRecordRepository, UsageFilter

CHAT_URL = "/v1/chat/completions"
ADMIN_URL = "/admin/usage"
ADMIN_HEADERS = {"X-Admin-Key": "sk-admin-test-key"}

TAGS = {
    "X-Client-App": "insightflow",
    "X-Agent-Name": "planner",
    "X-Run-Id": "run-2026-01-01.abc",
}


def _payload() -> dict:
    return {"messages": [{"role": "user", "content": "hi"}]}


class TestHeaderParsing:
    def test_all_three_tags_parsed(self) -> None:
        tags = parse_agent_tags(
            client_app="insightflow",
            agent_name="executor",
            run_id="run-1",
            max_length=64,
        )

        assert tags.client_app == "insightflow"
        assert tags.agent_name == "executor"
        assert tags.run_id == "run-1"
        assert tags.is_empty is False

    def test_missing_headers_are_none(self) -> None:
        tags = parse_agent_tags(
            client_app=None, agent_name=None, run_id=None, max_length=64
        )

        assert tags.is_empty is True

    def test_blank_value_treated_as_absent(self) -> None:
        tags = parse_agent_tags(
            client_app="   ", agent_name=None, run_id=None, max_length=64
        )

        assert tags.client_app is None

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        tags = parse_agent_tags(
            client_app="  insightflow  ", agent_name=None, run_id=None, max_length=64
        )

        assert tags.client_app == "insightflow"

    @pytest.mark.parametrize(
        "value",
        [
            "has space",
            "semi;colon",
            "quote'",
            'double"',
            "back\\slash",
            "<script>",
            "-leading-dash",
            ".leading-dot",
            "中文标签",
            "emoji🙂",
        ],
    )
    def test_invalid_characters_rejected(self, value: str) -> None:
        with pytest.raises(InvalidTrackingHeaderError):
            parse_agent_tags(
                client_app=value, agent_name=None, run_id=None, max_length=64
            )

    @pytest.mark.parametrize(
        "value",
        ["insightflow", "run-2026.01", "planner_1", "A1", "agent.v2-beta"],
    )
    def test_valid_characters_accepted(self, value: str) -> None:
        tags = parse_agent_tags(
            client_app=value, agent_name=None, run_id=None, max_length=64
        )

        assert tags.client_app == value

    def test_over_length_rejected_not_truncated(self) -> None:
        """超长直接拒绝：截断会让不同 run 混成同一个标识。"""
        with pytest.raises(InvalidTrackingHeaderError) as exc_info:
            parse_agent_tags(
                client_app="a" * 65, agent_name=None, run_id=None, max_length=64
            )

        assert "长度" in str(exc_info.value)

    def test_exactly_at_limit_accepted(self) -> None:
        tags = parse_agent_tags(
            client_app="a" * 64, agent_name=None, run_id=None, max_length=64
        )

        assert tags.client_app == "a" * 64


class TestTagsPersisted:
    async def test_tags_recorded_on_success(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload(), headers=TAGS)
        assert response.status_code == 200

        request_id = response.headers["X-Request-Id"]
        async with harness.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.client_app == "insightflow"
        assert record.agent_name == "planner"
        assert record.run_id == "run-2026-01-01.abc"

    async def test_tags_recorded_on_failure(self, make_harness) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        h = await make_harness(handler)
        response = await h.client.post(CHAT_URL, json=_payload(), headers=TAGS)

        request_id = response.headers["X-Request-Id"]
        async with h.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.http_status == 504
        assert record.client_app == "insightflow"
        assert record.agent_name == "planner"

    async def test_absent_headers_store_null(self, harness) -> None:
        response = await harness.client.post(CHAT_URL, json=_payload())
        request_id = response.headers["X-Request-Id"]

        async with harness.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        assert record.client_app is None
        assert record.agent_name is None
        assert record.run_id is None

    async def test_invalid_header_returns_400(self, harness) -> None:
        response = await harness.client.post(
            CHAT_URL,
            json=_payload(),
            headers={"X-Agent-Name": "bad name with spaces"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_tracking_header"

    async def test_invalid_header_never_reaches_upstream(self, harness) -> None:
        await harness.client.post(
            CHAT_URL, json=_payload(), headers={"X-Run-Id": "bad;value"}
        )

        assert harness.upstream_requests == []

    async def test_prompt_content_still_not_stored(self, harness) -> None:
        """标签入库不能顺手把 Prompt 也存进去。"""
        secret = "我的手机号是 13800001111"
        response = await harness.client.post(
            CHAT_URL,
            json={"messages": [{"role": "user", "content": secret}]},
            headers=TAGS,
        )
        request_id = response.headers["X-Request-Id"]

        async with harness.session_factory() as session:
            record = await CallRecordRepository(session).get_by_request_id(request_id)

        assert record is not None
        for value in vars(record).values():
            if isinstance(value, str):
                assert "13800001111" not in value


class TestTagsAreNotAuthentication:
    async def test_tags_do_not_bypass_client_key(self, make_harness) -> None:
        """带上追踪标签但没有合法凭据，依然被拒绝。"""
        h = await make_harness(authenticated=False)

        response = await h.client.post(CHAT_URL, json=_payload(), headers=TAGS)

        assert response.status_code == 401
        assert h.upstream_requests == []

    async def test_tags_do_not_grant_admin_access(self, harness) -> None:
        """追踪标签不能用于访问管理接口。"""
        response = await harness.client.get(ADMIN_URL, headers=TAGS)

        assert response.status_code == 401

    async def test_client_app_tag_is_not_trusted_for_auth(self, harness) -> None:
        """伪造高权限应用名不会带来任何额外能力。"""
        response = await harness.client.post(
            CHAT_URL,
            json=_payload(),
            headers={"X-Client-App": "admin", "X-Agent-Name": "root"},
        )

        assert response.status_code == 200
        # 仍然只能访问普通业务接口
        assert (await harness.client.get(ADMIN_URL)).status_code == 401


class TestAgentFiltering:
    async def _seed(self, harness) -> None:
        """写入三组不同标签的记录。"""
        await harness.client.post(
            CHAT_URL,
            json=_payload(),
            headers={
                "X-Client-App": "insightflow",
                "X-Agent-Name": "planner",
                "X-Run-Id": "run-a",
            },
        )
        await harness.client.post(
            CHAT_URL,
            json=_payload(),
            headers={
                "X-Client-App": "insightflow",
                "X-Agent-Name": "executor",
                "X-Run-Id": "run-a",
            },
        )
        await harness.client.post(
            CHAT_URL,
            json=_payload(),
            headers={
                "X-Client-App": "other-app",
                "X-Agent-Name": "critic",
                "X-Run-Id": "run-b",
            },
        )

    async def test_filter_by_client_app(self, harness) -> None:
        await self._seed(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?client_app=insightflow", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 2
        assert body["filters"]["client_app"] == "insightflow"

    async def test_filter_by_agent_name(self, harness) -> None:
        await self._seed(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?agent_name=executor", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 1

    async def test_filter_by_run_id(self, harness) -> None:
        await self._seed(harness)

        body = (
            await harness.client.get(f"{ADMIN_URL}?run_id=run-a", headers=ADMIN_HEADERS)
        ).json()

        assert body["totals"]["total_requests"] == 2

    async def test_filters_combine(self, harness) -> None:
        await self._seed(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?client_app=insightflow&agent_name=planner",
                headers=ADMIN_HEADERS,
            )
        ).json()

        assert body["totals"]["total_requests"] == 1

    async def test_unknown_run_id_returns_empty(self, harness) -> None:
        await self._seed(harness)

        body = (
            await harness.client.get(
                f"{ADMIN_URL}?run_id=does-not-exist", headers=ADMIN_HEADERS
            )
        ).json()

        assert body["totals"]["total_requests"] == 0
        assert body["by_model"] == []

    async def test_repository_filter_by_run_id(self, harness) -> None:
        await self._seed(harness)

        async with harness.session_factory() as session:
            totals = await CallRecordRepository(session).usage_totals(
                UsageFilter(run_id="run-a")
            )

        assert totals.total_requests == 2
