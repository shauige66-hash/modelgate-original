"""Docker 容器集成测试。

默认跳过：本机没有可用的 Docker daemon 时自动跳过，
因此没有安装 Docker 的机器上 `pytest` 依然全绿。

跑法：
    1) 确保 Docker Desktop 已启动（docker info 可用）
    2) docker compose up -d --build
    3) pytest tests/test_docker_integration.py -v

这些用例验证的是「真实容器里的端到端行为」：
容器是否起来、健康检查是否通过、数据库是否真的连上、
以及一次完整调用是否能在真实 PostgreSQL 中落库。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

# 允许覆盖，便于指向非默认端口
BASE_URL = os.getenv("MODELGATE_BASE_URL", "http://127.0.0.1:8000")
CLIENT_KEY = os.getenv("MODELGATE_CLIENT_KEY", "sk-client-local-dev-key")
ADMIN_KEY = os.getenv("MODELGATE_ADMIN_KEY", "sk-admin-local-dev-key")

# 等待容器就绪的最长时间
STARTUP_TIMEOUT_SECONDS = int(os.getenv("MODELGATE_STARTUP_TIMEOUT", "180"))


def _docker_available() -> bool:
    """Docker CLI 存在且 daemon 可用。"""
    if shutil.which("docker") is None:
        # PATH 里没有时，试一下 Docker Desktop 的默认安装位置
        candidate = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        if not os.path.exists(candidate):
            return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _http_get(path: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    """发一个 GET，返回 (状态码, 响应体)。不抛异常。"""
    request = urllib.request.Request(f"{BASE_URL}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _http_post(
    path: str, payload: dict, headers: dict[str, str] | None = None
) -> tuple[int, str]:
    """发一个 POST，返回 (状态码, 响应体)。不抛异常。"""
    body = json.dumps(payload).encode("utf-8")
    merged = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers=merged, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon 不可用（需要 Docker Desktop 正在运行）",
)


@pytest.fixture(scope="module")
def live_api() -> str:
    """确保 API 正在运行；未运行时给出明确的失败提示。"""
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    last_error = ""

    while time.time() < deadline:
        try:
            status, _ = _http_get("/health")
            if status == 200:
                return BASE_URL
        except OSError as exc:
            last_error = str(exc)
        time.sleep(2)

    pytest.fail(
        f"{STARTUP_TIMEOUT_SECONDS}s 内 {BASE_URL}/health 未就绪。"
        f"请先执行 `docker compose up -d --build`。最后错误：{last_error}"
    )


class TestContainerHealth:
    def test_health_endpoint(self, live_api: str) -> None:
        status, body = _http_get("/health")

        assert status == 200
        assert json.loads(body) == {"status": "ok"}

    def test_readiness_reports_database_connected(self, live_api: str) -> None:
        """容器内的应用必须真的连上了 PostgreSQL。"""
        status, body = _http_get("/health/ready")

        assert status == 200
        payload = json.loads(body)
        assert payload["status"] == "ok"
        assert payload["database"]["ok"] is True

    def test_auth_is_enforced_in_container(self, live_api: str) -> None:
        status, _ = _http_post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )

        assert status == 401

    def test_admin_usage_reachable(self, live_api: str) -> None:
        status, body = _http_get("/admin/usage", {"X-Admin-Key": ADMIN_KEY})

        assert status == 200
        assert "totals" in json.loads(body)


class TestPostgresInsideContainer:
    def test_call_record_table_exists(self, live_api: str) -> None:
        """真实容器里的 PostgreSQL 应该已经由 create_all 建好表。"""
        # 通过管理接口间接验证：能正常聚合说明表存在且可查询
        status, body = _http_get(
            f"/admin/usage?since=2000-01-01T00:00:00Z", {"X-Admin-Key": ADMIN_KEY}
        )

        assert status == 200
        payload = json.loads(body)
        assert "total_requests" in payload["totals"]
        assert payload["totals"]["estimated_cost_usd"] is not None

    def test_persisted_record_is_visible_in_usage(self, live_api: str) -> None:
        """发一次请求（上游认证可能失败），记录必须能在统计里看到。

        这里故意不依赖真实上游密钥：无论上游成功还是 401，
        网关都应该写入一条记录 —— 这正是持久化要验证的行为。
        """
        _, before_body = _http_get(
            "/admin/usage?since=2000-01-01T00:00:00Z", {"X-Admin-Key": ADMIN_KEY}
        )
        before = json.loads(before_body)["totals"]["total_requests"]

        status, _ = _http_post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello container"}]},
            {"Authorization": f"Bearer {CLIENT_KEY}"},
        )
        # 上游可能 200、也可能 502（未配置真实密钥）
        assert status in (200, 502), f"未预期的状态码: {status}"

        _, after_body = _http_get(
            "/admin/usage?since=2000-01-01T00:00:00Z", {"X-Admin-Key": ADMIN_KEY}
        )
        after = json.loads(after_body)["totals"]["total_requests"]

        assert after == before + 1, "调用记录未写入 PostgreSQL"
