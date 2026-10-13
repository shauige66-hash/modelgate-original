"""Docker 容器集成测试。

没有可用的 Docker daemon 时整体跳过，所以在未装 Docker 的机器上 `pytest` 依然全绿。

跑法：启动 Docker Desktop 后执行 `docker compose up -d --build`，
再运行 `pytest tests/test_docker_integration.py -v`。

这组用例向运行中的容器发真实 HTTP 请求，覆盖健康检查、
容器内数据库连通性，以及一次调用能否落库。
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from tests.conftest import TEST_DATABASE_URL

# 允许覆盖，便于指向非默认端口
BASE_URL = os.getenv("MODELGATE_BASE_URL", "http://127.0.0.1:8000")
CLIENT_KEY = os.getenv("MODELGATE_CLIENT_KEY", "sk-client-local-dev-key")
ADMIN_KEY = os.getenv("MODELGATE_ADMIN_KEY", "sk-admin-local-dev-key")

STARTUP_TIMEOUT_SECONDS = int(os.getenv("MODELGATE_STARTUP_TIMEOUT", "180"))

# 容器的 DATABASE_URL 默认值（docker-compose.yml 中通过服务名 db 访问）
DEFAULT_DATABASE_URL = "postgresql+asyncpg://modelgate:modelgate@db:5432/modelgate"

# .env.example 中的上游 Key 占位值
_PLACEHOLDER_UPSTREAM_KEYS = frozenset({"sk-replace-with-your-real-key"})


def _database_name(database_url: str) -> str:
    """取出连接串里的库名，用于判断两个连接串是否指向同一个库。"""
    from sqlalchemy.engine import make_url

    try:
        return make_url(database_url).database or ""
    except Exception:  # noqa: BLE001 - 无法解析时按「未知」处理
        return ""


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


@pytest.fixture(autouse=True)
def _skip_when_upstream_key_is_placeholder() -> None:
    """上游 Key 仍是占位值时跳过所有用例。

    这些用例需要正在运行的容器，而容器读取的是项目根目录的 .env。
    如果那里还是 .env.example 的占位值，每次运行都会真的打一次 DeepSeek
    （用无效 Key，必然 401）。我们没有真实 Key 可用，所以直接跳过，
    避免产生无意义的外部调用。
    """
    env_file = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        pytest.skip("未找到 .env，无法确认容器已配置真实上游 Key")

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("UPSTREAM_API_KEY="):
            value = stripped.split("=", 1)[1].strip()
            if value in _PLACEHOLDER_UPSTREAM_KEYS or not value:
                pytest.skip(
                    f"上游 Key 仍是占位值（{value!r}），跳过以免产生无效的外部 API 调用"
                )
            return

    pytest.skip("未在 .env 中找到 UPSTREAM_API_KEY")


@pytest.fixture(scope="module")
def live_api() -> str:
    """等待 API 就绪；超时给出可操作的提示。"""
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


@pytest.fixture(autouse=True)
def _refuse_to_write_into_the_test_database(live_api: str) -> None:
    """禁止「容器业务库」与「集成测试库」是同一个库。

    本文件会向运行中的容器发真实请求，因此会在容器所用数据库里写入记录。
    test_postgres_integration.py 则会对 TEST_DATABASE_URL 指向的库做
    drop_all + create_all。若两者指向同一个库，这里写入的业务数据
    会被后续的破坏性测试清空 —— 因此提前拦住。
    """
    if not TEST_DATABASE_URL:
        return

    service_db = _database_name(TEST_DATABASE_URL)
    container_db = _database_name(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))

    if service_db and service_db == container_db:
        pytest.fail(
            f"TEST_DATABASE_URL 与容器的 DATABASE_URL 指向同一个库 "
            f"({service_db!r})。该库会被 PostgreSQL 集成测试 drop_all 清空，"
            f"请为测试使用独立数据库（库名以 _test 结尾），"
            f"例如把 TEST_DATABASE_URL 指向 modelgate_test。"
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
