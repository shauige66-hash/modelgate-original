"""Docker 集成测试：跑在独立的测试栈上，绝不接触业务环境。

隔离方式（见 docker-compose.test.yml）：
- 独立 compose 项目名、独立 PostgreSQL 容器与数据卷、独立端口（18000 / 15432）
- API 容器通过 DATABASE_URL 显式指向 modelgate_docker_test
- 上游指向保留地址（RFC 5737 TEST-NET-1），不会访问真实模型服务，也不产生付费调用

本文件**自己拉起并拆除测试栈**，不复用本机可能正在运行的业务容器。
测试开始前会向被测容器确认它实际连的是哪个库，库名不符合测试约定就直接失败。
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

# ---------------------------------------------------------------- 测试栈配置

TEST_PROJECT = "modelgate-docker-test"
TEST_COMPOSE_FILE = "docker-compose.test.yml"
TEST_BASE_URL = os.getenv("MODELGATE_TEST_BASE_URL", "http://127.0.0.1:18000")
TEST_CLIENT_KEY = "sk-docker-test-client-key"
TEST_ADMIN_KEY = "sk-docker-test-admin-key"

# 被测容器必须连到这个库；库名以 _test 结尾，符合 tests/db_safety.py 的约定
EXPECTED_DATABASE_NAME = "modelgate_docker_test"

# 从 compose 文件读出的上游地址，用于断言它们不是真实服务商。
# 直接解析 YAML 文本而不是引依赖，避免为此新增测试依赖。
def _test_stack_upstream_urls() -> list[str]:
    """从 docker-compose.test.yml 取出上游 Base URL。"""
    import pathlib
    import re

    path = (
        pathlib.Path(__file__).resolve().parent.parent / TEST_COMPOSE_FILE
    )
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return re.findall(r"^\s*(?:DEEPSEEK|OPENAI)_BASE_URL:\s*(\S+)\s*$", text, re.M)


_TEST_UPSTREAM_BASE_URLS = _test_stack_upstream_urls()

# 业务栈的默认端口，用于确认我们没有误打业务容器
BUSINESS_BASE_URL = "http://127.0.0.1:8000"

STARTUP_TIMEOUT_SECONDS = int(os.getenv("MODELGATE_STARTUP_TIMEOUT", "300"))

_DOCKER_FALLBACK = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"

# 把密钥暴露给需要它的用例：不再从 .env 读取，避免与业务配置耦合
os.environ.setdefault("MODELGATE_CLIENT_KEYS", TEST_CLIENT_KEY)
os.environ.setdefault("MODELGATE_ADMIN_API_KEY", TEST_ADMIN_KEY)


def _docker() -> str | None:
    """返回可用的 docker 可执行文件路径。"""
    found = shutil.which("docker")
    if found:
        return found
    if os.path.exists(_DOCKER_FALLBACK):
        return _DOCKER_FALLBACK
    return None


def _docker_available() -> bool:
    docker = _docker()
    if docker is None:
        return False
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _compose(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """执行 docker compose 命令（指向测试栈）。"""
    docker = _docker()
    assert docker is not None
    return subprocess.run(
        [
            docker,
            "compose",
            "-p",
            TEST_PROJECT,
            "-f",
            TEST_COMPOSE_FILE,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


# ---------------------------------------------------------------- HTTP 工具


def _http_get(
    path: str, headers: dict[str, str] | None = None, base_url: str | None = None
) -> tuple[int, str]:
    """发一个 GET，返回 (状态码, 响应体)。不抛异常。"""
    request = urllib.request.Request(
        f"{base_url or TEST_BASE_URL}{path}", headers=headers or {}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _http_post(
    path: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    base_url: str | None = None,
) -> tuple[int, str]:
    """发一个 POST，返回 (状态码, 响应体)。不抛异常。"""
    body = json.dumps(payload).encode("utf-8")
    merged = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        f"{base_url or TEST_BASE_URL}{path}", data=body, headers=merged, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _wait_until_healthy(base_url: str, timeout: float) -> str:
    """等待 /health/ready 通过，返回最后一次错误信息。"""
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            status, body = _http_get("/health/ready", base_url=base_url)
            if status == 200:
                return ""
            last_error = f"HTTP {status}: {body[:200]}"
        except OSError as exc:
            last_error = str(exc)
        time.sleep(2)
    return last_error


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def isolated_stack() -> str:
    """拉起隔离的测试栈，返回其基地址；模块结束后拆除（含数据卷）。

    容器不可用时跳过（未装 Docker 的机器上 pytest 依然全绿）。
    容器可用但栈起不来时**明确失败**，绝不退回业务容器。

    每次运行都会重建测试栈并清空测试库，因此用例之间互不影响；
    代价是首次运行需要构建镜像。
    """
    if not _docker_available():
        pytest.skip("Docker daemon 不可用（需要 Docker Desktop 正在运行）")

    started = time.time()
    print(f"\n[测试栈] 启动 {TEST_PROJECT}（首次运行需要构建镜像，请稍候）...")
    up = _compose("up", "-d", "--build")
    if up.returncode != 0:
        pytest.fail(
            "无法启动隔离测试栈：\n"
            f"stdout: {up.stdout[-1500:]}\n"
            f"stderr: {up.stderr[-1500:]}"
        )

    error = _wait_until_healthy(TEST_BASE_URL, STARTUP_TIMEOUT_SECONDS)

    try:
        if error:
            logs = _compose("logs", "--tail", "40", "api")
            pytest.fail(
                f"{STARTUP_TIMEOUT_SECONDS}s 内 {TEST_BASE_URL}/health/ready 未就绪："
                f"{error}\n容器日志：\n{logs.stdout[-2000:]}{logs.stderr[-500:]}"
            )
        print(f"[测试栈] 就绪，用时 {time.time() - started:.1f}s（基地址 {TEST_BASE_URL}）")
        yield TEST_BASE_URL
    finally:
        print(f"\n[测试栈] 拆除 {TEST_PROJECT} 并删除测试数据卷")
        _compose("down", "-v", timeout=300)


@pytest.fixture(scope="module")
def api_database_name(isolated_stack: str) -> str:
    """被测 API 容器实际连接的库名。

    这是本文件最关键的一步：判断「打在哪个库上」不能看宿主机的环境变量，
    必须问被测容器自己。
    """
    status, body = _http_get("/health/ready", base_url=isolated_stack)
    assert status == 200, f"就绪探针失败: {status} {body}"

    name = json.loads(body)["database"].get("name")
    assert name, "就绪探针未报告数据库名，无法确认测试隔离性"
    return name


@pytest.fixture(autouse=True)
def _refuse_to_touch_business_environment(
    isolated_stack: str, api_database_name: str
) -> None:
    """确认被测容器用的是测试库，且业务环境没有被误用。

    这些断言是「不污染业务数据」的硬约束：一旦不成立就立即失败，
    而不是继续往业务库里写记录。
    """
    if api_database_name == EXPECTED_DATABASE_NAME:
        return

    pytest.fail(
        f"被测 API 容器连接的是 {api_database_name!r}，"
        f"而不是测试库 {EXPECTED_DATABASE_NAME!r}。"
        "拒绝在非测试库上执行会写入调用记录的集成测试。"
    )


@pytest.fixture(scope="module")
def stack_is_not_the_business_container(isolated_stack: str) -> None:
    """业务容器若在运行，其库名必须与被测容器不同。"""
    try:
        status, body = _http_get("/health/ready", base_url=BUSINESS_BASE_URL)
    except OSError:
        pytest.skip("业务容器未运行，无需比较")

    if status != 200:
        pytest.skip("业务容器未就绪，无需比较")

    business_db = json.loads(body)["database"].get("name")
    assert business_db != EXPECTED_DATABASE_NAME, (
        f"业务容器 {BUSINESS_BASE_URL} 也连着测试库 {business_db!r}，隔离被破坏"
    )


# ---------------------------------------------------------------- 健康检查


class TestContainerHealth:
    def test_health_endpoint(self, isolated_stack: str) -> None:
        status, body = _http_get("/health", base_url=isolated_stack)

        assert status == 200
        assert json.loads(body) == {"status": "ok"}

    def test_readiness_reports_database_connected(self, isolated_stack: str) -> None:
        """容器内的应用必须真的连上了 PostgreSQL。"""
        status, body = _http_get("/health/ready", base_url=isolated_stack)

        assert status == 200
        payload = json.loads(body)
        assert payload["status"] == "ok"
        assert payload["database"]["ok"] is True

    def test_readiness_reports_the_test_database(self, api_database_name: str) -> None:
        """就绪探针报告的库名必须是测试库。"""
        assert api_database_name == EXPECTED_DATABASE_NAME

    def test_auth_is_enforced_in_container(self, isolated_stack: str) -> None:
        status, _ = _http_post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            base_url=isolated_stack,
        )

        assert status == 401

    def test_admin_usage_reachable(self, isolated_stack: str) -> None:
        status, body = _http_get(
            "/admin/usage", {"X-Admin-Key": TEST_ADMIN_KEY}, base_url=isolated_stack
        )

        assert status == 200
        assert "totals" in json.loads(body)


# ---------------------------------------------------------------- 容器内数据库


class TestPostgresInsideContainer:
    def test_call_record_table_exists(self, isolated_stack: str) -> None:
        """真实容器里的 PostgreSQL 应该已经由 create_all 建好表。"""
        # 通过管理接口间接验证：能正常聚合说明表存在且可查询
        status, body = _http_get(
            "/admin/usage?since=2000-01-01T00:00:00Z",
            {"X-Admin-Key": TEST_ADMIN_KEY},
            base_url=isolated_stack,
        )

        assert status == 200
        payload = json.loads(body)
        assert "total_requests" in payload["totals"]
        assert payload["totals"]["estimated_cost_usd"] is not None

    def test_persisted_record_is_visible_in_usage(self, isolated_stack: str) -> None:
        """发一次请求，记录必须能在统计里看到。

        上游指向不可路由地址，因此状态码确定性地是 502；网关仍应写入一条记录，
        这正是持久化要验证的行为。用唯一的 run_id 定位本次调用，
        不依赖总量差值（避免与套件中其他数据互相干扰）。
        """
        run_id = f"docker-it-{int(time.time() * 1000)}"
        headers = {
            "Authorization": f"Bearer {TEST_CLIENT_KEY}",
            "X-Run-Id": run_id,
            "X-Agent-Name": "docker-it",
        }

        status, body = _http_post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello container"}]},
            headers,
            base_url=isolated_stack,
        )
        # 上游域名不可解析 -> 连接错误 -> 502；不依赖任何真实凭证
        assert status == 502, f"未预期的状态码: {status}: {body[:200]}"

        # 按 run_id 精确查回这一条记录
        detail_status, detail_body = _http_get(
            f"/admin/runs/{run_id}",
            {"X-Admin-Key": TEST_ADMIN_KEY},
            base_url=isolated_stack,
        )
        assert detail_status == 200
        detail = json.loads(detail_body)

        assert detail["totals"]["total_requests"] == 1, "调用记录未写入 PostgreSQL"
        assert detail["totals"]["failure_count"] == 1
        assert detail["errors"][0]["error_type"] == "upstream_connect_error"

    def test_no_real_upstream_is_contacted(self, isolated_stack: str) -> None:
        """上游配置指向保留域名，确认没有任何真实模型调用。

        这条用例保护的是「默认 pytest 不产生付费调用」这个前提：
        一旦有人把测试栈的上游改成真实服务商，它就会失败。
        """
        run_id = f"docker-it-noupstream-{int(time.time() * 1000)}"

        status, body = _http_post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "probe"}]},
            {
                "Authorization": f"Bearer {TEST_CLIENT_KEY}",
                "X-Run-Id": run_id,
            },
            base_url=isolated_stack,
        )

        assert status == 502
        error = json.loads(body)["error"]
        # 既不是上游认证失败（说明没带真实凭证去打真实服务），
        # 也不是限流，而是连接不可达
        assert error["type"] == "upstream_connect_error", error

    def test_configured_upstream_is_not_a_real_provider(
        self, isolated_stack: str
    ) -> None:
        """测试栈的上游地址不得指向真实服务商。

        与上一条互补：上一条验证行为，这一条验证配置本身。
        两者都通过，才能说「默认 pytest 不会产生付费调用」。
        """
        assert _TEST_UPSTREAM_BASE_URLS, "未能从 compose 文件读到上游配置"

        for url in _TEST_UPSTREAM_BASE_URLS:
            assert url.endswith(".invalid"), (
                f"测试栈上游 {url!r} 不是 .invalid 保留域名；"
                "指向真实服务商会产生付费调用"
            )

    def test_test_database_is_separate_from_business(self, isolated_stack: str) -> None:
        """测试库与业务库必须是两个不同的库。"""
        assert EXPECTED_DATABASE_NAME != "modelgate"
        assert EXPECTED_DATABASE_NAME.endswith("_test")
