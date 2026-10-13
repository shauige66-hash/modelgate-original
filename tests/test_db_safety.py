"""防误删守卫的回归测试（对应 tests/db_safety.py）。

这组用例不连接任何数据库，只验证守卫逻辑，可在任意机器上安全运行。
"""

from __future__ import annotations

import pytest

from tests.db_safety import (
    ALLOWED_SUFFIX,
    UnsafeTestDatabaseError,
    assert_safe_test_database_url,
    extract_database_name,
)

SAFE_URL = "postgresql+asyncpg://modelgate:pw@127.0.0.1:5432/modelgate_test"


class TestExtractDatabaseName:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (SAFE_URL, "modelgate_test"),
            (
                "postgresql+asyncpg://u:p@db:5432/modelgate_test",
                "modelgate_test",
            ),
            (
                "postgresql+asyncpg://u:p@127.0.0.1:5432/modelgate_test?ssl=require",
                "modelgate_test",
            ),
            # 密码含百分号编码时不能影响库名解析
            (
                "postgresql+asyncpg://u:p%40ss%2Fword@127.0.0.1:5432/app_test",
                "app_test",
            ),
        ],
    )
    def test_parses_database_name(self, url: str, expected: str) -> None:
        assert extract_database_name(url) == expected

    def test_unparseable_url_raises(self) -> None:
        with pytest.raises(UnsafeTestDatabaseError):
            extract_database_name("this is not a url")


class TestRejectsBusinessDatabase:
    """核心保护：正常业务库绝不能被当作破坏性测试目标。"""

    @pytest.mark.parametrize(
        "name",
        [
            "modelgate",  # 与 .env.example / docker-compose 的业务库同名
            "modelgate_prod",
            "production",
            "app",
            "users",
            "postgres",
            "template0",
            "template1",
        ],
    )
    def test_rejects_names_without_test_suffix(self, name: str) -> None:
        url = f"postgresql+asyncpg://modelgate:pw@127.0.0.1:5432/{name}"

        with pytest.raises(UnsafeTestDatabaseError) as exc_info:
            assert_safe_test_database_url(url)

        # 报错信息必须明确说明「已拒绝」，让人知道测试没有继续跑
        message = str(exc_info.value)
        assert "拒绝" in message, message

    def test_rejects_the_exact_url_from_the_old_readme(self) -> None:
        """README 中曾经的危险示例必须被拒绝。

        这正是本次修复的起因：复制旧示例会把业务库当测试库清空。
        """
        dangerous = "postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate"

        with pytest.raises(UnsafeTestDatabaseError):
            assert_safe_test_database_url(dangerous)


class TestAcceptsTestDatabase:
    @pytest.mark.parametrize(
        "name",
        [
            "modelgate_test",
            "modelgate_testdb_test",
            "app_test",
            "modelgate_TEST",  # 大小写不敏感
        ],
    )
    def test_accepts_test_suffixed_names(self, name: str) -> None:
        url = f"postgresql+asyncpg://modelgate:pw@127.0.0.1:5432/{name}"

        assert assert_safe_test_database_url(url) == name


class TestRejectsOtherUnsafeInputs:
    def test_rejects_sqlite(self) -> None:
        """SQLite 不需要这个变量；传进来多半是配置搞错了。"""
        with pytest.raises(UnsafeTestDatabaseError) as exc_info:
            assert_safe_test_database_url("sqlite+aiosqlite:///:memory:_test")

        assert "PostgreSQL" in str(exc_info.value)

    def test_accepts_sync_postgresql_driver(self) -> None:
        """同步驱动 postgresql:// 也放行 —— 判断依据是数据库名，不是驱动细节。

        本项目的引擎固定使用 asyncpg；这里放宽驱动检查是为了避免
        因为驱动写法不同而产生「看起来像误判」的拒绝，真正的保护是库名约定。
        """
        url = "postgresql://modelgate:pw@127.0.0.1:5432/modelgate_test"

        assert assert_safe_test_database_url(url) == "modelgate_test"

    def test_rejects_mysql(self) -> None:
        with pytest.raises(UnsafeTestDatabaseError):
            assert_safe_test_database_url("mysql+aiomysql://u:p@h:3306/x_test")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(UnsafeTestDatabaseError):
            assert_safe_test_database_url("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(UnsafeTestDatabaseError):
            assert_safe_test_database_url("   ")

    def test_rejects_url_without_database(self) -> None:
        with pytest.raises(UnsafeTestDatabaseError):
            assert_safe_test_database_url("postgresql+asyncpg://u:p@127.0.0.1:5432/")

    def test_error_message_contains_required_format(self) -> None:
        """报错信息要能直接照着改，所以必须给出目标格式。"""
        with pytest.raises(UnsafeTestDatabaseError) as exc_info:
            assert_safe_test_database_url(
                "postgresql+asyncpg://modelgate:pw@127.0.0.1:5432/modelgate"
            )

        message = str(exc_info.value)
        assert ALLOWED_SUFFIX in message
        assert "modelgate_test" in message


class TestConftestFailsFast:
    """conftest 必须在**导入阶段**就拒绝不安全的 TEST_DATABASE_URL。

    为什么不用子进程跑 pytest：
    那样在没装依赖或收集被跳过时会「什么都没验证就通过」。
    这里改为直接在子进程里 import tests.conftest，
    只验证一件事：危险配置会不会让导入本身失败。
    """

    def _import_conftest_with(self, test_database_url: str) -> "subprocess.CompletedProcess[str]":
        import os
        import subprocess
        import sys

        env = dict(os.environ)
        env["TEST_DATABASE_URL"] = test_database_url
        # Windows 默认控制台编码是 GBK，而我们的报错信息是 UTF-8 中文；
        # 不指定编码会让子进程解码失败（UnicodeDecodeError）。
        env["PYTHONIOENCODING"] = "utf-8"

        return subprocess.run(
            [sys.executable, "-c", "import tests.conftest"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    def test_dangerous_env_var_aborts_import(self) -> None:
        """指向业务库（modelgate）时必须导入失败，绝不可能执行到 drop_all。"""
        result = self._import_conftest_with(
            "postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate"
        )

        combined = result.stdout + result.stderr
        assert result.returncode != 0, (
            f"危险 TEST_DATABASE_URL 竟然通过了！输出：{combined}"
        )
        assert "ModelGate 测试安全保护" in combined, combined
        assert "UnsafeTestDatabaseError" in combined, combined

    def test_safe_env_var_allows_import(self) -> None:
        """库名以 _test 结尾时，导入 conftest 必须成功。"""
        result = self._import_conftest_with(
            "postgresql+asyncpg://modelgate:pw@127.0.0.1:5432/modelgate_test"
        )

        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"合法的测试库名被误拒了！输出：{combined}"
        )
        assert "ModelGate 测试安全保护" not in combined, combined

    def test_no_env_var_allows_import(self) -> None:
        """完全不设置该变量时，导入 conftest 必须成功（PostgreSQL 用例会 skip）。"""
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "TEST_DATABASE_URL"}
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, "-c", "import tests.conftest; print('IMPORT_OK')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "IMPORT_OK" in combined

