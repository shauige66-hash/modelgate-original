"""PostgreSQL 集成测试的防误删保护。

test_postgres_integration.py 会执行 drop_all + create_all，
若 TEST_DATABASE_URL 指向业务库就会清空真实数据。两道防线：

1. 静态检查：库名必须符合 *_test 约定。误把 DATABASE_URL 复制成
   TEST_DATABASE_URL 时会被挡住。
2. 运行时确认：连上后查询 current_database() 再核对一次，
   可发现「连接串写的是 A、实际连到 B」的情况
   （例如被 PGHOST/PGDATABASE 等环境变量改写）。

同时显式拒绝 SQLite 与 PostgreSQL 系统库。
"""

from __future__ import annotations

import re

from sqlalchemy.engine import make_url

ALLOWED_SUFFIX = "_test"

# 绝不能作为破坏性测试目标的名字
FORBIDDEN_EXACT = frozenset(
    {
        "postgres",
        "template0",
        "template1",
        "",
    }
)

# 小写字母/数字/下划线，且以 _test 结尾
_SAFE_NAME_PATTERN = re.compile(r"^[a-z0-9_]*_test$")

# 要求的连接串形状，写进报错信息便于直接复制修改
REQUIRED_FORMAT = (
    "postgresql+asyncpg://<user>:<password>@<host>:<port>/<database>_test"
)


class UnsafeTestDatabaseError(RuntimeError):
    """TEST_DATABASE_URL 指向的库不适合做破坏性测试。"""


def extract_database_name(database_url: str) -> str:
    """取出连接串中的数据库名。

    用 SQLAlchemy 的 make_url 解析，可正确处理百分号编码的密码与查询参数。
    """
    try:
        url = make_url(database_url)
    except Exception as exc:  # noqa: BLE001 - 解析失败一律视为不安全
        raise UnsafeTestDatabaseError(
            f"无法解析 TEST_DATABASE_URL（{exc}）。"
            f"要求的格式：{REQUIRED_FORMAT}"
        ) from exc

    return url.database or ""


def assert_safe_test_database_url(database_url: str) -> str:
    """校验连接串可安全用于破坏性测试，返回库名。

    不通过时抛出 UnsafeTestDatabaseError，并给出可照做的修复指引。
    """
    if not database_url.strip():
        raise UnsafeTestDatabaseError(
            "TEST_DATABASE_URL 为空。请显式提供一个专用的测试数据库，"
            f"格式：{REQUIRED_FORMAT}"
        )

    url = make_url(database_url)
    driver = url.drivername or ""

    # 集成测试的目的是验证生产使用的 PostgreSQL 行为
    if not driver.startswith("postgresql"):
        raise UnsafeTestDatabaseError(
            f"TEST_DATABASE_URL 必须使用 PostgreSQL（当前驱动：{driver!r}）。"
            f"单元测试请用内存 SQLite，无需设置该变量。"
        )

    name = (url.database or "").strip()

    # 先判断系统库，避免把它误报成「命名不合规」
    if name.lower() in FORBIDDEN_EXACT:
        raise UnsafeTestDatabaseError(
            f"拒绝把 {name!r} 作为测试数据库：它是 PostgreSQL 系统库，"
            f"不能被 drop_all 清空。"
        )

    if not _SAFE_NAME_PATTERN.match(name.lower()):
        raise UnsafeTestDatabaseError(
            f"拒绝把 {name!r} 作为测试数据库：数据库名必须以 {ALLOWED_SUFFIX!r} 结尾，"
            f"以表明它是可随时清空的专用测试库。\n"
            f"  当前：{name}\n"
            f"  应为：{name}_test 之类的名字\n"
            f"  完整格式：{REQUIRED_FORMAT}\n"
            f"这是为了防止误把业务数据库当作测试目标而被 drop_all 清空。"
        )

    return name


async def assert_connected_database_is_test(engine) -> str:
    """用 current_database() 核对实际连到的库。

    连接串看起来合法不代表连对了库，因此这里再确认一次。
    """
    from sqlalchemy import text

    async with engine.connect() as conn:
        actual = (await conn.execute(text("SELECT current_database()"))).scalar_one()

    if not isinstance(actual, str) or not _SAFE_NAME_PATTERN.match(actual.lower()):
        raise UnsafeTestDatabaseError(
            f"实际连接的数据库是 {actual!r}，不是以 {ALLOWED_SUFFIX!r} 结尾的测试库，"
            f"已中止破坏性操作（drop_all）。"
        )

    return actual
