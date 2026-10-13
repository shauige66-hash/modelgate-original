"""增量结构迁移。

项目没有引入 Alembic：
- 新建表由 `Base.metadata.create_all` 负责（它只建缺失的表）；
- 本模块补上 create_all 覆盖不到的部分：给**已存在**的表补列、补索引、补唯一约束，
  并在表缺失时用 IF NOT EXISTS 兜底建表。

为什么两者都做：迁移可能被单独调用（脚本、测试、运维手工执行），
若只依赖 create_all，单独跑迁移会静默什么都不做。

所有操作都先查现状或使用 IF NOT EXISTS，因此可重复运行、并发启动也安全。
加列前必须用 inspector 查列是否存在，因为 SQLite 不支持
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`。

语句在单个事务内执行，任一步失败则整体回滚，不会留下半成品结构。

绝不包含 DROP / TRUNCATE / DELETE / 类型收窄；
新增列一律 NULLABLE，避免旧行因缺少默认值而写入失败。

回滚：新增的表与列对旧代码无影响（旧代码不查询它们），保留即可。若要撤销：
    DROP TABLE budget_usage;
    ALTER TABLE call_records DROP COLUMN <列名>;
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.logging_config import get_logger

logger = get_logger(__name__)

# 表名 -> 兜底建表 DDL（仅在该表不存在时执行）。
# 与 app/db/models.py 的定义必须保持一致；类型需同时被 PostgreSQL 与 SQLite 接受。
# 唯一约束写在建表语句里：SQLite 不支持 ALTER TABLE 追加约束。
_BOOTSTRAP_TABLES: dict[str, str] = {
    "budget_usage": """
        CREATE TABLE IF NOT EXISTS budget_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope VARCHAR(16) NOT NULL,
            identity VARCHAR(160) NOT NULL,
            window_kind VARCHAR(16) NOT NULL,
            window_start DATE NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_budget_usage_identity_window
                UNIQUE (scope, identity, window_kind, window_start)
        )
    """,
}

# PostgreSQL 的自增写法与 SQLite 不同，需要替换。
_PG_ID_REPLACEMENT = ("id INTEGER PRIMARY KEY AUTOINCREMENT", "id BIGSERIAL PRIMARY KEY")

# 表名 -> {列名: 列类型}，用于给已存在的表补列
_ADDITIVE_COLUMNS: dict[str, dict[str, str]] = {
    "call_records": {
        # 实际处理该请求的服务商，以及请求的模型与上游返回的模型
        "provider": "VARCHAR(32)",
        "requested_model": "VARCHAR(128)",
        "response_model": "VARCHAR(128)",
        # 客户端提供的观测标签，不参与鉴权
        "client_app": "VARCHAR(64)",
        "agent_name": "VARCHAR(64)",
        "run_id": "VARCHAR(64)",
        # 额外发起的上游请求次数；默认不重试，因此通常为 0
        "retry_count": "INTEGER",
    },
}

# (索引名, 表名, 列定义)
_INDEXES: tuple[tuple[str, str, str], ...] = (
    ("ix_call_records_provider", "call_records", "provider"),
    ("ix_call_records_client_app", "call_records", "client_app"),
    ("ix_call_records_agent_name", "call_records", "agent_name"),
    ("ix_call_records_run_id", "call_records", "run_id"),
    # Run 级查询固定按「已认证客户端指纹 + run_id」过滤，再按时间排序；
    # 单列索引无法同时覆盖这三列，因此补一个复合索引。
    (
        "ix_call_records_client_run_created",
        "call_records",
        "key_fingerprint, run_id, created_at",
    ),
    ("ix_budget_usage_window_start", "budget_usage", "window_start"),
)

# (约束名, 表名, 列定义)。配额原子的基础：
# 同一身份在同一窗口只能有一行，否则 ON CONFLICT 扣减会插入多行、配额失效。
# 新建表时由 create_all 带上该约束，这里只处理升级上来的老库。
_UNIQUE_CONSTRAINTS: tuple[tuple[str, str, str], ...] = (
    (
        "uq_budget_usage_identity_window",
        "budget_usage",
        "scope, identity, window_kind, window_start",
    ),
)


def _inspector_columns(sync_conn, table: str) -> set[str]:
    if not inspect(sync_conn).has_table(table):
        return set()
    return {column["name"] for column in inspect(sync_conn).get_columns(table)}


def _inspector_indexes(sync_conn, table: str) -> set[str]:
    if not inspect(sync_conn).has_table(table):
        return set()
    return {index["name"] for index in inspect(sync_conn).get_indexes(table)}


def _inspector_unique_names(sync_conn, table: str) -> set[str]:
    """唯一约束在不同后端可能体现为约束或唯一索引，两者都算。"""
    if not inspect(sync_conn).has_table(table):
        return set()

    inspector = inspect(sync_conn)
    names = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints(table)
        if constraint.get("name")
    }
    names.update(
        index["name"]
        for index in inspector.get_indexes(table)
        if index.get("unique") and index.get("name")
    )
    return names


def _apply_sync(sync_conn) -> list[str]:
    """在同步连接上执行迁移，返回本次实际执行的变更。"""
    applied: list[str] = []

    for table, ddl in _BOOTSTRAP_TABLES.items():
        if inspect(sync_conn).has_table(table):
            continue

        statement = ddl
        if sync_conn.dialect.name == "postgresql":
            statement = ddl.replace(*_PG_ID_REPLACEMENT)

        sync_conn.execute(text(statement))
        applied.append(f"table:{table}")

    for table, columns in _ADDITIVE_COLUMNS.items():
        if not inspect(sync_conn).has_table(table):
            # 表还不存在时由 _BOOTSTRAP_TABLES 或 create_all 负责
            continue

        existing = _inspector_columns(sync_conn, table)
        for column, column_type in columns.items():
            if column in existing:
                continue
            sync_conn.execute(
                text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            )
            applied.append(f"{table}.{column}")

    for index_name, table, columns in _INDEXES:
        if not inspect(sync_conn).has_table(table):
            continue
        if index_name in _inspector_indexes(sync_conn, table):
            continue
        sync_conn.execute(
            text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({columns})")
        )
        applied.append(index_name)

    for name, table, columns in _UNIQUE_CONSTRAINTS:
        if not inspect(sync_conn).has_table(table):
            continue
        if name in _inspector_unique_names(sync_conn, table):
            continue

        if sync_conn.dialect.name == "postgresql":
            sync_conn.execute(
                text(f"ALTER TABLE {table} ADD CONSTRAINT {name} UNIQUE ({columns})")
            )
        else:
            # SQLite 不支持 ALTER TABLE 追加约束，用唯一索引等价实现，
            # ON CONFLICT 的冲突判定同样依赖唯一性。
            sync_conn.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {name} "
                    f"ON {table} ({columns})"
                )
            )
        applied.append(name)

    return applied


async def apply_additive_migrations(engine: AsyncEngine) -> list[str]:
    """补齐缺失的列、索引与唯一约束，返回本次新增的项。

    整个过程在单个事务内，失败即回滚；重复调用与并发启动都安全。
    """
    async with engine.begin() as conn:
        applied = await conn.run_sync(_apply_sync)

    if applied:
        logger.info("结构迁移完成，新增：%s", ", ".join(applied))
    else:
        logger.info("结构迁移：无需变更")
    return applied
