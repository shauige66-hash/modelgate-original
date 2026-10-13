# ModelGate

基于 **FastAPI** 的轻量级 LLM API 网关。

它在客户端与上游大模型服务（默认 DeepSeek，任何 OpenAI Chat Completions 兼容服务均可）之间做一层可观测的转发：统一鉴权、限制单进程并发、记录每次调用的 Token 与成本、并把调用明细落到 PostgreSQL。

---

## 目录

- [项目介绍与技术栈](#项目介绍与技术栈)
- [系统架构](#系统架构)
- [主要功能](#主要功能)
- [快速开始](#快速开始)
- [Docker Compose 部署](#docker-compose-部署)
- [API 使用示例](#api-使用示例)
- [运行测试](#运行测试)
- [项目结构](#项目结构)
- [配置项说明](#配置项说明)
- [设计取舍](#设计取舍)
- [已知限制](#已知限制)
- [未来改进方向](#未来改进方向)

---

## 项目介绍与技术栈

### 它解决什么问题

直接调用大模型 API 时，通常会遇到四件麻烦事：

1. **没有统一的调用入口和鉴权**，每接一个上游就要改一次客户端。
2. **没有用量账本**，一个月下来不知道钱花在哪、花了多少。
3. **上游变慢时容易把上游打爆**，缺少并发保护。
4. **失败原因不可追溯**，只看到「调用失败」，不知道是超时、限流还是上游 500。

ModelGate 用一层薄网关解决这四件事：转发 + 鉴权 + 并发保护 + 用量记账。

### 技术栈

| 组件 | 选型 | 用途 |
|---|---|---|
| Web 框架 | FastAPI | 异步路由、依赖注入、自动生成 OpenAPI 文档 |
| ASGI 服务器 | Uvicorn | 运行应用 |
| HTTP 客户端 | httpx (`AsyncClient`) | 异步访问上游，**复用连接池** |
| 并发控制 | asyncio.Semaphore | 限制单进程内同时在跑的上游请求数 |
| ORM | SQLAlchemy 2.x (async) | 异步 ORM，`Mapped[...]` 声明式模型 |
| 数据库 | PostgreSQL 16 + asyncpg | 持久化调用记录 |
| 配置 | pydantic-settings | 环境变量 / `.env` 注入，密钥用 `SecretStr` |
| 金额计算 | `decimal.Decimal` | 成本估算，避免浮点误差 |
| 测试 | pytest + pytest-asyncio + httpx MockTransport | 全部 mock 上游，不依赖真实付费 API |
| 部署 | Docker + Docker Compose | 一条命令拉起 API + 数据库 |

---

## 系统架构

```mermaid
flowchart LR
    subgraph Client["客户端"]
        C["应用 / curl / OpenAI SDK"]
    end

    subgraph Gateway["ModelGate（单个 FastAPI 进程）"]
        A["鉴权<br/>require_client_api_key"]
        L["并发限制器<br/>asyncio.Semaphore"]
        P["转发层<br/>httpx.AsyncClient<br/>共享连接池"]
        U["用量提取<br/>usage → tokens"]
        M["成本计算<br/>Decimal"]
        R["记录写入<br/>SQLAlchemy 仓储"]
    end

    subgraph Upstream["上游 LLM 服务"]
        D["DeepSeek / 任意兼容服务<br/>POST /chat/completions"]
    end

    subgraph DB["PostgreSQL"]
        T[("call_records")]
    end

    ADM["管理端<br/>GET /admin/usage"]

    C -->|"Bearer API Key"| A
    A -->|"通过"| L
    L -->|"取得槽位"| P
    P --> D
    D -->|"响应 + usage"| U
    U --> M
    M --> R
    R --> T
    P -->|"原样返回给客户端"| C
    ADM -->|"X-Admin-Key"| T
```

### 一次请求的生命周期

```mermaid
sequenceDiagram
    participant C as 客户端
    participant G as ModelGate
    participant S as 上游模型
    participant DB as PostgreSQL

    C->>G: POST /v1/chat/completions (+ API Key)
    G->>G: 校验 API Key（常数时间比较）
    alt 鉴权失败
        G-->>C: 401 / 403（不消耗上游额度）
    end

    G->>G: 生成 request_id，记录开始时间
    G->>G: 申请并发槽位（最多等 N 秒）

    alt 等待超时
        G-->>C: 503 concurrency_timeout
    end

    G->>S: POST /chat/completions（复用连接池）
    alt 上游超时 / 报错
        S--xG: Timeout / 4xx / 5xx
        G->>DB: 记录失败（error_type、NULL tokens）
        G-->>C: 504 / 502
    else 正常返回
        S-->>G: 200 + usage
        G->>G: 提取 usage → Decimal 估算成本
        G->>DB: 记录成功（tokens、cost、latency）
        G-->>C: 200 原始响应 + X-Request-Id
    end

    G->>G: 释放并发槽位（finally，异常也会释放）
```

---

## 主要功能

### 1. LLM API 网关

- `POST /v1/chat/completions`，请求与响应体兼容 OpenAI Chat Completions 风格。
- 通过 `httpx.AsyncClient` **异步**访问上游；客户端在应用启动时创建一次、关闭时释放，全程**复用连接池**。
- 上游地址、模型、密钥、超时、连接池大小均可通过环境变量配置。
- 只支持**非流式**请求。传 `stream: true` 会被显式拒绝（`422`），而不是假装支持却返回整包。

### 2. 异步并发控制

- 用 `asyncio.Semaphore` 限制同一进程内同时执行的上游请求数，上限由 `MAX_CONCURRENT_REQUESTS` 配置。
- 等待槽位有独立超时（`CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS`），超时返回 `503 concurrency_timeout`。
- 槽位在 `finally` 中释放：业务异常、上游超时、客户端主动断开都不会泄漏信号量，避免死锁。
- ⚠️ **这不是分布式限流。** 它只在单个进程内生效；多副本部署时每个副本各有一份配额，实际总并发是 `副本数 × 上限`。

### 3. PostgreSQL 持久化

- SQLAlchemy 2.x 异步 ORM，一行一条调用记录。
- 记录字段：`request_id`、模型、请求时间、耗时、HTTP 状态、输入/输出 Token、估算成本、错误类型，以及密钥指纹、消息条数、Prompt 字符数、客户端 IP。
- **成功与失败请求都会记录**，失败原因归类到 `error_type`（见 `app/errors.py` 的 `ErrorType`）。
- 每个请求一个会话；写入失败**不影响**已经拿到的模型响应（只记 error 日志），避免因为审计库抖动让用户白等一次推理。

### 4. Token 与成本统计

- 优先从上游响应的 `usage` 字段读取 `prompt_tokens` / `completion_tokens`。
- **上游没返回 `usage` 时，Token 记为 `NULL`（未知），绝不当成 0。** 这是刻意的选择：把未知记成 0 会让统计静默偏低，看起来像「免费」。
- 使用 `Decimal` 计算，单价单位为**美元 / 每百万 Token**，配置项显式标注单位。金额以字符串返回，避免 JSON 把精确值退化成浮点。
- `GET /admin/usage` 返回调用次数、Token 总量、估算成本、失败数量，并附 `data_quality` 字段告知有多少条记录未计入汇总。

### 5. API 安全

- 客户端 Key（`/v1/*`）与管理 Key（`/admin/*`）**互相独立**。
- 密钥只从环境变量 / `.env` 读取，源码中不写死任何密钥。
- 比较密钥使用 `secrets.compare_digest`（常数时间），且对白名单全量比较，不因命中位置提前返回。
- 密钥字段使用 pydantic `SecretStr`，`repr()` 与日志中只会显示 `**********`。
- 日志与数据库都**不保存明文密钥**（只保存 SHA-256 前 12 位指纹）与**完整用户 Prompt**（只保存条数与字符数）。
- 提供 [`.env.example`](.env.example)；真实 `.env` 已被 `.gitignore` 忽略。

### 6. Docker 部署

- 多阶段 `Dockerfile`：依赖在 builder 阶段安装，运行镜像不带编译工具链；以非 root 用户运行。
- `docker-compose.yml` 编排 API + PostgreSQL 16。
- 数据库有 `pg_isready` healthcheck，API 通过 `depends_on: condition: service_healthy` 等待数据库真正就绪。
- API 容器自带 `HEALTHCHECK`，打不依赖数据库的 `/health`。

### 7. 自动化测试

- pytest + pytest-asyncio，全部通过 `httpx.MockTransport` 模拟上游，**不依赖真实付费模型 API**。
- 覆盖：健康检查、基础转发、鉴权成功与失败、上游超时与各类异常、并发上限是否生效、Decimal 成本计算、用量统计接口、调用记录持久化。
- 默认可在内存 SQLite 上跑完；设置 `TEST_DATABASE_URL` 后额外运行真实 PostgreSQL 集成测试。

---

## 快速开始

### 前置条件

- Python 3.11+（开发环境使用 3.13）
- 一个上游 API Key（如 [DeepSeek](https://platform.deepseek.com/)）；如果只想跑测试，**不需要**真实 Key

### 1. 准备虚拟环境与依赖

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，至少要改这几项：

```dotenv
UPSTREAM_API_KEY=sk-你的真实密钥
CLIENT_API_KEYS=sk-client-local-dev-key
ADMIN_API_KEY=sk-admin-local-dev-key
DATABASE_URL=postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate
```

> 本机直接跑（不用 Docker）时，注意把 `DATABASE_URL` 里的主机名从 `db` 改成 `127.0.0.1`。

### 3. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后：

- 存活探针：<http://127.0.0.1:8000/health>
- 就绪探针：<http://127.0.0.1:8000/health/ready>
- 交互式 API 文档：<http://127.0.0.1:8000/docs>

> 数据库连不上时服务**仍会启动**，但 `/health/ready` 返回 `503`。这样在容器编排里能一眼区分「应用没起来」和「数据库没好」，而不用让容器无限重启。

---

## Docker Compose 部署

### 1. 配置

```powershell
Copy-Item .env.example .env
# 至少填入 UPSTREAM_API_KEY
```

### 2. 启动

```powershell
docker compose up --build
```

后台运行：

```powershell
docker compose up --build -d
```

### 3. 验证

```powershell
# API 存活
curl http://127.0.0.1:8000/health

# 数据库与并发水位
curl http://127.0.0.1:8000/health/ready

# 查看服务状态（db 应为 healthy）
docker compose ps
```

### 4. 常用运维命令

```powershell
docker compose logs -f api          # 跟踪 API 日志
docker compose logs -f db           # 跟踪数据库日志
docker compose exec db psql -U modelgate -d modelgate -c "SELECT count(*) FROM call_records;"
docker compose down                 # 停止（保留数据）
docker compose down -v              # 停止并删除数据卷
```

启动顺序由 healthcheck 保证：`db` 通过 `pg_isready` 后 `api` 才开始启动。

### ⚠️ 默认凭据只适用于本机开发

`docker-compose.yml` 与 `.env.example` 里的 `modelgate:modelgate`、`sk-client-local-dev-key`、
`sk-admin-local-dev-key` 都是**本地开发占位值，强度极低**。

为降低误暴露风险，两个端口都已限制为仅监听宿主机回环地址，局域网/公网无法直接访问：

```yaml
db:
  ports:
    - "127.0.0.1:5432:5432"    # 不用 "5432:5432"
api:
  ports:
    - "127.0.0.1:8000:8000"    # 不用 "8000:8000"
```

`api` 之所以也必须收紧：默认的 `CLIENT_API_KEYS` / `ADMIN_API_KEY` 是弱占位值，
若 `8000` 暴露到局域网，任何人都能拿默认 Key 调用网关、消耗你的模型额度，
甚至通过 `/admin/usage` 读取你的用量统计。

本机访问不受影响（`curl http://127.0.0.1:8000/health` 正常）。

如果你要把服务放到本机以外的环境（局域网、云主机、CI runner），**必须**先做这三件事：

1. 修改 `POSTGRES_PASSWORD`（不要再用 `modelgate`）
2. 更换 `CLIENT_API_KEYS` 与 `ADMIN_API_KEY`（生成足够随机的值）
3. 不要把 `.env` 提交到任何仓库（它已在 `.gitignore` 中）

完成以上三步后，再把端口显式改回 `"8000:8000"` / `"5432:5432"` 对外提供服务。

---

## API 使用示例

示例中的 `$CLIENT_KEY` / `$ADMIN_KEY` 请替换为你 `.env` 里的真实值。

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

### 聊天补全

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "你是一个简洁的助手。"},
      {"role": "user", "content": "用一句话解释什么是 API 网关。"}
    ],
    "temperature": 0.7
  }'
```

响应（原样透传上游，并带上网关生成的调用 ID）：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "deepseek-chat",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "API 网关是..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 28, "completion_tokens": 35, "total_tokens": 63}
}
```

响应头中的 `X-Request-Id` 就是数据库里那条记录的 `request_id`，可用于对账。

也可以用 `X-API-Key` 头代替 `Authorization`：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CLIENT_KEY" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

### 查询用量统计

```bash
curl "http://127.0.0.1:8000/admin/usage?since=2026-01-01T00:00:00Z" \
  -H "X-Admin-Key: $ADMIN_KEY"
```

```json
{
  "range": {"since": "2026-01-01T00:00:00+00:00", "until": null, "model_filter": null},
  "totals": {
    "total_requests": 12,
    "success_count": 11,
    "failure_count": 1,
    "prompt_tokens": 1320,
    "completion_tokens": 880,
    "total_tokens": 2200,
    "estimated_cost_usd": "0.001324",
    "price_unit": "USD per 1,000,000 tokens"
  },
  "data_quality": {
    "records_with_unknown_tokens": 1,
    "records_without_cost": 1,
    "note": "上游未返回 usage 的记录 Token 记为未知，不会按 0 计入总量；上述两个字段用于提示统计的完整性"
  },
  "by_model": [
    {
      "model": "deepseek-chat",
      "total_requests": 12,
      "success_count": 11,
      "failure_count": 1,
      "prompt_tokens": 1320,
      "completion_tokens": 880,
      "estimated_cost_usd": "0.001324"
    }
  ],
  "errors": [{"error_type": "upstream_timeout", "count": 1}]
}
```

上例中 11 次成功调用的成本为 `1320 × 0.27 / 1e6 + 880 × 1.10 / 1e6 ≈ 0.001324` 美元；那次失败调用没有 Token 数据，因此被 `records_with_unknown_tokens` 单独标出，而不是按 0 计入。

可用查询参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `since` | 24 小时前 | ISO 8601，可不带时区（按 UTC 处理） |
| `until` | 无上限 | ISO 8601 |
| `model` | 全部 | 只看某个模型 |
| `group_by_model` | `true` | 是否返回按模型拆分的明细 |
| `top_errors` | `5` | 返回前 N 个错误类型；设 `0` 可关闭 |

`estimated_cost_usd` 是**估算值**（基于配置的单价），不是账单金额。

### 错误响应格式

所有网关层错误统一为：

```json
{
  "error": {
    "message": "上游模型服务返回 401: Invalid API key",
    "type": "upstream_auth_error",
    "code": 502
  }
}
```

| 状态码 | `type` | 含义 |
|---|---|---|
| 401 | — | 未提供 API Key |
| 403 | — | API Key 不正确 |
| 422 | — | 请求体不合法（如 `stream: true`、`messages` 为空） |
| 502 | `upstream_http_error` | 上游返回 4xx/5xx |
| 502 | `upstream_auth_error` | 上游拒绝我们的密钥（401/403） |
| 502 | `upstream_connect_error` | 连不上上游 |
| 502 | `upstream_invalid_response` | 上游响应不是合法 JSON 或缺少 `choices` |
| 503 | `concurrency_timeout` | 等待并发槽位超时 |
| 503 | `upstream_not_configured` | 服务端未配置上游密钥 |
| 504 | `upstream_timeout` | 上游响应超时 |

---

## 运行测试

测试全部使用 mock 上游，**不会调用真实付费 API**。

### 安装测试依赖

```powershell
pip install -r requirements-dev.txt
```

### 运行全部测试

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

默认结果：**145 passed, 15 skipped**（共收集 160 个用例）。

跳过项需要外部环境：9 个 PostgreSQL 集成测试缺 `TEST_DATABASE_URL`，
6 个容器集成测试缺 Docker 或真实上游 Key（详见对应章节）。

**如果 Docker + 专用测试库就绪**，先创建测试库（库名必须以 `_test` 结尾，见下一节）：

```powershell
docker compose up -d --build
docker compose exec db psql -U modelgate -d postgres -c "CREATE DATABASE modelgate_test;"

$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
.\.venv\Scripts\python.exe -m pytest -q
# -> 154 passed, 6 skipped
```

| 运行条件 | 结果 |
|---|---|
| 仅单元测试（无 Docker、无测试库） | `145 passed, 15 skipped` |
| 加上专用测试库 | `154 passed, 6 skipped` |
| 再加上真实上游 Key（容器集成测试） | `160 passed` |

> 容器集成测试会向运行中的容器发真实请求，因此要求 `.env` 中配置了
> 真实 `UPSTREAM_API_KEY`；仍是占位值时这 6 个用例会自动跳过，避免产生无效的付费调用。

### ⚠️ PostgreSQL 集成测试是破坏性的

`tests/test_postgres_integration.py` 会执行 `drop_all` + `create_all`，
**清空目标库里的 `call_records` 表**。

为防止误删业务数据，`TEST_DATABASE_URL` 受到强制校验（实现见 [tests/db_safety.py](tests/db_safety.py)）：

1. **库名必须以 `_test` 结尾**，否则测试在**导入阶段直接报错**，绝不会执行到 `drop_all`。
2. 连上数据库后再查一次 `current_database()` 复核库名。
3. 只接受 PostgreSQL 驱动，并拒绝 `postgres` / `template0` / `template1` 系统库。

因此下面这种写法会被**明确拒绝**（这也正是本次修复掉的历史错误示例）：

```powershell
# ❌ 错误：这是业务库，会被拒绝
$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate"
# UnsafeTestDatabaseError: 拒绝把 'modelgate' 作为测试数据库：
# 数据库名必须以 '_test' 结尾 ...

# ✅ 正确：专用测试库
$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
```

创建测试库（与业务库 `modelgate` 完全分开）：

```powershell
docker compose exec db psql -U modelgate -d postgres -c "CREATE DATABASE modelgate_test;"
```

`tests/test_docker_integration.py` 另有一道防护：若 `TEST_DATABASE_URL` 与容器使用的
数据库是同一个库，它会直接失败并提示改用独立测试库 ——
因为容器集成测试会写入记录，而这些数据随后会被上面的 `drop_all` 清空。

### 只跑某一部分

```powershell
pytest tests/test_cost.py -v                    # Token 与 Decimal 成本
pytest tests/test_auth.py -v                    # 鉴权
pytest tests/test_concurrency.py -v             # 并发上限与超时
pytest tests/test_upstream_errors.py -v         # 上游异常处理
pytest tests/test_persistence.py -v             # 调用记录持久化
pytest tests/test_admin_usage.py -v             # 用量统计接口
```

### 运行 PostgreSQL 集成测试

单元测试跑在内存 SQLite 上（快、零依赖），但 SQLite 与 PostgreSQL 的方言和类型行为并不完全一致。要验证**生产真正使用的组合**：

```powershell
docker compose up -d db

# 必须使用专用测试库（库名以 _test 结尾），否则测试会拒绝运行
docker compose exec db psql -U modelgate -d postgres -c "CREATE DATABASE modelgate_test;"
$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
.\.venv\Scripts\python.exe -m pytest tests/test_postgres_integration.py -v
```

这些用例验证建表、`Numeric(18,6)` 精度、`timestamptz` 时区、唯一约束、聚合查询与索引是否真实存在。

### 运行 Docker 容器集成测试

验证「真实容器里的端到端行为」：容器是否起来、健康检查是否通过、容器内应用是否真的连上了 PostgreSQL、以及一次调用能否在真实数据库中落库。

```powershell
docker compose up -d --build

pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_docker_integration.py -v
```

未运行 Docker 时这 6 个用例会自动跳过，不会让本地 `pytest` 变红。

> **副作用说明（非破坏性）**：`test_persisted_record_is_visible_in_usage` 会向
> **正在运行的容器**（`:8000`）真实发一次请求，因为 Docker 集成测试的用途就是验证线上部署形态。
> 该请求会在容器所用的数据库里新增调用记录（每跑一次多约 2 条），
> 但**不会删除或清空任何数据** —— 与上面 `drop_all` 的 PostgreSQL 集成测试完全不同。
> 若不想污染业务库，可以先把容器指向 `modelgate_test`，或跑完手工清理这些记录。

### 测试覆盖范围对照

| 要求 | 对应文件 |
|---|---|
| 健康检查及基本代理请求 | `tests/test_health_and_proxy.py` |
| API Key 鉴权成功与失败 | `tests/test_auth.py` |
| 上游超时与异常处理 | `tests/test_upstream_errors.py` |
| 并发上限是否生效 | `tests/test_concurrency.py` |
| Token 和 Decimal 成本计算 | `tests/test_cost.py` |
| 上游错误脱敏（不泄露密钥） | `tests/test_upstream_secret_leak.py` |
| 测试库防误删保护 | `tests/test_db_safety.py` |
| 用量统计接口 | `tests/test_admin_usage.py` |
| PostgreSQL 调用记录持久化 | `tests/test_persistence.py`（SQLite）+ `tests/test_postgres_integration.py`（真实 PostgreSQL） |
| 容器集成 | `tests/test_docker_integration.py` |

---

## 项目结构

```
modelgate-original/
├── app/
│   ├── main.py                 # 应用入口：lifespan 管理共享资源、注册路由与异常处理
│   ├── config.py               # pydantic-settings 配置，密钥用 SecretStr
│   ├── dependencies.py         # 依赖注入：从 app.state 取单例资源
│   ├── security.py             # 鉴权、密钥指纹、Prompt 摘要
│   ├── errors.py               # ErrorType 枚举与统一错误响应
│   ├── request_context.py      # ContextVar 传递 request_id 给异常处理器
│   ├── schemas.py              # 请求/响应 Pydantic 模型
│   ├── upstream.py             # httpx 上游客户端（连接池复用）
│   ├── concurrency.py          # asyncio.Semaphore 并发限制器
│   ├── cost.py                 # Decimal 成本计算
│   ├── redaction.py            # 密钥脱敏（上游错误文本落日志前必须过这里）
│   ├── logging_config.py       # 日志配置
│   ├── api/
│   │   └── routes/
│   │       ├── health.py       # /health、/health/ready
│   │       ├── chat.py         # /v1/chat/completions
│   │       └── admin.py        # /admin/usage
│   └── db/
│       ├── models.py           # CallRecord ORM 模型
│       ├── session.py          # 异步引擎与会话工厂
│       └── repository.py       # 数据访问与聚合查询
├── tests/
│   ├── conftest.py                          # fixtures：测试应用、Mock 上游、内存数据库
│   ├── db_safety.py                         # 测试库防误删守卫（库名必须 _test 结尾）
│   ├── test_health_and_proxy.py
│   ├── test_auth.py
│   ├── test_upstream_errors.py
│   ├── test_upstream_secret_leak.py         # 上游敏感信息不外泄回归测试
│   ├── test_db_safety.py                    # 防误删守卫的回归测试
│   ├── test_concurrency.py
│   ├── test_cost.py
│   ├── test_persistence.py
│   ├── test_admin_usage.py
│   ├── test_postgres_integration.py
│   └── test_docker_integration.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

---

## 配置项说明

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `APP_NAME` | `ModelGate` | 应用名 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `UPSTREAM_BASE_URL` | `https://api.deepseek.com` | 上游地址 |
| `UPSTREAM_API_KEY` | 空 | **必填**，上游密钥 |
| `UPSTREAM_MODEL` | `deepseek-chat` | 客户端未指定 `model` 时的默认模型 |
| `UPSTREAM_TIMEOUT_SECONDS` | `60` | 单次上游请求超时（秒） |
| `UPSTREAM_MAX_CONNECTIONS` | `100` | 连接池上限 |
| `UPSTREAM_MAX_KEEPALIVE_CONNECTIONS` | `20` | 保持的连接数 |
| `MAX_CONCURRENT_REQUESTS` | `8` | 单进程并发上限 |
| `CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS` | `10` | 等待并发槽位的上限（秒） |
| `CLIENT_API_KEYS` | 空 | 客户端密钥，多个用英文逗号分隔 |
| `ADMIN_API_KEY` | 空 | 管理密钥，须与客户端密钥不同 |
| `DATABASE_URL` | `postgresql+asyncpg://modelgate:modelgate@db:5432/modelgate` | 数据库连接串 |
| `DB_POOL_SIZE` | `10` | 连接池大小（SQLite 下自动忽略） |
| `DB_MAX_OVERFLOW` | `20` | 连接池溢出上限 |
| `DB_ECHO` | `false` | 是否打印 SQL（调试用） |
| `PRICE_INPUT_PER_MILLION_USD` | `0.27` | 输入单价（美元/百万 Token） |
| `PRICE_OUTPUT_PER_MILLION_USD` | `1.10` | 输出单价（美元/百万 Token） |

> 单价只是**估算口径**。真实价格请以 DeepSeek 官方文档为准；本项目按「输入/输出」两档简化，未区分缓存命中与未命中。

---

## 设计取舍


**1. Token 未知用 `NULL` 而不是 0。**
统计时 `SUM()` 会跳过 `NULL`，所以「Token 总量」只累加已知值，可能偏小。为了不让偏小被误读成准确账单，接口额外返回 `records_with_unknown_tokens` 与 `records_without_cost`。反过来，如果记成 0，使用者根本无从发现数据不完整。

**2. 落库失败不影响响应。**
模型推理已经完成、钱已经花了，此时因为审计库抖动而给客户端返回 500 是得不偿失的。所以记录写入失败只记 `error` 日志。代价是审计可能缺失 —— 这是一个明确的取舍，不是遗漏。

**3. 用 `create_all` 而不是 Alembic。**
项目只有一张表，引入迁移框架会增加上手成本。局限见下方「已知限制」。

**4. 配置通过 `app.state` 而不是模块级全局变量。**
`create_app()` 接受可选的 `settings` 参数，测试可以注入自己的配置。最初的实现让鉴权依赖直接调用模块级缓存的 `get_settings()`，结果测试注入的配置被忽略（这个 bug 是被测试抓出来的）。现在依赖统一从 `app.state.settings` 读取，单一来源。

**5. 请求 ID 用 `ContextVar` 传递。**
异常处理器拿不到路由里的局部变量，失败响应也需要带 `X-Request-Id`（否则客户端无法用返回的 ID 对账数据库）。`ContextVar` 在 asyncio 中按任务隔离，并发请求之间不会串。

**6. 显式拒绝 `stream: true`。**
比「假装支持、实际返回整包」更诚实。

---

## 已知限制


1. **并发控制不是分布式限流。** `asyncio.Semaphore` 只在单进程内生效。多副本部署时总并发 = 副本数 × `MAX_CONCURRENT_REQUESTS`。真正的全局限流需要 Redis 之类的共享计数器。
2. **不支持流式（SSE）。** 只实现了非流式转发，`stream: true` 会被拒绝。流式代理需要处理增量解析、心跳、客户端中断等情况。
3. **没有重试与熔断。** 上游瞬时失败会直接返回给客户端，没有自动重试、退避或熔断器。上游持续故障时每个请求都会真实打过去。
4. **只有全局一套单价。** 成本按配置的输入/输出单价估算，**未按模型区分**，也不区分缓存命中。多模型、多价格的场景会算不准。
5. **没有数据库迁移工具。** 用 `create_all` 建表，只对「表不存在」有效。修改模型字段后需要手工 `ALTER TABLE` 或重建库，不适合需要平滑升级的场景。
6. **没有请求体大小限制与 Prompt 注入防护。** 网关做透传，不审查内容。
7. **没有速率限制（rate limiting）。** 只有并发上限，没有「每 Key 每分钟 N 次」这类配额。
8. **管理接口没有分页。** `/admin/usage` 返回聚合结果；明细查询（`list_recent`）只存在于仓储层，未暴露为 HTTP 接口。
9. **无分布式追踪。** 有 `request_id` 和结构化日志，但没有接入 OpenTelemetry。
10. **Docker 镜像未做漏洞扫描**，也未固定基础镜像的 digest。
11. **客户端密钥是静态的**，没有签发/轮换机制，也没有按 Key 计费归属（只存了指纹用于排查）。

---

## 未来改进方向

按「投入产出比」排序

1. **按模型维护价格表。** 把单价从单个配置项换成 `{model: {input, output}}` 字典，成本计算时按记录的实际模型取价，未命中时回落到默认价并标记为估算。
2. **接入 Alembic。** 让表结构变更可版本化、可回滚。
3. **加入速率限制。** 先做单进程令牌桶（按 Key），再演进到 Redis 共享计数。
4. **实现流式代理。** 用 `StreamingResponse` + `httpx.stream()` 转发 SSE；难点在于流式响应通常拿不到 `usage`，需要按字符估算或要求上游开启 `stream_options.include_usage`。
5. **重试与熔断。** 只对幂等的、明确可重试的错误（连接错误、429、5xx）做有限次指数退避；连续失败则短暂熔断。
6. **调用明细查询接口。** 加上分页与过滤，暴露 `GET /admin/calls`。
7. **可观测性。** 接入 OpenTelemetry，把 `request_id` 串成 trace，并导出 Prometheus 指标（QPS、延迟直方图、错误率、并发水位）。
8. **多上游与路由。** 支持按模型路由到不同上游（DeepSeek / OpenAI / 本地 vLLM），并做故障转移。
9. **CI 流水线。** GitHub Actions 跑 lint + pytest + Docker 构建，并加上 PostgreSQL service container 跑集成测试。
10. **密钥生命周期管理。** 支持多 Key、过期时间、按 Key 的用量配额。

---
