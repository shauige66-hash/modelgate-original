# ModelGate

基于 **FastAPI** 的轻量级 LLM API 网关，面向 Agent 系统的基础设施组件。

它在客户端与多家上游大模型服务之间做一层可观测的转发：用统一的 OpenAI 兼容接口访问
DeepSeek、OpenAI 等不同服务商，同时负责鉴权、单进程并发保护、**配额治理**、
分模型与分服务商的 Token / 成本统计，并把调用明细落到 PostgreSQL。

通过 `X-Client-App` / `X-Agent-Name` / `X-Run-Id` 三个请求头，可以把调用关联到
多智能体工作流，并按 run 聚合出「哪个 Agent 节点花了多少 Token、多少钱、平均耗时多久」。

> 只支持非流式（`stream: false`）。流式代理尚未实现，传 `stream: true` 会返回 422。

**适合什么场景**：多个 Agent / 应用需要统一入口访问多个模型服务商，
并且需要按调用方或按一次工作流限制调用量、盘点成本、定位失败原因。

**不适合什么场景**：需要跨副本的分布式限流（当前配额依赖数据库但并发保护是单进程的）、
需要 Token 或金额的硬性预算控制（当前只做请求计数）、需要流式输出。

---

## 目录

- [项目介绍与技术栈](#项目介绍与技术栈)
- [系统架构](#系统架构)
- [主要功能](#主要功能)
- [快速开始](#快速开始)
- [Docker Compose 部署](#docker-compose-部署)
- [API 使用示例](#api-使用示例)
- [Agent 集成](#agent-集成)
- [配额与可靠性](#配额与可靠性)
- [性能基准](#性能基准)
- [运行测试](#运行测试)
- [项目结构](#项目结构)
- [配置项说明](#配置项说明)
- [数据库结构变更](#数据库结构变更)
- [设计取舍](#设计取舍)
- [已知限制](#已知限制)
- [未来改进方向](#未来改进方向)

---

## 项目介绍与技术栈

### 它解决什么问题

直接调用大模型 API 时，通常会遇到五件麻烦事：

1. **没有统一的调用入口和鉴权**，每接一个上游就要改一次客户端。
2. **多服务商切换成本高**，模型名、Base URL、密钥散落在各业务代码里。
3. **没有用量账本**，一个月下来不知道钱花在哪、花了多少。
4. **上游变慢时容易把上游打爆**，缺少并发保护。
5. **失败原因不可追溯**，只看到「调用失败」，不知道是超时、限流还是上游 500。

ModelGate 用一层薄网关解决这几件事：统一入口 + 多服务商路由 + 鉴权 + 并发保护 + 用量记账。

### 技术栈

| 组件 | 选型 | 用途 |
|---|---|---|
| Web 框架 | FastAPI | 异步路由、依赖注入、自动生成 OpenAPI 文档 |
| ASGI 服务器 | Uvicorn | 运行应用 |
| HTTP 客户端 | httpx (`AsyncClient`) | 异步访问上游，**每个服务商独立连接池** |
| 并发控制 | asyncio.Semaphore | 限制单进程内同时在跑的上游请求数 |
| 模型路由 | 自建注册表（`app/providers.py`） | 模型名 → 服务商，未登记即拒绝 |
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
        AG["Agent 工作流<br/>X-Client-App / X-Agent-Name / X-Run-Id"]
    end

    subgraph Gateway["ModelGate（单个 FastAPI 进程）"]
        A["鉴权<br/>require_client_api_key"]
        RT["模型注册表<br/>model → provider"]
        Q["配额预扣<br/>独立短事务，提交后才调上游"]
        L["并发限制器<br/>asyncio.Semaphore"]
        P["转发层<br/>每个服务商一个<br/>httpx.AsyncClient<br/>默认不重试"]
        U["用量提取<br/>usage → tokens"]
        M["成本计算<br/>按服务商/模型取价<br/>Decimal"]
        R["记录写入<br/>SQLAlchemy 仓储"]
    end

    subgraph Upstream["上游 LLM 服务"]
        D["DeepSeek<br/>独立 Key + 连接池"]
        O["OpenAI<br/>独立 Key + 连接池"]
    end

    subgraph DB["PostgreSQL"]
        T[("call_records")]
        B[("budget_usage")]
    end

    ADM["管理端<br/>/admin/usage<br/>/admin/runs<br/>/admin/budget"]

    C -->|"Bearer API Key"| A
    AG -.->|"观测标签"| A
    A -->|"通过"| RT
    RT -->|"未登记 → 400"| C
    RT -->|"已登记"| Q
    Q -->|"超限 → 429（不调上游）"| C
    Q -->|"扣减并提交"| B
    Q --> L
    L -->|"取得槽位"| P
    P --> D
    P --> O
    D -->|"响应 + usage"| U
    O -->|"响应 + usage"| U
    U --> M
    M --> R
    R --> T
    P -->|"原样返回给客户端"| C
    ADM -->|"X-Admin-Key"| T
```

模型名只用于查表，不参与 URL 或凭证的构造 —— 上游目标只能来自配置，
调用方无法通过 `model` 字段把请求指向任意地址。

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

    G->>G: 校验追踪标签（X-Client-App 等）
    alt 标签不合法
        G-->>C: 400 invalid_tracking_header
    end

    G->>G: 查模型注册表
    alt 模型未登记
        G-->>C: 400 unknown_model（不调用任何上游）
    end
    alt 服务商未配置 Key
        G-->>C: 503 upstream_not_configured
    end

    G->>G: 生成 request_id，记录开始时间
    G->>DB: 配额预扣（独立短事务，立即提交）
    alt 配额超限
        G-->>C: 429 quota_exceeded（不调用上游）
    end
    Note over G,DB: 此时配额行锁已释放，不会把锁带进上游等待

    G->>G: 申请并发槽位（最多等 N 秒）

    alt 等待超时
        G-->>C: 503 concurrency_timeout
    end

    G->>S: POST /chat/completions（该服务商的连接池与 Key）
    alt 上游超时 / 报错
        S--xG: Timeout / 4xx / 5xx
        G->>DB: 记录失败（provider、error_type、NULL tokens）
        G-->>C: 504 / 502
    else 正常返回
        S-->>G: 200 + usage
        G->>G: 提取 usage → 按该模型单价用 Decimal 估算成本
        G->>DB: 记录成功（provider、模型、tokens、cost、latency、标签）
        G-->>C: 200 原始响应 + X-Request-Id
    end

    G->>G: 释放并发槽位（finally，异常也会释放）
```

---

## 主要功能

### 1. LLM API 网关

- `POST /v1/chat/completions`，请求与响应体兼容 OpenAI Chat Completions 风格。
- 通过 `httpx.AsyncClient` **异步**访问上游；每个服务商一个客户端，在应用启动时创建、关闭时释放，全程复用连接池。
- 只支持**非流式**请求。传 `stream: true` 会被显式拒绝（`422`），而不是假装支持却返回整包。

### 2. 多服务商模型路由

- 支持同时接入 **DeepSeek** 与 **OpenAI**，客户端继续只调一个 `/v1/chat/completions`。
- 由 `model` 字段查**模型注册表**决定服务商：`MODEL_REGISTRY=deepseek-chat:deepseek,gpt-4o-mini:openai`。
- 每个服务商持有**独立的 Base URL、API Key 与 HTTP 连接池**，任一服务商的连接池耗尽不影响另一个。
- 客户端只需 ModelGate 的 Client API Key，**永远接触不到真实上游 Key**。
- 模型名只用于查表，**不参与 URL 与凭证构造**；上游地址只能来自 `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL`。
- **未登记的模型返回 `400 unknown_model`**，不会静默回落到默认服务商。
- 新增模型只需改配置（`MODEL_REGISTRY`），不需要改代码。
- 某个服务商未配置 Key 时，只有路由到它的请求返回 `503`，其他服务商不受影响。
- `GET /health/providers` 可查询已登记的模型与各服务商配置状态（不含任何凭证）。

### 3. 异步并发控制

- 用 `asyncio.Semaphore` 限制同一进程内同时执行的上游请求数，上限由 `MAX_CONCURRENT_REQUESTS` 配置。
- 等待槽位有独立超时（`CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS`），超时返回 `503 concurrency_timeout`。
- 槽位在 `finally` 中释放：业务异常、上游超时、客户端主动断开都不会泄漏信号量，避免死锁。
- ⚠️ **这不是分布式限流。** 它只在单个进程内生效；多副本部署时每个副本各有一份配额，实际总并发是 `副本数 × 上限`。

### 4. PostgreSQL 持久化

- SQLAlchemy 2.x 异步 ORM，一行一条调用记录。
- 记录字段：`request_id`、`provider`、请求模型与实际响应模型、请求时间、耗时、HTTP 状态、输入/输出 Token、估算成本、错误类型，以及密钥指纹、消息条数、Prompt 字符数、客户端 IP、Agent 追踪标签。
- **成功与失败请求都会记录**，失败原因归类到 `error_type`（见 `app/errors.py` 的 `ErrorType`）。
- 每个请求一个会话；写入失败**不影响**已经拿到的模型响应（只记 error 日志），避免因为审计库抖动让用户白等一次推理。
- 已有库通过增量迁移补列，历史数据保持不变（见[数据库结构变更](#数据库结构变更)）。

### 5. 分模型与分服务商成本统计

- 优先从上游响应的 `usage` 字段读取 `prompt_tokens` / `completion_tokens`。
- **上游没返回 `usage` 时，Token 记为 `NULL`（未知），绝不当成 0。** 这是刻意的选择：把未知记成 0 会让统计静默偏低，看起来像「免费」。
- 单价按服务商配置，支持用 `MODEL_PRICES` 覆盖单个模型；单位统一为**美元 / 每百万 Token**。这些是**估算口径，不是官方报价**，请按各服务商实际价格填写。
- **价格未配置时成本记为 `NULL`（未知），不伪造为 0**，因此「成本偏低」不会被误读成「很便宜」。
- 使用 `Decimal` 计算；金额以字符串返回，避免 JSON 把精确值退化成浮点。
- `GET /admin/usage` 支持按 **provider / model / 时间范围 / 应用名 / Agent 名 / Run ID** 筛选，返回请求数、成功/失败数、Token、估算成本与**平均耗时**，并可按模型和服务商分组。
- 响应附 `data_quality` 字段，告知有多少条记录因 usage 或价格缺失而未计入汇总。

### 6. Agent 调用追踪

- 支持三个可选请求头，用于把调用关联到多智能体工作流：`X-Client-App`、`X-Agent-Name`、`X-Run-Id`。
- 有长度上限（`AGENT_TAG_MAX_LENGTH`，默认 64）与字符集校验；不合法返回 `400`，**不静默截断**（截断会让不同 run 混成同一个标识）。
- 字段写入 `call_records`，可通过 `/admin/usage` 按应用名、Agent 名、Run ID 查询。
- ⚠️ 这些是**客户端自报的观测标签**，只用于统计与检索，**不作为身份认证或权限判断依据**。任何人能调用网关就能随意填写。

### 7. API 安全

- 客户端 Key（`/v1/*`）与管理 Key（`/admin/*`）**互相独立**。
- 密钥只从环境变量 / `.env` 读取，源码中不写死任何密钥。
- 比较密钥使用 `secrets.compare_digest`（常数时间），且对白名单全量比较，不因命中位置提前返回。
- 密钥字段使用 pydantic `SecretStr`，`repr()` 与日志中只会显示 `**********`。
- 日志与数据库都**不保存明文密钥**（只保存 SHA-256 前 12 位指纹）与**完整用户 Prompt**（只保存条数与字符数）。
- 上游错误原文可能回显我们发出的 Key 片段，因此**只脱敏后写日志**，返回给客户端的是固定描述（见 `app/redaction.py`）。
- 提供 [`.env.example`](.env.example)；真实 `.env` 已被 `.gitignore` 忽略。

### 8. Docker 部署

- 多阶段 `Dockerfile`：依赖在 builder 阶段安装，运行镜像不带编译工具链；以非 root 用户运行。
- `docker-compose.yml` 编排 API + PostgreSQL 16。
- 数据库有 `pg_isready` healthcheck，API 通过 `depends_on: condition: service_healthy` 等待数据库真正就绪。
- API 容器自带 `HEALTHCHECK`，打不依赖数据库的 `/health`。

### 9. 自动化测试

- pytest + pytest-asyncio，全部通过 `httpx.MockTransport` 模拟上游，**不依赖真实付费模型 API**。
- 覆盖：模型路由、服务商凭证隔离、未知模型拒绝、健康检查、鉴权、上游超时与各类异常、并发上限、Decimal 成本计算、持久化、用量统计与筛选、Agent 追踪、增量迁移。
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
# 模型注册表：只有登记在此的模型才会被转发
MODEL_REGISTRY=deepseek-chat:deepseek,deepseek-reasoner:deepseek,gpt-4o-mini:openai
DEFAULT_MODEL_NAME=deepseek-chat

# 服务商密钥（未配置的服务商对应的模型会返回 503）
DEEPSEEK_API_KEY=sk-你的真实密钥
OPENAI_API_KEY=

# 单价：留空表示未知，成本记为 NULL 而不是 0
DEEPSEEK_PRICE_INPUT_PER_MILLION_USD=0.27
DEEPSEEK_PRICE_OUTPUT_PER_MILLION_USD=1.10

CLIENT_API_KEYS=sk-client-local-dev-key
ADMIN_API_KEY=sk-admin-local-dev-key
DATABASE_URL=postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate
```

> 本机直接跑（不用 Docker）时，注意把 `DATABASE_URL` 里的主机名从 `db` 改成 `127.0.0.1`。
> 只想接一个服务商也可以：把 `MODEL_REGISTRY` 里不需要的模型删掉即可，
> 未配置 Key 的服务商不会影响其他服务商。

### 3. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后：

- 存活探针：<http://127.0.0.1:8000/health>
- 就绪探针：<http://127.0.0.1:8000/health/ready>
- 模型列表：<http://127.0.0.1:8000/health/providers>
- 交互式 API 文档：<http://127.0.0.1:8000/docs>

> 数据库连不上时服务**仍会启动**，但 `/health/ready` 返回 `503`。这样在容器编排里能一眼区分「应用没起来」和「数据库没好」，而不用让容器无限重启。

---

## Docker Compose 部署

### 1. 配置

```powershell
Copy-Item .env.example .env
# 至少填入 DEEPSEEK_API_KEY
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

**切换到另一个服务商**只需改 `model`，接口、鉴权与响应格式完全一致：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say OK"}]
  }'
```

未登记的模型会返回 400，并列出可用模型：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -d '{"model": "not-registered", "messages": [{"role": "user", "content": "hi"}]}'
# 400 {"error": {"message": "模型 'not-registered' 未在网关登记；可用模型：deepseek-chat, ...", "type": "unknown_model"}}
```

也可以用 `X-API-Key` 头代替 `Authorization`：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CLIENT_KEY" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

### 带 Agent 追踪标签调用

三个标签都是可选的，用于把调用关联到多智能体工作流：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -H "X-Client-App: insightflow" \
  -H "X-Agent-Name: planner" \
  -H "X-Run-Id: run-2026-01-01-abc" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}'
```

随后可按这些标签查询（例如只看某次工作流的开销）：

```bash
curl "http://127.0.0.1:8000/admin/usage?client_app=insightflow&run_id=run-2026-01-01-abc" \
  -H "X-Admin-Key: $ADMIN_KEY"
```

标签只允许字母、数字、点、下划线、连字符，最长 64（可配）。不合法会返回 400 而不是被截断。

### 查询用量统计

```bash
curl "http://127.0.0.1:8000/admin/usage?since=2026-01-01T00:00:00Z" \
  -H "X-Admin-Key: $ADMIN_KEY"
```

```json
{
  "range": {"since": "2026-01-01T00:00:00+00:00", "until": null},
  "filters": {"model": null, "provider": null, "client_app": null, "agent_name": null, "run_id": null},
  "totals": {
    "total_requests": 12,
    "success_count": 11,
    "failure_count": 1,
    "prompt_tokens": 1320,
    "completion_tokens": 880,
    "total_tokens": 2200,
    "estimated_cost_usd": "0.001232",
    "avg_latency_ms": 812,
    "price_unit": "USD per 1,000,000 tokens"
  },
  "data_quality": {
    "records_with_unknown_tokens": 1,
    "records_without_cost": 1,
    "note": "上游未返回 usage 或模型价格未配置时，Token 与成本记为未知，不会按 0 计入总量；上述两个字段用于提示统计的完整性"
  },
  "by_provider": [
    {"key": "deepseek", "total_requests": 9, "success_count": 8, "failure_count": 1,
     "prompt_tokens": 990, "completion_tokens": 660, "total_tokens": 1650,
     "estimated_cost_usd": "0.000924", "avg_latency_ms": 780},
    {"key": "openai", "total_requests": 3, "success_count": 3, "failure_count": 0,
     "prompt_tokens": 330, "completion_tokens": 220, "total_tokens": 550,
     "estimated_cost_usd": "0.000308", "avg_latency_ms": 910}
  ],
  "by_model": [
    {"key": "deepseek-chat", "total_requests": 9, "success_count": 8, "failure_count": 1,
     "prompt_tokens": 990, "completion_tokens": 660, "total_tokens": 1650,
     "estimated_cost_usd": "0.000924", "avg_latency_ms": 780}
  ],
  "errors": [{"error_type": "upstream_timeout", "count": 1}]
}
```

上例按 DeepSeek `0.20 / 0.80`、OpenAI `1.50 / 6.00`（美元/百万 Token）估算。
那一次失败调用没有 Token 数据，因此被 `records_with_unknown_tokens` 单独标出，而不是按 0 计入。
若某模型没配价格，它的成本会是 `null` 并计入 `records_without_cost` —— 不会伪装成 0。

可用查询参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `since` | 24 小时前 | ISO 8601，可不带时区（按 UTC 处理） |
| `until` | 无上限 | ISO 8601 |
| `model` | 全部 | 只看某个模型（同时匹配请求模型与实际响应模型） |
| `provider` | 全部 | 只看某个服务商，如 `deepseek` |
| `client_app` | 全部 | 按 `X-Client-App` 筛选 |
| `agent_name` | 全部 | 按 `X-Agent-Name` 筛选 |
| `run_id` | 全部 | 按 `X-Run-Id` 筛选 |
| `group_by_model` | `true` | 是否返回按模型拆分的明细 |
| `group_by_provider` | `true` | 是否返回按服务商拆分的明细 |
| `top_errors` | `5` | 返回前 N 个错误类型；设 `0` 可关闭 |

`estimated_cost_usd` 是**估算值**（基于配置的单价），不是账单金额。
`avg_latency_ms` 在区间内没有记录时为 `null`（无样本），不是 0。
`records_with_retries` 表示有多少条记录发生过重试，其成本只算最后一次尝试（见[配额与可靠性](#配额与可靠性)）。

### 查询一次 Agent 工作流

按 `run_id` 聚合，用于回答「这次任务用了多少次调用、花了多少、各节点分布如何」：

```bash
curl "http://127.0.0.1:8000/admin/runs/run-2026-01-01-abc" \
  -H "X-Admin-Key: $ADMIN_KEY"
```

```json
{
  "run_id": "run-2026-01-01-abc",
  "client_scope": "all_clients",
  "isolation_note": "run_id 由调用方自报。未指定 client 指纹时，不同客户端的同名 run 会被合并统计",
  "totals": {
    "total_requests": 5,
    "success_count": 5,
    "failure_count": 0,
    "prompt_tokens": 60,
    "completion_tokens": 40,
    "total_tokens": 100,
    "estimated_cost_usd": "0.000045",
    "avg_latency_ms": 57
  },
  "data_quality": {
    "records_with_unknown_tokens": 0,
    "records_without_cost": 0,
    "records_with_retries": 0
  },
  "by_agent": [
    {"key": "executor", "total_requests": 3, "success_count": 3, "failure_count": 0,
     "prompt_tokens": 36, "completion_tokens": 24, "total_tokens": 60,
     "estimated_cost_usd": "0.000027", "avg_latency_ms": 56},
    {"key": "critic", "total_requests": 1, "success_count": 1, "failure_count": 0,
     "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
     "estimated_cost_usd": "0.000009", "avg_latency_ms": 60},
    {"key": "planner", "total_requests": 1, "success_count": 1, "failure_count": 0,
     "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
     "estimated_cost_usd": "0.000009", "avg_latency_ms": 59}
  ],
  "by_provider": [{"key": "deepseek", "total_requests": 5, "estimated_cost_usd": "0.000045", "...": "..."}],
  "by_model": [{"key": "deepseek-chat", "total_requests": 5, "estimated_cost_usd": "0.000045", "...": "..."}],
  "errors": []
}
```

**同名 run 的隔离**：`run_id` 是客户端自报的标签，不同客户端可能用同一个值。
要只看某个客户端的调用，带上该客户端的 Key 指纹：

```bash
# 指纹可从 /admin/budget 的 entries 里看到（形如 ***7aac169c，取后 8 位无法直接反推完整指纹，
# 因此这里建议改为在服务端按 client_app 过滤，或直接查该客户端的 run 列表）
curl "http://127.0.0.1:8000/admin/runs?client_app=insightflow" \
  -H "X-Admin-Key: $ADMIN_KEY"
```

`GET /admin/runs` 返回工作流列表，支持 `limit` / `offset` 分页、`since` / `until` 时间范围
与 `client_app` 过滤，按最近活动时间倒序。

### 查询配额用量

```bash
curl "http://127.0.0.1:8000/admin/budget" -H "X-Admin-Key: $ADMIN_KEY"
```

```json
{
  "client_window_date": "2026-01-01",
  "reset_policy": {
    "client": "每个 UTC 自然日 00:00 重置",
    "run": "不重置，覆盖整个 Run 生命周期"
  },
  "limits": {"client_daily_request_quota": 1000, "run_max_llm_calls": 20},
  "note": "配额仅限制请求次数，不包含 Token 或金额预算；0 或 null 表示该维度未启用限制",
  "entries": [
    {"scope": "client", "identity": "***7aac169c", "window_kind": "daily",
     "window_start": "2026-01-01", "used": 42, "limit": 1000, "remaining": 958},
    {"scope": "run", "identity": "run-2026-01-01-abc", "window_kind": "run_total",
     "window_start": null, "used": 5, "limit": 20, "remaining": 15}
  ]
}
```

client 维度的 `identity` 是 API Key 指纹，只展示后 8 位；run 维度只展示客户端自己提供的 `run_id`。

### 错误响应格式

所有网关层错误统一为：

```json
{
  "error": {
    "message": "上游模型服务返回 401",
    "type": "upstream_auth_error",
    "code": 502
  }
}
```

上游错误只返回固定描述，不拼接上游原文 —— 因为上游可能在错误信息里回显我们发出的 Key 片段。

| 状态码 | `type` | 含义 |
|---|---|---|
| 400 | `unknown_model` | `model` 未登记在 `MODEL_REGISTRY` |
| 400 | `invalid_tracking_header` | `X-Client-App` / `X-Agent-Name` / `X-Run-Id` 不合法 |
| 401 | — | 未提供 API Key |
| 403 | — | API Key 不正确 |
| 422 | — | 请求体不合法（如 `stream: true`、`messages` 为空） |
| 429 | `quota_exceeded` | 超出**本网关**配额，未调用上游；带 `scope` / `limit` / `used` |
| 502 | `upstream_http_error` | 上游返回其他 4xx/5xx |
| 502 | `upstream_rate_limited` | 上游返回 429；有 `Retry-After` 时一并透传 |
| 502 | `upstream_auth_error` | 上游拒绝该服务商的密钥（401/403） |
| 502 | `upstream_connect_error` | 连不上上游 |
| 502 | `upstream_invalid_response` | 上游响应不是合法 JSON 或缺少 `choices` |
| 503 | `concurrency_timeout` | 等待并发槽位超时 |
| 503 | `upstream_not_configured` | 该服务商未配置 API Key |
| 504 | `upstream_timeout` | 上游响应超时 |

错误同时带**归属**（内部统计与日志用）：`client` / `gateway` / `upstream`。
上游限流（`upstream_rate_limited`）归属 `upstream`，
网关配额拒绝（`quota_exceeded`）归属 `gateway` —— 两者都是 429 语义，但责任方不同。

---

## Agent 集成

`examples/` 下提供了一个可运行的演示，不是示意代码：

```powershell
# 用 Mock 上游 + 独立测试库跑通，不需要真实 API Key
$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
python -m examples.demo
```

它会启动 Mock 上游与一个指向测试库的网关，模拟 Planner → Executor ×3 → Critic
三个节点共享同一个 `run_id`，最后打印管理接口给出的聚合结果：

```
Run run-dbba202afe52 聚合结果（来自 GET /admin/runs/run-dbba202afe52）
  调用次数 : 5  (成功 5 / 失败 0)
  Token    : 输入 60 + 输出 40 = 100
  估算成本 : 0.000045 USD
  平均耗时 : 57 ms

  各 Agent 节点分布：
    executor     调用 3   Token 60     成本 0.000027     平均 56 ms
    critic       调用 1   Token 20     成本 0.000009     平均 60 ms
    planner      调用 1   Token 20     成本 0.000009     平均 59 ms
```

### 接入自己的 Agent 运行时

`examples/modelgate_client.py` 是完整的适配层，只做一件事：注入追踪标签。

```python
from examples.modelgate_client import ModelGateClient, ModelGateConfig

client = ModelGateClient(ModelGateConfig.from_env(), run_id="run-abc", client_app="myapp")
await client.complete(agent_name="planner", messages=[...], model="deepseek-chat")
summary = await client.run_summary()   # 取回本次 Run 的聚合
```

### 边界划分

| 由 ModelGate 负责 | 由上层 Agent 运行时负责 |
|---|---|
| 模型名 → 服务商路由 | Agent 循环与终止条件 |
| 上游凭证保管（Agent 侧永远拿不到上游 Key） | Checkpoint 与状态恢复 |
| 单进程并发保护 | 工具定义与执行 |
| 请求次数配额 | 上下文窗口管理、消息裁剪 |
| Token / 成本统计与 run 聚合 | 重试语义下的业务幂等（见已知限制） |
| 失败分类与结构化日志 | 多 Agent 编排策略 |

ModelGate **不实现** Agent 循环、检查点、工具执行或上下文管理 —— 那些属于上层运行时。
网关只提供「统一出入口 + 治理 + 观测」。

---

## 运行测试

测试全部使用 Mock 上游，**不会调用真实付费 API**。
需要数据库的测试只使用 `TEST_DATABASE_URL` 指向的独立测试库（守卫强制库名以 `_test` 结尾）。

### 安装测试依赖

```powershell
pip install -r requirements-dev.txt
```

### 运行全部测试

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

默认结果：**370 passed, 19 skipped**（共收集 389 个用例）。

跳过项都需要外部环境：9 个 PostgreSQL 集成测试与 10 个配额并发测试
缺 `TEST_DATABASE_URL`。Docker 集成测试会自动拉起独立测试栈，
无需预先启动业务容器。

**如果 Docker + 专用测试库就绪**，先创建测试库（库名必须以 `_test` 结尾，见下一节）：

```powershell
docker compose up -d --build
docker compose exec db psql -U modelgate -d postgres -c "CREATE DATABASE modelgate_test;"

$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
.\.venv\Scripts\python.exe -m pytest -q
# -> 389 passed
```

| 运行条件 | 结果 |
|---|---|
| 仅单元测试（无 Docker、无测试库） | `370 passed, 19 skipped` |
| 加上专用测试库（真实 PostgreSQL） | `389 passed` |

> 配额并发测试必须跑在真实 PostgreSQL 上才有意义：内存 SQLite 的 `StaticPool`
> 只有一条连接，那样的「并发」其实是串行，测不出行级竞争。

### Docker 集成测试用独立测试栈

`tests/test_docker_integration.py` **自己拉起一个完全隔离的 compose 栈**
（[docker-compose.test.yml](docker-compose.test.yml)），不复用任何正在运行的业务容器：

| 隔离点 | 业务栈 | 测试栈 |
|---|---|---|
| compose 项目名 | `modelgate-original` | `modelgate-docker-test` |
| 容器名 | `modelgate-api` / `modelgate-db` | `modelgate-test-api` / `modelgate-test-db` |
| 端口 | 8000 / 5432 | **18000 / 15432** |
| 数据库 | `modelgate` | `modelgate_docker_test` |
| 数据卷 | `modelgate_pgdata` | `modelgate-docker-test-pgdata`（测试后删除） |
| 上游 | 真实服务商 | `http://upstream.invalid`（RFC 6761 保留域名） |

关键设计：

- **不看宿主机的环境变量判断打到了哪个库**，而是请求被测容器的 `/health/ready`，
  由它报告 `current_database()`。库名不是测试库就直接失败。
- 上游指向 `.invalid` 保留域名，DNS 必然解析失败：**不会访问真实模型服务，
  也不产生付费调用**，且失败立即且确定（不像不可路由地址要等到超时）。
  有专门用例断言配置与行为两个层面。
- 测试栈起不来时**明确失败并打印容器日志**，绝不退回业务环境。
- 模块结束后 `down -v` 拆除容器并删除测试数据卷。
- 未安装 Docker 时整体跳过，`pytest` 仍然全绿。

因此在配置了真实 API Key 的机器上跑完整套件，**业务库不会被写入任何记录**。

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
| **多服务商模型路由与凭证隔离** | `tests/test_provider_routing.py` |
| **Agent 追踪字段记录与查询** | `tests/test_agent_tracking.py` |
| **数据库增量迁移（历史数据保留）** | `tests/test_migrations.py` |
| API Key 鉴权成功与失败 | `tests/test_auth.py` |
| 上游超时与异常处理 | `tests/test_upstream_errors.py` |
| 并发上限是否生效 | `tests/test_concurrency.py` |
| Token 和 Decimal 成本计算 | `tests/test_cost.py` |
| 上游错误脱敏（不泄露密钥） | `tests/test_upstream_secret_leak.py` |
| 测试库防误删保护 | `tests/test_db_safety.py` |
| 用量统计接口与筛选 | `tests/test_admin_usage.py` |
| **配额治理与并发竞争** | `tests/test_budget.py` + `tests/test_budget_concurrency_pg.py`（真实 PostgreSQL） |
| **重试策略与故障注入** | `tests/test_reliability.py` |
| **Run 级聚合与隔离** | `tests/test_run_tracking.py` |
| **配额账本与统计一致性** | `tests/test_data_consistency.py` |
| PostgreSQL 调用记录持久化 | `tests/test_persistence.py`（SQLite）+ `tests/test_postgres_integration.py`（真实 PostgreSQL） |
| 容器集成 | `tests/test_docker_integration.py` |

---

## 配额与可靠性

### 配额

两类配额，单位都是**请求次数**：

| 维度 | 身份 | 窗口 | 配置 |
|---|---|---|---|
| client | 已认证 API Key 的指纹 | UTC 自然日，每日 00:00 重置 | `CLIENT_DAILY_REQUEST_QUOTA` |
| run | API Key 指纹 + `X-Run-Id` | **整个 Run 生命周期，不重置** | `RUN_MAX_LLM_CALLS` |

身份只来自服务端可验证的信息：client 维度用鉴权通过的 Key 指纹，
**不使用客户端自填的 `X-Client-App`**。run 维度额外要求 `run_id`，
它是标签、不参与鉴权，但和指纹组合后可以隔离不同客户端的同名 run。

**并发安全**依赖数据库而非应用层判断。扣减是单条语句：

```sql
INSERT INTO budget_usage (scope, identity, window_kind, window_start, used)
VALUES (:scope, :identity, :kind, :window, 1)
ON CONFLICT (scope, identity, window_kind, window_start)
DO UPDATE SET used = budget_usage.used + 1
WHERE budget_usage.used < :limit
RETURNING used
```

超限时 `DO UPDATE` 的 `WHERE` 不成立 → 不返回行 → 判定为超限。
行锁由数据库保证，N 个并发请求不可能同时通过检查。

**事务边界**：预扣在独立 session 中完成并**立即提交**，
之后才发起上游 HTTP 请求。因此持有配额行锁的时间与上游耗时无关，
同一配额下的请求不会串行阻塞。若 run 维度超限，本次已扣的 client 额度会整体回滚，
避免出现「run 被拒但 client 额度已消耗」的错账。

**哪些请求不消耗配额**：认证失败（401/403）、请求体校验失败（422）、
未知模型（400）、追踪标签非法（400）、服务商未配置（503）—— 这些都发生在预扣之前。

**失败的上游调用会消耗配额**：配额是预扣的，请求确实发生并占用了上游资源。
事后补偿会让「预留」退化成「事后统计」，失去并发保护。

### 配额不是费用预算

本模块**只统计请求次数**，没有实现金额预留与结算。
`CLIENT_DAILY_REQUEST_QUOTA=1000` 意味着「一天最多 1000 次调用」，
**不意味着**花费上限 —— 单次调用的成本可能相差几个数量级。
要控制费用，请结合 `/admin/usage` 的 `estimated_cost_usd` 与告警，而不是把配额当预算。

### 重试

**默认不重试**（`RETRY_MAX_ATTEMPTS=1`）。LLM 请求不是幂等的：
读取超时意味着请求已经发出，上游可能已在推理、甚至已经计费。

| 错误 | 默认 | 说明 |
|---|---|---|
| `upstream_connect_error` | 可重试（开启后） | 请求确定未送达，重试无重复推理风险 |
| `upstream_timeout` | **不重试** | 需 `RETRY_ON_READ_TIMEOUT=true` 显式开启 |
| `upstream_rate_limited` / `upstream_http_error` | **不重试** | 需 `RETRY_ON_UPSTREAM_ERROR=true` 显式开启 |
| `upstream_auth_error` 等 | **永不重试** | 重试没有意义 |

**不做自动故障切换。** 同一模型名在不同服务商上的行为、价格、输出格式都不同，
静默切换会让调用方拿到意料之外的模型输出。

### 上游限流的识别

上游返回 429 归类为 `upstream_rate_limited`，与本网关自己的配额拒绝
`quota_exceeded`（HTTP 429）明确区分 —— 两者都是「限流」，但责任方完全不同。
上游给出合法的整数 `Retry-After` 时会透传给调用方（响应头 + 错误体
`retry_after_seconds`），**网关自己不会据此自动等待或重试**，避免替调用方长时间阻塞。

### 重试与成本核算的边界

一次**逻辑请求**可能包含多次**上游尝试**：

- `call_records` 一行 = 一次逻辑请求
- `retry_count` = 额外发起的上游请求次数（默认配置下恒为 0）
- 上游实际尝试次数 = `retry_count + 1`

发生过重试时，记录里保存的 `usage` **只来自最后一次尝试**。
中间尝试是否已在上一级计费，网关无从得知，因此：

> `/admin/usage` 与 `/admin/runs/{id}` 的 `estimated_cost_usd`
> 对含重试的记录来说是**下限**，不是完整账单。

为此统计接口返回 `records_with_retries`，让调用方知道有多少条记录存在这个缺口。
本项目不为此引入更复杂的计费系统。

---

## 性能基准

完整报告见 [benchmarks/README.md](benchmarks/README.md)。要点：

```powershell
docker compose exec db psql -U modelgate -d postgres -c "CREATE DATABASE modelgate_test;"
$env:TEST_DATABASE_URL="postgresql+asyncpg://modelgate:modelgate@127.0.0.1:5432/modelgate_test"
python -m benchmarks.acceptance --requests 1000 --concurrency 10 20 50 --repeats 3 --label acceptance
```

实测（真实 PostgreSQL，Mock 上游注入 30ms，每档 1000 请求 × 3 轮，18,000 次请求 0 错误）：

| 场景 | 并发 | 中位 RPS | 各轮 RPS | p50 | p99 | 错误率 |
|---|---|---|---|---|---|---|
| 配额关 | 10 | 66.3 | 60.1 / 66.3 / 69.6 | 162 ms | 239 ms | 0% |
| 配额关 | 20 | 63.0 | 61.8 / 63.0 / 66.0 | 316 ms | 457 ms | 0% |
| 配额关 | 50 | 64.1 | 62.7 / 64.1 / 65.5 | 739 ms | 1054 ms | 0% |
| 配额开 | 10 | 50.7 | 49.9 / 50.7 / 53.0 | 187 ms | 270 ms | 0% |
| 配额开 | 20 | 43.9 | 41.4 / 43.9 / 50.1 | 453 ms | 1062 ms | 0% |
| 配额开 | 50 | 48.4 | 44.3 / 48.4 / 49.5 | 712 ms | 3680 ms | 0% |

**分项归因**：Mock 上游直连 253–296 RPS；网关鉴权拒绝（不碰数据库）~250 RPS；
网关完整链路 63–66 RPS；再加配额预扣降到 44–51 RPS。
吞吐在 c=10 就已饱和，说明瓶颈是串行化环节而非并发度不足。

**配额开销**：约 −16 RPS（−23% ~ −30%），三种并发下比例接近，
符合「每请求多一次短事务 + 一次额外连接取用」的定性。

### 事务边界是实测过的

用连接池事件观测一次请求（`benchmarks/probe_connection_scope.py`）：

```
配额预扣    持有连接  18.7 ms -> 归还
上游调用    300 ms，期间无任何数据库连接被持有
调用记录    持有连接  23.5 ms -> 归还
```

预扣事务在发起上游请求前就已提交并归还连接，**行锁不跨上游调用持有** ——
这是设计意图，也是实测结果。

### 测量限制（重要）

压测客户端、网关、Mock 上游、PostgreSQL 全部跑在同一台同时运行 IDE 与浏览器的机器上。
机器饱和时会观察到 502。**已排除**以下原因（均有直接证据）：
Mock 上游主动返回 502、PostgreSQL 连接耗尽、配额行锁争用、
SQLAlchemy 连接池耗尽、上游连接池排队（那会是 504）。

**高度怀疑**是同机压测环境与连接资源竞争引起：Windows IPv4 动态端口范围只有
16,384 个（49152–65535），高负载时实测 TIME_WAIT 峰值达 23,801，
失败样本中约八成是**无 body、无 Server 头的裸 502**（不是 ModelGate 生成的响应），
同时压测客户端出现大量 `ConnectError`。**但这尚需在独立环境验证，
本轮不能断言为已证实的唯一根因**（未定位裸 502 的产生者，
也未在失败瞬间同步采样端口计数）。

因此**没有为了压测通过而调大任何超时或连接池参数**。完整证据链与未闭合的缺口见
[benchmarks/README.md](benchmarks/README.md)。

**这些数字不代表 DeepSeek / OpenAI 真实服务的表现**，测的是网关自身处理与持久化开销。
**也不能作为生产容量依据**：单机 Mock 环境、四个进程同机争抢 CPU。
本轮没有做任何针对性优化，因此没有可宣称的优化收益。

### 已知瓶颈（本轮未优化）

1. 每请求一次 PostgreSQL 写入是主要开销：完整链路（63–66 RPS）远低于不碰数据库的鉴权路径（~250 RPS）。
2. 配额预扣每请求多一个短事务与一次额外连接取用，同批次对照中吞吐下降约 23%–30%。
3. `upstream.py` 的 httpx 客户端未显式设置 `max_connections`，使用 httpx 默认值 100；
   把 `MAX_CONCURRENT_REQUESTS` 调到接近该值时需同步调整，否则会转为 `PoolTimeout`（504）。
   这是基于配置关系的推理，不是本轮实测结论。

本轮**没有**做针对性优化：测量环境噪声过大，任何「优化收益」都无法可靠归因。
先把测量方法做对，再谈优化。

---

## 项目结构

```
modelgate-original/
├── app/
│   ├── main.py                 # 应用入口：lifespan 管理共享资源、注册路由与异常处理
│   ├── config.py               # pydantic-settings 配置，密钥用 SecretStr
│   ├── providers.py            # 模型注册表与服务商路由（模型名 -> 服务商 + 价格）
│   ├── provider_pool.py        # 各服务商的 httpx 客户端装配与生命周期
│   ├── dependencies.py         # 依赖注入：从 app.state 取单例资源
│   ├── security.py             # 鉴权、密钥指纹、Prompt 摘要
│   ├── agent_tracking.py       # X-Client-App / X-Agent-Name / X-Run-Id 解析与校验
│   ├── budget.py               # 配额治理：client / run 两维度，短事务预扣
│   ├── retry.py                # 重试策略：默认关闭，错误按安全等级分类
│   ├── errors.py               # ErrorType / ErrorOrigin 与统一错误响应
│   ├── request_context.py      # ContextVar 传递 request_id 给异常处理器
│   ├── schemas.py              # 请求/响应 Pydantic 模型
│   ├── upstream.py             # 上游调用与错误翻译（连接池按服务商隔离）
│   ├── concurrency.py          # asyncio.Semaphore 并发限制器
│   ├── cost.py                 # Decimal 成本计算
│   ├── redaction.py            # 密钥脱敏（上游错误文本落日志前必须过这里）
│   ├── logging_config.py       # 日志配置与结构化字段
│   ├── api/
│   │   └── routes/
│   │       ├── health.py       # /health、/health/ready、/health/providers
│   │       ├── chat.py         # /v1/chat/completions
│   │       └── admin.py        # /admin/usage、/admin/runs、/admin/budget
│   └── db/
│       ├── models.py           # CallRecord 与 BudgetUsage ORM 模型
│       ├── session.py          # 异步引擎与会话工厂
│       ├── migrations.py       # 增量结构迁移（建表 / 加列 / 加索引）
│       ├── budget_repository.py # 配额的原子扣减
│       └── repository.py       # 数据访问与聚合查询
├── benchmarks/
│   ├── mock_provider.py        # 可控延迟与故障率的 Mock 上游
│   ├── load.py                 # 并发压测与百分位计算（含失败分层采样）
│   ├── acceptance.py           # 主基准：同批次配额对照，记录每轮明细
│   ├── probe_connection_scope.py # 观测请求期间是否持有数据库连接
│   ├── diagnose.py             # 分段诊断，定位瓶颈
│   ├── run_benchmark.py        # 注入故障场景与落库完整率
│   └── results/                # 原始 JSON 结果
├── examples/
│   ├── modelgate_client.py     # 最小适配层（注入追踪标签）
│   ├── run_workflow.py         # Planner/Executor/Critic 共享 run_id 演示
│   └── demo.py                 # 一键跑通演示（Mock 上游 + 独立测试库）
├── tests/
│   ├── conftest.py                          # fixtures：双服务商 Mock 上游、内存/文件数据库
│   ├── db_safety.py                         # 测试库防误删守卫（库名必须 _test 结尾）
│   ├── docker/init-test-db.sql              # 测试栈的 PostgreSQL 初始化脚本
│   ├── test_provider_routing.py             # 模型路由、凭证隔离、未知模型
│   ├── test_agent_tracking.py               # 追踪标签校验、入库与筛选
│   ├── test_budget.py                       # 配额原子性、窗口语义、429 路径
│   ├── test_budget_concurrency_pg.py        # 真实 PostgreSQL 并发竞争验证
│   ├── test_reliability.py                  # 重试策略与故障注入矩阵
│   ├── test_run_tracking.py                 # Run 级聚合、隔离与分页
│   ├── test_data_consistency.py             # 配额账本、未知值、统计自洽
│   ├── test_migrations.py                   # 增量迁移与历史数据保留
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
│   └── test_docker_integration.py           # 自建隔离测试栈，不接触业务环境
├── Dockerfile
├── docker-compose.yml
├── docker-compose.test.yml                  # Docker 集成测试专用栈（独立项目名/端口/库）
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

---

## 配置项说明

### 应用与鉴权

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `APP_NAME` | `ModelGate` | 应用名 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `CLIENT_API_KEYS` | 空 | 客户端密钥，多个用英文逗号分隔 |
| `ADMIN_API_KEY` | 空 | 管理密钥，须与客户端密钥不同 |
| `AGENT_TAG_MAX_LENGTH` | `64` | 追踪标签的长度上限 |

### 模型路由

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_REGISTRY` | `deepseek-chat:deepseek,deepseek-reasoner:deepseek` | 模型注册表，格式 `模型名:服务商` |
| `DEFAULT_MODEL_NAME` | `deepseek-chat` | 未指定 `model` 时使用，**必须已登记** |
| `MODEL_PRICES` | 空 | 单模型价格覆盖，格式 `模型名:输入价:输出价` |

### 服务商

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek 地址 |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek 密钥；留空则该服务商返回 503 |
| `DEEPSEEK_PRICE_INPUT_PER_MILLION_USD` | 空 | DeepSeek 输入单价（美元/百万 Token） |
| `DEEPSEEK_PRICE_OUTPUT_PER_MILLION_USD` | 空 | DeepSeek 输出单价 |
| `OPENAI_BASE_URL` | `https://api.openai.com` | OpenAI 地址 |
| `OPENAI_API_KEY` | 空 | OpenAI 密钥；留空则该服务商返回 503 |
| `OPENAI_PRICE_INPUT_PER_MILLION_USD` | 空 | OpenAI 输入单价 |
| `OPENAI_PRICE_OUTPUT_PER_MILLION_USD` | 空 | OpenAI 输出单价 |

> 价格留空或填 `0` 表示**未知**，成本会记为 `NULL` 而不是 0。
> 上表默认值不含任何价格——必须由你按服务商实际价格填写。

### 上游与并发

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `UPSTREAM_TIMEOUT_SECONDS` | `60` | 单次上游请求超时（秒） |
| `UPSTREAM_MAX_CONNECTIONS` | `100` | 每个服务商的连接池上限 |
| `UPSTREAM_MAX_KEEPALIVE_CONNECTIONS` | `20` | 保持的连接数 |
| `MAX_CONCURRENT_REQUESTS` | `8` | 单进程并发上限 |
| `CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS` | `10` | 等待并发槽位的上限（秒） |

### 配额

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `CLIENT_DAILY_REQUEST_QUOTA` | `0` | 每个已认证客户端的每日调用次数上限；`0` 表示不限制 |
| `RUN_MAX_LLM_CALLS` | `0` | 单个 Run 的总调用次数上限，**不随日期重置**；`0` 表示不限制 |

> 只限制请求次数，不是 Token 或金额预算。详见[配额与可靠性](#配额与可靠性)。

### 重试

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `RETRY_MAX_ATTEMPTS` | `1` | `1` 表示不重试 |
| `RETRY_BACKOFF_BASE_SECONDS` | `0.5` | 第 n 次重试等待 `base × 2^(n-1)` |
| `RETRY_BACKOFF_MAX_SECONDS` | `8` | 单次重试等待上限 |
| `RETRY_ON_READ_TIMEOUT` | `false` | 是否重试读取超时（有重复计费风险） |
| `RETRY_ON_UPSTREAM_ERROR` | `false` | 是否重试上游 429 / 5xx |

### 数据库

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://modelgate:modelgate@db:5432/modelgate` | 数据库连接串 |
| `DB_POOL_SIZE` | `10` | 连接池大小（SQLite 下自动忽略） |
| `DB_MAX_OVERFLOW` | `20` | 连接池溢出上限 |
| `DB_ECHO` | `false` | 是否打印 SQL（调试用） |

> 开启配额后每个请求会额外占用一个短事务连接，
> 并发较高时需相应调整 `DB_POOL_SIZE`、`DB_MAX_OVERFLOW` 与 `MAX_CONCURRENT_REQUESTS`。
> 具体表现见 [benchmarks/README.md](benchmarks/README.md) 的瓶颈说明。

### 从单服务商配置迁移

旧版本使用单个 `UPSTREAM_*` 与全局 `PRICE_*`，对应的新写法：

| 旧变量 | 新变量 |
|---|---|
| `UPSTREAM_BASE_URL` | `DEEPSEEK_BASE_URL` |
| `UPSTREAM_API_KEY` | `DEEPSEEK_API_KEY` |
| `UPSTREAM_MODEL` | `MODEL_REGISTRY` + `DEFAULT_MODEL_NAME` |
| `PRICE_INPUT_PER_MILLION_USD` | `DEEPSEEK_PRICE_INPUT_PER_MILLION_USD` |
| `PRICE_OUTPUT_PER_MILLION_USD` | `DEEPSEEK_PRICE_OUTPUT_PER_MILLION_USD` |

新增的 `MODEL_REGISTRY` 必须包含 `DEFAULT_MODEL_NAME`，否则启动时直接报错
（避免默认模型未登记导致所有请求 400）。旧变量不再被读取。

---

## 数据库结构变更

本次升级为 `call_records` 增加了 7 个可空列，并新增 `budget_usage` 表与 5 个索引。

`call_records` 新列：`provider`、`requested_model`、`response_model`、`retry_count`、
`client_app`、`agent_name`、`run_id`。

索引：`provider`、`client_app`、`agent_name`、`run_id`，
以及 Run 级查询用的复合索引 `(key_fingerprint, run_id, created_at)`。

新增表 `budget_usage`：配额计数，含唯一约束
`(scope, identity, window_kind, window_start)` —— 没有它 `ON CONFLICT` 的原子扣减会失效。

迁移由 [app/db/migrations.py](app/db/migrations.py) 在应用启动时自动执行：

- **只加列、加索引、建缺失的表**，先查现有结构或使用 `IF NOT EXISTS`，
  可重复执行、**并发启动也安全**（多个副本同时启动不会互相冲突）。
- 新列全部 `NULLABLE`，历史行保持原样（新列为 `NULL`），**不删除、不清空任何数据**。
- 不含任何 `DROP` / `TRUNCATE` / `DELETE` / 类型收窄。
- 全部语句在**单个事务**内执行，任一步失败则整体回滚，不会留下半成品结构。
- 启动日志会打印本次实际变更，例如 `结构迁移完成，新增：call_records.provider, ...`。

迁移可被单独调用（脚本、运维手工执行），不依赖 `create_all` 先建表。

### 验证迁移

```powershell
docker exec modelgate-db psql -U modelgate -d modelgate -c "\d call_records"
docker exec modelgate-db psql -U modelgate -d modelgate -c "\d budget_usage"
docker exec modelgate-db psql -U modelgate -d modelgate -c "SELECT count(*) FROM call_records;"
```

### 回滚

新增的表与列对旧代码没有影响（旧代码不查询它们），保留即可。若要撤销：

```sql
DROP TABLE budget_usage;
ALTER TABLE call_records DROP COLUMN provider;
ALTER TABLE call_records DROP COLUMN requested_model;
ALTER TABLE call_records DROP COLUMN response_model;
ALTER TABLE call_records DROP COLUMN retry_count;
ALTER TABLE call_records DROP COLUMN client_app;
ALTER TABLE call_records DROP COLUMN agent_name;
ALTER TABLE call_records DROP COLUMN run_id;
```

删除这些列不会影响其余字段的数据。**回滚前请先备份**，
并注意删除 `budget_usage` 会让所有 Run 与客户端的已用额度清零 ——
也就是这些 Run 会重新获得额度。若不希望如此，只回滚代码、保留该表。

### 配额数据的增长

`budget_usage` 每个「身份 + 窗口」一行：client 维度每天每客户端一行，
run 维度每个 run 一行且**永不过期**（因为 Run 配额不重置）。

长期运行后这张表会持续增长。当前**没有**实现清理机制，
若需要，必须满足两个条件：

1. **只能删除已完成的 Run。** 直接按时间删除会让仍在使用中的 Run 重新获得额度。
   判定「已完成」需要上层运行时上报，网关自身无法知道。
2. **不得影响 `call_records`。** 配额行与调用记录是两套数据，
   清理配额不应级联删除历史调用记录。

一个安全的近似做法：只删除 `updated_at` 早于保留期（如 30 天）**且**该 run 在
`call_records` 中已无最近活动的行。这仍不完美，但不会误伤活跃 Run。

---

## 设计取舍


**1. Token 未知用 `NULL` 而不是 0。**
统计时 `SUM()` 会跳过 `NULL`，所以「Token 总量」只累加已知值，可能偏小。为了不让偏小被误读成准确账单，接口额外返回 `records_with_unknown_tokens` 与 `records_without_cost`。反过来，如果记成 0，使用者根本无从发现数据不完整。

**2. 落库失败不影响响应。**
模型推理已经完成、钱已经花了，此时因为审计库抖动而给客户端返回 500 是得不偿失的。所以记录写入失败只记 `error` 日志。代价是审计可能缺失 —— 这是一个明确的取舍，不是遗漏。

**3. 用 `create_all` + 自建增量迁移，而不是 Alembic。**
项目只有一张表，引入迁移框架会增加上手成本。`create_all` 负责建新表，[app/db/migrations.py](app/db/migrations.py) 负责给已存在的表补列，足够当前规模。局限见「已知限制」。

**4. 配置通过 `app.state` 而不是模块级全局变量。**
`create_app()` 接受可选的 `settings` 参数，测试可以注入自己的配置。最初的实现让鉴权依赖直接调用模块级缓存的 `get_settings()`，结果测试注入的配置被忽略（这个 bug 是被测试抓出来的）。现在依赖统一从 `app.state.settings` 读取，单一来源。

**5. 请求 ID 用 `ContextVar` 传递。**
异常处理器拿不到路由里的局部变量，失败响应也需要带 `X-Request-Id`（否则客户端无法用返回的 ID 对账数据库）。`ContextVar` 在 asyncio 中按任务隔离，并发请求之间不会串。

**6. 显式拒绝 `stream: true`。**
比「假装支持、实际返回整包」更诚实。

**7. 模型名只用于查表，不参与 URL 构造。**
上游地址只能来自配置里的 `*_BASE_URL`。这样即使调用方传入 `../../admin` 之类的字符串，也只会得到 400，不可能改变请求目标。

**8. 未登记的模型返回 400 而不是回落到默认服务商。**
静默回落会让调用方以为自己在用 A 模型、实际用了 B，成本与效果都对不上，比直接报错危险得多。

**9. 追踪标签非法就拒绝，不截断。**
截断会把两个不同的 `run_id` 变成同一个，统计出来的工作流开销就是错的。宁可让调用方修正输入。

**10. 价格未配置记 `NULL`，不记 0。**
价格缺失时把成本算成 0 会让人以为「这个模型免费」。记 `NULL` 并在 `data_quality` 里报出来，问题才会被发现。

**11. 配额用数据库行锁，而不是应用层判断。**
「先查用量再决定放行」在并发下必然超发：N 个请求会同时读到「未超限」。
因此把判断和扣减合并成单条 `INSERT ... ON CONFLICT DO UPDATE ... WHERE used < limit`，
由数据库串行化。代价是每请求多一次写往返 —— 这个代价在基准里是可测量的。

**12. 配额的锁不跨上游请求持有。**
预扣在独立 session 中提交后才发起上游调用。如果图省事把预扣和上游调用放在同一事务里，
同一配额下的所有请求会因为等待行锁而串行，吞吐直接塌掉。

**13. run 配额不随时间重置，client 配额按日重置。**
两者的语义本来就不同：「这次工作流最多调 20 次」和「这个客户端今天最多调 1000 次」。
所以 run 维度用固定哨兵日期做窗口键，而不是复用按日窗口 —— 否则一次跨零点的工作流会重新获得额度。

**14. 上游 429 与网关 429 分开归类。**
两者对外都是「限流」，但一个是上游的问题、一个是我们的策略。
混在一起会让「今天为什么这么多 429」无法回答。上游的 `Retry-After` 只透传，
网关不据此自动等待 —— 那会把调用方的等待时间绑在网关的请求生命周期上。

**15. 重试默认关闭，且按错误分安全等级。**
LLM 请求不是幂等的。读取超时后上游可能已经在推理并计费，自动重试会造成重复计费。
因此只有「连接建立失败」（确定未送达）才算安全，其余必须显式开启。

**16. 含重试的成本标记为下限，而不是估算补齐。**
中间尝试是否已计费我们无从得知。与其编一个数字，不如用 `records_with_retries`
把缺口暴露出来，让使用者知道这个统计不完整。

**17. 基准先修测量方法，不做无法归因的优化。**
第一次跑基准时 `quota_on` 比 `quota_off` 还快 —— 这说明机器负载噪声压过了被测差异。
在这种情况下宣称「优化了 X%」是没有意义的。因此本轮先补上重复取中位数、
分段诊断和噪声说明，把瓶颈定位清楚，优化留到能可靠测量之后。

**18. 不引入 Redis。**
配额需要跨请求、跨副本共享状态，最自然的方案是 Redis。
但 PostgreSQL 已经是必需依赖，用它的行锁就足以实现原子配额，
再加一个中间件只会增加部署与运维成本。只有当写入路径成为瓶颈时才值得重新评估 ——
基准里它确实是瓶颈之一，但那是「数量级」问题，不是「有没有」问题。

---

## 已知限制


1. **并发控制不是分布式限流。** `asyncio.Semaphore` 只在单进程内生效。多副本部署时总并发 = 副本数 × `MAX_CONCURRENT_REQUESTS`。真正的分布式并发控制需要共享计数器。
2. **配额是请求次数，不是费用预算。** 没有金额预留与结算机制，`CLIENT_DAILY_REQUEST_QUOTA` 不构成成本硬上限。单次调用成本可能相差几个数量级，要控费用必须结合 `estimated_cost_usd` 与告警。
3. **配额是固定窗口。** client 维度按 UTC 自然日，因此在零点边界可能出现短时间双倍额度。滑动窗口需要额外的时间桶与清理逻辑，本轮未做。
4. **含重试的调用成本是下限。** 一次逻辑请求发生多次上游尝试时，记录里的 `usage` 只来自最后一次尝试；中间尝试是否已计费网关无从得知。用 `records_with_retries` 暴露这个缺口，不做估算补偿。
5. **不支持流式（SSE）。** 只实现了非流式转发，`stream: true` 会被拒绝。流式代理需要处理增量解析、心跳、客户端中断等情况。
6. **默认不重试，也没有熔断。** 上游瞬时失败直接返回给客户端。开启重试后也只对连接建立失败是安全的，读取超时与 429/5xx 存在重复推理与重复计费风险。
7. **不做自动故障切换。** 同一模型名在不同服务商上的行为与价格不同，静默切换会让调用方拿到意料之外的输出。
8. **价格需手工维护。** 不自动同步官方价格，也不区分缓存命中、批处理折扣或阶梯定价。因此 `estimated_cost_usd` 只是**估算**，不是账单。
9. **模型注册表是静态的。** 新增模型要改配置并重启，没有从服务商 `/models` 接口动态拉取的能力。
10. **迁移只支持加列与建表。** 改列类型、加非空约束、数据回填都需要手工 SQL，也不具备版本化回滚能力。
11. **`budget_usage` 会持续增长。** 每个 Run 一行且不重置，当前没有清理机制；设计清理时需要保证不会让活跃 Run 重新获得额度（见[数据库结构变更](#数据库结构变更)）。
12. **没有请求体大小限制与 Prompt 注入防护。** 网关做透传，不审查内容。
13. **管理接口分页有限。** `/admin/runs` 支持 `limit`/`offset`；`/admin/usage` 返回聚合结果；调用明细（`list_recent`）只存在于仓储层，未暴露为 HTTP 接口。
14. **追踪标签未做可信化。** 它们是客户端自报的字符串，网关不做归属校验，因此不能用于计费归属或权限判断。`run_id` 只是标签，不同客户端可以复用同一个值 —— Run 级查询需要带上 `key_fingerprint` 才能正确隔离。
15. **无分布式追踪。** 有 `request_id` 与结构化日志，但没有接入 OpenTelemetry。
16. **Docker 镜像未做漏洞扫描**，也未固定基础镜像的 digest。
17. **客户端密钥是静态的**，没有签发/轮换机制。

---

## 未来改进方向

按「投入产出比」排序

1. **调优持久化路径。** 基准显示每请求一次写入是主要开销、配额预扣再叠加一个短事务（见 [benchmarks/README.md](benchmarks/README.md)）。可考虑批量写入、调整连接池与并发上限的配比，并用同一套基准验证收益。
2. **价格表与自动同步。** 把价格搬到独立的 `{model: {input, output}}` 结构，并支持区分缓存命中价。
3. **配额清理机制。** 按「已完成 Run + 保留期」两个条件删除 `budget_usage` 行，避免误伤活跃 Run。
4. **Token / 金额预算。** 在请求预扣前预留额度、响应后按实际 usage 结算。需要处理「实际用量超出预留」的补偿路径，比请求计数复杂得多。
5. **滑动窗口配额。** 用时间桶替代固定日期键，消除零点边界的双倍额度。
6. **故障转移。** 同一个模型可登记主备服务商，主上游连续失败时短暂切到备用 —— 前提是有足够的验证，不能默认开启。
7. **接入 Alembic。** 让表结构变更可版本化、可回滚，替代当前的加列式迁移。
8. **实现流式代理。** 用 `StreamingResponse` + `httpx.stream()` 转发 SSE；难点在于流式响应通常拿不到 `usage`，需要按字符估算或要求上游开启 `stream_options.include_usage`。
9. **调用明细查询接口。** 暴露 `GET /admin/calls`，可按 Agent 标签与 `run_id` 追查单次工作流。
10. **可观测性。** 接入 OpenTelemetry，把 `request_id` 与 `run_id` 串成 trace，导出 Prometheus 指标（QPS、延迟直方图、错误率、配额拒绝率、分服务商成功率）。
11. **CI 流水线。** GitHub Actions 跑 lint + pytest + Docker 构建，并加上 PostgreSQL service container 跑集成测试与基准回归。

---
