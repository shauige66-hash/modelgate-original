# ModelGate 运行镜像
# 采用多阶段构建：builder 负责装依赖，运行镜像不带编译工具链，体积更小。
FROM python:3.13-slim AS builder

WORKDIR /build

# 只装运行依赖，不装开发依赖
COPY requirements.txt .

# --prefix 把依赖装到独立目录，方便整体拷贝到运行镜像
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------
FROM python:3.13-slim

# PYTHONUNBUFFERED: 让日志实时输出，docker logs 才能及时看到
# PYTHONDONTWRITEBYTECODE: 镜像内不需要 .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# curl 供 HEALTHCHECK 使用
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 拷贝 builder 阶段装好的依赖
COPY --from=builder /install /usr/local

# 拷贝应用代码
COPY app ./app

# 以非 root 用户运行，降低容器被攻破后的影响面
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 存活检查打 /health（不依赖数据库），避免数据库抖动导致容器被反复重启
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
