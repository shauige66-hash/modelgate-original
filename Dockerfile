# ModelGate 运行镜像
# 多阶段构建：builder 安装依赖，运行镜像不带编译工具链。
FROM python:3.13-slim AS builder

WORKDIR /build

# 只安装运行依赖
COPY requirements.txt .

# --prefix 将依赖装到独立目录，便于整体拷贝
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# curl 供 HEALTHCHECK 使用
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app ./app

# 非 root 运行，降低容器被攻破后的影响面
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 打 /health（不依赖数据库），避免数据库抖动导致容器反复重启
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
