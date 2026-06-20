FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/backend/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/uv.lock backend/README.md /app/backend/
WORKDIR /app/backend
RUN uv sync --frozen --no-dev --no-install-project

COPY backend /app/backend
RUN uv sync --frozen --no-dev

COPY frontend /app/frontend

EXPOSE 8000 5173

CMD ["bash", "-c", "\
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
backend_pid=$!; \
python -m http.server 5173 --bind 0.0.0.0 --directory /app/frontend & \
frontend_pid=$!; \
trap 'kill $backend_pid $frontend_pid' INT TERM; \
wait -n $backend_pid $frontend_pid; \
status=$?; \
kill $backend_pid $frontend_pid 2>/dev/null; \
exit $status"]