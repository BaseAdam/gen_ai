# Build stage: uv installs the dependencies and the app into a virtual environment.
FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependencies first, so this layer stays cached until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the app itself, installed as a regular package (not editable).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Runtime stage: Python, the virtual environment and the data, without uv or the sources.
FROM python:3.14-slim
RUN useradd --uid 1000 --create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY offers.json ./
COPY scripts ./scripts
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# A numeric user id lets platforms such as Kubernetes check that it is not root.
USER 1000
EXPOSE 8000
CMD ["uvicorn", "offers.api:app", "--host", "0.0.0.0", "--port", "8000"]
