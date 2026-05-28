# syntax=docker/dockerfile:1.7

# Pin a slim base image: reproducible builds, small surface area.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.3.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# System deps: only what's needed for building wheels (libgomp1 for sklearn).
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

# Dependencies first (cache layer) — рекомпиляция образа при изменении только кода
# не будет переустанавливать пакеты.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-cache \
    && pip install --no-cache-dir pytest

# Project code
COPY src/ ./src/
COPY tests/ ./tests/

# Non-root user — снижает риск выхода из контейнера через уязвимости рантайма.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models /app/mlruns /mlflow \
    && chown -R appuser:appuser /app /mlflow
USER appuser

# По умолчанию — прогон тестов: образ сам себя валидирует при старте.
CMD ["pytest", "tests/", "-v"]
