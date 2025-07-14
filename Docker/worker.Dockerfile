# Dockerfile specifically for Celery Worker
# This builds a container optimized for file processing tasks

# Use Python 3.13 slim image as base
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for file processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    file \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev --prerelease=allow

# Production worker stage
FROM python:3.13-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    HOME="/home/appuser"

# Install runtime dependencies for file processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    file \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from base stage
COPY --from=base /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Copy alembic configuration and migration files (in case worker needs DB access)
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Copy .env files to the app directory
COPY .env* ./

# Create directories for file processing and set permissions
RUN mkdir -p /app/temp_processing && \
    mkdir -p /app/history && \
    mkdir -p /app/uploads && \
    mkdir -p /home/appuser/.mem0 && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Health check for worker (check if celery is responding)
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD celery -A src.app.celery.worker:app inspect ping --timeout=10 || exit 1

# Default command for production worker
CMD ["celery", "-A", "src.app.celery.worker:app", "worker", "--loglevel=info", "--concurrency=2", "--queues=celery,file_processing"]

# Development worker stage
FROM base AS development

# Install development dependencies
RUN uv sync --frozen --prerelease=allow

# Install additional dev tools for file processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    file \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory for development
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy application code
COPY . .

# Set permissions for development
RUN mkdir -p /app/temp_processing && \
    mkdir -p /app/uploads && \
    mkdir -p /home/appuser/.mem0 && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

# Switch to non-root user
USER appuser

# Default command for development worker (with auto-reload)
CMD ["celery", "-A", "src.app.celery.worker:app", "worker", "--loglevel=debug", "--concurrency=1", "--autoreload", "--queues=celery,file_processing"]
