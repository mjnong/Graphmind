# Use Python 3.13 slim image for production
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev --prerelease=allow

# Production stage
FROM python:3.13-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    HOME="/home/appuser"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from base stage
COPY --from=base /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Copy alembic configuration and migration files
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Copy .env files to the app directory
COPY .env* ./

# Create necessary directories and set permissions
RUN mkdir -p /app/history && \
    mkdir -p /app/uploads && \
    mkdir -p /home/appuser/.mem0 && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base AS development

# Install development dependencies
RUN uv sync --frozen --prerelease=allow

# Create non-root user with home directory for development
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy application code
COPY . .

# Set permissions for development
RUN mkdir -p /app/uploads && \
    mkdir -p /home/appuser/.mem0 && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

# Switch to non-root user
USER appuser

# Default command for development
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]