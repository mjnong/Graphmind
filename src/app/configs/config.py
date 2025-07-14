import functools
import sys
from enum import StrEnum

from dotenv import load_dotenv
from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings
from sqlalchemy import URL

load_dotenv(
    override=True,  # Override existing environment variables
)


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRACE = "trace"


class Config(BaseSettings):
    # General configuration
    local_path: str = "/app/uploads"  # Local storage path within the app directory
    local_base_url: str = "/files/"  # Default base URL for local files
    storage_backend: str = "local"  # Options: "local", "s3", "azure"
    s3_bucket: str = "graphrag"  # Default S3 bucket name
    
    # Azure Storage configuration
    azure_storage_account: str = ""  # Azure Storage Account name
    azure_storage_key: str = ""  # Azure Storage Account key
    azure_container_name: str = "uploads"  # Azure Blob container name
    azure_prefix: str = ""  # Optional prefix for blob names
    azure_public: bool = False  # Whether to use public URLs instead of SAS URLs
    
    # Security configuration
    max_file_size_mb: int = 100  # Maximum file size in MB
    max_files_per_upload: int = 10  # Maximum files per upload request
    rate_limit_per_minute: int = 60  # Maximum requests per minute per IP
    rate_limit_burst: int = 10  # Burst allowance for rate limiting
    
    # Database connection pooling configuration
    db_pool_size: int = 20  # Number of connections to maintain in pool
    db_max_overflow: int = 30  # Number of additional connections allowed
    db_pool_timeout: int = 30  # Timeout in seconds for getting connection
    db_pool_recycle: int = 3600  # Recycle connections after 1 hour
    db_pool_pre_ping: bool = True  # Validate connections before use
    
    # Environment configuration
    environment: str = "production"  # Options: "development", "production"
    run_migrations_on_startup: bool = True  # Whether to run migrations on startup
    # Environment configuration
    database_env: str = "local"  # Options: "local", "development", "production"
    
    # Worker scaling configuration
    celery_workers: int = 4  # Number of Celery worker processes
    celery_concurrency: int = 2  # Concurrent tasks per worker
    celery_prefetch_multiplier: int = 1  # Tasks prefetched per worker
    celery_max_tasks_per_child: int = 1000  # Restart worker after N tasks
    
    # Load balancing configuration
    app_workers: int = 1  # Number of FastAPI app instances
    worker_connections: int = 1000  # Max connections per worker
    
    # Redis configuration
    redis_url: str = "redis://broker:6379"  # Default Redis URL
    dragonfly_host: str = "broker"  # Default Dragonfly host
    dragonfly_port: int = 6379  # Default Dragonfly port
    celery_broker_url: str = "redis://broker:6379/0"  # Default Celery broker URL
    celery_result_backend: str = "redis://broker:6379/1"

    # PostgresDB configuration
    postgres_host: str = ""
    postgres_port: int = 5432
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_database: str = ""
    postgres_ssl_mode: str = "disable"  # Options: "disable", "prefer", "require"
    database_url: str = ""

    # Neo4j configuration
    neo4j_uri: str = "bolt://neo4j:7687"  # Full URI for Docker environment
    neo4j_username: str = "neo4j"
    neo4j_password: str = "mem0graph"  # Match the password from docker-compose

    # FastAPI configuration
    fastapi_host: str = "localhost"  # e.g., "localhost" or "0.0.0"
    fastapi_port: int = 8000  # Default
    graphrag_log_level: LogLevel = LogLevel.INFO  # Default log level for FastAPI

    # OpenAI configuration
    openai_api_key: str = "your_openai_api_key"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    @field_validator("graphrag_log_level", mode="before")
    @classmethod
    def validate_graphrag_log_level(cls, v) -> LogLevel:
        if isinstance(v, LogLevel):
            return v
        if isinstance(v, str):
            # Try to find the enum by value (case-insensitive)
            v_lower = v.lower()
            for level in LogLevel:
                if level.value.lower() == v_lower:
                    return level
            # If not found, raise an error with helpful message
            valid_levels = [level.value for level in LogLevel]
            raise ValueError(
                f"graphrag_log_level must be one of {valid_levels}, got '{v}'"
            )
        raise ValueError(
            f"graphrag_log_level must be a string or LogLevel enum, got {type(v)}"
        )
    
    @field_validator("database_url", mode="after")
    @classmethod
    def validate_database_url(cls, v, info) -> str:
        if not v:
            # Get the values from the validation context
            data = info.data
            
            # Build connection parameters
            connect_args = {}
            if data.get("postgres_ssl_mode", "prefer") == "require":
                connect_args["sslmode"] = "require"
            elif data.get("postgres_ssl_mode", "prefer") == "prefer":
                connect_args["sslmode"] = "prefer"
            
            url = URL.create(
                drivername="postgresql+psycopg2",
                username=data.get("postgres_user", ""),
                password=data.get("postgres_password", ""),
                host=data.get("postgres_host", ""),
                port=int(data.get("postgres_port", 5432)),
                database=data.get("postgres_database", ""),
                query=connect_args if connect_args else {}
            )
            return url.render_as_string(hide_password=False)
        return v


    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    try:
        return Config()
    except ValidationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
