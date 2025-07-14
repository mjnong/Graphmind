from contextlib import asynccontextmanager
import logging
import signal
import sys

from fastapi import FastAPI, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from src.app.routes import health, files, error_handling
import uvicorn
import asyncio
from src.app.configs.config import get_config

logger = logging.getLogger(__name__)

def handle_shutdown_signal(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

    # Exit gracefully
    sys.exit(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register dependencies for health checks
    from src.app.health import DependencyType, register_dependency
    register_dependency("database_migrations", DependencyType.MIGRATION, {
        "description": "Alembic database schema migrations"
    })
    
    # Ensure upload directory exists
    import os
    upload_dir = get_config().local_path
    os.makedirs(upload_dir, exist_ok=True)
    logger.info(f"Upload directory ensured: {upload_dir}")
    
    # Run Database migrations using Alembic (if enabled)
    config = get_config()
    if config.run_migrations_on_startup:
        from src.app.db.alembic_runner import run_migrations
        logger.info("Checking database migrations...")
        try:
            run_migrations()
        except Exception as e:
            logger.error(f"Failed to run database migrations: {e}")
            # Don't raise the exception to prevent app from crashing
            # The health check will show the migration status
    else:
        logger.info("Skipping database migrations (disabled by configuration)")
    
    # Upload directory already created during app initialization
    logger.info(f"Upload directory confirmed ready: {get_config().local_path}")
    
    yield
    # Cleanup actions can be added here if needed

app = FastAPI(
    title="GraphRAG API",
    description="API for GraphRAG, a custom graph-based RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files at a different path to avoid conflict with /files API routes
app.mount("/static", staticfiles.StaticFiles(directory=get_config().local_path), name="static")

app.include_router(health.router)
app.include_router(files.router)
app.include_router(error_handling.router)  # Include the error handling router

async def main():
    """
    Main entry point for the FastAPI application.
    
    This function is called when the application starts.
    """
    logger.info("Starting GraphRAG API...")
    # Additional startup tasks can be added here if needed
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    # Add additional signal handling for development reloads
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handle_shutdown_signal)
        
    config = uvicorn.Config(
        app,
        host=get_config().fastapi_host,
        port=get_config().fastapi_port,
        ws_ping_interval=10,
        ws_ping_timeout=20,
        log_level=get_config().graphrag_log_level.value,
        use_colors=True,
        access_log=True,
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())