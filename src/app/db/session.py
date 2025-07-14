from typing import Generator, Dict, Any
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool

from src.app.configs.config import get_config

from .base import Base

logger = logging.getLogger(__name__)

def create_database_engine():
    """Create database engine with optimized connection pooling."""
    config = get_config()
    
    # Base engine configuration
    engine_kwargs: Dict[str, Any] = {
        'echo': config.environment == "development",  # SQL logging in dev
        'future': True,  # Use SQLAlchemy 2.0 style
    }
    
    # Determine if we're using a real database (PostgreSQL) or SQLite
    database_url = config.database_url
    is_sqlite = database_url.startswith('sqlite')
    
    if config.environment == "production" and not is_sqlite:
        # Use QueuePool for production PostgreSQL
        engine_kwargs.update({
            'poolclass': QueuePool,
            'pool_size': config.db_pool_size,
            'max_overflow': config.db_max_overflow,
            'pool_timeout': config.db_pool_timeout,
            'pool_recycle': config.db_pool_recycle,
            'pool_pre_ping': config.db_pool_pre_ping,
        })
        logger.info(f"Database engine configured with QueuePool: "
                   f"pool_size={config.db_pool_size}, "
                   f"max_overflow={config.db_max_overflow}")
    else:
        # Use StaticPool for development/SQLite (no pooling parameters)
        engine_kwargs.update({
            'poolclass': StaticPool,
            'connect_args': {"check_same_thread": False} if is_sqlite else {}
        })
        logger.info("Database engine configured with StaticPool for development/SQLite")
    
    engine = create_engine(database_url, **engine_kwargs)
    
    # Add connection event listeners for monitoring
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_connection, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Database connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def on_checkin(dbapi_connection, connection_record):
        logger.debug("Database connection returned to pool")
    
    return engine

# Create the engine with connection pooling
engine = create_database_engine()

# Configure session with optimized settings
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine,
    expire_on_commit=False  # Avoid lazy loading issues in async contexts
)


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get a new database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # If this script is run directly, initialize the database
    init_db()
    print("Database initialized successfully.")
