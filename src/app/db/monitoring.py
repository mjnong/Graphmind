"""Database monitoring and health check utilities."""

import asyncio
import logging
import time
from typing import Dict, Any
from sqlalchemy import text

from src.app.db.session import engine

logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Monitor database connection pool and performance."""
    
    @staticmethod
    def get_pool_status() -> Dict[str, Any]:
        """Get current database connection pool status."""
        pool = engine.pool
        
        try:
            # Get basic pool information
            status = {
                "pool_class": pool.__class__.__name__,
                "engine_url_masked": str(engine.url).replace(str(engine.url.password) if engine.url.password else "", "***"),
            }
            
            # Try to get pool statistics with error handling
            try:
                # Use getattr for safer attribute access
                if hasattr(pool, '_pool'):
                    inner_pool = getattr(pool, '_pool', None)
                    if inner_pool and hasattr(inner_pool, 'qsize'):
                        status["queue_size"] = inner_pool.qsize()
                    else:
                        status["queue_size"] = "unknown"
                    
                # Check for overflow attribute safely
                overflow = getattr(pool, '_overflow', None)
                if overflow is not None:
                    status["overflow_count"] = overflow
                    
                # Try common pool status methods
                status["pool_status"] = "active"
                
            except Exception as attr_error:
                logger.debug(f"Could not access pool attributes: {attr_error}")
                status["pool_status"] = "limited_info"
                
            return status
            
        except Exception as e:
            logger.warning(f"Could not get pool status: {e}")
            return {
                "pool_class": "unknown",
                "error": str(e),
                "pool_status": "error"
            }
    
    @staticmethod
    async def check_database_health() -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        try:
            # Test basic connectivity
            with engine.begin() as conn:
                result = conn.execute(text("SELECT 1 as healthy")).fetchone()
                if result and result[0] == 1:
                    connectivity = "healthy"
                else:
                    connectivity = "unhealthy"
            
            # Get pool status
            pool_status = DatabaseMonitor.get_pool_status()
            
            # Test transaction performance
            start_time = time.time()
            with engine.begin() as conn:
                conn.execute(text("SELECT COUNT(*) FROM information_schema.tables"))
            transaction_time = time.time() - start_time
            
            return {
                "status": "healthy" if connectivity == "healthy" else "unhealthy",
                "connectivity": connectivity,
                "pool_status": pool_status,
                "transaction_time_ms": round(transaction_time * 1000, 2),
                "engine_url": str(engine.url).replace(str(engine.url.password), "***") if engine.url.password else str(engine.url),
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_status": DatabaseMonitor.get_pool_status(),
            }
    
    @staticmethod
    def log_pool_stats():
        """Log current pool statistics for monitoring."""
        try:
            stats = DatabaseMonitor.get_pool_status()
            logger.info(f"DB Pool Stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to log pool stats: {e}")


async def periodic_pool_monitoring():
    """Periodically monitor and log database pool status."""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            DatabaseMonitor.log_pool_stats()
        except Exception as e:
            logger.error(f"Pool monitoring error: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error
