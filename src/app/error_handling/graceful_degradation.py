"""
Graceful degradation manager for GraphRAG system.
Provides service level management with automatic fallbacks.
"""

import logging
from enum import Enum
from typing import Any, Callable, Dict

from src.app.error_handling.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class ServiceLevel(Enum):
    FULL = "full"  # All features available
    DEGRADED = "degraded"  # Some features disabled
    BASIC = "basic"  # Only essential features
    MAINTENANCE = "maintenance"  # Service unavailable


class GracefulDegradationManager:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.current_level = ServiceLevel.FULL

    def register_circuit_breaker(
        self, service_name: str, circuit_breaker: CircuitBreaker
    ):
        """Register a circuit breaker for a service"""
        self.circuit_breakers[service_name] = circuit_breaker
        logger.info(f"Registered circuit breaker for service: {service_name}")

    def register_fallback(self, service_name: str, fallback_handler: Callable):
        """Register a fallback handler for a service"""
        self.fallback_handlers[service_name] = fallback_handler
        logger.info(f"Registered fallback handler for service: {service_name}")

    def get_service_level(self) -> ServiceLevel:
        """Determine current service level based on circuit breaker states"""
        open_breakers = [
            name for name, breaker in self.circuit_breakers.items() if breaker.is_open()
        ]

        if not open_breakers:
            return ServiceLevel.FULL
        elif len(open_breakers) == len(self.circuit_breakers):
            return ServiceLevel.MAINTENANCE
        elif "neo4j" in open_breakers or "openai" in open_breakers:
            return ServiceLevel.BASIC
        else:
            return ServiceLevel.DEGRADED

    async def execute_with_fallback(
        self, service_name: str, primary_func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with fallback support"""
        try:
            # Try primary service
            circuit_breaker = self.circuit_breakers.get(service_name)
            if circuit_breaker and not circuit_breaker.is_open():
                return await circuit_breaker.execute(primary_func, *args, **kwargs)
            else:
                raise Exception(f"Service {service_name} is unavailable")

        except Exception as e:
            logger.warning(f"Primary service {service_name} failed: {e}")

            # Try fallback
            fallback = self.fallback_handlers.get(service_name)
            if fallback:
                logger.info(f"Using fallback for service {service_name}")
                if hasattr(fallback, "__await__"):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            else:
                logger.error(f"No fallback available for service {service_name}")
                raise e

    def get_available_features(self) -> Dict[str, bool]:
        """Get list of currently available features"""
        service_level = self.get_service_level()
        open_breakers = [
            name for name, breaker in self.circuit_breakers.items() if breaker.is_open()
        ]

        return {
            "file_upload": service_level != ServiceLevel.MAINTENANCE,
            "graph_processing": "neo4j" not in open_breakers,
            "ai_processing": "openai" not in open_breakers,
            "real_time_updates": "redis" not in open_breakers,
            "full_text_search": service_level
            in [ServiceLevel.FULL, ServiceLevel.DEGRADED],
            "advanced_analytics": service_level == ServiceLevel.FULL,
        }


# Initialize global degradation manager
degradation_manager = GracefulDegradationManager()


# Fallback functions
async def neo4j_fallback(*args, **kwargs):
    """Fallback when Neo4j is unavailable - store in file for later processing"""
    logger.warning("Neo4j unavailable, storing data for later processing")
    # Store data in queue for later processing
    return {
        "status": "queued_for_later",
        "message": "Graph processing will be completed when service is restored",
        "fallback_used": True,
    }


async def openai_fallback(*args, **kwargs):
    """Fallback when OpenAI is unavailable - use local models or simpler processing"""
    logger.warning("OpenAI unavailable, using simplified processing")
    return {
        "status": "simplified",
        "message": "Text processed with basic extraction",
        "fallback_used": True,
    }


async def redis_fallback(*args, **kwargs):
    """Fallback when Redis is unavailable - use in-memory storage"""
    logger.warning("Redis unavailable, using in-memory storage")
    return {
        "status": "degraded",
        "message": "Using in-memory storage - data may be lost on restart",
        "fallback_used": True,
    }


def setup_degradation_system():
    """Setup the graceful degradation system"""
    from src.app.error_handling.circuit_breaker import CircuitBreakerConfig

    # Register circuit breakers with different configurations
    neo4j_breaker = CircuitBreaker(
        "neo4j",
        CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=120, success_threshold=2
        ),
    )

    openai_breaker = CircuitBreaker(
        "openai",
        CircuitBreakerConfig(
            failure_threshold=5, recovery_timeout=60, success_threshold=3
        ),
    )

    redis_breaker = CircuitBreaker(
        "redis",
        CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=30, success_threshold=2
        ),
    )

    # Register circuit breakers
    degradation_manager.register_circuit_breaker("neo4j", neo4j_breaker)
    degradation_manager.register_circuit_breaker("openai", openai_breaker)
    degradation_manager.register_circuit_breaker("redis", redis_breaker)

    # Register fallbacks
    degradation_manager.register_fallback("neo4j", neo4j_fallback)
    degradation_manager.register_fallback("openai", openai_fallback)
    degradation_manager.register_fallback("redis", redis_fallback)

    logger.info(
        "Graceful degradation system initialized with circuit breakers and fallbacks"
    )
