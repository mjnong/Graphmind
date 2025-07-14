"""
Circuit breaker implementation for GraphRAG system.
Provides automatic service protection with configurable failure thresholds.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from redis import Redis

from src.app.configs.config import get_config

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit breaker triggered
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: int = 30  # Individual operation timeout


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.redis = Redis(
            host=get_config().dragonfly_host,
            port=get_config().dragonfly_port,
            db=3,
            decode_responses=True,
        )
        self._key_prefix = f"circuit_breaker:{name}"

    def _get_state(self) -> CircuitState:
        """Get current circuit breaker state from Redis"""
        state = self.redis.get(f"{self._key_prefix}:state")
        if state:
            return CircuitState(state)
        return CircuitState.CLOSED

    def _set_state(self, state: CircuitState):
        """Set circuit breaker state in Redis"""
        self.redis.set(
            f"{self._key_prefix}:state", state.value, ex=3600
        )  # Expire in 1 hour

    def _get_failure_count(self) -> int:
        """Get current failure count"""
        count = self.redis.get(f"{self._key_prefix}:failures")
        if count is not None:
            try:
                return int(str(count))  # Explicit string conversion for type safety
            except (ValueError, TypeError):
                return 0
        return 0

    def _increment_failures(self):
        """Increment failure count"""
        key = f"{self._key_prefix}:failures"
        self.redis.incr(key)
        self.redis.expire(key, 3600)  # Expire in 1 hour

    def _reset_failures(self):
        """Reset failure count"""
        self.redis.delete(f"{self._key_prefix}:failures")

    def _get_last_failure_time(self) -> float:
        """Get timestamp of last failure"""
        timestamp = self.redis.get(f"{self._key_prefix}:last_failure")
        if timestamp:
            try:
                return float(
                    str(timestamp)
                )  # Explicit string conversion for type safety
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    def _set_last_failure_time(self):
        """Set timestamp of last failure"""
        self.redis.set(f"{self._key_prefix}:last_failure", time.time(), ex=3600)

    def _get_success_count(self) -> int:
        """Get success count in half-open state"""
        count = self.redis.get(f"{self._key_prefix}:successes")
        if count is not None:
            try:
                return int(str(count))  # Explicit string conversion for type safety
            except (ValueError, TypeError):
                return 0
        return 0

    def _increment_successes(self):
        """Increment success count"""
        key = f"{self._key_prefix}:successes"
        self.redis.incr(key)
        self.redis.expire(key, 3600)

    def _reset_successes(self):
        """Reset success count"""
        self.redis.delete(f"{self._key_prefix}:successes")

    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        state = self._get_state()

        if state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (
                time.time() - self._get_last_failure_time()
                > self.config.recovery_timeout
            ):
                self._set_state(CircuitState.HALF_OPEN)
                self._reset_successes()
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                return False
            return True

        return False

    def record_success(self):
        """Record a successful operation"""
        state = self._get_state()

        if state == CircuitState.HALF_OPEN:
            self._increment_successes()
            if self._get_success_count() >= self.config.success_threshold:
                self._set_state(CircuitState.CLOSED)
                self._reset_failures()
                self._reset_successes()
                logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
        elif state == CircuitState.CLOSED:
            # Reset failure count on success
            self._reset_failures()

    def record_failure(self):
        """Record a failed operation"""
        state = self._get_state()

        if state == CircuitState.HALF_OPEN:
            # Go back to open state
            self._set_state(CircuitState.OPEN)
            self._set_last_failure_time()
            logger.warning(f"Circuit breaker {self.name} transitioning back to OPEN")
        elif state == CircuitState.CLOSED:
            self._increment_failures()
            if self._get_failure_count() >= self.config.failure_threshold:
                self._set_state(CircuitState.OPEN)
                self._set_last_failure_time()
                logger.error(
                    f"Circuit breaker {self.name} transitioning to OPEN after {self.config.failure_threshold} failures"
                )

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.is_open():
            raise Exception(
                f"Circuit breaker {self.name} is OPEN - operation not allowed"
            )

        try:
            # Handle both async and sync functions
            if hasattr(func, "__call__"):
                if hasattr(func, "__await__"):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            else:
                result = func

            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self._get_state().value,
            "failure_count": self._get_failure_count(),
            "success_count": self._get_success_count(),
            "last_failure_time": self._get_last_failure_time(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
            },
        }

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self._get_state()

    def reset(self):
        """Manually reset the circuit breaker to closed state"""
        self._set_state(CircuitState.CLOSED)
        self._reset_failures()
        self._reset_successes()
        logger.info(f"Circuit breaker {self.name} has been manually reset")
