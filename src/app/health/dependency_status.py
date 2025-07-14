import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, List


logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies that can be tracked."""
    DATABASE = "database"
    MIGRATION = "migration"
    EXTERNAL_API = "external_api"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"
    AUTHENTICATION = "authentication"
    CUSTOM = "custom"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DependencyHealth:
    """Track the health status of a single dependency."""
    name: str
    dependency_type: DependencyType
    status: HealthStatus = HealthStatus.UNKNOWN
    last_checked: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if the dependency is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def is_ready(self) -> bool:
        """Check if the dependency is ready for use (healthy or degraded)."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.dependency_type.value,
            "status": self.status.value,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "error": self.error_message,
            "metadata": self.metadata
        }


class DependencyHealthTracker:
    """Central tracker for all application dependencies."""
    
    def __init__(self):
        self._dependencies: Dict[str, DependencyHealth] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_dependency(
        self, 
        name: str, 
        dependency_type: DependencyType,
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Register a new dependency to track."""
        self._dependencies[name] = DependencyHealth(
            name=name,
            dependency_type=dependency_type,
            metadata=metadata or {}
        )
        self._logger.info(f"Registered dependency: {name} ({dependency_type.value})")
    
    def set_healthy(
        self, 
        name: str, 
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Mark a dependency as healthy."""
        if name not in self._dependencies:
            self._logger.warning(f"Dependency {name} not registered, auto-registering")
            self.register_dependency(name, DependencyType.CUSTOM)
        
        dep = self._dependencies[name]
        dep.status = HealthStatus.HEALTHY
        dep.last_checked = datetime.now(timezone.utc)
        dep.last_success = datetime.now(timezone.utc)
        dep.error_message = None
        if metadata:
            dep.metadata.update(metadata)
        
        self._logger.info(f"Dependency {name} marked as healthy")
    
    def set_degraded(
        self, 
        name: str, 
        message: str, 
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Mark a dependency as degraded (partially working)."""
        if name not in self._dependencies:
            self._logger.warning(f"Dependency {name} not registered, auto-registering")
            self.register_dependency(name, DependencyType.CUSTOM)
        
        dep = self._dependencies[name]
        dep.status = HealthStatus.DEGRADED
        dep.last_checked = datetime.now(timezone.utc)
        dep.error_message = message
        if metadata:
            dep.metadata.update(metadata)
        
        self._logger.warning(f"Dependency {name} marked as degraded: {message}")
    
    def set_unhealthy(
        self, 
        name: str, 
        error_message: str, 
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Mark a dependency as unhealthy."""
        if name not in self._dependencies:
            self._logger.warning(f"Dependency {name} not registered, auto-registering")
            self.register_dependency(name, DependencyType.CUSTOM)
        
        dep = self._dependencies[name]
        dep.status = HealthStatus.UNHEALTHY
        dep.last_checked = datetime.now(timezone.utc)
        dep.error_message = error_message
        if metadata:
            dep.metadata.update(metadata)
        
        self._logger.error(f"Dependency {name} marked as unhealthy: {error_message}")
    
    def get_dependency(self, name: str) -> Optional[DependencyHealth]:
        """Get the health status of a specific dependency."""
        return self._dependencies.get(name)
    
    def get_all_dependencies(self) -> Dict[str, DependencyHealth]:
        """Get all tracked dependencies."""
        return self._dependencies.copy()
    
    def get_dependencies_by_type(self, dependency_type: DependencyType) -> List[DependencyHealth]:
        """Get all dependencies of a specific type."""
        return [
            dep for dep in self._dependencies.values() 
            if dep.dependency_type == dependency_type
        ]
    
    def is_application_ready(self) -> bool:
        """Check if the application is ready based on all dependencies."""
        if not self._dependencies:
            return False
        
        # Application is ready if all dependencies are either healthy or degraded
        return all(dep.is_ready for dep in self._dependencies.values())
    
    def is_application_healthy(self) -> bool:
        """Check if the application is fully healthy based on all dependencies."""
        if not self._dependencies:
            return False
        
        # Application is healthy only if all dependencies are healthy
        return all(dep.is_healthy for dep in self._dependencies.values())
    
    def get_health_summary(self) -> Dict:
        """Get a comprehensive health summary for the readiness endpoint."""
        dependencies_dict = {
            name: dep.to_dict() 
            for name, dep in self._dependencies.items()
        }
        
        healthy_count = sum(1 for dep in self._dependencies.values() if dep.is_healthy)
        degraded_count = sum(1 for dep in self._dependencies.values() if dep.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for dep in self._dependencies.values() if dep.status == HealthStatus.UNHEALTHY)
        unknown_count = sum(1 for dep in self._dependencies.values() if dep.status == HealthStatus.UNKNOWN)
        
        return {
            "overall_status": "ready" if self.is_application_ready() else "not ready",
            "overall_health": "healthy" if self.is_application_healthy() else "degraded" if self.is_application_ready() else "unhealthy",
            "summary": {
                "total_dependencies": len(self._dependencies),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "unknown": unknown_count
            },
            "dependencies": dependencies_dict
        }
    
    def reset_all(self) -> None:
        """Reset all dependencies to unknown state."""
        for dep in self._dependencies.values():
            dep.status = HealthStatus.UNKNOWN
            dep.last_checked = None
            dep.last_success = None
            dep.error_message = None
        self._logger.info("All dependencies reset to unknown state")


# Global dependency health tracker instance
_dependency_tracker = DependencyHealthTracker()


def get_dependency_tracker() -> DependencyHealthTracker:
    """Get the global dependency tracker instance."""
    return _dependency_tracker


# Convenience functions for common operations
def register_dependency(name: str, dependency_type: DependencyType, metadata: Optional[Dict[str, str]] = None) -> None:
    """Register a dependency with the global tracker."""
    _dependency_tracker.register_dependency(name, dependency_type, metadata)


def set_dependency_healthy(name: str, metadata: Optional[Dict[str, str]] = None) -> None:
    """Mark a dependency as healthy."""
    _dependency_tracker.set_healthy(name, metadata)


def set_dependency_degraded(name: str, message: str, metadata: Optional[Dict[str, str]] = None) -> None:
    """Mark a dependency as degraded."""
    _dependency_tracker.set_degraded(name, message, metadata)


def set_dependency_unhealthy(name: str, error_message: str, metadata: Optional[Dict[str, str]] = None) -> None:
    """Mark a dependency as unhealthy."""
    _dependency_tracker.set_unhealthy(name, error_message, metadata)
