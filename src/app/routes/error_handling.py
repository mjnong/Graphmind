"""
Error handling management API endpoints.
Provides monitoring and control over error handling systems.
"""

import logging
from typing import Any, Dict, List

from celery import current_app as celery_app
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.app.error_handling.dead_letter_queue import dlq_handler
from src.app.error_handling.graceful_degradation import degradation_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/error-handling",
    tags=["Error Handling"],
    responses={404: {"description": "Not found"}},
)


# Pydantic models for request/response
class RetryTaskRequest(BaseModel):
    task_ids: List[str]


class RetryTaskResponse(BaseModel):
    results: Dict[str, bool]
    message: str


class CircuitBreakerStatusResponse(BaseModel):
    service_name: str
    state: str
    failure_count: int
    last_failure_time: str
    next_attempt_time: str


class SystemHealthResponse(BaseModel):
    service_level: str
    available_features: Dict[str, bool]
    circuit_breakers: List[CircuitBreakerStatusResponse]
    dlq_stats: Dict[str, Any]


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get overall system health and error handling status"""
    try:
        # Get service level
        service_level = degradation_manager.get_service_level()
        available_features = degradation_manager.get_available_features()

        # Get circuit breaker statuses
        circuit_breaker_statuses = []
        for service_name, breaker in degradation_manager.circuit_breakers.items():
            stats = breaker.get_stats()
            status = CircuitBreakerStatusResponse(
                service_name=service_name,
                state=stats["state"],
                failure_count=stats["failure_count"],
                last_failure_time=str(stats["last_failure_time"])
                if stats["last_failure_time"]
                else "",
                next_attempt_time="",  # This isn't tracked in current implementation
            )
            circuit_breaker_statuses.append(status)

        # Get DLQ stats
        dlq_stats = dlq_handler.get_dlq_stats()

        return SystemHealthResponse(
            service_level=service_level.value,
            available_features=available_features,
            circuit_breakers=circuit_breaker_statuses,
            dlq_stats=dlq_stats,
        )

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failed-tasks")
async def get_failed_tasks(limit: int = 100, offset: int = 0):
    """Get failed tasks from the dead letter queue"""
    try:
        failed_tasks = dlq_handler.get_failed_tasks(limit=limit, offset=offset)
        return {
            "failed_tasks": [task.to_dict() for task in failed_tasks],
            "total_count": len(failed_tasks),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error getting failed tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failed-tasks/{task_id}")
async def get_failed_task(task_id: str):
    """Get specific failed task details"""
    try:
        failed_task = dlq_handler.get_failed_task_by_id(task_id)
        if not failed_task:
            raise HTTPException(status_code=404, detail="Failed task not found")

        return failed_task.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting failed task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retry-tasks", response_model=RetryTaskResponse)
async def retry_failed_tasks(request: RetryTaskRequest):
    """Retry one or more failed tasks"""
    try:
        results = dlq_handler.bulk_retry_failed_tasks(request.task_ids, celery_app)

        success_count = sum(1 for success in results.values() if success)
        total_count = len(request.task_ids)

        return RetryTaskResponse(
            results=results,
            message=f"Retried {success_count}/{total_count} tasks successfully",
        )
    except Exception as e:
        logger.error(f"Error retrying failed tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/failed-tasks/{task_id}")
async def remove_failed_task(task_id: str):
    """Remove a failed task from the dead letter queue"""
    try:
        success = dlq_handler.remove_failed_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Failed task not found")

        return {"message": f"Successfully removed task {task_id} from DLQ"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing failed task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str):
    """Manually reset a circuit breaker"""
    try:
        breaker = degradation_manager.circuit_breakers.get(service_name)
        if not breaker:
            raise HTTPException(
                status_code=404, detail=f"Circuit breaker for {service_name} not found"
            )

        breaker.reset()
        return {"message": f"Circuit breaker for {service_name} has been reset"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting circuit breaker for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuit-breakers/{service_name}")
async def get_circuit_breaker_status(service_name: str):
    """Get detailed status of a specific circuit breaker"""
    try:
        breaker = degradation_manager.circuit_breakers.get(service_name)
        if not breaker:
            raise HTTPException(
                status_code=404, detail=f"Circuit breaker for {service_name} not found"
            )

        stats = breaker.get_stats()

        return {
            "service_name": service_name,
            "state": stats["state"],
            "failure_count": stats["failure_count"],
            "success_count": stats["success_count"],
            "last_failure_time": str(stats["last_failure_time"])
            if stats["last_failure_time"]
            else None,
            "next_attempt_time": None,  # Not currently tracked
            "config": stats["config"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting circuit breaker status for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-dlq")
async def cleanup_dead_letter_queue(background_tasks: BackgroundTasks):
    """Clean up old tasks from the dead letter queue"""
    try:

        def cleanup_task():
            removed_count = dlq_handler.cleanup_old_tasks()
            logger.info(f"DLQ cleanup removed {removed_count} old tasks")

        background_tasks.add_task(cleanup_task)

        return {"message": "DLQ cleanup task has been scheduled"}
    except Exception as e:
        logger.error(f"Error scheduling DLQ cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/dlq")
async def get_dlq_statistics():
    """Get detailed statistics about the dead letter queue"""
    try:
        stats = dlq_handler.get_dlq_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting DLQ statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/service-level")
async def get_current_service_level():
    """Get current service degradation level"""
    try:
        service_level = degradation_manager.get_service_level()
        available_features = degradation_manager.get_available_features()

        return {
            "service_level": service_level.value,
            "available_features": available_features,
            "description": {
                "full": "All features available",
                "degraded": "Some features disabled",
                "basic": "Only essential features",
                "maintenance": "Service unavailable",
            }.get(service_level.value, "Unknown"),
        }
    except Exception as e:
        logger.error(f"Error getting service level: {e}")
        raise HTTPException(status_code=500, detail=str(e))
