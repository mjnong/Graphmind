import logging

from fastapi import HTTPException
from fastapi.routing import APIRouter

from src.app.health import get_dependency_tracker

logger = logging.getLogger("graphrag.health")
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def root():
    """
    Root endpoint for the health check.
    
    Returns:
        dict: A simple message indicating the health check endpoint.
    """
    return {"message": "Health check endpoint. Use /health/live for detailed status."}

@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Returns:
        dict: A simple message indicating the readiness check endpoint.
    """
    dependency_tracker = get_dependency_tracker()
    try:
        # Check if the application is ready based on registered dependencies
        if not dependency_tracker.is_application_ready():
            raise HTTPException(status_code=503, detail="Application is not ready")
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=500, detail="Readiness check failed")
    return {"message": "Readiness check endpoint. Use /health/live for detailed status."}

@router.get("/live")
async def health_check():
    """
    Perform a health check of the application.
    
    Returns:
        dict: A dictionary containing the health status of the application.
    """
    dependency_tracker = get_dependency_tracker()
    try:
        # Get the dependency tracker to check the health of registered dependencies
        health_status = dependency_tracker.is_application_healthy()
        return {"status": "healthy" if health_status else "unhealthy", "code": 200, "dependencies": dependency_tracker.get_all_dependencies()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")