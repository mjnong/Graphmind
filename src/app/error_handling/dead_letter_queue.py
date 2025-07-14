"""
Dead Letter Queue (DLQ) handler for failed tasks.
Provides task recovery and manual retry capabilities.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from redis import Redis

from src.app.configs.config import get_config

logger = logging.getLogger(__name__)
settings = get_config()


def parse_redis_url(redis_url: str) -> Dict[str, Any]:
    """Parse Redis URL and return connection parameters"""
    if redis_url.startswith("redis://"):
        # Format: redis://host:port/db
        url_parts = redis_url.replace("redis://", "").split("/")
        host_port = url_parts[0].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 6379
        db = int(url_parts[1]) if len(url_parts) > 1 else 0
        return {"host": host, "port": port, "db": db}
    else:
        return {"host": "localhost", "port": 6379, "db": 0}


class FailedTask:
    def __init__(
        self,
        task_id: str,
        task_name: str,
        args: List,
        kwargs: Dict,
        error: str,
        failed_at: datetime,
        retry_count: int = 0,
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.args = args
        self.kwargs = kwargs
        self.error = error
        self.failed_at = failed_at
        self.retry_count = retry_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "error": self.error,
            "failed_at": self.failed_at.isoformat(),
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailedTask":
        return cls(
            task_id=data["task_id"],
            task_name=data["task_name"],
            args=data["args"],
            kwargs=data["kwargs"],
            error=data["error"],
            failed_at=datetime.fromisoformat(data["failed_at"]),
            retry_count=data["retry_count"],
        )


class DeadLetterQueueHandler:
    def __init__(self, redis_client: Optional[Redis] = None):
        if redis_client:
            self.redis_client = redis_client
        else:
            # Parse Redis URL from settings
            redis_params = parse_redis_url(settings.redis_url)
            self.redis_client = Redis(
                host=redis_params["host"],
                port=redis_params["port"],
                db=redis_params["db"],
                decode_responses=True,
            )
        self.dlq_key = "dlq:failed_tasks"
        self.retention_days = 30  # Keep failed tasks for 30 days

    def add_failed_task(
        self,
        task_id: str,
        task_name: str,
        args: List,
        kwargs: Dict,
        error: str,
        retry_count: int = 0,
    ):
        """Add a failed task to the dead letter queue"""
        failed_task = FailedTask(
            task_id=task_id,
            task_name=task_name,
            args=args,
            kwargs=kwargs,
            error=error,
            failed_at=datetime.utcnow(),
            retry_count=retry_count,
        )

        task_json = json.dumps(failed_task.to_dict())

        # Add to sorted set with timestamp as score for easy retrieval and cleanup
        timestamp = failed_task.failed_at.timestamp()
        self.redis_client.zadd(self.dlq_key, {task_json: timestamp})

        logger.warning(f"Added task {task_id} to dead letter queue: {error}")

    def get_failed_tasks(self, limit: int = 100, offset: int = 0) -> List[FailedTask]:
        """Get failed tasks from the dead letter queue"""
        try:
            # Get tasks ordered by failure time (newest first)
            task_data_result = self.redis_client.zrevrange(
                self.dlq_key, offset, offset + limit - 1, withscores=False
            )

            failed_tasks = []
            # Handle Redis response type
            try:
                task_data = task_data_result or []
                for task_json in task_data:  # type: ignore
                    try:
                        task_dict = json.loads(str(task_json))
                        failed_tasks.append(FailedTask.from_dict(task_dict))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing failed task data: {e}")
            except TypeError:
                # Handle case where Redis returns different type
                logger.debug("Redis returned unexpected type, returning empty list")

            return failed_tasks
        except Exception as e:
            logger.error(f"Error retrieving failed tasks: {e}")
            return []

    def get_failed_task_by_id(self, task_id: str) -> Optional[FailedTask]:
        """Get a specific failed task by ID"""
        failed_tasks = self.get_failed_tasks(limit=1000)  # Get more tasks for search
        for task in failed_tasks:
            if task.task_id == task_id:
                return task
        return None

    def remove_failed_task(self, task_id: str) -> bool:
        """Remove a failed task from the dead letter queue"""
        try:
            # Find and remove the task
            task_data_result = self.redis_client.zrange(
                self.dlq_key, 0, -1, withscores=False
            )

            try:
                task_data = task_data_result or []
                for task_json in task_data:  # type: ignore
                    try:
                        task_dict = json.loads(str(task_json))
                        if task_dict["task_id"] == task_id:
                            self.redis_client.zrem(self.dlq_key, task_json)
                            logger.info(
                                f"Removed task {task_id} from dead letter queue"
                            )
                            return True
                    except json.JSONDecodeError:
                        continue
            except TypeError:
                logger.debug("Redis returned unexpected type in remove_failed_task")

            return False
        except Exception as e:
            logger.error(f"Error removing failed task {task_id}: {e}")
            return False

    def retry_failed_task(self, task_id: str, celery_app) -> bool:
        """Retry a failed task from the dead letter queue"""
        try:
            failed_task = self.get_failed_task_by_id(task_id)
            if not failed_task:
                logger.error(f"Failed task {task_id} not found in DLQ")
                return False

            # Remove from DLQ first
            if not self.remove_failed_task(task_id):
                logger.error(f"Could not remove task {task_id} from DLQ")
                return False

            # Retry the task with incremented retry count
            task_func = getattr(celery_app, failed_task.task_name, None)
            if task_func:
                task_func.apply_async(
                    args=failed_task.args,
                    kwargs=failed_task.kwargs,
                    task_id=f"{task_id}_retry_{failed_task.retry_count + 1}",
                )
                logger.info(f"Retried task {task_id} from dead letter queue")
                return True
            else:
                logger.error(f"Task function {failed_task.task_name} not found")
                return False

        except Exception as e:
            logger.error(f"Error retrying failed task {task_id}: {e}")
            return False

    def bulk_retry_failed_tasks(
        self, task_ids: List[str], celery_app
    ) -> Dict[str, bool]:
        """Retry multiple failed tasks"""
        results = {}
        for task_id in task_ids:
            results[task_id] = self.retry_failed_task(task_id, celery_app)
        return results

    def cleanup_old_tasks(self):
        """Remove tasks older than retention period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
            cutoff_timestamp = cutoff_time.timestamp()

            # Remove tasks older than cutoff
            removed_count_result = self.redis_client.zremrangebyscore(
                self.dlq_key, 0, cutoff_timestamp
            )

            # Handle Redis response type
            removed_count = int(removed_count_result) if removed_count_result else 0  # type: ignore

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old failed tasks from DLQ")

            return removed_count
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")
            return 0

    def get_dlq_stats(self) -> Dict[str, Any]:
        """Get statistics about the dead letter queue"""
        try:
            total_tasks = self.redis_client.zcard(self.dlq_key)

            # Get task distribution by error type
            failed_tasks = self.get_failed_tasks(limit=1000)
            error_counts = {}
            task_type_counts = {}

            for task in failed_tasks:
                # Count by error type
                error_type = (
                    task.error.split(":")[0] if ":" in task.error else task.error
                )
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

                # Count by task type
                task_type_counts[task.task_name] = (
                    task_type_counts.get(task.task_name, 0) + 1
                )

            return {
                "total_failed_tasks": total_tasks,
                "error_distribution": error_counts,
                "task_type_distribution": task_type_counts,
                "oldest_task": failed_tasks[-1].failed_at.isoformat()
                if failed_tasks
                else None,
                "newest_task": failed_tasks[0].failed_at.isoformat()
                if failed_tasks
                else None,
            }
        except Exception as e:
            logger.error(f"Error getting DLQ stats: {e}")
            return {"error": str(e)}


# Initialize global DLQ handler
dlq_handler = DeadLetterQueueHandler()
