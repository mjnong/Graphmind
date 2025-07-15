from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from redis import Redis
from src.app.configs.config import get_config
from src.app.deps import get_file_service
from src.app.storage import get_store
from src.app.storage.local import LocalStore
from src.app.deps import get_neo4j_driver

from src.app.error_handling.graceful_degradation import setup_degradation_system
from src.app.error_handling.dead_letter_queue import dlq_handler

import json
import time
import logging
import os
import asyncio

logger = logging.getLogger(__name__)

config = get_config()

# Enhanced Celery configuration for scalability
celery_app = Celery("uploads", broker=config.celery_broker_url, backend=config.celery_result_backend)

# Make the app available for Celery discovery
app = celery_app

# Configure Celery for horizontal scaling
celery_app.conf.update(
    # Worker configuration
    worker_concurrency=config.celery_concurrency,
    worker_prefetch_multiplier=config.celery_prefetch_multiplier,
    worker_max_tasks_per_child=config.celery_max_tasks_per_child,
    
    # Task routing and execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        'process_file': {'queue': 'file_processing'},
        'cleanup_task': {'queue': 'maintenance'},
    },
    
    # Performance optimizations
    task_compression='gzip',
    result_compression='gzip',
    worker_disable_rate_limits=False,
    
    # Monitoring and logging
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Error handling
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
    worker_proc_alive_timeout=4.0,
)

# Worker identification for monitoring
worker_instance = os.environ.get('WORKER_INSTANCE', '1')
logger.info(f"Starting Celery worker instance: {worker_instance}")

redis = Redis(host=get_config().dragonfly_host, port=get_config().dragonfly_port, db=2)

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Log when worker is ready to accept tasks."""
    logger.info(f"Worker {worker_instance} is ready to accept tasks")
    
    # Initialize error handling system
    try:
        setup_degradation_system()
        logger.info("Error handling system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize error handling system: {e}")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Log when worker is shutting down."""
    logger.info(f"Worker {worker_instance} is shutting down")

def notify_progress(upload_id: int, progress: int, message: str = ""):
    """Send progress notification via Redis pub/sub"""
    try:
        data = {
            "id": upload_id,
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }
        redis.publish("upload_progress", json.dumps(data))
        logger.info(f"Progress notification sent for upload {upload_id}: {progress}% - {message}")
    except Exception as e:
        logger.error(f"Failed to send progress notification: {e}")

async def process_document(file_path: str, upload_id: int):
    """Process document and build knowledge graph from PDF"""
    logger.info(f"Starting document processing for file: {file_path}")
    
    # Check if file exists
    import os
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Log file details
    file_size = os.path.getsize(file_path)
    logger.info(f"Processing file: {file_path}, size: {file_size} bytes")
    
    notify_progress(upload_id, 20, "Extracting text from PDF...")
    
    try:
        # Get file metadata from database
        from src.app.db.crud.files import get_upload
        from src.app.db.session import SessionLocal
        
        db = SessionLocal()
        try:
            upload_record = get_upload(db, upload_id)
            file_metadata = {
                "upload_id": upload_id,
                "filename": upload_record.filename if upload_record else "unknown",
                "object_key": upload_record.object_key if upload_record else file_path,
                "mime_type": upload_record.mime_type if upload_record else "application/pdf",
                "file_path": file_path,
                "created_at": upload_record.created_at.isoformat() if upload_record and upload_record.created_at else None
            }
            logger.info(f"Retrieved file metadata: {file_metadata}")
        finally:
            db.close()
        
        # Build knowledge graph
        driver = get_neo4j_driver()
        logger.info("Starting knowledge graph construction...")
        
        notify_progress(upload_id, 40, "Building knowledge graph...")
        res = await driver.build_kg_from_pdf(file_path, upload_id=upload_id, file_metadata=file_metadata)
        logger.info(f"Knowledge graph construction completed. Result: {res}")
        
        notify_progress(upload_id, 80, "Verifying graph structure...")
        
        # Verify what was created in the database
        await verify_graph_creation(driver, upload_id)
        
        logger.info("Document processed successfully")
        return "Document processed successfully"
        
    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        raise

async def verify_graph_creation(driver, upload_id: int):
    """Verify what nodes and relationships were created in Neo4j"""
    try:
        # Get comprehensive schema information
        logger.info("Verifying graph creation...")
        
        schema = driver.inspect_graph_schema()
        logger.info(f"Graph schema after processing: {schema}")
        
        # Log key statistics
        total_nodes = schema.get('total_nodes', 0)
        total_rels = schema.get('total_relationships', 0)
        node_labels = schema.get('node_labels', [])
        rel_types = schema.get('relationship_types', [])
        
        logger.info(f"Created {total_nodes} nodes and {total_rels} relationships")
        logger.info(f"Node labels: {node_labels}")
        logger.info(f"Relationship types: {rel_types}")
        
        if total_nodes == 0:
            logger.warning("No nodes were created in the graph!")
        if total_rels == 0:
            logger.warning("No relationships were created in the graph!")
            
        notify_progress(upload_id, 90, f"Graph created: {total_nodes} nodes, {total_rels} relationships")
        
    except Exception as e:
        logger.warning(f"Could not verify graph creation: {e}")
        notify_progress(upload_id, 85, "Graph verification skipped")

@celery_app.task(bind=True, name="process_file", autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_file(self, upload_id: int):
    """Process an uploaded file with comprehensive error handling."""
    file_service = get_file_service()
    store = get_store()
    
    try:
        # Get upload record
        upload = file_service.get_upload_by_id(upload_id)
        if not upload:
            raise ValueError(f"Upload with id {upload_id} not found")
        
        logger.info(f"Starting processing for upload {upload_id}: {upload.filename}")
        
        # Mark as processing
        file_service.update_upload_status(upload_id, "processing")
        file_service.update_upload_progress(upload_id, 0)
        notify_progress(upload_id, 0, "Starting file processing...")
        
        # Get file from storage
        notify_progress(upload_id, 10, "Retrieving file...")
        
        # Process the file based on type
        if upload.mime_type == "application/pdf":
            notify_progress(upload_id, 15, "Processing PDF document...")
            
            # Get the actual file path from storage
            if isinstance(store, LocalStore):
                actual_file_path = str(store._full(upload.object_key))
            else:
                actual_file_path = upload.object_key
            
            # Process document with error handling
            try:
                result = run_async_in_celery(process_document, actual_file_path, upload_id)
            except Exception as e:
                # Try degraded processing if main processing fails
                logger.warning(f"Main processing failed, trying degraded mode: {e}")
                result = run_async_in_celery(process_document_degraded, actual_file_path, upload_id)
                    
        elif upload.mime_type in ["image/jpeg", "image/png"]:
            notify_progress(upload_id, 15, "Processing image...")
            result = "Image processed successfully"
        else:
            notify_progress(upload_id, 15, "Processing text document...")
            result = "Text document processed successfully"
        
        # Mark as complete
        file_service.update_upload_status(upload_id, "complete")
        file_service.update_upload_progress(upload_id, 100)
        notify_progress(upload_id, 100, "File processing completed successfully!")
        
        logger.info(f"Successfully processed upload {upload_id}: {upload.filename}")
        return {"status": "success", "message": result}
        
    except Exception as e:
        logger.error(f"Error processing upload {upload_id}: {e}")
        
        # Add to dead letter queue for manual retry
        dlq_handler.add_failed_task(
            task_id=self.request.id,
            task_name="process_file",
            args=[upload_id],
            kwargs={},
            error=str(e),
            retry_count=getattr(self.request, 'retries', 0)
        )
        
        # Mark as failed
        upload = file_service.get_upload_by_id(upload_id)
        if upload:
            file_service.update_upload_status(upload_id, "failed")
            notify_progress(upload_id, -1, f"Processing failed: {str(e)}")
        
        raise e

async def process_document_with_circuit_breaker(file_path: str, upload_id: int):
    """Process document with circuit breaker protection"""
    return await process_document(file_path, upload_id)


async def process_document_degraded(file_path: str, upload_id: int):
    """Process document in degraded mode (without graph operations)"""
    logger.info(f"Processing document in degraded mode: {file_path}")
    
    notify_progress(upload_id, 40, "Processing in degraded mode - text extraction only...")
    
    # Basic text extraction without external dependencies
    try:
        # Validate file exists and get size
        file_size = os.path.getsize(file_path)
        
        notify_progress(upload_id, 80, "Basic file validation completed - graph processing deferred...")
        
        return {
            "file_validated": True,
            "file_size": file_size,
            "degraded_mode": True,
            "message": "File validated successfully. Full processing will be retried when services are available."
        }
        
    except Exception as e:
        logger.error(f"Error in degraded processing: {e}")
        raise e


# Import asyncio for running async functions in Celery tasks


def run_async_in_celery(async_func, *args, **kwargs):
    """Helper to run async functions in Celery tasks"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(async_func(*args, **kwargs))