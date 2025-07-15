import logging
from datetime import datetime
from typing import Annotated
from fastapi.routing import APIRouter
from fastapi import File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect, Request, Depends
from src.app.storage import get_store
from src.app.configs.config import get_config
from src.app.deps import get_file_service, get_neo4j_driver
from src.app.db.schemas.file_metadata import UploadCreate
from src.app.celery.worker import celery_app
from src.app.security.file_validator import ValidationError, get_file_validator
from src.app.security.rate_limiter import rate_limit_dependency
import redis.asyncio as redis
import uuid

logger = logging.getLogger("graphrag.files")
router = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={404: {"description": "Not found"}},
)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time upload progress updates"""
    await websocket.accept()
    redis_client = redis.from_url(get_config().redis_url)
    pubsub = redis_client.pubsub()
    
    try:
        await pubsub.subscribe("upload_progress")
        logger.info("Subscribed to upload_progress channel")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                # Process the message from the Redis pubsub
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                logger.info(f"Received message: {data}")
                await websocket.send_text(data)
            elif message["type"] == "subscribe":
                logger.info("Subscribed to upload_progress channel")
            elif message["type"] == "unsubscribe":
                logger.info("Unsubscribed from upload_progress channel")
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    finally:
        await pubsub.unsubscribe("upload_progress")
        await redis_client.close()

@router.post("/upload")
async def upload_file(
    request: Request,
    file: Annotated[UploadFile, File(description="The file to upload")],
    metadata: Annotated[str, Form(description="JSON metadata for the file upload")] = "{}",
    _: None = Depends(rate_limit_dependency)  # Rate limiting dependency
):
    """
    Upload a file to the server and start processing.
    Includes comprehensive security validation and rate limiting.
    """
    try:
        # 1. File validation using security module
        validator = get_file_validator()
        validation_result = await validator.validate_file(file)
        
        logger.info(f"File validation passed: {validation_result['filename']}")
        
        # 2. Generate unique object key
        file_uuid = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y/%m/%d")
        object_key = f"{timestamp}/{file_uuid}_{file.filename}"
        
        # 3. Store file
        store = get_store()
        await file.read()  # Read for storage
        await file.seek(0)  # Reset file pointer
        store.put(file.file, object_key)
        
        # 4. Create database record
        file_service = get_file_service()
        upload_data = UploadCreate(
            filename=file.filename or "unknown",  # Validation ensures filename exists
            mime_type=validation_result["detected_type"] or file.content_type or "application/octet-stream",
            object_key=object_key,
            status="uploaded",
            progress=0
        )
        upload_record = file_service.create_upload(upload_data)
        
        # 5. Start processing task
        task_result = celery_app.send_task("process_file", args=[upload_record.id], queue="file_processing")
        logger.info(f"Dispatched task {task_result.id} for upload {upload_record.id} to queue 'file_processing'")
        
        logger.info(f"Uploaded and queued file: {file.filename}, ID: {upload_record.id}")
        
        return {
            "id": upload_record.id,
            "filename": file.filename,
            "size": validation_result["size"],
            "detected_type": validation_result["detected_type"],
            "object_key": object_key,
            "status": "uploaded",
            "validation": {
                "passed": validation_result["valid"],
                "detected_type": validation_result["detected_type"],
                "size_mb": round(validation_result["size"] / 1024 / 1024, 2)
            },
            "message": "File uploaded successfully and processing started"
        }
        
    except ValidationError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.post("/upload-multiple")
async def upload_multiple_files(
    request: Request,
    files: Annotated[list[UploadFile], File(description="The files to upload")],
    metadata: Annotated[str, Form(description="JSON metadata for the file upload")] = "{}",
    _: None = Depends(rate_limit_dependency)  # Rate limiting dependency
):
    """
    Upload multiple files to the server and start processing.
    Includes comprehensive security validation and rate limiting.
    """
    try:
        # 1. Preliminary validation for multiple files
        validator = get_file_validator()
        validator.validate_multiple_files(files)  # Will raise ValidationError if limits exceeded
        logger.info(f"Preliminary validation for {len(files)} files passed")
        
        file_service = get_file_service()
        store = get_store()
        uploaded_files = []
        validation_errors = []
        
        for i, file in enumerate(files):
            try:
                # 2. Individual file validation
                validation_result = await validator.validate_file(file)
                logger.info(f"File {i} validation passed: {file.filename}")
                
                # 3. Generate unique object key
                file_uuid = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y/%m/%d")
                object_key = f"{timestamp}/{file_uuid}_{file.filename}"
                
                # 4. Store file
                await file.read()  # Read for storage
                await file.seek(0)  # Reset file pointer
                store.put(file.file, object_key)
                
                # 5. Create database record
                upload_data = UploadCreate(
                    filename=file.filename or f"file_{i}",
                    mime_type=validation_result["detected_type"] or file.content_type or "application/octet-stream",
                    object_key=object_key,
                    status="uploaded",
                    progress=0
                )
                upload_record = file_service.create_upload(upload_data)
                
                # 6. Start processing task
                task_result = celery_app.send_task("process_file", args=[upload_record.id], queue="file_processing")
                logger.info(f"Dispatched task {task_result.id} for upload {upload_record.id} to queue 'file_processing'")
                
                uploaded_files.append({
                    "id": upload_record.id,
                    "filename": file.filename,
                    "size": validation_result["size"],
                    "detected_type": validation_result["detected_type"],
                    "object_key": object_key,
                    "status": "uploaded",
                    "validation": {
                        "passed": validation_result["valid"],
                        "detected_type": validation_result["detected_type"],
                        "size_mb": round(validation_result["size"] / 1024 / 1024, 2)
                    }
                })
                
                logger.info(f"Uploaded and queued file: {file.filename}, ID: {upload_record.id}")
                
            except ValidationError as e:
                validation_errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
                logger.warning(f"File {i} validation failed: {str(e)}")
                continue  # Skip this file, continue with others
        
        # Return results
        result = {
            "files": uploaded_files,
            "total_uploaded": len(uploaded_files),
            "total_attempted": len(files),
            "message": f"Successfully uploaded {len(uploaded_files)} out of {len(files)} files"
        }
        
        if validation_errors:
            result["validation_errors"] = validation_errors
            result["message"] += f". {len(validation_errors)} files failed validation."
        
        return result
        
    except ValidationError as e:
        logger.warning(f"Multiple file validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.get("/upload/{upload_id}")
async def get_upload_status(upload_id: int):
    """
    Get the status of an upload by ID.
    """
    try:
        file_service = get_file_service()
        upload = file_service.get_upload_by_id(upload_id)
        
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {
            "id": upload.id,
            "filename": upload.filename,
            "status": upload.status,
            "progress": upload.progress,
            "created_at": upload.created_at.isoformat(),
            "updated_at": upload.updated_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get upload status: {str(e)}")


@router.get("/uploads")
async def list_uploads(skip: int = 0, limit: int = 100):
    """
    List all uploads with pagination.
    """
    try:
        file_service = get_file_service()
        uploads = file_service.get_uploads(skip=skip, limit=limit)
        
        return {
            "uploads": [
                {
                    "id": upload.id,
                    "filename": upload.filename,
                    "status": upload.status,
                    "progress": upload.progress,
                    "created_at": upload.created_at.isoformat(),
                    "updated_at": upload.updated_at.isoformat()
                }
                for upload in uploads
            ],
            "count": len(uploads)
        }
    except Exception as e:
        logger.error(f"Error listing uploads: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")
    

@router.delete("/upload/{upload_id}")
async def delete_upload(upload_id: int):
    """
    Delete an upload by ID and clean up associated graph data.
    """
    try:
        file_service = get_file_service()
        neo4j_driver = get_neo4j_driver()
        
        # Get the upload record to retrieve the object_key before deletion
        upload = file_service.get_upload_by_id(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        # Initialize deletion stats
        deletion_stats = {"error": "Graph cleanup not attempted"}
        
        # Delete from Neo4j graph first
        try:
            deletion_stats = await neo4j_driver.delete_file_from_graph(upload_id)
            logger.info(f"Deleted graph data for upload {upload_id}: {deletion_stats}")
        except Exception as e:
            logger.warning(f"Error deleting graph data for upload {upload_id}: {e}")
            deletion_stats = {"error": str(e)}
            # Continue with file deletion even if graph cleanup fails
        
        # Delete from storage using the object_key
        storage_cleanup = "success"
        try:
            get_store().delete(upload.object_key)
        except Exception as e:
            logger.warning(f"Error deleting file from storage: {e}")
            storage_cleanup = f"error: {str(e)}"
        
        # Delete from database
        success = file_service.delete_upload(upload_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {
            "message": "Upload deleted successfully",
            "upload_id": upload_id,
            "graph_cleanup": deletion_stats,
            "storage_cleanup": storage_cleanup
        }
    except Exception as e:
        logger.error(f"Error deleting upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete upload: {str(e)}")


@router.get("/graph/schema")
async def get_graph_schema():
    """
    Get the current Neo4j graph schema to understand what was created.
    """
    try:
        driver = get_neo4j_driver()
        schema = driver.inspect_graph_schema()
        return {
            "schema": schema,
            "message": "Graph schema retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting graph schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph schema: {str(e)}")

@router.post("/graph/query")
async def query_graph(query_text: str = Form(...)):
    """
    Query the knowledge graph with a text query.
    """
    try:
        driver = get_neo4j_driver()
        result = driver.query_kg(query_text)
        return {
            "query": query_text,
            "result": result,
            "message": "Query executed successfully"
        }
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query graph: {str(e)}")

@router.post("/graph/retrieve")
async def retrieve_context(
    query_text: str = Form(...), 
    top_k: int = Form(5, description="Number of top results to retrieve")
):
    """
    Directly retrieve context from the knowledge graph without LLM generation.
    Returns structured data perfect for feeding to your own agent or streaming LLM.
    """
    try:
        driver = get_neo4j_driver()
        result = driver.retrieve_context(query_text, top_k=top_k)
        return {
            "query": query_text,
            "retrieval_result": result,
            "message": "Context retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve context: {str(e)}")

@router.get("/graph/context")
async def get_context_for_query(
    query: str,
    top_k: int = 5
):
    """
    Get combined context string for a query - convenient for feeding to streaming LLMs.
    Returns just the text content ready to be used in prompts.
    """
    try:
        driver = get_neo4j_driver()
        context = driver.get_context_for_query(query, top_k=top_k)
        return {
            "query": query,
            "context": context,
            "top_k": top_k,
            "message": "Context retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")

@router.post("/graph/compare")
async def compare_retrieval_methods(query_text: str = Form(...)):
    """
    Compare direct retrieval vs GraphRAG with LLM generation.
    Shows the difference between raw context and LLM-generated responses.
    """
    try:
        driver = get_neo4j_driver()
        
        # Method 1: Direct retrieval (raw context)
        retrieval_result = driver.retrieve_context(query_text, top_k=5)
        
        # Method 2: GraphRAG with LLM generation (traditional RAG)
        try:
            graphrag_result = driver.query_kg(query_text)
            llm_answer = graphrag_result.answer if hasattr(graphrag_result, 'answer') else str(graphrag_result)
        except Exception as e:
            llm_answer = f"LLM generation failed: {str(e)}"
        
        return {
            "query": query_text,
            "direct_retrieval": {
                "method": "Direct context retrieval (no LLM)",
                "combined_context": retrieval_result["combined_context"],
                "total_chunks": retrieval_result["total_items"],
                "use_case": "Feed to your own agent/streaming LLM"
            },
            "graphrag_with_llm": {
                "method": "GraphRAG with LLM generation", 
                "generated_answer": llm_answer,
                "use_case": "Complete RAG pipeline with answer generation"
            },
            "message": "Comparison completed successfully"
        }
    except Exception as e:
        logger.error(f"Error comparing retrieval methods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare methods: {str(e)}")

@router.get("/security/config")
async def get_security_config():
    """
    Get current security configuration settings.
    Useful for frontend to know the limits.
    """
    try:
        config = get_config()
        return {
            "file_limits": {
                "max_file_size_mb": config.max_file_size_mb,
                "max_files_per_upload": config.max_files_per_upload,
                "allowed_extensions": [".jpg", ".jpeg", ".png", ".pdf", ".txt"],
                "allowed_mime_types": [
                    "image/jpeg", "image/png", "application/pdf", "text/plain"
                ]
            },
            "rate_limits": {
                "requests_per_minute": config.rate_limit_per_minute,
                "burst_allowance": config.rate_limit_burst
            },
            "validation_features": {
                "magic_number_checking": True,
                "file_size_validation": True,
                "content_type_validation": True,
                "extension_validation": True,
                "content_safety_checks": True
            },
            "message": "Security configuration retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting security config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get security config: {str(e)}")


@router.get("/graph/file-summary/{upload_id}")
async def get_file_graph_summary(upload_id: int):
    """
    Get a summary of graph data for a specific file upload.
    Useful for understanding what's in the graph before deletion.
    """
    try:
        driver = get_neo4j_driver()
        summary = driver.get_file_graph_summary(upload_id)
        return {
            "upload_id": upload_id,
            "summary": summary,
            "message": "File graph summary retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting file graph summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file graph summary: {str(e)}")

@router.get("/graph/orphaned-nodes")
async def check_orphaned_nodes():
    """
    Check for orphaned graph nodes that have upload_id but no corresponding database record.
    Useful for maintenance and cleanup.
    """
    try:
        driver = get_neo4j_driver()
        cleanup_stats = driver.cleanup_orphaned_nodes()
        return {
            "cleanup_stats": cleanup_stats,
            "message": "Orphaned nodes check completed"
        }
    except Exception as e:
        logger.error(f"Error checking orphaned nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check orphaned nodes: {str(e)}")

@router.get("/graph/stats")
async def get_graph_statistics():
    """
    Get comprehensive statistics about the current graph state.
    """
    try:
        driver = get_neo4j_driver()
        stats = driver.get_graph_stats()
        return {
            "statistics": stats,
            "message": "Graph statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")

@router.post("/graph/context-for-llm")
async def get_context_for_llm(
    query_text: str = Form(...),
    top_k: int = Form(5, description="Number of top results to retrieve"),
    format_type: str = Form("combined", description="Format type: 'combined' for text, 'prompt' for full prompt, 'structured' for detailed data")
):
    """
    Get context formatted specifically for feeding into LLM agents or streaming APIs.
    Different format options for different use cases.
    """
    try:
        driver = get_neo4j_driver()
        
        if format_type == "combined":
            # Just the combined context text
            context = driver.get_context_for_query(query_text, top_k=top_k)
            return {
                "query": query_text,
                "context": context,
                "format": "combined_text",
                "message": "Context text retrieved successfully"
            }
        elif format_type == "prompt":
            # Full prompt ready for LLM
            prompt = driver.get_llm_ready_prompt(query_text, top_k=top_k)
            return {
                "query": query_text,
                "prompt": prompt,
                "format": "llm_ready_prompt",
                "message": "LLM prompt retrieved successfully"
            }
        else:
            # Full structured data
            result = driver.retrieve_context(query_text, top_k=top_k)
            return {
                "query": query_text,
                "structured_result": result,
                "format": "structured_data",
                "message": "Structured context retrieved successfully"
            }
            
    except Exception as e:
        logger.error(f"Error getting context for LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context for LLM: {str(e)}")