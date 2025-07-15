from mcp.server.fastmcp import FastMCP
import logging
from typing import Optional
from src.app.deps import get_file_service, get_neo4j_driver

logger = logging.getLogger("graphrag.mcp")

mcp = FastMCP(
    name="Graphmind MCP",
    description="A multi-channel processing application for Graphmind.",
    version="1.0.0"
)

@mcp.resource(
    uri="resource://files/rag",
    name="get_rag_files",
    title="Get RAG Files",
    description="Retrieve files from the RAG system.",
)
def get_rag_files():
    """
    Retrieve all uploaded files from the RAG system.
    
    Returns:
        Dict: List of uploaded files with their metadata.
    """
    try:
        file_service = get_file_service()
        uploads = file_service.get_uploads(skip=0, limit=100)
        
        files_data = []
        for upload in uploads:
            files_data.append({
                "id": upload.id,
                "filename": upload.filename,
                "status": upload.status,
                "progress": upload.progress,
                "created_at": upload.created_at.isoformat(),
                "updated_at": upload.updated_at.isoformat()
            })
        
        return {
            "files": files_data,
            "count": len(files_data),
            "message": "RAG files retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error retrieving RAG files: {e}")
        return {"error": f"Failed to retrieve RAG files: {str(e)}"}

@mcp.tool(
    name="delete_rag_file",
    description="Delete a file from the RAG system.",
    title="Delete RAG File",
)
async def delete_rag_file(file_id: str):
    """
    Deletes a file from the RAG system by its ID.
    
    Args:
        file_id (str): The ID of the file to delete.
    
    Returns:
        Dict: Confirmation message with deletion details.
    """
    try:
        upload_id = int(file_id)
        file_service = get_file_service()
        neo4j_driver = get_neo4j_driver()
        
        # Get the upload record to retrieve the object_key before deletion
        upload = file_service.get_upload_by_id(upload_id)
        if not upload:
            return {"error": "Upload not found", "file_id": file_id}
        
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
        from src.app.storage import get_store
        storage_cleanup = "success"
        try:
            get_store().delete(upload.object_key)
        except Exception as e:
            logger.warning(f"Error deleting file from storage: {e}")
            storage_cleanup = f"error: {str(e)}"
        
        # Delete from database
        success = file_service.delete_upload(upload_id)
        
        if not success:
            return {"error": "Upload not found in database", "file_id": file_id}
        
        return {
            "message": "Upload deleted successfully",
            "upload_id": upload_id,
            "graph_cleanup": deletion_stats,
            "storage_cleanup": storage_cleanup
        }
    except ValueError:
        return {"error": "Invalid file_id format. Must be a number.", "file_id": file_id}
    except Exception as e:
        logger.error(f"Error deleting upload: {e}")
        return {"error": f"Failed to delete upload: {str(e)}", "file_id": file_id}


@mcp.tool(
    name="query_graphmind",
    description="Query the Graphmind system to get a generated response based on context.",
    title="Query Graphmind",
)
async def query_graphmind(query: str):
    """
    Queries the Graphmind system with the given query string.

    Args:
        query (str): The query string to send to the Graphmind system.

    Returns:
        Dict: The response from the Graphmind system.
    """
    try:
        driver = get_neo4j_driver()
        result = driver.query_kg(query)
        return {
            "query": query,
            "result": result,
            "message": "Query executed successfully"
        }
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        return {"error": f"Failed to query graph: {str(e)}", "query": query}

@mcp.tool(
    name="retrieve_context_from_graphmind",
    description="Retrieve context from the Graphmind system.",
    title="Get Context from Graphmind",
)
async def retrieve_context_from_graphmind(query: str, top_k: int = 5):
    """
    Retrieves context from the Graphmind system based on the given query.

    Args:
        query (str): The query string to send to the Graphmind system.
        top_k (int): Number of top results to retrieve (default: 5).

    Returns:
        Dict: The context retrieved from the Graphmind system.
    """
    try:
        driver = get_neo4j_driver()
        result = driver.retrieve_context(query, top_k=top_k)
        return {
            "query": query,
            "retrieval_result": result,
            "message": "Context retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return {"error": f"Failed to retrieve context: {str(e)}", "query": query}

@mcp.tool(
    name="search_database_for_rag_files",
    description="Search the database for RAG files based on a query.",
    title="Search RAG Files in Database",
)
async def search_database_for_rag_files(name: Optional[str] = None, status: Optional[str] = None, limit: int = 100):
    """
    Searches the database for RAG files matching the given criteria.

    Args:
        name (str, optional): The name of the RAG file to search for.
        status (str, optional): The status to filter by (e.g., 'uploaded', 'processing', 'completed').
        limit (int): Maximum number of results to return (default: 100).

    Returns:
        Dict: The search results from the database.
    """
    try:
        file_service = get_file_service()
        uploads = file_service.get_uploads(skip=0, limit=limit)
        
        # Filter results based on criteria
        filtered_uploads = []
        for upload in uploads:
            include = True
            
            if name and name.lower() not in upload.filename.lower():
                include = False
            
            if status and upload.status != status:
                include = False
            
            if include:
                filtered_uploads.append({
                    "id": upload.id,
                    "filename": upload.filename,
                    "status": upload.status,
                    "progress": upload.progress,
                    "mime_type": getattr(upload, 'mime_type', 'unknown'),
                    "created_at": upload.created_at.isoformat(),
                    "updated_at": upload.updated_at.isoformat()
                })
        
        return {
            "uploads": filtered_uploads,
            "count": len(filtered_uploads),
            "total_searched": len(uploads),
            "filters": {
                "name": name,
                "status": status,
                "limit": limit
            },
            "message": f"Found {len(filtered_uploads)} matching files"
        }
    except Exception as e:
        logger.error(f"Error searching database for RAG files: {e}")
        return {"error": f"Failed to search database: {str(e)}"}

@mcp.tool(
    name="get_upload_status",
    description="Get the status of a specific upload by ID.",
    title="Get Upload Status",
)
async def get_upload_status(upload_id: str):
    """
    Get the status of an upload by ID.
    
    Args:
        upload_id (str): The ID of the upload to check.
    
    Returns:
        Dict: The upload status information.
    """
    try:
        upload_id_int = int(upload_id)
        file_service = get_file_service()
        upload = file_service.get_upload_by_id(upload_id_int)
        
        if not upload:
            return {"error": "Upload not found", "upload_id": upload_id}
        
        return {
            "id": upload.id,
            "filename": upload.filename,
            "status": upload.status,
            "progress": upload.progress,
            "created_at": upload.created_at.isoformat(),
            "updated_at": upload.updated_at.isoformat()
        }
    except ValueError:
        return {"error": "Invalid upload_id format. Must be a number.", "upload_id": upload_id}
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        return {"error": f"Failed to get upload status: {str(e)}", "upload_id": upload_id}


@mcp.tool(
    name="get_graph_schema",
    description="Get the current Neo4j graph schema to understand what was created.",
    title="Get Graph Schema",
)
async def get_graph_schema():
    """
    Get the current Neo4j graph schema to understand what was created.
    
    Returns:
        Dict: The graph schema information.
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
        return {"error": f"Failed to get graph schema: {str(e)}"}


@mcp.tool(
    name="get_graph_statistics",
    description="Get comprehensive statistics about the current graph state.",
    title="Get Graph Statistics",
)
async def get_graph_statistics():
    """
    Get comprehensive statistics about the current graph state.
    
    Returns:
        Dict: The graph statistics.
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
        return {"error": f"Failed to get graph statistics: {str(e)}"}


@mcp.tool(
    name="get_file_graph_summary",
    description="Get a summary of graph data for a specific file upload.",
    title="Get File Graph Summary",
)
async def get_file_graph_summary(upload_id: str):
    """
    Get a summary of graph data for a specific file upload.
    Useful for understanding what's in the graph before deletion.
    
    Args:
        upload_id (str): The ID of the upload to get graph summary for.
    
    Returns:
        Dict: The file graph summary.
    """
    try:
        upload_id_int = int(upload_id)
        driver = get_neo4j_driver()
        summary = driver.get_file_graph_summary(upload_id_int)
        return {
            "upload_id": upload_id_int,
            "summary": summary,
            "message": "File graph summary retrieved successfully"
        }
    except ValueError:
        return {"error": "Invalid upload_id format. Must be a number.", "upload_id": upload_id}
    except Exception as e:
        logger.error(f"Error getting file graph summary: {e}")
        return {"error": f"Failed to get file graph summary: {str(e)}", "upload_id": upload_id}


@mcp.tool(
    name="cleanup_orphaned_nodes",
    description="Check for and cleanup orphaned graph nodes that have upload_id but no corresponding database record.",
    title="Cleanup Orphaned Nodes",
)
async def cleanup_orphaned_nodes():
    """
    Check for orphaned graph nodes that have upload_id but no corresponding database record.
    Useful for maintenance and cleanup.
    
    Returns:
        Dict: Cleanup statistics.
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
        return {"error": f"Failed to check orphaned nodes: {str(e)}"}
