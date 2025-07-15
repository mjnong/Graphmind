# Graphmind MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for Graphmind, providing tool-based access to the GraphRAG functionality.

## Overview

The MCP server exposes the same functionality as the REST API endpoints but through the MCP protocol, allowing AI assistants and agents to interact with your GraphRAG system programmatically.

## Available Tools

### File Management
- **get_rag_files** (Resource): Retrieve all uploaded files from the RAG system
- **delete_rag_file**: Delete a file from the RAG system by ID
- **search_database_for_rag_files**: Search for files by name, status, etc.
- **get_upload_status**: Get the status of a specific upload by ID

### Graph Operations
- **query_graphmind**: Query the knowledge graph with LLM generation
- **retrieve_context_from_graphmind**: Retrieve raw context from the graph

### Graph Analytics
- **get_graph_schema**: Get the current Neo4j graph schema
- **get_graph_statistics**: Get comprehensive graph statistics
- **get_file_graph_summary**: Get graph summary for a specific file
- **cleanup_orphaned_nodes**: Cleanup orphaned graph nodes

## Running the MCP Server

### Direct Execution
```bash
cd /path/to/your/project
python -m src.app.mcp.toolbox
```

### Using with Claude Desktop

Add this configuration to your Claude Desktop config file:

```json
{
  "mcpServers": {
    "graphmind": {
      "command": "python",
      "args": ["/path/to/your/project/src/app/mcp/toolbox.py"],
      "env": {
        "PYTHONPATH": "/path/to/your/project"
      }
    }
  }
}
```

## API Equivalence

Each MCP tool corresponds to REST API endpoints:

| MCP Tool | REST Endpoint |
|----------|---------------|
| get_rag_files | GET /files/uploads |
| delete_rag_file | DELETE /files/upload/{id} |
| query_graphmind | POST /files/graph/query |
| retrieve_context_from_graphmind | POST /files/graph/retrieve |
| get_graph_schema | GET /files/graph/schema |
| get_graph_statistics | GET /files/graph/stats |

## Error Handling

All tools return structured responses with either success data or error information:

```python
# Success response
{
    "result": "...",
    "message": "Operation completed successfully"
}

# Error response  
{
    "error": "Description of what went wrong",
    "query": "original_query"  # when applicable
}
```

## Integration Notes

- The MCP server shares the same dependencies and database connections as the main FastAPI application
- File uploads are not directly supported through MCP - use the REST API for uploads
- WebSocket functionality (progress updates) is only available through the REST API
- Rate limiting is not applied to MCP tools (unlike the REST API)
