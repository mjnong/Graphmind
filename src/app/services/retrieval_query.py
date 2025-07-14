"""
Neo4j Retrieval Query for GraphRAG

This query is designed to work with the actual Neo4j schema created by the SlideTextSplitter.
It retrieves chunk nodes and their connected entities to provide context for RAG applications.

New Direct Retrieval Capabilities:
- `/files/graph/retrieve` (POST): Get detailed retrieval results with metadata
- `/files/graph/context` (GET): Get simple combined context string for streaming LLMs
- `Neo4jDriver.retrieve_context()`: Direct access to retriever without LLM generation
- `Neo4jDriver.get_context_for_query()`: Get just the context string

These methods allow you to:
1. Retrieve raw context without LLM generation
2. Feed context to your own agents or streaming LLMs
3. Have full control over the prompt and response generation

Example usage for direct retrieval:

1. Using the API endpoints:
   GET /files/graph/context?query=your_question&top_k=5
   POST /files/graph/retrieve with form data: query_text=your_question&top_k=5

2. Using the driver directly:
   from src.app.services.neo4j_driver import get_driver
   driver = get_driver()

   # Get detailed results
   results = driver.retrieve_context("your question", top_k=5)

   # Get just the context string for your own LLM
   context = driver.get_context_for_query("your question", top_k=5)
   your_prompt = f"Context: {context}\n\nQuestion: your question\n\nAnswer:"
   # Feed to your streaming LLM or agent
"""

QUERY = """
// Simplified query based on actual schema (from Neo4j warnings)
// We know 'text' exists (not in warnings), other properties don't
WITH node AS chunk

// Use only properties that actually exist
WITH chunk,
     coalesce(chunk.text, 'No text found') AS chunk_text,
     coalesce(chunk.local_context, '') AS context_info

// Get connected entities (1 hop) - be flexible about relationship directions
OPTIONAL MATCH (chunk)-[r1]->(entity1)
OPTIONAL MATCH (chunk)<-[r2]-(entity2) 
WITH chunk, chunk_text, context_info,
     collect(DISTINCT {
         relationship_type: type(r1),
         direction: 'outgoing',
         entity_name: coalesce(entity1.name, entity1.id, 'Entity'),
         entity_labels: labels(entity1)
     }) + 
     collect(DISTINCT {
         relationship_type: type(r2), 
         direction: 'incoming',
         entity_name: coalesce(entity2.name, entity2.id, 'Entity'),
         entity_labels: labels(entity2)
     }) AS all_connections

// Build simple result
WITH chunk_text +
     CASE WHEN context_info <> '' 
          THEN '\n\n[Context: ' + context_info + ']' 
          ELSE '' 
     END +
     CASE WHEN size([c IN all_connections WHERE c.relationship_type IS NOT NULL]) > 0
          THEN '\n\n[Connected Entities]\n' +
               reduce(s = '', c IN [conn IN all_connections WHERE conn.relationship_type IS NOT NULL] |
                 s + '• ' + c.entity_name + 
                 CASE WHEN size(c.entity_labels) > 0 THEN ' (' + head(c.entity_labels) + ')' ELSE '' END +
                 ' via ' + c.relationship_type + 
                 ' (' + c.direction + ')\n'
               )
          ELSE ''
     END AS result_text

RETURN result_text AS info
"""