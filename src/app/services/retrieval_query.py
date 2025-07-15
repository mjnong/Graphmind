"""
Simplified Neo4j Retrieval Query for GraphRAG - Medical & Scientific Applications

This query is designed for DEEP CONTEXTUAL UNDERSTANDING with multi-hop exploration,
returning clean, structured data that's ready for LLM consumption without complex parsing.

⚠️ IMPORTANT: This query works with nodes AND scores that have already been filtered by vector/fulltext search.
The 'node' and 'score' parameters come from the retriever's search results, not all nodes in the database.

🔥 SCORE HANDLING: The query properly preserves and returns similarity scores from the HybridCypherRetriever,
enabling accurate similarity-based filtering for quality control.

🏥 MEDICAL-GRADE FEATURES:
- ✅ 3-hop relationship exploration for deep medical insights
- ✅ Multi-layered entity connections (Direct → Extended → Deep)
- ✅ Comprehensive context building for complex medical knowledge
- ✅ Robust property handling (only uses existing Neo4j GraphRAG properties)
- ✅ Clean structured output ready for LLM processing

Key Medical Use Cases:
- 🩺 Drug-Disease-Gene interactions across multiple papers
- 🧬 Protein-pathway-phenotype relationships in research literature
- 🏥 Clinical symptom-diagnosis-treatment knowledge networks
- 📊 Research finding connections across studies and publications
- 🔬 Scientific concept relationships in complex domains

Advanced Features:
- 📄 Clean document content without formatting overhead
- 🔍 Local context information when available  
- 🔗 Direct entity relationships (1-hop) as structured lists
- 🌐 Extended knowledge network (2-hop) for broader context
- 🧠 Deep knowledge insights (3-hop) for comprehensive understanding
- Structured output perfect for LLM understanding and post-processing
- 🎯 Accurate similarity scores preserved from HybridCypherRetriever
- 🚀 Optimized for performance with minimal string concatenation
"""

QUERY = """
// Advanced Medical-Grade GraphRAG retrieval query with multi-hop exploration
// Designed for deep contextual understanding in medical/scientific applications
// The 'node' and 'score' parameters come from the HybridCypherRetriever's search results

WITH node AS chunk, score AS similarity_score

// 1) Get the chunk's text content and metadata (only for relevant chunks)
WITH chunk, similarity_score,
     coalesce(chunk.document_title, 'No title available') AS document_title,
     coalesce(chunk.excerpt_keywords, '') AS excerpt_keywords,
     coalesce(chunk.questions_this_excerpt_can_answer, '') AS questions_this_excerpt_can_answer,
     coalesce(chunk.section_summary, 'No text available') AS section_summary,
     coalesce(chunk.text, 'No text available') AS chunk_text,
     coalesce(chunk.local_context, '') AS local_context,
     coalesce(chunk.filename, coalesce(chunk.file_path, 'Unknown')) AS page_info,
     coalesce(chunk.upload_id, 'Unknown') AS file_source

// 2) Only proceed if we have actual text content (filter out empty/irrelevant results)
WHERE chunk_text <> 'No text available' AND trim(chunk_text) <> ''

// 3) DEEP EXPLORATION: Multi-hop entity relationships for comprehensive context
// Direct relationships (1st hop)
OPTIONAL MATCH (chunk)-[r1]-(entity1:__Entity__)
WHERE entity1.name IS NOT NULL AND trim(entity1.name) <> ''

// Extended relationships (2nd hop) - Critical for medical knowledge discovery
OPTIONAL MATCH (entity1)-[r2]-(entity2:__Entity__)
WHERE entity1 <> entity2 AND entity2 <> chunk 
  AND entity2.name IS NOT NULL AND trim(entity2.name) <> ''

// Tertiary relationships (3rd hop) - For very deep medical connections
OPTIONAL MATCH (entity2)-[r3]-(entity3:__Entity__)
WHERE entity3 <> entity2 AND entity3 <> entity1 AND entity3 <> chunk
  AND entity3.name IS NOT NULL AND trim(entity3.name) <> ''
  AND size([e IN [entity1, entity2] WHERE e.name = entity3.name]) = 0

// 4) Collect relationships by depth for medical context building
WITH chunk, similarity_score, chunk_text, document_title, excerpt_keywords, 
     questions_this_excerpt_can_answer, section_summary, local_context, page_info, file_source,
     
     // Direct relationships (immediate connections)
     collect(DISTINCT {
         entity: entity1.name,
         relationship: type(r1),
         direction: CASE 
             WHEN startNode(r1) = chunk THEN 'outgoing'
             ELSE 'incoming'
         END,
         hop: 1
     }) AS direct_relations,
     
     // Second-hop relationships (extended medical context)
     collect(DISTINCT {
         entity1: entity1.name,
         entity2: entity2.name,
         relationship: type(r2),
         hop: 2
     }) AS extended_relations,
     
     // Third-hop relationships (deep medical insights)
     collect(DISTINCT {
         entity2: entity2.name,
         entity3: entity3.name,
         relationship: type(r3),
         hop: 3
     }) AS deep_relations

// 5) Filter out null relationships and prepare clean data

// 6) Return clean, structured content ready for LLM consumption
RETURN 
    // Main document content
    chunk_text AS info,
    
    // Preserved similarity score
    similarity_score AS score,
    
    // Aggregated metadata for enhanced context
    {
        document_title: document_title,
        excerpt_keywords: excerpt_keywords,
        questions_this_excerpt_can_answer: questions_this_excerpt_can_answer,
        section_summary: section_summary,
        local_context: local_context,
        source_file: page_info,
        upload_id: file_source,
        content_type: CASE 
            WHEN trim(local_context) <> '' THEN 'contextualized'
            ELSE 'standalone'
        END,
        has_keywords: CASE 
            WHEN trim(excerpt_keywords) <> '' THEN true
            ELSE false
        END,
        has_questions: CASE 
            WHEN trim(questions_this_excerpt_can_answer) <> '' THEN true
            ELSE false
        END
    } AS metadata,
    
    // Relationship data as structured lists
    [rel IN direct_relations WHERE rel.entity IS NOT NULL AND trim(rel.entity) <> '' | 
     rel.entity + ' (' + rel.relationship + 
     CASE WHEN rel.direction = 'outgoing' THEN ' →)' ELSE ' ←)' END
    ] AS direct_relations,
    
    [rel IN extended_relations WHERE rel.entity1 IS NOT NULL AND rel.entity2 IS NOT NULL AND trim(rel.entity1) <> '' AND trim(rel.entity2) <> '' |
     rel.entity1 + ' → ' + rel.entity2 + ' (' + rel.relationship + ')'
    ] AS extended_relations,
    
    [rel IN deep_relations WHERE rel.entity2 IS NOT NULL AND rel.entity3 IS NOT NULL AND trim(rel.entity2) <> '' AND trim(rel.entity3) <> '' |
     rel.entity2 + ' → ' + rel.entity3 + ' (' + rel.relationship + ')'
    ] AS deep_relations
"""

# Advanced fallback query without APOC functions but with deep exploration
QUERY_NO_APOC = """
// Medical-grade retrieval query without APOC dependencies
// Works with HybridCypherRetriever's pre-filtered nodes
// Includes multi-hop exploration for comprehensive medical context

WITH node AS chunk, score AS similarity_score

// 1) Get the chunk's text content and metadata (only for relevant chunks)
WITH chunk, similarity_score,
     coalesce(chunk.document_title, 'No title available') AS document_title,
     coalesce(chunk.excerpt_keywords, '') AS excerpt_keywords,
     coalesce(chunk.questions_this_excerpt_can_answer, '') AS questions_this_excerpt_can_answer,
     coalesce(chunk.section_summary, 'No text available') AS section_summary,
     coalesce(chunk.text, 'No text available') AS chunk_text,
     coalesce(chunk.local_context, '') AS local_context,
     coalesce(chunk.filename, coalesce(chunk.file_path, 'Unknown')) AS page_info,
     coalesce(chunk.upload_id, 'Unknown') AS file_source

// 2) Only proceed if we have actual text content
WHERE chunk_text <> 'No text available' AND trim(chunk_text) <> ''

// 3) Multi-hop entity relationships for medical context
OPTIONAL MATCH (chunk)-[r1]-(entity1:__Entity__)
WHERE entity1.name IS NOT NULL AND trim(entity1.name) <> ''

OPTIONAL MATCH (entity1)-[r2]-(entity2:__Entity__)
WHERE entity1 <> entity2 AND entity2 <> chunk 
  AND entity2.name IS NOT NULL AND trim(entity2.name) <> ''

// 4) Build clean result for medical applications
WITH chunk_text, document_title, excerpt_keywords, questions_this_excerpt_can_answer, 
     section_summary, local_context, page_info, file_source, similarity_score,
     collect(DISTINCT {
         entity: entity1.name,
         relationship: type(r1),
         direction: CASE 
             WHEN startNode(r1) = chunk THEN '→'
             ELSE '←'
         END,
         hop: 1
     }) AS direct_relations,
     collect(DISTINCT {
         entity1: entity1.name,
         entity2: entity2.name,
         relationship: type(r2),
         hop: 2
     }) AS extended_relations

// 5) Return clean, structured data
RETURN 
    chunk_text AS info,
    similarity_score AS score,
    
    // Aggregated metadata for enhanced context
    {
        document_title: document_title,
        excerpt_keywords: excerpt_keywords,
        questions_this_excerpt_can_answer: questions_this_excerpt_can_answer,
        section_summary: section_summary,
        local_context: local_context,
        source_file: page_info,
        upload_id: file_source,
        content_type: CASE 
            WHEN trim(local_context) <> '' THEN 'contextualized'
            ELSE 'standalone'
        END,
        has_keywords: CASE 
            WHEN trim(excerpt_keywords) <> '' THEN true
            ELSE false
        END,
        has_questions: CASE 
            WHEN trim(questions_this_excerpt_can_answer) <> '' THEN true
            ELSE false
        END
    } AS metadata,
    
    [rel IN direct_relations WHERE rel.entity IS NOT NULL | 
     rel.entity + ' (' + rel.relationship + ' ' + rel.direction + ')'
    ] AS direct_relations,
    
    [rel IN extended_relations WHERE rel.entity1 IS NOT NULL AND rel.entity2 IS NOT NULL |
     rel.entity1 + ' → ' + rel.entity2 + ' (' + rel.relationship + ')'
    ] AS extended_relations
"""