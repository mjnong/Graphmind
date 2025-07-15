import logging
from functools import lru_cache
from typing import List

import neo4j
import neo4j.exceptions
from llama_index.core import Document
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.slide import SlideNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever

from src.app.configs.config import get_config
from src.app.services.retrieval_query import QUERY
from src.app.services.template import INPUTS, TEMPLATE

logger = logging.getLogger("graphrag.neo4j_driver")

# Global driver instance for singleton pattern
_driver_instance = None


class SlideTextSplitterAdapter(TextSplitter):
    def __init__(
        self, slide_parser: SlideNodeParser, file_metadata: dict | None = None
    ):
        self._slide_parser = slide_parser
        self._file_metadata = file_metadata or {}
        self._transformations = [
            SentenceSplitter(),
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=3),
            SummaryExtractor(summaries=["prev", "self"]),
            KeywordExtractor(keywords=10),
        ]
        self._ingestion_pipeline = IngestionPipeline(transformations=self._transformations)

    async def run(self, text: str) -> TextChunks:
        # SlideNodeParser only works on Document objects
        doc = Document(text=text)
        logger.info(
            f"SlideTextSplitter: Processing document with {len(text)} characters"
        )

        # NOTE: get_nodes_from_documents is blocking; run it in a threadpool
        # so we don't block the event loop.
        from anyio import to_thread

        nodes = await to_thread.run_sync(
            self._slide_parser.get_nodes_from_documents, [doc]
        )

        logger.info(f"SlideTextSplitter: Created {len(nodes)} nodes from document")

        nodes = await self._ingestion_pipeline.arun(nodes=nodes)
        chunks: List[TextChunk] = []
        for idx, node in enumerate(nodes):
            node_text = getattr(node, "text", "") or ""
            node_metadata = getattr(node, "metadata", {}) or {}

            logger.info(f"Node {idx}:")
            logger.info(f"  Text length: {len(node_text)}")
            logger.info(f"  Text preview: {node_text[:100]}...")
            logger.info(f"  Node metadata keys: {list(node_metadata.keys())}")
            logger.info(f"  Node metadata: {node_metadata}")

            chunk = TextChunk(
                text=node_text,  # type: ignore
                index=idx,
                metadata={
                    "local_context": node_metadata.get("local_context", ""),
                    **node_metadata,  # keep anything else (page, doc_id…)
                    **self._file_metadata,  # Add file tracking metadata
                },
            )

            logger.info(
                f"  Created TextChunk with metadata keys: {list(chunk.metadata.keys() if chunk.metadata else [])}"
            )
            logger.info(f"  TextChunk metadata: {chunk.metadata}")
            chunks.append(chunk)

        logger.info(f"SlideTextSplitter: Returning {len(chunks)} TextChunks")
        return TextChunks(chunks=chunks)

    def split_text(self, text: str) -> TextChunks:
        # Synchronous version for use in Celery workers
        # SlideNodeParser only works on Document objects
        doc = Document(text=text)
        logger.info(
            f"SlideTextSplitter (sync): Processing document with {len(text)} characters"
        )

        # Call the synchronous method directly since we're in a worker context
        nodes = self._slide_parser.get_nodes_from_documents([doc])
        logger.info(
            f"SlideTextSplitter (sync): Created {len(nodes)} nodes from document"
        )

        chunks: List[TextChunk] = []
        for idx, node in enumerate(nodes):
            node_text = getattr(node, "text", "") or ""
            node_metadata = getattr(node, "metadata", {}) or {}

            logger.info(f"Node {idx} (sync):")
            logger.info(f"  Text length: {len(node_text)}")
            logger.info(f"  Text preview: {node_text[:100]}...")
            logger.info(f"  Node metadata keys: {list(node_metadata.keys())}")
            logger.info(f"  Node metadata: {node_metadata}")

            chunk = TextChunk(
                text=node_text,  # type: ignore
                index=idx,
                metadata={
                    "local_context": node_metadata.get("local_context", ""),
                    **node_metadata,  # keep anything else (page, doc_id…)
                    **self._file_metadata,  # Add file tracking metadata
                },
            )

            logger.info(
                f"  Created TextChunk with metadata keys: {list(chunk.metadata.keys() if chunk.metadata else [])}"
            )
            logger.info(f"  TextChunk metadata: {chunk.metadata}")
            chunks.append(chunk)

        logger.info(f"SlideTextSplitter (sync): Returning {len(chunks)} TextChunks")
        return TextChunks(chunks=chunks)


class Neo4jDriver:
    def __init__(self):
        cfg = get_config()
        self._cfg = cfg
        self._driver = None
        self._pipeline = None
        self._retriver = None
        self._graphrag = None
        self._llm = None
        self._embedder = None
        self._slide_llm = None
        self._token_counter = None
        self._index_name = "text_embeddings"
        self._connection_lock = None  # Will be set to asyncio.Lock when needed

    def _get_connection_lock(self):
        """Get or create an asyncio lock for thread-safe connection management"""
        import asyncio

        if self._connection_lock is None:
            try:
                self._connection_lock = asyncio.Lock()
            except RuntimeError:
                # No event loop running, will use without lock
                pass
        return self._connection_lock

    async def _ensure_connected_async(self):
        """Async version of _ensure_connected for proper concurrency handling"""
        lock = self._get_connection_lock()
        if lock:
            async with lock:
                self._ensure_connected()
        else:
            self._ensure_connected()

    def _ensure_connected(self):
        """Ensure Neo4j connection is established"""
        if self._driver is None:
            try:
                self._driver = neo4j.GraphDatabase.driver(
                    self._cfg.neo4j_uri,
                    auth=(self._cfg.neo4j_username, self._cfg.neo4j_password),
                )
                # Test the connection
                with self._driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Successfully connected to Neo4j")
                self._initialize_shared_components()
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                if self._driver:
                    self._driver.close()
                    self._driver = None
                raise

    async def _test_openai_connectivity(self):
        """Test OpenAI API connectivity before initializing components"""
        try:
            import aiohttp

            logger.info("Testing OpenAI API connectivity...")

            # Test basic HTTP connectivity to OpenAI
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    "Authorization": f"Bearer {self._cfg.openai_api_key}",
                    "Content-Type": "application/json",
                }

                # Test with a simple models list request
                async with session.get(
                    "https://api.openai.com/v1/models", headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info("✓ OpenAI API connectivity test successful")
                        return True
                    else:
                        logger.error(f"✗ OpenAI API returned status {response.status}")
                        response_text = await response.text()
                        logger.error(f"Response: {response_text}")
                        return False

        except Exception as e:
            logger.error(f"✗ OpenAI API connectivity test failed: {e}")
            logger.error(
                "This may indicate network connectivity issues from the container"
            )

            # Provide helpful debugging information
            logger.info("Debugging information:")
            logger.info("1. Check if the container has internet access")
            logger.info("2. Verify DNS resolution for api.openai.com")
            logger.info("3. Check if there are firewall/proxy restrictions")
            logger.info("4. Verify the OpenAI API key is valid")

            return False

    def _initialize_shared_components(self):
        """Initialize shared components that can be reused across requests"""
        if self._llm is not None or self._driver is None:
            return  # Already initialized or driver not available

        self._llm = OpenAILLM(
            model_name=self._cfg.openai_model,
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0,  # turning temperature down for more deterministic results
            },
        )
        self._embedder = OpenAIEmbeddings(model=self._cfg.openai_embedding_model)
        self._token_counter = TokenCounter()

        # Create the llm instance for SlideNodeParser
        self._slide_llm = OpenAI(
            model=self._cfg.openai_model,
            api_key=self._cfg.openai_api_key,
        )

        # Log configuration details for debugging (but hide sensitive data)
        logger.info("Initializing Neo4j GraphRAG shared components...")
        logger.info(f"OpenAI model: {self._cfg.openai_model}")
        logger.info(f"OpenAI embedding model: {self._cfg.openai_embedding_model}")
        logger.info(
            f"OpenAI API key configured: {'Yes' if self._cfg.openai_api_key and self._cfg.openai_api_key != 'your_openai_api_key' else 'No (using placeholder)'}"
        )

        if (
            not self._cfg.openai_api_key
            or self._cfg.openai_api_key == "your_openai_api_key"
        ):
            logger.error(
                "OpenAI API key is not properly configured! Please set OPENAI_API_KEY environment variable."
            )
            raise ValueError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY environment variable."
            )

        try:
            create_vector_index(
                driver=self._driver,  # At this point _driver is guaranteed to be not None
                name=self._index_name,
                label="Chunk",  # all your chunks share this label
                embedding_property="embedding",
                dimensions=1536,  # must match your model
                similarity_fn="cosine",
            )
        except neo4j.exceptions.ClientError as e:
            logger.warning(f"Warning creating vector index (might already exist): {e}")

        # Create fulltext index for better text search
        try:
            create_fulltext_index(
                driver=self._driver,
                name="fulltext_chunks",
                label="Chunk",
                node_properties=["text"],
            )
        except neo4j.exceptions.ClientError as e:
            logger.warning(
                f"Warning creating fulltext index (might already exist): {e}"
            )

        # Initialize the retriever for querying (shared across requests)
        self._retriver = HybridCypherRetriever(
            driver=self._driver,
            vector_index_name=self._index_name,
            fulltext_index_name="fulltext_chunks",
            retrieval_query=QUERY,
            embedder=self._embedder,
        )

        # Initialize GraphRAG for complete RAG pipeline (shared across requests)
        self._graphrag = GraphRAG(
            retriever=self._retriver,
            llm=OpenAILLM(
                model_name=self._cfg.openai_model,
            ),
            prompt_template=RagTemplate(template=TEMPLATE, expected_inputs=INPUTS),
        )

    def _create_pipeline_for_request(self, file_metadata: dict | None = None):
        """
        Create a new pipeline instance for a specific request with optional metadata.
        This ensures metadata isolation between concurrent requests.

        :param file_metadata: Optional metadata to inject into the pipeline
        :return: New pipeline instance with request-specific metadata
        """
        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

        # Ensure shared components are initialized
        if self._llm is None or self._embedder is None:
            self._initialize_shared_components()

        # Type assertions to ensure components are not None
        assert self._llm is not None, "LLM not initialized"
        assert self._driver is not None, "Driver not initialized"
        assert self._embedder is not None, "Embedder not initialized"
        assert self._slide_llm is not None, "Slide LLM not initialized"
        assert self._token_counter is not None, "Token counter not initialized"

        logger.info(f"Creating new pipeline for request with metadata: {file_metadata}")

        # Create a new pipeline with request-specific metadata
        pipeline = SimpleKGPipeline(
            llm=self._llm,  # Shared LLM
            driver=self._driver,  # Shared driver connection
            text_splitter=SlideTextSplitterAdapter(
                SlideNodeParser.from_defaults(
                    llm=self._slide_llm,  # Shared slide LLM
                    token_counter=self._token_counter,  # Shared token counter
                    chunk_size=200,
                    window_size=5,
                ),
                file_metadata=file_metadata,  # Request-specific metadata
            ),
            embedder=self._embedder,  # Shared embedder
            from_pdf=True,
        )

        return pipeline

    def _init_pipeline_with_metadata(self, file_metadata: dict):
        """Initialize pipeline with file metadata for tracking"""
        logger.info(f"Creating pipeline with file metadata: {file_metadata}")
        # Don't store the pipeline as instance variable to avoid conflicts
        return self._create_pipeline_for_request(file_metadata)

    def _init_pipeline(self):
        """Initialize pipeline without file metadata"""
        logger.info("Creating pipeline without file metadata")
        # Don't store the pipeline as instance variable to avoid conflicts
        return self._create_pipeline_for_request()

    async def build_kg_from_pdf(
        self,
        pdf_path: str,
        upload_id: int | None = None,
        file_metadata: dict | None = None,
    ):
        """
        Build a knowledge graph from a PDF file.
        :param pdf_path: Path to the PDF file.
        :param upload_id: Database upload ID for tracking
        :param file_metadata: Optional metadata to attach to the nodes.
        :return: The ID of the created knowledge graph.
        """
        logger.info(f"Starting knowledge graph building for: {pdf_path}")
        logger.info(f"Upload ID: {upload_id}, File metadata: {file_metadata}")

        self._ensure_connected()

        # Create pipeline with file metadata if provided
        if file_metadata or upload_id:
            metadata_to_inject = file_metadata.copy() if file_metadata else {}
            if upload_id:
                metadata_to_inject.update(
                    {"upload_id": upload_id, "file_path": pdf_path}
                )
            logger.info(f"Injecting metadata into graph nodes: {metadata_to_inject}")
            pipeline = self._create_pipeline_for_request(metadata_to_inject)
        else:
            pipeline = self._create_pipeline_for_request()

        if pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        # Test OpenAI connectivity before processing
        connectivity_ok = await self._test_openai_connectivity()
        if not connectivity_ok:
            raise RuntimeError(
                "OpenAI API connectivity test failed. Cannot proceed with knowledge graph building."
            )

        try:
            # Log before processing
            logger.info("Pipeline is ready, starting async processing...")
            logger.info("=" * 50)
            logger.info("STARTING SLIDE TEXT SPLITTER PROCESSING")
            logger.info("=" * 50)

            # Use the async run method for proper async/await handling
            result = await pipeline.run_async(pdf_path)

            logger.info("=" * 50)
            logger.info("SLIDE TEXT SPLITTER PROCESSING COMPLETED")
            logger.info("=" * 50)

            # Log the result
            logger.info(f"Pipeline processing completed. Result type: {type(result)}")
            if hasattr(result, "__dict__"):
                logger.info(f"Result attributes: {vars(result)}")
            else:
                logger.info(f"Result value: {result}")

            # Get some statistics about what was created
            await self._log_graph_statistics()

            return result

        except Exception as e:
            logger.error(f"Error in knowledge graph building: {e}", exc_info=True)
            raise

    async def delete_file_from_graph(self, upload_id: int) -> dict:
        """
        Delete all graph nodes and relationships associated with a specific file upload.

        :param upload_id: The database upload ID to delete from the graph
        :return: Dictionary with deletion statistics
        """
        logger.info(f"Deleting graph data for upload_id: {upload_id}")
        await self._ensure_connected_async()

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        deletion_stats = {
            "upload_id": upload_id,
            "nodes_deleted": 0,
            "relationships_deleted": 0,
            "chunks_deleted": 0,
            "entities_deleted": 0,
            "success": False,
        }

        try:
            with self._driver.session() as session:
                # First, count what we're about to delete for logging
                count_query = """
                MATCH (n) WHERE n.upload_id = $upload_id
                OPTIONAL MATCH (n)-[r]-()
                RETURN 
                    count(DISTINCT n) as node_count,
                    count(DISTINCT r) as rel_count,
                    count(DISTINCT CASE WHEN any(label IN labels(n) WHERE toLower(label) CONTAINS 'chunk') THEN n END) as chunk_count,
                    count(DISTINCT CASE WHEN any(label IN labels(n) WHERE toLower(label) CONTAINS 'entity') THEN n END) as entity_count
                """

                result = session.run(count_query, upload_id=upload_id)
                stats = result.single()

                if stats:
                    logger.info(
                        f"Found {stats['node_count']} nodes, {stats['rel_count']} relationships, "
                        f"{stats['chunk_count']} chunks, {stats['entity_count']} entities to delete"
                    )

                # Phase 1: Collect all connected entities before deleting main nodes
                connected_entities_query = """
                MATCH (main_node) WHERE main_node.upload_id = $upload_id
                OPTIONAL MATCH (main_node)-[r]-(connected_entity:__Entity__)
                WITH collect(DISTINCT connected_entity) as entities_to_check, 
                     collect(DISTINCT main_node) as main_nodes,
                     collect(DISTINCT r) as all_rels
                RETURN entities_to_check, main_nodes, all_rels
                """
                
                result = session.run(connected_entities_query, upload_id=upload_id)
                connection_data = result.single()
                connected_entities = connection_data["entities_to_check"] if connection_data else []
                
                logger.info(f"Found {len(connected_entities)} connected __Entity__ nodes to check for orphaning")

                # Phase 2: Delete relationships first to avoid constraint issues
                rel_delete_query = """
                MATCH (n)-[r]-(m) 
                WHERE n.upload_id = $upload_id OR m.upload_id = $upload_id
                DELETE r
                RETURN count(r) as deleted_rels
                """

                result = session.run(rel_delete_query, upload_id=upload_id)
                rel_result = result.single()
                deletion_stats["relationships_deleted"] = (
                    rel_result["deleted_rels"] if rel_result else 0
                )

                # Phase 3: Delete all nodes with the upload_id (including __KGBuilder__ and other system nodes)
                node_delete_query = """
                MATCH (n) WHERE n.upload_id = $upload_id
                WITH n, labels(n) as node_labels
                DELETE n
                RETURN count(n) as deleted_nodes, 
                       count(CASE WHEN any(label IN node_labels WHERE toLower(label) CONTAINS 'chunk') THEN 1 END) as deleted_chunks,
                       count(CASE WHEN any(label IN node_labels WHERE toLower(label) CONTAINS 'entity') THEN 1 END) as deleted_entities
                """

                result = session.run(node_delete_query, upload_id=upload_id)
                node_result = result.single()

                if node_result:
                    deletion_stats["nodes_deleted"] = node_result["deleted_nodes"]
                    deletion_stats["chunks_deleted"] = node_result["deleted_chunks"]
                    deletion_stats["entities_deleted"] = node_result["deleted_entities"]

                # Phase 4: Clean up orphaned __Entity__ nodes that have no remaining connections
                orphaned_entity_cleanup_query = """
                MATCH (entity:__Entity__)
                WHERE NOT (entity)--() // No connections at all
                DELETE entity
                RETURN count(entity) as orphaned_entities_deleted
                """
                
                result = session.run(orphaned_entity_cleanup_query)
                orphaned_result = result.single()
                deletion_stats["orphaned_entities_deleted"] = orphaned_result["orphaned_entities_deleted"] if orphaned_result else 0

                # Phase 5: Additional cleanup: Remove any __KGBuilder__ nodes that might reference this upload
                kg_builder_cleanup_query = """
                MATCH (kb:__KGBuilder__) 
                WHERE kb.file_path CONTAINS $upload_id_str OR kb.upload_id = $upload_id
                DELETE kb
                RETURN count(kb) as deleted_kg_builders
                """
                
                result = session.run(kg_builder_cleanup_query, 
                                   upload_id=upload_id, 
                                   upload_id_str=str(upload_id))
                kb_result = result.single()
                deletion_stats["kg_builder_nodes_deleted"] = kb_result["deleted_kg_builders"] if kb_result else 0

                # Phase 6: Final cleanup: Remove any remaining nodes that might have been missed
                final_cleanup_query = """
                MATCH (n) 
                WHERE (n.file_path IS NOT NULL AND n.file_path CONTAINS $upload_id_str) 
                   OR (n.document_id IS NOT NULL AND n.document_id CONTAINS $upload_id_str)
                   OR (n.source IS NOT NULL AND n.source CONTAINS $upload_id_str)
                DELETE n
                RETURN count(n) as final_cleanup_count
                """
                
                result = session.run(final_cleanup_query, upload_id_str=str(upload_id))
                final_result = result.single()
                deletion_stats["final_cleanup_count"] = final_result["final_cleanup_count"] if final_result else 0

                deletion_stats["success"] = True

                logger.info(
                    f"Successfully deleted graph data for upload_id {upload_id}: "
                    f"{deletion_stats['nodes_deleted']} nodes, "
                    f"{deletion_stats['relationships_deleted']} relationships"
                )

        except Exception as e:
            logger.error(
                f"Error deleting graph data for upload_id {upload_id}: {e}",
                exc_info=True,
            )
            deletion_stats["error"] = str(e)
            raise

        return deletion_stats

    def get_file_graph_summary(self, upload_id: int) -> dict:
        """
        Get a summary of graph data associated with a specific file upload.

        :param upload_id: The database upload ID to check
        :return: Dictionary with graph summary statistics
        """
        logger.info(f"Getting graph summary for upload_id: {upload_id}")
        self._ensure_connected()

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        summary = {
            "upload_id": upload_id,
            "exists": False,
            "total_nodes": 0,
            "total_relationships": 0,
            "node_types": {},
            "sample_properties": {},
        }

        try:
            with self._driver.session() as session:
                # Check if any nodes exist for this upload_id
                check_query = """
                MATCH (n) WHERE n.upload_id = $upload_id
                OPTIONAL MATCH (n)-[r]-()
                RETURN 
                    count(DISTINCT n) as node_count,
                    count(DISTINCT r) as rel_count,
                    collect(DISTINCT labels(n)) as all_labels,
                    collect(DISTINCT keys(n))[0..10] as sample_properties
                """

                result = session.run(check_query, upload_id=upload_id)
                data = result.single()

                if data and data["node_count"] > 0:
                    summary["exists"] = True
                    summary["total_nodes"] = data["node_count"]
                    summary["total_relationships"] = data["rel_count"]

                    # Count nodes by type
                    for label_set in data["all_labels"]:
                        for label in label_set:
                            if label in summary["node_types"]:
                                summary["node_types"][label] += 1
                            else:
                                summary["node_types"][label] = 1

                    summary["sample_properties"] = data["sample_properties"]

                logger.info(f"Graph summary for upload_id {upload_id}: {summary}")

        except Exception as e:
            logger.error(f"Error getting graph summary for upload_id {upload_id}: {e}")
            summary["error"] = str(e)

        return summary

    def cleanup_orphaned_nodes(self) -> dict:
        """
        Find and optionally clean up graph nodes that have upload_id but no corresponding database record.
        This is useful for maintenance and finding orphaned data.

        :return: Dictionary with cleanup statistics
        """
        logger.info("Checking for orphaned graph nodes...")
        self._ensure_connected()

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        cleanup_stats = {
            "orphaned_nodes_found": 0,
            "unique_upload_ids": [],
            "node_types": {},
            "cleanup_performed": False,
        }

        try:
            with self._driver.session() as session:
                # Find all unique upload_ids in the graph
                upload_ids_query = """
                MATCH (n) WHERE n.upload_id IS NOT NULL
                RETURN DISTINCT n.upload_id as upload_id, count(n) as node_count, collect(DISTINCT labels(n)) as labels
                ORDER BY upload_id
                """

                result = session.run(upload_ids_query)
                upload_data = []

                for record in result:
                    upload_id = record["upload_id"]
                    node_count = record["node_count"]
                    labels = record["labels"]

                    upload_data.append(
                        {
                            "upload_id": upload_id,
                            "node_count": node_count,
                            "labels": labels,
                        }
                    )

                cleanup_stats["unique_upload_ids"] = [
                    item["upload_id"] for item in upload_data
                ]
                cleanup_stats["upload_data"] = upload_data

                logger.info(
                    f"Found {len(upload_data)} unique upload_ids in graph: {cleanup_stats['unique_upload_ids']}"
                )

                # Note: To check if these upload_ids exist in the database, you would need to
                # query your uploads table. This could be done by the calling code.

        except Exception as e:
            logger.error(f"Error checking for orphaned nodes: {e}")
            cleanup_stats["error"] = str(e)

        return cleanup_stats

    async def _log_graph_statistics(self):
        """Log statistics about the created graph"""
        try:
            if self._driver is None:
                return

            logger.info("Gathering graph statistics...")

            with self._driver.session() as session:
                # Count nodes by label
                result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = [record["label"] for record in result]
                logger.info(f"Node labels in database: {labels}")

                # Count relationships by type
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                rel_types = [record["relationshipType"] for record in result]
                logger.info(f"Relationship types in database: {rel_types}")

                # Count total nodes and relationships
                result = session.run(
                    "MATCH (n) OPTIONAL MATCH ()-[r]-() RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as rels"
                )
                stats = result.single()
                if stats:
                    logger.info(
                        f"Total nodes: {stats['nodes']}, Total relationships: {stats['rels']}"
                    )

        except Exception as e:
            logger.error(f"Error gathering graph statistics: {e}")

    def close(self):
        """Close the Neo4j driver connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def query_kg(self, query: str) -> RagResultModel:
        """
        Query the knowledge graph using the GraphRAG pipeline.

        :param query: The user's query string
        :return: RagResultModel containing the answer and context
        """
        logger.info(f"Querying knowledge graph with: {query}")
        self._ensure_connected()

        if self._graphrag is None:
            raise RuntimeError(
                "GraphRAG not initialized. Ensure components are properly set up."
            )

        try:
            result = self._graphrag.search(
                query_text=query, retriever_config={"top_k": 3}
            )

            logger.info(f"Query completed. Result type: {type(result)}")
            if hasattr(result, "answer"):
                logger.info(
                    f"Answer length: {len(result.answer) if result.answer else 0}"
                )

            return result

        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}", exc_info=True)
            raise

    def retrieve_context(self, query: str, top_k: int = 5, min_score: float = 0.3) -> dict:
        """
        Retrieve relevant context from the knowledge graph without generating an answer.
        Optimized for feeding into other LLM instances or agents.

        :param query: The user's query string
        :param top_k: Number of top results to return
        :param min_score: Minimum similarity score (0.0-1.0) for results to be included
        :return: Dictionary with structured context data
        """
        logger.info(f"Retrieving context for query: {query} (top_k={top_k}, min_score={min_score})")
        self._ensure_connected()

        if self._retriver is None:
            raise RuntimeError(
                "Retriever not initialized. Ensure components are properly set up."
            )

        try:
            retriever_result = self._retriver.search(query_text=query, top_k=top_k)
            
            print(f"Retriever returned {len(retriever_result.items) if retriever_result and hasattr(retriever_result, 'items') else 0} items")
            print(f"Filtering results with min_score={min_score}")
            print(f"Items: {retriever_result}")

            context_items = []
            combined_content = []
            total_score = 0.0

            # Handle case where retriever_result might be None or not have items
            if retriever_result is None:
                logger.warning("Retriever returned None result")
                return self._empty_context_result(query, min_score)

            if hasattr(retriever_result, "items") and retriever_result.items:
                for i, item in enumerate(retriever_result.items):
                    # Safely extract item attributes
                    raw_content = getattr(item, "content", "").strip() if item else ""
                    metadata = getattr(item, "metadata", {}) if item else {}
                    item_score = getattr(item, "score", 0.0) if item else 0.0

                    # Handle case where metadata might be None
                    if metadata is None:
                        metadata = {}

                    # Parse the clean result from our simplified Cypher query
                    print(f"DEBUG: Raw content preview: {raw_content[:200]}...")
                    print(f"DEBUG: Item score from retriever: {item_score}")
                    content, score, parsed_metadata = self._parse_retriever_result(raw_content, item_score)

                    # Merge the parsed metadata with any existing metadata
                    enhanced_metadata = {**(metadata or {}), **parsed_metadata}
                    
                    # Extract relationship data for additional context
                    relationships = {
                        'direct': parsed_metadata.get('direct_relations', []),
                        'extended': parsed_metadata.get('extended_relations', []),
                        'deep': parsed_metadata.get('deep_relations', [])
                    }
                    
                    # Try to extract additional structured data if available
                    # Our new query returns additional fields that might be in metadata
                    additional_context = []
                    
                    # Add relationship information to additional context
                    if any(relationships.values()):
                        if relationships['direct']:
                            additional_context.extend([f"Direct: {rel}" for rel in relationships['direct']])
                        if relationships['extended']:
                            additional_context.extend([f"Extended: {rel}" for rel in relationships['extended']])
                        if relationships['deep']:
                            additional_context.extend([f"Deep: {rel}" for rel in relationships['deep']])

                    # Filter by minimum score threshold
                    if score < min_score:
                        logger.debug(f"Filtering out low-score result: score={score:.3f} < min_score={min_score}")
                        continue

                    # Skip empty content
                    if not content or content.strip() == "":
                        logger.debug("Filtering out empty content result")
                        continue

                    context_item = {
                        "rank": len(context_items) + 1,  # Re-rank after filtering
                        "content": content,
                        "metadata": enhanced_metadata,
                        "score": score,
                        "source": {
                            "file_id": enhanced_metadata.get("file_id") or metadata.get("file_id", "unknown"),
                            "upload_id": enhanced_metadata.get("upload_id") or metadata.get("upload_id", "unknown"),
                            "filename": enhanced_metadata.get("source_file") or metadata.get("filename", "unknown"),
                            "chunk_index": metadata.get("index", len(context_items)),
                        },
                        "document_info": {
                            "title": enhanced_metadata.get("document_title", "No title available"),
                            "section_summary": enhanced_metadata.get("section_summary", ""),
                            "keywords": enhanced_metadata.get("excerpt_keywords", ""),
                            "questions": enhanced_metadata.get("questions_this_excerpt_can_answer", ""),
                            "local_context": enhanced_metadata.get("local_context", ""),
                            "content_type": enhanced_metadata.get("content_type", "unknown"),
                            "has_keywords": enhanced_metadata.get("has_keywords", False),
                            "has_questions": enhanced_metadata.get("has_questions", False),
                        },
                        "relationships": {
                            "direct": relationships["direct"],
                            "extended": relationships["extended"], 
                            "deep": relationships["deep"],
                            "all_formatted": additional_context if additional_context else [],
                        },
                    }

                    context_items.append(context_item)

                    # Add to combined content with clear delimiters
                    if content:
                        combined_content.append(f"[Source {len(context_items)}]: {content}")
                        total_score += score
            else:
                logger.warning(f"No retrieval items found for query: {query}")

            # Log filtering results
            total_raw_results = len(retriever_result.items) if retriever_result and hasattr(retriever_result, "items") and retriever_result.items else 0
            logger.info(f"Filtered {total_raw_results} raw results to {len(context_items)} results above min_score={min_score}")

            # Create a formatted context string perfect for LLM prompts
            formatted_context = "\n\n".join(combined_content)

            result = {
                "query": query,
                "total_items": len(context_items),
                "total_raw_results": total_raw_results,
                "min_score_threshold": min_score,
                "filtered_count": total_raw_results - len(context_items),
                "avg_score": total_score / len(context_items) if context_items else 0.0,
                "context_items": context_items,
                "combined_context": formatted_context,
                "llm_ready_prompt": f"Based on the following context, please answer the question: '{query}'\n\nContext:\n{formatted_context}",
                "metadata_summary": {
                    "unique_files": len(
                        set(item["source"]["file_id"] for item in context_items)
                    ),
                    "unique_uploads": len(
                        set(item["source"]["upload_id"] for item in context_items)
                    ),
                    "score_range": {
                        "min": min(
                            (item["score"] for item in context_items), default=0.0
                        ),
                        "max": max(
                            (item["score"] for item in context_items), default=0.0
                        ),
                    },
                },
            }

            logger.info(
                f"Retrieved {len(context_items)} context items with avg score {result['avg_score']:.3f}"
            )
            logger.info(
                f"Context spans {result['metadata_summary']['unique_files']} files from {result['metadata_summary']['unique_uploads']} uploads"
            )

            return result

        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            raise

    def format_context_with_metadata(self, context_items: list[dict]) -> str:
        """
        Format context items with their enhanced metadata for display or logging.
        
        :param context_items: List of context items from retrieve_context
        :return: Formatted string with content and metadata
        """
        if not context_items:
            return "No context items to format."
        
        formatted_sections = []
        
        for item in context_items:
            content = item.get("content", "")
            metadata = item.get("document_info", {})
            relationships = item.get("relationships", {})
            source = item.get("source", {})
            score = item.get("score", 0.0)
            
            section = []
            section.append(f"=== Context Item {item.get('rank', '?')} (Score: {score:.3f}) ===")
            
            # Document information
            if metadata.get("title") and metadata["title"] != "No title available":
                section.append(f"Title: {metadata['title']}")
            
            if metadata.get("keywords"):
                section.append(f"Keywords: {metadata['keywords']}")
            
            if metadata.get("questions"):
                section.append(f"Relevant Questions: {metadata['questions']}")
            
            # Source information
            if source.get("filename") and source["filename"] != "unknown":
                section.append(f"Source: {source['filename']}")
            
            # Content type
            content_type = metadata.get("content_type", "unknown")
            section.append(f"Content Type: {content_type}")
            
            # Main content
            section.append("\nContent:")
            section.append(content)
            
            # Local context if available
            if metadata.get("local_context"):
                section.append(f"\nLocal Context: {metadata['local_context']}")
            
            # Section summary if available
            if metadata.get("section_summary") and metadata["section_summary"] != "No text available":
                section.append(f"\nSection Summary: {metadata['section_summary']}")
            
            # Relationships
            if relationships:
                if relationships.get("direct"):
                    section.append(f"\nDirect Relations: {', '.join(relationships['direct'])}")
                if relationships.get("extended"):
                    section.append(f"Extended Relations: {', '.join(relationships['extended'])}")
                if relationships.get("deep"):
                    section.append(f"Deep Relations: {', '.join(relationships['deep'])}")
            
            formatted_sections.append("\n".join(section))
        
        return "\n\n" + "\n\n".join(formatted_sections) + "\n"

    def get_context_for_query(self, query: str, top_k: int = 5, min_score: float = 0.3, include_metadata: bool = False) -> str:
        """
        Get a formatted context string ready for feeding into LLMs or agents.
        This is a convenience method that returns just the combined context text.

        :param query: The user's query string
        :param top_k: Number of top results to return
        :param min_score: Minimum similarity score for results to be included
        :param include_metadata: Whether to include enhanced metadata in the output
        :return: Formatted context string ready for LLM prompts
        """
        try:
            retrieval_result = self.retrieve_context(query, top_k, min_score)
            if retrieval_result and isinstance(retrieval_result, dict):
                if include_metadata and retrieval_result.get("context_items"):
                    # Return rich metadata format
                    return self.format_context_with_metadata(retrieval_result["context_items"])
                else:
                    # Return simple combined context
                    return retrieval_result.get("combined_context", "")
            else:
                logger.error(f"retrieve_context returned invalid result: {type(retrieval_result)}")
                return ""
        except Exception as e:
            logger.error(f"Error getting context for query: {e}")
            return ""

    def get_llm_ready_prompt(self, query: str, top_k: int = 5, min_score: float = 0.3, include_metadata: bool = False) -> str:
        """
        Get a complete prompt ready for sending to an LLM or agent.

        :param query: The user's query string
        :param top_k: Number of top results to return
        :param min_score: Minimum similarity score for results to be included
        :param include_metadata: Whether to include enhanced metadata in the context
        :return: Complete prompt with context and question
        """
        try:
            retrieval_result = self.retrieve_context(query, top_k, min_score)
            if retrieval_result and isinstance(retrieval_result, dict):
                if include_metadata and retrieval_result.get("context_items"):
                    # Use rich metadata formatting
                    context = self.format_context_with_metadata(retrieval_result["context_items"])
                    return f"Based on the following context with metadata:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context and relationship information provided."
                else:
                    # Use simple combined context
                    return retrieval_result.get("llm_ready_prompt", f"Please answer: {query}")
            else:
                logger.error(f"retrieve_context returned invalid result: {type(retrieval_result)}")
                return f"Please answer: {query}"
        except Exception as e:
            logger.error(f"Error in get_llm_ready_prompt: {e}", exc_info=True)
            return f"Please answer: {query}"

    def inspect_graph_schema(self) -> dict:
        """
        Inspect the current graph schema to understand what's in the database.

        :return: Dictionary containing schema information
        """
        logger.info("Inspecting graph schema...")
        self._ensure_connected()

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        schema_info = {
            "node_labels": [],
            "relationship_types": [],
            "node_properties": {},
            "relationship_properties": {},
            "indexes": [],
            "constraints": [],
        }

        try:
            with self._driver.session() as session:
                # Get node labels
                result = session.run(
                    "CALL db.labels() YIELD label RETURN label ORDER BY label"
                )
                schema_info["node_labels"] = [record["label"] for record in result]

                # Get relationship types
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
                )
                schema_info["relationship_types"] = [
                    record["relationshipType"] for record in result
                ]

                # Get node properties for each label
                for label in schema_info["node_labels"]:
                    result = session.run(
                        "MATCH (n) WHERE $label IN labels(n) RETURN keys(n) as props LIMIT 100",
                        label=label,
                    )
                    props = set()
                    for record in result:
                        props.update(record["props"])
                    schema_info["node_properties"][label] = sorted(list(props))

                # Get relationship properties for each type
                for rel_type in schema_info["relationship_types"]:
                    result = session.run(
                        "MATCH ()-[r]-() WHERE type(r) = $rel_type RETURN keys(r) as props LIMIT 100",
                        rel_type=rel_type,
                    )
                    props = set()
                    for record in result:
                        props.update(record["props"])
                    schema_info["relationship_properties"][rel_type] = sorted(
                        list(props)
                    )

                # Get indexes
                result = session.run("SHOW INDEXES")
                schema_info["indexes"] = [dict(record) for record in result]

                # Get constraints
                result = session.run("SHOW CONSTRAINTS")
                schema_info["constraints"] = [dict(record) for record in result]

            logger.info(
                f"Schema inspection complete. Found {len(schema_info['node_labels'])} node labels, "
                f"{len(schema_info['relationship_types'])} relationship types"
            )

        except Exception as e:
            logger.error(f"Error inspecting graph schema: {e}")
            schema_info["error"] = str(e)

        return schema_info

    def get_graph_stats(self) -> dict:
        """
        Get comprehensive statistics about the current graph.

        :return: Dictionary containing graph statistics
        """
        logger.info("Getting graph statistics...")
        self._ensure_connected()

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        stats = {
            "total_nodes": 0,
            "total_relationships": 0,
            "nodes_by_label": {},
            "relationships_by_type": {},
            "nodes_with_upload_id": 0,
            "unique_upload_ids": [],
        }

        try:
            with self._driver.session() as session:
                # Get total counts
                result = session.run(
                    "MATCH (n) OPTIONAL MATCH ()-[r]-() RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as rels"
                )
                totals = result.single()
                if totals:
                    stats["total_nodes"] = totals["nodes"]
                    stats["total_relationships"] = totals["rels"]

                # Count nodes by label
                result = session.run(
                    "MATCH (n) RETURN labels(n) as labels, count(n) as count"
                )
                for record in result:
                    labels = record["labels"]
                    count = record["count"]
                    for label in labels:
                        if label in stats["nodes_by_label"]:
                            stats["nodes_by_label"][label] += count
                        else:
                            stats["nodes_by_label"][label] = count

                # Count relationships by type
                result = session.run(
                    "MATCH ()-[r]-() RETURN type(r) as rel_type, count(r) as count"
                )
                for record in result:
                    stats["relationships_by_type"][record["rel_type"]] = record["count"]

                # Count nodes with upload_id
                result = session.run(
                    "MATCH (n) WHERE n.upload_id IS NOT NULL RETURN count(n) as count, collect(DISTINCT n.upload_id) as upload_ids"
                )
                upload_data = result.single()
                if upload_data:
                    stats["nodes_with_upload_id"] = upload_data["count"]
                    stats["unique_upload_ids"] = upload_data["upload_ids"]

            logger.info(
                f"Graph statistics: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships"
            )

        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            stats["error"] = str(e)

        return stats

    def _empty_context_result(self, query: str, min_score: float = 0.3) -> dict:
        """
        Return an empty context result when retrieval fails.
        
        :param query: The original query
        :param min_score: The minimum score threshold used
        :return: Empty context result structure
        """
        return {
            "query": query,
            "total_items": 0,
            "total_raw_results": 0,
            "min_score_threshold": min_score,
            "filtered_count": 0,
            "avg_score": 0.0,
            "context_items": [],
            "combined_context": "",
            "llm_ready_prompt": f"Please answer: {query}",
            "metadata_summary": {
                "unique_files": 0,
                "unique_uploads": 0,
                "score_range": {
                    "min": 0.0,
                    "max": 0.0,
                },
            },
        }

    def _parse_retriever_result(self, raw_content: str, fallback_score: float = 0.0) -> tuple[str, float, dict]:
        """
        Parse the clean structured result from our simplified Cypher query.
        
        :param raw_content: The content from the retriever (may be wrapped in Record)
        :param fallback_score: Fallback score if parsing fails
        :return: Tuple of (clean_content, score, metadata_dict)
        """
        if not raw_content or not raw_content.strip():
            return "", fallback_score, {}
        
        try:
            content = raw_content.strip()
            extracted_metadata = {}
            extracted_relationships = {}
            
            # Check if it's still wrapped in a Record object
            if content.startswith('<Record '):
                # Try to extract the actual content from the Record
                # Look for patterns like: <Record info='actual content' score=X.XX metadata={...} ...>
                import re
                
                # First try to extract info field
                info_match = re.search(r"info='([^']*)'", content)
                if not info_match:
                    info_match = re.search(r'info="([^"]*)"', content)
                
                if info_match:
                    clean_content = info_match.group(1)
                    # Unescape the content
                    clean_content = clean_content.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                else:
                    # Fallback: use the raw content without Record wrapper
                    clean_content = content
                
                # Try to extract score from Record
                score_match = re.search(r'score=([0-9]*\.?[0-9]+)', content)
                if score_match:
                    try:
                        extracted_score = float(score_match.group(1))
                    except (ValueError, IndexError):
                        extracted_score = fallback_score
                else:
                    extracted_score = fallback_score
                
                # Try to extract metadata from Record
                # Look for metadata={...} pattern
                metadata_match = re.search(r'metadata=\{([^}]*)\}', content)
                if metadata_match:
                    metadata_str = metadata_match.group(1)
                    # Simple parsing of key-value pairs
                    # This is a basic implementation - could be enhanced for complex nested structures
                    pairs = metadata_str.split(',')
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip("'\"")
                            value = value.strip().strip("'\"")
                            extracted_metadata[key] = value
                
                # Try to extract relationship arrays
                for rel_type in ['direct_relations', 'extended_relations', 'deep_relations']:
                    rel_pattern = rf'{rel_type}=\[([^\]]*)\]'
                    rel_match = re.search(rel_pattern, content)
                    if rel_match:
                        rel_content = rel_match.group(1)
                        # Parse the list items
                        if rel_content.strip():
                            # Split by comma but be careful of nested quotes
                            relations = []
                            current_item = ""
                            in_quotes = False
                            for char in rel_content:
                                if char == "'" and not in_quotes:
                                    in_quotes = True
                                elif char == "'" and in_quotes:
                                    in_quotes = False
                                elif char == ',' and not in_quotes:
                                    if current_item.strip():
                                        relations.append(current_item.strip().strip("'\""))
                                    current_item = ""
                                    continue
                                current_item += char
                            
                            # Add the last item
                            if current_item.strip():
                                relations.append(current_item.strip().strip("'\""))
                            
                            extracted_relationships[rel_type] = relations
                        else:
                            extracted_relationships[rel_type] = []
                
            else:
                # Content is already clean
                clean_content = content
                extracted_score = fallback_score
            
            print(f"DEBUG: Clean content preview: {clean_content[:200]}...")
            print(f"DEBUG: Extracted/fallback score: {extracted_score}")
            print(f"DEBUG: Extracted metadata: {extracted_metadata}")
            print(f"DEBUG: Extracted relationships: {extracted_relationships}")
            
            # Combine metadata and relationships
            full_metadata = {**extracted_metadata, **extracted_relationships}
            
            return clean_content, extracted_score, full_metadata
            
        except Exception as e:
            logger.warning(f"Error parsing retriever result: {e}")
            return raw_content.strip(), fallback_score, {}


# Global driver instance for singleton pattern
_driver_instance = None


@lru_cache(maxsize=1)
def get_driver() -> "Neo4jDriver":
    """
    Get a singleton instance of the Neo4j driver.
    This ensures proper connection pooling and prevents multiple driver instances.
    """
    global _driver_instance
    if _driver_instance is None:
        _driver_instance = Neo4jDriver()
    return _driver_instance
