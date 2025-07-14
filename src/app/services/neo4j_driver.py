import logging
from typing import List

import neo4j
import neo4j.exceptions
from llama_index.core import Document
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.slide import SlideNodeParser
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever

from src.app.services.template import INPUTS, TEMPLATE
from src.app.configs.config import get_config
from src.app.services.retrieval_query import QUERY

logger = logging.getLogger("graphrag.neo4j_driver")


class SlideTextSplitterAdapter(TextSplitter):
    def __init__(self, slide_parser: SlideNodeParser):
        self._slide_parser = slide_parser

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
        self._llm = None
        self._embedder = None
        self._index_name = "text_embeddings"

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
                self._initialize_components()
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

    def _initialize_components(self):
        """Initialize Neo4j-dependent components"""
        if self._pipeline is not None or self._driver is None:
            return  # Already initialized or driver not available

        self._llm = OpenAILLM(
            model_name=self._cfg.openai_model,
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0,  # turning temperature down for more deterministic results
            },
        )
        self._embedder = OpenAIEmbeddings(model=self._cfg.openai_embedding_model)
        llm = OpenAI(
            model=self._cfg.openai_model,
            api_key=self._cfg.openai_api_key,
        )
        self._token_counter = TokenCounter()
        self._pipeline = SimpleKGPipeline(
            llm=self._llm,
            driver=self._driver,  # At this point _driver is guaranteed to be not None
            text_splitter=SlideTextSplitterAdapter(
                SlideNodeParser.from_defaults(
                    llm=llm,
                    token_counter=self._token_counter,
                    chunk_size=200,
                    window_size=5,
                )
            ),
            embedder=self._embedder,
            from_pdf=True,
        )

        # Log configuration details for debugging (but hide sensitive data)
        logger.info("Initializing Neo4j GraphRAG components...")
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
            logger.warning(
                f"Failed to create vector index: {e}. "
                "This may be because the index already exists."
            )
        except Exception as e:
            logger.error(f"Unexpected error creating vector index: {e}")
            raise

        try:
            create_fulltext_index(
                driver=self._driver,  # At this point _driver is guaranteed to be not None
                name=f"{self._index_name}_fulltext",
                label="Chunk",  # all your chunks share this label
                node_properties=[
                    "text",
                    "local_context",
                    "metadata",
                ],  # properties to index
            )
        except neo4j.exceptions.ClientError as e:
            logger.warning(
                f"Failed to create fulltext index: {e}. "
                "This may be because the index already exists."
            )
        except Exception as e:
            logger.error(f"Unexpected error creating fulltext index: {e}")
            raise

        self._retriver = HybridCypherRetriever(
            driver=self._driver,  # At this point _driver is guaranteed to be not None
            vector_index_name=self._index_name,
            fulltext_index_name=f"{self._index_name}_fulltext",
            retrieval_query=QUERY,
            embedder=self._embedder,
        )

        prompt_template = RagTemplate(
            template=TEMPLATE,
            expected_inputs=INPUTS,  # Define expected inputs for the template
        )

        self._graphrag = GraphRAG(
            retriever=self._retriver,
            llm=OpenAILLM(model_name=self._cfg.openai_model),
            prompt_template=prompt_template,
        )

    async def build_kg_from_pdf(self, pdf_path: str):
        """
        Build a knowledge graph from a PDF file.
        :param pdf_path: Path to the PDF file.
        :param metadata: Optional metadata to attach to the nodes.
        :return: The ID of the created knowledge graph.
        """
        logger.info(f"Starting knowledge graph building for: {pdf_path}")
        self._ensure_connected()
        if self._pipeline is None:
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
            result = await self._pipeline.run_async(pdf_path)

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
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()
                logger.info(
                    f"Total nodes: {node_count['node_count'] if node_count else 0}"
                )

                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()
                logger.info(
                    f"Total relationships: {rel_count['rel_count'] if rel_count else 0}"
                )

                # Sample some nodes to see their structure - especially chunk nodes
                result = session.run(
                    "MATCH (n) RETURN labels(n) as labels, keys(n) as properties LIMIT 5"
                )
                samples = []
                for record in result:
                    samples.append(
                        {"labels": record["labels"], "properties": record["properties"]}
                    )
                logger.info(f"Sample node structures: {samples}")

                # Specifically look at chunk nodes and their properties
                chunk_query = """
                    MATCH (n) 
                    WHERE any(label IN labels(n) WHERE toLower(label) CONTAINS 'chunk')
                    RETURN labels(n) as labels, keys(n) as properties, 
                           coalesce(n.text[0..100], 'No text') as sample_text,
                           coalesce(n.local_context, 'No local_context') as context_sample,
                           properties(n) as all_properties
                    LIMIT 5
                """
                result = session.run(chunk_query)
                chunk_samples = []
                for record in result:
                    chunk_samples.append(
                        {
                            "labels": record["labels"],
                            "properties": record["properties"],
                            "sample_text": record["sample_text"],
                            "context_sample": record["context_sample"],
                            "all_properties": record["all_properties"],
                        }
                    )

                if chunk_samples:
                    logger.info("Detailed chunk node analysis:")
                    for i, chunk in enumerate(chunk_samples):
                        logger.info(f"  Chunk {i}:")
                        logger.info(f"    Labels: {chunk['labels']}")
                        logger.info(f"    Property keys: {chunk['properties']}")
                        logger.info(f"    Text sample: {chunk['sample_text']}")
                        logger.info(f"    Context sample: {chunk['context_sample']}")
                        logger.info(f"    All properties: {chunk['all_properties']}")
                    logger.info(
                        "✓ Chunk nodes found - SlideTextSplitter data appears to be stored"
                    )

                    # Check if metadata from SlideTextSplitter is preserved
                    first_chunk_props = chunk_samples[0]["all_properties"]
                    metadata_keys = [
                        k
                        for k in first_chunk_props.keys()
                        if k not in ["text", "embedding", "id"]
                    ]
                    logger.info(f"Metadata properties preserved: {metadata_keys}")

                else:
                    logger.warning(
                        "✗ No chunk nodes found - SlideTextSplitter data may not be stored properly"
                    )

        except Exception as e:
            logger.warning(f"Could not gather graph statistics: {e}")

    def query_kg(self, query: str) -> RagResultModel:
        """
        Query the knowledge graph.
        :param query: The query string.
        :return: A list of documents matching the query.
        """
        self._ensure_connected()
        if self._retriver is None:
            raise RuntimeError("Retriever not initialized")
        return self._graphrag.search(
            query_text=query, retriever_config={"top_k": 5}, return_context=True
        )

    def inspect_graph_schema(self) -> dict:
        """
        Inspect the current graph schema to understand what was created.
        This helps debug issues with retrieval queries.
        """
        self._ensure_connected()
        if self._driver is None:
            return {"error": "No driver connection"}

        schema_info = {}

        try:
            with self._driver.session() as session:
                # Get all node labels
                result = session.run(
                    "CALL db.labels() YIELD label RETURN collect(label) as labels"
                )
                record = result.single()
                schema_info["node_labels"] = record["labels"] if record else []

                # Get all relationship types
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
                )
                record = result.single()
                schema_info["relationship_types"] = record["types"] if record else []

                # Get property keys
                result = session.run(
                    "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys"
                )
                record = result.single()
                schema_info["property_keys"] = record["keys"] if record else []

                # Get schema information for each label
                schema_info["label_schemas"] = {}
                for label in schema_info["node_labels"]:
                    query = "MATCH (n:" + label + ") RETURN keys(n) as props LIMIT 10"
                    result = session.run(query)
                    prop_sets = [record["props"] for record in result]
                    # Get unique properties across all nodes of this label
                    all_props = set()
                    for prop_set in prop_sets:
                        all_props.update(prop_set)
                    schema_info["label_schemas"][label] = list(all_props)

                # Get some sample relationship patterns
                schema_info["relationship_patterns"] = {}
                for rel_type in schema_info["relationship_types"]:
                    query = """
                        MATCH (a)-[r]->(b) WHERE type(r) = $rel_type
                        RETURN labels(a) as start_labels, labels(b) as end_labels, keys(r) as rel_props
                        LIMIT 5
                    """
                    result = session.run(query, rel_type=rel_type)
                    patterns = []
                    for record in result:
                        patterns.append(
                            {
                                "start_labels": record["start_labels"],
                                "end_labels": record["end_labels"],
                                "relationship_properties": record["rel_props"],
                            }
                        )
                    schema_info["relationship_patterns"][rel_type] = patterns

                # Count nodes and relationships
                result = session.run("MATCH (n) RETURN count(n) as count")
                record = result.single()
                schema_info["total_nodes"] = record["count"] if record else 0

                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = result.single()
                schema_info["total_relationships"] = record["count"] if record else 0

        except Exception as e:
            schema_info["error"] = str(e)
            logger.error(f"Error inspecting graph schema: {e}")

        return schema_info

    def test_retrieval_query(self, test_query: str = "test") -> dict:
        """
        Test the retrieval query against the actual schema to debug issues.
        This helps understand what properties and relationships actually exist.
        """
        self._ensure_connected()
        if self._driver is None:
            return {"error": "No driver connection"}

        test_results = {}

        try:
            with self._driver.session() as session:
                # First, let's see what chunk nodes actually look like
                result = session.run("""
                    MATCH (n)
                    WHERE any(label IN labels(n) WHERE label CONTAINS 'Chunk' OR label CONTAINS 'chunk')
                    RETURN labels(n) as labels, keys(n) as properties, n.text[0..100] as sample_text
                    LIMIT 3
                """)

                chunk_samples = []
                for record in result:
                    chunk_samples.append(
                        {
                            "labels": record["labels"],
                            "properties": record["properties"],
                            "sample_text": record["sample_text"],
                        }
                    )
                test_results["chunk_samples"] = chunk_samples

                # Test a simple version of our retrieval query on actual data
                if chunk_samples:
                    # Get the actual chunk label
                    chunk_label = None
                    for sample in chunk_samples:
                        for label in sample["labels"]:
                            if "chunk" in label.lower():
                                chunk_label = label
                                break
                        if chunk_label:
                            break

                    if chunk_label:
                        # Test our query with actual chunk nodes - use a simpler approach
                        try:
                            result = session.run("""
                                MATCH (node)
                                WHERE any(label IN labels(node) WHERE toLower(label) CONTAINS 'chunk')
                                WITH node AS chunk LIMIT 1
                                
                                // Test property access
                                WITH chunk,
                                     coalesce(chunk.text, chunk.content, 'No text') AS text_content,
                                     coalesce(chunk.local_context, chunk.context, '') AS context_content,
                                     properties(chunk) AS all_props
                                
                                // Test relationships
                                OPTIONAL MATCH (chunk)-[r]->(connected)
                                WITH chunk, text_content, context_content, all_props,
                                     collect({
                                         rel_type: type(r),
                                         connected_labels: labels(connected),
                                         connected_props: keys(connected)
                                     }) AS connections
                                
                                RETURN {
                                    chunk_properties: all_props,
                                    text_sample: text_content[0..200],
                                    context_sample: context_content[0..100], 
                                    connection_count: size([c IN connections WHERE c.rel_type IS NOT NULL]),
                                    sample_connections: connections[0..3]
                                } AS test_result
                            """)

                            record = result.single()
                            if record:
                                test_results["query_test"] = record["test_result"]
                            else:
                                test_results["query_test"] = (
                                    "No results from test query"
                                )
                        except Exception as query_error:
                            test_results["query_test"] = (
                                f"Query test failed: {query_error}"
                            )
                    else:
                        test_results["query_test"] = "No chunk label identified"
                else:
                    test_results["query_test"] = "No chunk nodes found in database"

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Error testing retrieval query: {e}")

        return test_results

    def close(self):
        if self._driver:
            self._driver.close()

    def retrieve_context(self, query: str, top_k: int = 5) -> dict:
        """
        Directly retrieve context from the knowledge graph without LLM generation.
        This allows you to get the raw retrieved chunks/context and feed them to your own agent or streaming LLM.

        :param query: The query string for retrieval
        :param top_k: Number of top results to retrieve (default: 5)
        :return: Dictionary containing retrieved items and metadata
        """
        self._ensure_connected()
        if self._retriver is None:
            raise RuntimeError("Retriever not initialized")

        logger.info(f"Direct retrieval for query: '{query}' (top_k={top_k})")

        try:
            # Use the retriever directly without GraphRAG LLM processing
            retriever_result = self._retriver.search(query_text=query, top_k=top_k)

            # Extract the raw context items
            context_items = []
            for item in retriever_result.items:
                context_items.append(
                    {
                        "content": item.content,
                        "metadata": getattr(item, "metadata", {}),
                        # Include score if available (for vector/hybrid search)
                        "score": getattr(item, "score", None),
                    }
                )

            # Combine all content into a single context string for convenience
            combined_context = "\n\n".join(
                item.content for item in retriever_result.items
            )

            result = {
                "query": query,
                "items": context_items,
                "combined_context": combined_context,
                "total_items": len(context_items),
                "retriever_metadata": {
                    "top_k": top_k,
                    "retriever_type": type(self._retriver).__name__,
                },
            }

            logger.info(f"Retrieved {len(context_items)} context items for query")
            return result

        except Exception as e:
            logger.error(f"Error in direct retrieval: {e}", exc_info=True)
            raise

    def get_context_for_query(self, query: str, top_k: int = 5) -> str:
        """
        Get combined context string for a query - convenient for feeding to streaming LLMs.

        :param query: The query string for retrieval
        :param top_k: Number of top results to retrieve (default: 5)
        :return: Combined context string ready to be used in prompts
        """
        retrieval_result = self.retrieve_context(query, top_k=top_k)
        return retrieval_result["combined_context"]


# Global driver instance - initialized lazily
_driver_instance = None


def get_driver() -> Neo4jDriver:
    """Get the global Neo4j driver instance"""
    global _driver_instance
    if _driver_instance is None:
        _driver_instance = Neo4jDriver()
    return _driver_instance
