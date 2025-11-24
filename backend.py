from math import log
import re
import uuid
import json
from collections import defaultdict
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from pydantic import BaseModel
from openai import OpenAI
from main import fetch_related_graph
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from config import Config
from loguru import logger

# Pydantic Models for Extraction
class SingleRelationship(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[SingleRelationship]

class GraphRAGEngine:
    def __init__(self):
        logger.info("Initializing GraphRAGEngine...")
        try:
            # Initialize Neo4j, Qdrant, and LLM clients
            self.neo4j_driver = GraphDatabase.driver(
                Config.NEO4J_URI, 
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
            )
            self.qdrant_client = QdrantClient(url=Config.QDRANT_URL)
            self.llm_client = OpenAI(
                base_url=Config.OLLAMA_BASE_URL, 
                api_key=Config.OLLAMA_API_KEY
            )
            
            # Ensure Vector Collection Exists
            self._create_collection_if_not_exists()
            logger.info("GraphRAGEngine initialized successfully.")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise e

    def _create_collection_if_not_exists(self):
        logger.info(f"Checking for existing Qdrant collection: {Config.COLLECTION_NAME}")
        try:
            self.qdrant_client.get_collection(collection_name=Config.COLLECTION_NAME)
            logger.info(f"Collection '{Config.COLLECTION_NAME}' already exists.")
        except Exception as e:
            if 'Not found: Collection' in str(e):
                logger.info(f"Collection '{Config.COLLECTION_NAME}' not found. Creating new collection.")
                self.qdrant_client.create_collection(
                    collection_name=Config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=Config.VECTOR_DIMENSION, 
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{Config.COLLECTION_NAME}' created successfully.")
            else:
                logger.error(f"Error checking/creating collection: {e}")
                raise e
    
    def get_embeddings(self, text):
        logger.info(f"Generating embedding for chunk of size {len(text)}")
        response = self.llm_client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text
        )
        logger.info("Embedding generated successfully.")
        return response.data[0].embedding
    
    def _extract_graph_components(self, raw_data):
        logger.info("Starting graph component extraction...")
        prompt = f"""
        Extract the nodes and relationships from the following text:\n{raw_data}
        """
        
        completion = self.llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system", 
                    "content": """You are a precision graph relationship extractor. Extract all relationships from the text and format them as a JSON object with this exact structure:
                    {
                        "graph": [
                            {
                                "node": "Person/Entity",
                                "target_node": "Related Entity",
                                "relationship": "Type of Relationship"
                            }
                        ]
                    }"""
                },
                {"role": "user", "content": prompt}
            ]
        )
        # Log the raw LLM output for debuging JSON parsing issues
        raw_output = completion.choices[0].message.content
        logger.debug(f"Raw LLM Output: {raw_output[:500]}...")

        parsed_response = GraphComponents.model_validate_json(raw_output)

        nodes = {}
        relationships = []
        for entry in parsed_response.graph:
            # Assign UUIDs to nodes if not already present
            if entry.node not in nodes:
                nodes[entry.node] = str(uuid.uuid4())
            if entry.target_node and entry.target_node not in nodes:
                nodes[entry.target_node] = str(uuid.uuid4())

            if entry.target_node and entry.relationship:
                relationships.append({
                    "source": nodes[entry.node],
                    "target": nodes[entry.target_node],
                    "relationship": entry.relationship
                })
        logger.info(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships.")
        return nodes, relationships
    
    def ingest_text(self, raw_text):
        """Full pipeline to ingest text into both Neo4j and Qdrant."""
        logger.info(f"Ingestion started for {len(raw_text)} characters of text.")
        try:
            # Step 1: Extraction
            nodes, relationships = self._extract_graph_components(raw_text)

            # Step 2: Neo4j Ingestion
            logger.info(f"Ingesting {len(nodes)} nodes and {len(relationships)} relationships into Neo4j...")
            with self.neo4j_driver.session() as session:
                # Create nodes
                for name, node_id in nodes.items():
                    session.run(
                        "MERGE (n:Entity {id: $id}) SET n.name = $name",
                        id=node_id, name=name
                    )
                # Create relationships
                for rel in relationships:
                    session.run(
                        """
                        MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
                        MERGE (a)-[r:RELATIONSHIP {type: $relationship}]->(b)
                        """,
                        source_id=rel["source"],
                        target_id=rel["target"],
                        relationship=rel["relationship"]
                    )
            logger.info("Neo4j ingestion completed.")

            # Step 3: Qdrant Ingestion
            logger.info("Step 3: Ingesting vectors into Qdrant...")
            # Split text by newlines for embedding chunks
            text_chunks = [p for p in raw_text.split('\n') if p.strip()]
            for chunk in text_chunks:
            embedding = self.get_embeddings(chunk)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"id": node_id}
                ) for node_id in nodes.values()
            ]
            self.qdrant_client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=points
            )
            logger.info("Qdrant ingestion completed.")
            return f"Ingested {len(nodes)} nodes and {len(relationships)} relationships."
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            raise e
    
    # Extracted function from original query_graph to manage final generation and logging
    def _generate_answer(self, user_query, subgraph):
        # Format graph context
        nodes_set = set()
        edges_list = []
        # ... (subgraph formatting logic as per original fetch_related_graph logic)
        
        # Note: The original provided script had this logic in the main function:
        for entry in subgraph:
            entity = entry["entity"]
            related = entry["related_node"]
            relationship = entry["relationship"]

            nodes_set.add(entity["name"])
            nodes_set.add(related["name"])

            # Use .get() for robustness in case 'type' key is missing
            r_type = relationship.get('type', 'RELATES_TO')
            edges_list.append(f"{entity['name']} -[{r_type}]-> {related['name']}")
            
        context_str = f"Nodes: {', '.join(nodes_set)}\nEdges: {'; '.join(edges_list)}"
        logger.debug(f"Graph context generated: {context_str}") # ADDED LOG

        prompt = f"""
        Context Graph:
        {context_str}
        
        Question: {user_query}
        
        Answer the question using ONLY the Context Graph provided.
        """
        
        logger.info("Calling LLM for final answer generation...") # ADDED LOG
        response = self.llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided graph context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content, context_str

    def query_graph(self, query_text):
        """Retrieves context and answers the question"""
        logger.info(f"Querying graph with question: {query_text}")
        try:
            # 1. Retriever Search (Vector Search)
            retriever = QdrantNeo4jRetriever(
                driver = self.neo4j_driver,
                client = self.qdrant_client,
                collection_name = Config.COLLECTION_NAME,
                id_property_external = "id",
                id_property_neo4j = "id"
            )
            query_vector = self.get_embeddings(query_text)
            logger.info("Running Qdrant vector search for relevant chunks.")
            results = retriever.search(query_vector=query_vector, top_k=5)
            logger.info(f"Qdrant search returned {len(results)} results.")

            # 2. Extract Entity IDs
            entity_ids = [item.content.split("id': '")[1].split("'")[0] for item in results]
            logger.info(f"Extracted entity IDs: {len(entity_ids)} IDs found.")

            # 3. Fetch Related Graph (Neo4j Traversal)
            logger.info("Fetching related graph from Neo4j.")
            subgraph = fetch_related_graph(self.neo4j_driver, entity_ids)

            # 4. Generate Answer using context
            answer, context_str = self._generate_answer(query_text, subgraph)
            logger.info("Answer generation completed.")
            return answer, context_str
        except Exception as e:
            logger.exception("FATAL ERROR: Query pipeline failed.")
            raise e