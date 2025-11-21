import re
import uuid
import json
from collections import defaultdict
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from pydantic import BaseModel
from openai import OpenAI
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from config import Config

# Pydantic Models for Extraction
class SingleRelationship(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[SingleRelationship]

class GraphRAGEngine:
    def __init__(self):
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

    def _create_collection_if_not_exists(self):
        try:
            self.qdrant_client.get_collection(collection_name=Config.COLLECTION_NAME)
        except Exception:
            print(f"Collection '{Config.COLLECTION_NAME}' not found. Creating new collection.")
            self.qdrant_client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=Config.VECTOR_DIMENSION, 
                    distance=models.Distance.COSINE
                )
            )
    
    def get_embeddings(self, text):
        response = self.llm_client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _extract_graph_components(self, raw_data):
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
        parsed_response = GraphComponents.model_validate_json(completion.choices[0].message.content)

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
        return nodes, relationships
    
    def ingest_text(self, raw_text):
        """Full pipeline to ingest text into both Neo4j and Qdrant."""
        print("Step 1: Extracting Graph Components...")
        nodes, relationships = self._extract_graph_components(raw_text)

        print("Step 2: Ingesting into Neo4j...")
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

        print("Step 3: Ingesting into Qdrant...")
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
        return f"Ingested {len(nodes)} nodes and {len(relationships)} relationships."
    
    def query_graph(self, query_text):
        """Retrieves context and answers the question"""
        # 1. Retriever Search (Vector Search)
        retriever = QdrantNeo4jRetriever(
            driver = self.neo4j_driver,
            client = self.qdrant_client,
            collection_name = Config.COLLECTION_NAME,
            id_property_external = "id",
            id_property_neo4j = "id"
        )
        query_vector = self.get_embeddings(query_text)
        results = retriever.search(query_vector=query_vector, top_k=5)

        # 2. Extract Entity IDs
        entity_ids = [item.content.split("id': '")[1].split("'")[0] for item in results]

        # 3. Fetch Related Graph (Neo4j Traversal)
        subgraph_query = """
        MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
        WHERE e.id IN $entity_ids
        RETURN e, r1 as r, n1 as related, r2, n2
        UNION
        MATCH (e:Entity)-[r]-(related)
        WHERE e.id IN $entity_ids
        RETURN e, r, related, null as r2, null as n2
        """

        nodes_set = set()
        edges_list = []

        with self.neo4j_driver.session() as session:
            result = session.run(subgraph_query, entity_ids=entity_ids)
            for record in result:
                e_name = record["e"]["name"]
                rel_name = record["related"]["name"]
                r_type = record["r"]["type"]
                nodes_set.add(e_name)
                nodes_set.add(rel_name)
                edges_list.append(f"{e_name} -[{r_type}]-> {rel_name}")

        # Generate Answer
        context_str = f"Nodes: {', '.join(nodes_set)}\nEdges: {'; '.join(edges_list)}"
        prompt = f"""
        Context Graph:
        {context_str}

        Question: {query_text}

        Answer the question using ONLY the Context Graph provided.
        """

        response = self.llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided graph context."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content, context_str