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
    
    