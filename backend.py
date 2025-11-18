import uuid
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from openai import OpenAI
from config import Config

class GraphRAGEngine:
    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver(Config.NEO4J_URI, auth=Config.NEO4J_AUTH)
        self.qdrant_client = QdrantClient(url=Config.QDRANT_URL)
        self.llm_client = OpenAI(base_url=Config.OLLAMA_URL, api_key="ollama")
        self.collection_name = "graphRAG_production"

        # Ensure collection exists on init
        self._init_qdrant()

def _init_qdrant(self):