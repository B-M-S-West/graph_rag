import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class to manage environment variables."""

    SERVER_HOST = os.getenv("SERVER_HOST", "localhost")

    # Neo4j Configuration
    NEO4J_URI = f"bolt://{SERVER_HOST}:{os.getenv('NEO4J_PORT', '7687')}"
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Qdrant Configuration
    QDRANT_URL = f"http://{SERVER_HOST}:{os.getenv('QDRANT_PORT', '6333')}"
    COLLECTION_NAME = "graphRAGstoreds"
    VECTOR_DIMENSION = 768

    # LLM / Ollama Configuration
    OLLAMA_BASE_URL = f"http://{SERVER_HOST}:11434/v1"
    OLLAMA_API_KEY = "ollama"
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    LLM_MODEL = "gemma3:1b-it-qat"
