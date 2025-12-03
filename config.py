import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class to manage environment variables."""

    # Neo4j Configuration (Cloud)
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "graphRAGstoreds"
    VECTOR_DIMENSION = 768

    # LLM / Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "gemma3:1b-it-qat"
