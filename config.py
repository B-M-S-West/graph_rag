from curses.ascii import EM
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class to manage environment variables."""
    NEO4J_URI = f"http://{os.getenv('SERVER_HOST')}:{os.getenv('NEO4J_PORT')}"
    NEO4J_AUTH = (os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    QDRANT_URL = f"http://{os.getenv('SERVER_HOST')}:{os.getenv('QDRANT_PORT')}"
    OLLAMA_URL = f"http://{os.getenv('SERVER_HOST')}:11434/v1"
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    LLM_MODEL = "gemma3:1b-it-qat"

