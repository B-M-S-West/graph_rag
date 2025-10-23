from email.mime import base
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from collections import defaultdict
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
server_host = os.getenv("SERVER_HOST")
qdrant_name = os.getenv("QDRANT_NAME")
qdrant_port = int(os.getenv("QDRANT_PORT"))
neo4j_name = os.getenv("NEO4J_NAME")
neo4j_port = int(os.getenv("NEO4J_PORT"))
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

qdrant_url = f"http://{server_host}:{qdrant_port}"
neo4j_url = f"http://{server_host}:{neo4j_port}"

# Using Ollama for locally run models
client = OpenAI(
    base_url=f"http://{server_host}:11434/v1",
    api_key="ollama"
)

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(
    neo4j_url,
    auth=(neo4j_user, neo4j_password)   
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url
)

class single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[single]

def openai_llm_parser(prompt):
    completion = client.chat.completions.create(
        model="gemma3:1b-it-qat",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 
                """You are a precision graph relationship extractor. Extract all relationships from the text and format them as a JSON object object with this exact structure:
                {
                    "graph": [
                        {"node": "Person/Entity",
                        "target_node": "Related Entity",
                        "relationship": "Type of relationship"},
                        ...more relationships...
                    ]
                }
                Include ALL relationships mentioned in the text, including implicit ones. Be thorough and precise.
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return GraphComponents.model_validate(completion.choices[0].message.content)
