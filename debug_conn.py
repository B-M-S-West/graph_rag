import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()


def test_connections():
    # 1. Test Neo4j
    neo4j_uri = os.getenv("NEO4J_URI")
    print(f"Testing Neo4j connection to: {neo4j_uri}...")
    try:
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=5,  # Short timeout for testing
        )
        driver.verify_connectivity()
        print("✅ Neo4j Connected Successfully!")
    except Exception as e:
        print(f"❌ Neo4j Connection Failed: {e}")

    # 2. Test Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    print(f"\nTesting Qdrant connection to: {qdrant_url}...")
    try:
        client = QdrantClient(
            url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"), timeout=5
        )
        collections = client.get_collections()
        print("✅ Qdrant Connected Successfully!")
    except Exception as e:
        print(f"❌ Qdrant Connection Failed: {e}")


if __name__ == "__main__":
    test_connections()
