import os
import time
from turtle import st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()


def test_connections():
    # 1. Test Neo4j
    neo4j_uri = os.getenv("NEO4J_URI")
    print(f"Testing Neo4j connection to: {neo4j_uri}...")
    start_time = time.perf_counter()
    try:
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            connection_timeout=5,  # Short timeout for testing
        )
        driver.verify_connectivity()
        end_time = time.perf_counter()
        print("✅ Neo4j Connected Successfully!")
        print(f"Connection Time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        end_time = time.perf_counter()
        print(f"❌ Neo4j Connection Failed: {e}")
        print(f"Elapsed Time: {end_time - start_time:.2f} seconds")

    # 2. Test Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    print(f"\nTesting Qdrant connection to: {qdrant_url}...")
    start_time = time.perf_counter()
    try:
        client = QdrantClient(
            url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"), timeout=5
        )
        collections = client.get_collections()
        end_time = time.perf_counter()
        print("✅ Qdrant Connected Successfully!")
        print(f"Connection Time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        end_time = time.perf_counter()
        print(f"❌ Qdrant Connection Failed: {e}")
        print(f"Elapsed Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    test_connections()
