import re
import json
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

    return GraphComponents.model_validate_json(completion.choices[0].message.content)

def extract_graph_components(raw_data):
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed_response = openai_llm_parser(prompt) # Should return a list of dictionaries
    parsed_response = parsed_response.graph # Should be the 'graph' structure is a key in the parsed response

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node # Get target node if available
        relationship = entry.relationship # Get relationship if available

        # Add nodes to the dictionary with a uniquie ID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "relationship": relationship
            })

    return nodes, relationships

def ingest_to_neo4j(nodes, relationships):
    """
    Ingest nodes and relationships into Neo4j database.
    """

    with neo4j_driver.session() as session:
        # Create nodes in Neo4j
        for name, node_id in nodes.items():
            session.run(
                "CREATE (n:Entity {id: $id, name: $name})",
                id=node_id,
                name=name
            )
        # Create relationships in Neo4j
        for relationship in relationships:
            session.run(
                """
                MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
                CREATE (a)-[:RELATIONSHIP {type: $relationship}]->(b)
                """,
                source_id=relationship["source"],
                target_id=relationship["target"],
                relationship=relationship["relationship"]
            )

    return nodes

def create_collection(client, collection_name, vector_dimension):
    """
    Create a Qdrant collection if it doesn't exist.
    """
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' does not exist. Creating new collection.")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE
                )
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"An error occurred: {e}")

def openai_embeddings(texts):
    response = client.embeddings.create(
        model="nomic-embed-text:latest", 
        input=texts
    )
    
    return response.data[0].embedding

def ingest_to_qdrant(collection_name, raw_data, node_id_mapping):
    """
    Ingest data into Qdrant collection.
    """
    embeddings = [openai_embeddings(paragraph) for paragraph in raw_data.split("\n")]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"id": node_id}
            }
            for node_id, embedding in zip(node_id_mapping.values(), embeddings)
        ]
    )

def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id"
    )

    results = retriever.search(query_vector=openai_embeddings(query), top_k=5)
    return results

def fetch_related_graph(neo4j_client, entity_ids):
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"],
            })
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"],
                })
    return subgraph

def format_graph_context(subgraph):
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} -[{relationship['type']}]-> {related['name']}")

    return {"nodes": list(nodes), "edges": edges}

def graphRAG_run(graph_context, user_query):
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:
    
    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """

    try:
        response = client.chat.completions.create(
            model="gemma3:1b-it-qat",
            messages=[
                {
                    "role": "system",
                    "content": "Provide the answer for the following question:"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message
    
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
    
if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv('.env')
    print("Environment variables loaded.")

    print("Initializing clients...")
    neo4j_driver = GraphDatabase.driver(
        neo4j_url,
        auth=(neo4j_user, neo4j_password)   
    )
    qdrant_client = QdrantClient(
        url=qdrant_url
    )
    print("Clients initialized.")

    print("Creating collection...")
    collection_name = "graphRAGstoreds"
    vector_dimension = 768 # 768 for nomic-embed-text
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified.")

    print("Extracting graph components...")

    raw_data = """Alice is a data scientist at TechCorp's Seattle office.
    Bob and Carol collaborate on the Alpha project.
    Carol transferred to the New York office last year.
    Dave mentors both Alice and Bob.
    TechCorp's headquarters is in Seattle.
    Carol leads the East Coast team.
    Dave started his career in Seattle.
    The Alpha project is managed from New York.
    Alice previously worked with Carol at DataCo.
    Bob joined the team after Dave's recommendation.
    Eve runs the West Coast operations from Seattle.
    Frank works with Carol on client relations.
    The New York office expanded under Carol's leadership.
    Dave's team spans multiple locations.
    Alice visits Seattle monthly for team meetings.
    Bob's expertise is crucial for the Alpha project.
    Carol implemented new processes in New York.
    Eve and Dave collaborated on previous projects.
    Frank reports to the New York office.
    TechCorp's main AI research is in Seattle.
    The Alpha project revolutionized East Coast operations.
    Dave oversees projects in both offices.
    Bob's contributions are mainly remote.
    Carol's team grew significantly after moving to New York.
    Seattle remains the technology hub for TechCorp."""

    nodes, relationships = extract_graph_components(raw_data)
    print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships.")

    print("Nodes:", nodes)
    print("Relationships:", relationships)

    print("Ingesting data into Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Data ingested into Neo4j.")

    print("Ingesting data into Qdrant...")
    ingest_to_qdrant(collection_name, raw_data, node_id_mapping)
    print("Data ingested into Qdrant.")

    query = "How is Bob connected to New York?"
    print("Starting retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)

    print("Extracting entity IDs...")
    entity_ids = [
        item.content.split("'id': '")[1].split("'")[0]
        for item in retriever_result.items
    ]
    print("Entity IDs:", entity_ids)

    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)

    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)

    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)
    