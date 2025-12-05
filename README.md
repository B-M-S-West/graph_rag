# GraphRAG Explorer

GraphRAG Explorer is a Streamlit front end for experimenting with Retrieval-Augmented Generation on top of a hybrid Neo4j + Qdrant knowledge graph. Paste raw text, let the app extract entities and relationships with an LLM, and then chat against the graph-aware context that is stored across both databases.

## Prerequisites
- Python 3.12 (managed via `uv`)
- Running Neo4j instance reachable via `bolt://` or `neo4j+s://`
- Running Qdrant instance (local Docker or cloud)
- Ollama or another OpenAI-compatible endpoint that serves the embedding and chat models referenced in `config.py`
- Optional: `direnv` or similar to load environment variables automatically

## Install uv
If you do not already have the `uv` package manager:

```bash
curl -Ls https://astral.sh/install.sh | sh
# or: pip install uv
```

Ensure `uv` knows about Python 3.12:

```bash
uv python install 3.12
```

## Initial Setup
1. Clone the repository and move into it:
	```bash
	git clone https://github.com/B-M-S-West/graph_rag.git
	cd graph_rag
	```
2. Install dependencies and create the virtual environment:
	```bash
	uv sync
	```
	This reads `pyproject.toml`, creates `.venv/`, and installs the pinned dependencies. Re-run `uv sync` whenever dependencies change.
3. Create a `.env` file (copy from `.env.example` if available) with the required secrets:

	| Variable | Description |
	| --- | --- |
	| `NEO4J_URI` | Bolt connection string, e.g. `neo4j://localhost:7687` or `neo4j+s://<host>:7687` |
	| `NEO4J_USER` | Neo4j username (default `neo4j`) |
	| `NEO4J_PASSWORD` | Neo4j password |
	| `QDRANT_URL` | Qdrant endpoint URL, e.g. `http://localhost:6333` |
	| `QDRANT_API_KEY` | Qdrant API key (omit or leave blank for local installs without auth) |
	| `OLLAMA_BASE_URL` | Base URL for the Ollama/OpenAI-compatible server, default `http://localhost:11434/v1` |
	| `OLLAMA_API_KEY` | API key or token for the LLM endpoint |

	The application uses `python-dotenv` to load this file automatically.

## Running the Streamlit App
Launch the UI with:

```bash
uv run streamlit run app.py
```

Streamlit boots a local server (by default on `http://localhost:8501`). Use the sidebar to ingest raw text. The ingestion pipeline:

1. Calls an LLM to extract entities and labeled relationships.
2. Writes nodes and edges into Neo4j.
3. Generates embeddings for each entity and stores them as vectors in Qdrant.

Chat questions are embedded, matched against Qdrant, expanded via Neo4j traversals, and finally answered with the retrieved graph context. Logging is written to `graphrag_app.log` for diagnostics.

## Verifying External Services
Before running the UI, you can verify connectivity to Neo4j and Qdrant:

```bash
uv run python debug_conn.py
```

The script prints timing and connection status for each service to help troubleshoot credentials, firewalls, or networking issues.

## Key Modules
- `app.py`: Streamlit front end that manages the UI, caching, and logging.
- `backend.py`: `GraphRAGEngine` combines extraction, graph storage, vector storage, and query-time retrieval/answering.
- `config.py`: Centralizes environment variable loading and default model names.
- `debug_conn.py`: Quick health check for Neo4j and Qdrant.

## Development with uv
- Add a new dependency: `uv add <package>` (updates `pyproject.toml` and regenerates the lock file automatically).
- Run ad-hoc scripts: `uv run python <script.py>`.
- Drop into the environment shell: `uv run -- python -i` or `uv run <command>` as required.

## Troubleshooting
- **Neo4j authentication errors**: confirm the credentials in `.env` and that the user has write permissions.
- **Qdrant connection timeouts**: ensure the service is reachable and, for Docker setups, that ports are exposed.
- **LLM failures**: check that the configured `OLLAMA_BASE_URL` is live and the referenced models (`nomic-embed-text`, `gemma3:1b-it-qat`) are pulled or available.
- **Streamlit caching issues**: use `st.cache_resource.clear()` from the Streamlit command palette or restart the server if infrastructure settings change.

## License
This project is distributed under the terms specified in the repository. Review `LICENSE` (if present) for details.

