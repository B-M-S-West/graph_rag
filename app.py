# app.py
import streamlit as st
from backend import GraphRAGEngine
from loguru import logger

# Global Logger Configuration
logger.add("graphrag_app.log", rotation="1 MB", level="INFO")

# Page Configuration
st.set_page_config(page_title="GraphRAG Explorer", layout="wide")


@st.cache_resource
def get_engine():
    # Cached to prevent reloading connections on every interaction
    print("--- DEBUG: Starting engine initialization. ---")
    logger.info("Initializing GraphRAGEngine instance...")
    try:
        engine = GraphRAGEngine()
    except Exception as e:
        logger.error(f"Error initializing GraphRAGEngine: {e}")
        print(f"--- FATAL ERROR: Engine initialization failed with: {e} ---")
        raise
    logger.info("GraphRAGEngine instance created successfully.")
    print("--- DEBUG: Engine initialization completed. ---")
    return engine

engine = get_engine()

st.title("🕸️ GraphRAG Explorer")
st.markdown("Interact with your Neo4j + Qdrant Knowledge Graph.")

# --- Sidebar: Data Ingestion ---
with st.sidebar:
    st.header("📥 Ingest Knowledge")
    st.info("Paste text here to extract nodes and build the graph.")

    raw_text = st.text_area(
        "Raw Text Input", height=250, placeholder="Alice works at TechCorp..."
    )

    if st.button("Ingest Data"):
        if not raw_text.strip():
            st.warning("Please enter some text.")
            logger.warning("Ingestion attempt blocked: empty text input.")
        else:
            logger.info(
                f"User requested ingestion of text (length: {len(raw_text)} characters)."
            )
            with st.spinner("Extracting Entities & Relationships..."):
                try:
                    result_msg = engine.ingest_text(raw_text)
                    st.success(result_msg)
                    logger.success("Data ingestion completed successfully.")
                except Exception:
                    st.error("Ingestion failed. Check graphrag_app.log for details.")
                    logger.error("Ingestion failed in Streamlit UI layer")

# --- Main Area: Chat Interface ---
st.subheader("💬 Chat with your Graph")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("View Graph Context Used"):
                st.code(message["context"])

# Chat Logic
if prompt := st.chat_input("Ask a question about the ingested data..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    logger.info(f"User submitted query: {prompt}")

    # 2. Generate & Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, context = engine.query_graph(prompt)
                st.markdown(answer)
                with st.expander("View Graph Context Used"):
                    st.code(context)

                logger.success("Generated response successfully.")

                # Save to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context}
                )
            except Exception:
                st.error("Query failed. Check graphrag_app.log for details.")
                logger.error("Query failed in Streamlit UI layer.")
