# app.py
import streamlit as st
from backend import GraphRAGEngine

# Page Configuration
st.set_page_config(page_title="GraphRAG Explorer", layout="wide")


@st.cache_resource
def get_engine():
    # Cached to prevent reloading connections on every interaction
    return GraphRAGEngine()


try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to connect to backend: {e}")
    st.stop()

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
        else:
            with st.spinner("Extracting Entities & Relationships..."):
                try:
                    result_msg = engine.ingest_text(raw_text)
                    st.success(result_msg)
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

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

    # 2. Generate & Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, context = engine.query(prompt)
                st.markdown(answer)
                with st.expander("View Graph Context Used"):
                    st.code(context)

                # Save to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context}
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")

