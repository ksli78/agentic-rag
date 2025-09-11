# -*- coding: utf-8 -*-
import streamlit as st
from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.ollama import Ollama
from agno.vectordb.lancedb import LanceDb, SearchType

st.set_page_config(page_title="Agentic RAG with EmbeddingGemma", layout="wide")
st.title("Agentic RAG with Google's EmbeddingGemma (local)")

@st.cache_resource
def load_kb(urls):
    return PDFUrlKnowledgeBase(
        urls=urls,
        vector_db=LanceDb(
            table_name="kb",
            uri="data/lancedb",
            search_type=SearchType.vector,
            embedder=OllamaEmbedder(
                id="embeddinggemma:latest",
                dimensions=768  # NOTE: no base_url here
            ),
        ),
    )

if "urls" not in st.session_state:
    st.session_state["urls"] = []

with st.sidebar:
    st.header("Add Knowledge Sources")
    new_url = st.text_input("PDF URL", placeholder="https://example.com/file.pdf")
    if st.button("+ Add URL"):
        if new_url:
            st.session_state["urls"].append(new_url)
            st.success(f"Added: {new_url}")
        else:
            st.error("Please enter a URL")

kb = load_kb(st.session_state["urls"])

if st.button("Build/Update KB"):
    with st.spinner("Loading and embedding..."):
        kb.load(recreate=False, upsert=True)
    st.success("KB ready")

agent = Agent(
    model=Ollama(id="llama3.2:latest"),  # NOTE: no base_url here either
    knowledge=kb,
    instructions=[
        "Search the knowledge base for relevant information and base your answers on it.",
        "Be clear, and generate well-structured answers.",
    ],
    search_knowledge=True,
    markdown=True,
)

query = st.text_input("Ask a question about your documents:")
if st.button("Get Answer"):
    if not query:
        st.error("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            out = ""
            box = st.empty()
            for chunk in agent.run(query, stream=True):
                if chunk.content:
                    out += chunk.content
                    box.markdown(out)
