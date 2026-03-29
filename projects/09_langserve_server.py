"""
Notebook 9: LangServe RAG server (final assessment).

Run this file directly to start the FastAPI server:
    python 09_langserve_server.py

Endpoints:
    POST /basic_chat/invoke   — plain LLM, no retrieval
    POST /retriever/invoke    — takes a query string, returns Document list
    POST /generator/invoke    — takes {input, context}, returns answer string

Client-side usage (from notebook or frontend):
    from langserve import RemoteRunnable
    retriever = RemoteRunnable("http://localhost:9012/retriever/")
    generator = RemoteRunnable("http://localhost:9012/generator/")
"""

from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG LangServe",
    version="1.0",
    description="RAG endpoints for the NVIDIA DLI course final assessment",
)

# PRE-ASSESSMENT: basic chat with no retrieval
add_routes(app, instruct_llm, path="/basic_chat")

# ---------------------------------------------------------------------------
# Load vector store
# ---------------------------------------------------------------------------

docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
retriever = docstore.as_retriever()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def docs2str(docs, title="Document") -> str:
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, "metadata", {}).get("Title", title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, "page_content", str(doc)) + "\n"
    return out_str

# ---------------------------------------------------------------------------
# Generator prompt
# ---------------------------------------------------------------------------

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User just asked: {input}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Keep it conversational.)"
    "\n\nUser Question: {input}"
)

# ---------------------------------------------------------------------------
# ASSESSMENT endpoints
# ---------------------------------------------------------------------------

# /retriever: takes a string query, returns list of Document objects
retriever_chain = RunnableLambda(lambda x: retriever.invoke(x))

# /generator: takes {'input': str, 'context': str}, returns answer string
generator_chain = chat_prompt | llm

add_routes(app, generator_chain, path="/generator")
add_routes(app, retriever_chain, path="/retriever")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
