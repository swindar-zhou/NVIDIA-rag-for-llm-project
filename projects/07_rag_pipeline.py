"""
Notebook 7: Full RAG pipeline with FAISS vector store and conversation memory.

Demonstrates:
- Building and saving a multi-paper FAISS docstore
- Retrieval chain: semantic history (convstore) + document context (docstore)
- LongContextReorder to optimize LLM attention
- Conversation memory via vector store
"""

import json
from operator import itemgetter

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.document_transformers import LongContextReorder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils import docs2str, default_FAISS, aggregate_vstores, RPrint, queue_fake_streaming_gradio

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# ---------------------------------------------------------------------------
# Build the document store
# ---------------------------------------------------------------------------

# arXiv paper IDs: (Attention, BERT, RAG, MRKL, Mistral, LLM-as-Judge)
PAPER_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.11401",  # RAG paper
    "2205.00445",  # MRKL
    "2310.06825",  # Mistral
    "2306.05685",  # LLM-as-a-Judge
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)


def build_docstore(paper_ids=PAPER_IDS, save_path="docstore_index"):
    """Load arXiv papers, chunk, embed, and save a FAISS docstore."""
    from langchain.document_loaders import ArxivLoader

    print(f"Loading {len(paper_ids)} papers...")
    docs = []
    for pid in paper_ids:
        print(f"  {pid}...")
        doc = ArxivLoader(query=pid).load()
        content = json.dumps(doc[0].page_content)
        if "References" in content:
            doc[0].page_content = content[: content.index("References")]
        docs.append(doc)

    docs_chunks = [text_splitter.split_documents(d) for d in docs]
    docs_chunks = [[c for c in chunks if len(c.page_content) > 200] for chunks in docs_chunks]

    print("Building FAISS vectorstores...")
    vecstores = [FAISS.from_documents(chunks, embedder) for chunks in docs_chunks]
    docstore = aggregate_vstores(vecstores, embedder)

    print(f"Saving to {save_path}/...")
    docstore.save_local(save_path)
    print(f"Done. {len(docstore.docstore._dict)} chunks indexed.")
    return docstore


def load_docstore(path="docstore_index"):
    return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

long_reorder = RunnableLambda(LongContextReorder().transform_documents)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document chatbot. Help the user as they ask questions about documents."
     " User just asked: {input}\n\n"
     " Conversation History Retrieval:\n{history}\n\n"
     " Document Retrieval:\n{context}\n\n"
     " (Answer only from retrieval. Only cite sources that are used. Keep it conversational.)"),
    ("user", "{input}"),
])


def build_retrieval_chain(convstore, docstore):
    """
    Build the retrieval chain that fetches:
      - history: semantically relevant past exchanges (from convstore)
      - context: relevant document chunks (from docstore)
    """
    return (
        {"input": (lambda x: x)}
        | RunnableAssign({
            "history": itemgetter("input") | convstore.as_retriever() | long_reorder | docs2str
        })
        | RunnableAssign({
            "context": itemgetter("input") | docstore.as_retriever() | long_reorder | docs2str
        })
    )


def save_memory_and_get_output(d: dict, vstore: FAISS) -> str:
    """Store the exchange as two entries in the conversation vector store."""
    vstore.add_texts([
        f"User previously said: {d.get('input')}",
        f"Agent previously said: {d.get('output')}",
    ])
    return d.get("output")


def make_chat_gen(retrieval_chain, stream_chain, convstore):
    """Return a chat_gen function bound to the given chains and stores."""
    def chat_gen(message: str, history=None, return_buffer: bool = True):
        buffer = ""
        retrieval = retrieval_chain.invoke(message)
        for token in stream_chain.stream(retrieval):
            buffer += token
            yield buffer if return_buffer else token
        save_memory_and_get_output({"input": message, "output": buffer}, convstore)

    return chat_gen


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_rag_chatbot(docstore_path="docstore_index"):
    """
    Run an interactive RAG chatbot over the indexed papers.

    Build the docstore first if it doesn't exist:
        docstore = build_docstore()
    """
    print("Loading docstore...")
    docstore = load_docstore(docstore_path)
    doc_titles = list({
        v.metadata.get("Title", "Unknown")
        for v in docstore.docstore._dict.values()
    })
    print(f"Loaded {len(docstore.docstore._dict)} chunks from: {doc_titles}")

    convstore = default_FAISS(embedder)
    retrieval_chain = build_retrieval_chain(convstore, docstore)
    stream_chain = chat_prompt | RPrint() | instruct_llm | StrOutputParser()
    chat_gen = make_chat_gen(retrieval_chain, stream_chain, convstore)

    initial_msg = f"Hello! I have access to papers on: {', '.join(doc_titles)}. How can I help?"
    history = [[None, initial_msg]]
    queue_fake_streaming_gradio(chat_gen, history=history, max_questions=6)


if __name__ == "__main__":
    import os

    if not os.path.exists("docstore_index"):
        print("Docstore not found — building from arXiv papers (this takes a few minutes)...")
        build_docstore()

    run_rag_chatbot()
