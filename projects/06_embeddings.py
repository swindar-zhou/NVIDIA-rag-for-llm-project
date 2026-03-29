"""
Notebook 6: Embeddings, semantic similarity, and document expansion.

Demonstrates:
- NVIDIAEmbeddings dual-encoder (query vs document path)
- Cosine similarity cross-matrix visualization
- Document expansion (expound_chain) to enrich short documents before embedding
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

QUERIES = [
    "What's the weather like in Rocky Mountains?",
    "What kinds of food is Italy known for?",
    "What's my name? I bet you don't remember...",
    "What's the point of life anyways?",
    "The point of life is to have fun :D",
]

DOCUMENTS = [
    "Kamchatka's weather is cold, with long, severe winters.",
    "Italy is famous for pasta, pizza, gelato, and espresso.",
    "I can't recall personal names, only provide information.",
    "Life's purpose varies, often seen as personal fulfillment.",
    "Enjoying life's moments is indeed a wonderful approach.",
]

# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def embed_queries_and_docs(queries, documents):
    """Embed queries and documents using their respective paths."""
    q_embs = [embedder.embed_query(q) for q in queries]
    d_embs = embedder.embed_documents(documents)
    return q_embs, d_embs


def plot_cross_similarity(emb1, emb2, title="Cross-Similarity Matrix"):
    """Visualize cosine similarity between two sets of embeddings."""
    matrix = cosine_similarity(np.array(emb1), np.array(emb2))
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Greens", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.title(title)
    plt.xlabel("Documents")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.show()
    return matrix


# ---------------------------------------------------------------------------
# Document expansion (expound_chain)
# ---------------------------------------------------------------------------

expound_prompt = ChatPromptTemplate.from_template(
    "Generate part of a longer story that could reasonably answer all"
    " of these questions somewhere in its contents: {questions}\n"
    " Make sure the passage only answers the following concretely: {q1}."
    " Give it some weird formatting, and try not to answer the others."
    " Do not include any commentary like 'Here is your response'"
)

# Pipe: format prompt → call LLM → extract string
expound_chain = expound_prompt | instruct_llm | StrOutputParser()


def expand_documents(queries, verbose=True):
    """
    Expand each short query into a longer story passage.

    Each expansion answers its own query concretely but only vaguely hints
    at the others — this makes retrieval harder and tests embedding quality.
    """
    all_questions = "\n".join(queries)
    longer_docs = []

    for i, q in enumerate(queries):
        doc = expound_chain.invoke({
            "questions": all_questions,  # context: all queries exist in the story
            "q1": q,                     # this specific query must be answered clearly
        })
        if verbose:
            print(f"\n[Query {i+1}] {q}")
            print(f"[Document {i+1}] {doc[:200]}...")
            print("-" * 64)
        longer_docs.append(doc)

    return longer_docs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_basic_similarity():
    print("Embedding queries and documents...")
    q_embs, d_embs = embed_queries_and_docs(QUERIES, DOCUMENTS)
    matrix = plot_cross_similarity(q_embs, d_embs, "Short Doc Similarity")
    print(f"\nDiagonal (matching pairs): {[round(matrix[i][i], 3) for i in range(len(QUERIES))]}")
    return q_embs, d_embs


def demo_expound_and_compare():
    print("Expanding documents with LLM...")
    longer_docs = expand_documents(QUERIES)

    print("\nEmbedding expanded documents...")
    longer_docs_cut = [d[:2048] for d in longer_docs]
    q_long_embs = [embedder.embed_query(d) for d in longer_docs_cut]
    d_long_embs = embedder.embed_documents(longer_docs_cut)

    print("\nShort doc similarity:")
    q_embs, d_embs = embed_queries_and_docs(QUERIES, DOCUMENTS)
    short_matrix = cosine_similarity(np.array(q_embs), np.array(d_embs))
    print(f"  Diagonal: {[round(short_matrix[i][i], 3) for i in range(len(QUERIES))]}")

    print("\nExpanded doc similarity:")
    long_matrix = cosine_similarity(np.array(q_long_embs), np.array(d_long_embs))
    print(f"  Diagonal: {[round(long_matrix[i][i], 3) for i in range(len(QUERIES))]}")

    return longer_docs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo: basic similarity ===")
    demo_basic_similarity()

    print("\n=== Demo: expand docs and compare ===")
    demo_expound_and_compare()
