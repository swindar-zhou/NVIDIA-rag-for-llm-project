"""
Notebook 8: Evaluating RAG quality with LLM-as-a-Judge.

Demonstrates:
- Reconstructing a clean RAG chain (no conversation memory) for evaluation
- Generating synthetic QA pairs from the docstore
- Scoring RAG answers against ground truth using pairwise LLM evaluation
"""

import random
from operator import itemgetter

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS

from utils import docs2str

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Load docstore
# ---------------------------------------------------------------------------

def load_docstore(path="docstore_index"):
    return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)


def format_chunk(doc) -> str:
    return (
        f"Paper: {doc.metadata.get('Title', 'unknown')}"
        f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\nPage Body: {doc.page_content}"
    )


# ---------------------------------------------------------------------------
# RAG chain (no memory — isolation for evaluation)
# ---------------------------------------------------------------------------

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User just asked: {input}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Keep it conversational.)"
    "\n\nUser Question: {input}"
)


def output_puller(inputs):
    """Extract string output from a dict or stream of dicts with 'output' key."""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get("output"):
            yield token.get("output")


def build_rag_chain(docstore):
    long_reorder = RunnableLambda(LongContextReorder().transform_documents)
    context_getter = itemgetter("input") | docstore.as_retriever() | long_reorder | docs2str
    retrieval_chain = {"input": (lambda x: x)} | RunnableAssign({"context": context_getter})
    generator_chain = {"output": chat_prompt | llm} | RunnableLambda(output_puller)
    return retrieval_chain | generator_chain


# ---------------------------------------------------------------------------
# Synthetic QA generation
# ---------------------------------------------------------------------------

synth_sys = (
    "Use the documents provided by the user to generate an interesting question-answer pair."
    " Try to use both documents if possible, and rely more on the document bodies than the summary."
    " Use the format:\nQuestion: (good question, 1-3 sentences, detailed)\n\nAnswer: (answer derived from the documents)"
    " DO NOT SAY: 'Here is an interesting question pair' or similar. FOLLOW FORMAT!"
)

simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("user", "INPUT: {input}"),
])


def generate_synth_qa(docs, num_questions=3):
    """Generate synthetic QA pairs by asking the LLM to create questions from random doc pairs."""
    synth_questions, synth_answers = [], []

    for i in range(num_questions):
        doc1, doc2 = random.sample(docs, 2)
        user_msg = f"Document1: {format_chunk(doc1)}\n\nDocument2: {format_chunk(doc2)}"
        qa_pair = (simple_prompt | llm).invoke({"system": synth_sys, "input": user_msg})
        parts = qa_pair.split("\n\n")
        synth_questions.append(parts[0])
        synth_answers.append(parts[1] if len(parts) > 1 else "")
        print(f"\n--- QA Pair {i+1} ---")
        print(synth_questions[-1])
        print(synth_answers[-1])

    return synth_questions, synth_answers


# ---------------------------------------------------------------------------
# LLM-as-a-Judge evaluation
# ---------------------------------------------------------------------------

eval_prompt = ChatPromptTemplate.from_template(
    "INSTRUCTION:\n"
    "Evaluate the following Question-Answer pair for human preference and consistency.\n"
    "Assume the first answer is ground truth and correct.\n"
    "Assume the second answer may or may not be true.\n"
    "[1] The second answer lies, does not answer the question, or is inferior to the first.\n"
    "[2] The second answer is better than the first and does not introduce inconsistencies.\n\n"
    "Output Format:\n[Score] Justification\n\n"
    "{qa_trio}\n\nEVALUATION:"
)


def evaluate_rag(rag_chain, synth_questions, synth_answers):
    """Generate RAG answers and score them against synthetic ground truth."""
    rag_answers = []
    for i, q in enumerate(synth_questions):
        rag_answer = ""
        for token in rag_chain.stream(q):
            rag_answer += token
        rag_answers.append(rag_answer)
        print(f"\n--- QA Pair {i+1} ---")
        print(q)
        print(f"RAG Answer: {rag_answer[:300]}...")

    pref_scores = []
    for q, a_synth, a_rag in zip(synth_questions, synth_answers, rag_answers):
        trio = (
            f"Question: {q}\n\n"
            f"Answer 1 (Ground Truth): {a_synth}\n\n"
            f"Answer 2 (New Answer): {a_rag}"
        )
        score = (eval_prompt | llm).invoke({"qa_trio": trio})
        pref_scores.append(score)
        print(f"\nEvaluation: {score[:200]}")

    preference_score = sum("[2]" in s for s in pref_scores) / len(pref_scores)
    print(f"\n=== Preference Score: {preference_score:.2f} ===")
    print("(fraction of RAG answers rated better than synthetic ground truth)")
    return preference_score, rag_answers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading docstore...")
    docstore = load_docstore()
    docs = list(docstore.docstore._dict.values())
    print(f"Loaded {len(docs)} chunks.")

    rag_chain = build_rag_chain(docstore)

    print("\n=== Generating synthetic QA pairs ===")
    synth_questions, synth_answers = generate_synth_qa(docs, num_questions=3)

    print("\n=== Evaluating RAG chain ===")
    preference_score, rag_answers = evaluate_rag(rag_chain, synth_questions, synth_answers)
