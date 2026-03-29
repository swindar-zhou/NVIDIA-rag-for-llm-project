"""
Notebook 5: Progressive document summarization using RSummarizer.

Demonstrates:
- Loading and chunking documents (ArxivLoader)
- RSummarizer: iterative slot-filling over document chunks
- Same loop pattern as notebook 4, applied to documents instead of conversations
"""

from typing import List

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import RExtract

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

instruct_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1").bind(max_tokens=4096)
instruct_llm = instruct_model | StrOutputParser()

# ---------------------------------------------------------------------------
# Document summary schema
# ---------------------------------------------------------------------------

class DocumentSummaryBase(BaseModel):
    running_summary: str = Field(
        "",
        description="Running description of the document. Do not override; only update!",
    )
    main_ideas: List[str] = Field(
        [],
        description="Most important information from the document (max 3)",
    )
    loose_ends: List[str] = Field(
        [],
        description="Open questions not yet answered that would enrich the summary (max 3)",
    )


summary_prompt = ChatPromptTemplate.from_template(
    "You are generating a running summary of the document. Make it readable by a technical user."
    " After this, the old knowledge base will be replaced by the new one."
    " Keep it short, but as dense and useful as possible!"
    " The information should flow from chunk to (loose ends or main ideas) to running_summary."
    " The updated knowledge base keeps all information from running_summary here: {info_base}."
    "\n\n{format_instructions}. Follow the format precisely, including quotations and commas"
    "\n\nWithout losing any of the info, update the knowledge base with the following: {input}"
)

# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

# ---------------------------------------------------------------------------
# RSummarizer
# ---------------------------------------------------------------------------

latest_summary = ""  # global checkpoint — inspect if loop crashes


def RSummarizer(knowledge, llm, prompt, verbose=False):
    """
    Returns a Runnable that summarizes a list of Document chunks
    into a single DocumentSummaryBase (or compatible Pydantic schema).

    Args:
        knowledge: An initial (empty) Pydantic model instance — defines the schema.
        llm:       Language model to use for extraction.
        prompt:    Prompt template that uses {info_base}, {input}, {format_instructions}.
        verbose:   Print intermediate state after each chunk.
    """
    def summarize_docs(docs):
        global latest_summary

        # knowledge.__class__ gives us DocumentSummaryBase (or whatever was passed)
        parse_chain = RunnableAssign(
            {"info_base": RExtract(knowledge.__class__, llm, prompt)}
        )

        # start with the provided (empty) knowledge instance
        state = {"info_base": knowledge}

        for i, doc in enumerate(docs):
            # feed old summary + new chunk → get updated summary
            state = parse_chain.invoke({**state, "input": doc.page_content})

            assert "info_base" in state
            if verbose:
                print(f"\n--- Chunk {i+1}/{len(docs)} ---")
                print(f"Running summary: {state['info_base'].running_summary[:200]}...")
                latest_summary = state["info_base"]

        return state["info_base"]

    return RunnableLambda(summarize_docs)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def summarize_paper(arxiv_id: str = "2404.16130", num_chunks: int = 15):
    """
    Load an arXiv paper, chunk it, and produce a running summary.

    Default paper: GraphRAG (2404.16130)
    """
    from langchain.document_loaders import ArxivLoader

    print(f"Loading arXiv paper {arxiv_id}...")
    documents = ArxivLoader(query=arxiv_id).load()

    # strip references section if present
    content = documents[0].page_content
    if "References" in content:
        documents[0].page_content = content[:content.index("References")]

    docs_split = text_splitter.split_documents(documents)
    docs_split = [d for d in docs_split if len(d.page_content) > 200]
    print(f"Split into {len(docs_split)} chunks. Summarizing first {num_chunks}...")

    summarizer = RSummarizer(DocumentSummaryBase(), instruct_llm, summary_prompt, verbose=True)
    summary = summarizer.invoke(docs_split[:num_chunks])

    print("\n=== FINAL SUMMARY ===")
    print(f"Running Summary:\n{summary.running_summary}")
    print(f"\nMain Ideas:")
    for idea in summary.main_ideas:
        print(f"  - {idea}")
    print(f"\nLoose Ends:")
    for end in summary.loose_ends:
        print(f"  - {end}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summarize_paper()
