"""
Shared utilities used across multiple projects.
"""
from functools import partial
from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.output_parsers import PydanticOutputParser


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_and_return(x, preface=""):
    print(f"{preface}{x}")
    return x

def RPrint(preface=""):
    """Passthrough runnable that prints the value with an optional label."""
    return RunnableLambda(partial(print_and_return, preface=preface))


# ---------------------------------------------------------------------------
# Dictionary coercion helpers
# ---------------------------------------------------------------------------

def make_dictionary(v, key):
    if isinstance(v, dict):
        return v
    return {key: v}

def RInput(key="input"):
    """Coerce any value into {'input': value} (or given key)."""
    return RunnableLambda(partial(make_dictionary, key=key))

def ROutput(key="output"):
    """Coerce any value into {'output': value} (or given key)."""
    return RunnableLambda(partial(make_dictionary, key=key))


# ---------------------------------------------------------------------------
# RExtract: slot-filling extraction into a Pydantic schema
# ---------------------------------------------------------------------------

def RExtract(pydantic_class, llm, prompt):
    """
    Creates a Runnable that extracts structured data from conversation context
    into a Pydantic model via slot-filling.

    Input dict must contain the fields required by `prompt`.
    Returns an updated instance of `pydantic_class`.

    Usage:
        extractor = RExtract(MySchema, instruct_llm, my_prompt)
        result = extractor.invoke({"input": "...", "my_field": old_value, ...})
        # result is a MySchema instance
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign(
        {"format_instructions": lambda x: parser.get_format_instructions()}
    )

    def preparse(string):
        if "{" not in string:
            string = "{" + string
        if "}" not in string:
            string = string + "}"
        return (
            string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\\]", "]")
            .replace("\\[", "[")
        )

    return instruct_merge | prompt | llm | preparse | parser


# ---------------------------------------------------------------------------
# Document utilities
# ---------------------------------------------------------------------------

def docs2str(docs, title="Document"):
    """Convert a list of LangChain Documents into a formatted context string."""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, "metadata", {}).get("Title", title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, "page_content", str(doc)) + "\n"
    return out_str


# ---------------------------------------------------------------------------
# FAISS utilities
# ---------------------------------------------------------------------------

def default_FAISS(embedder):
    """Create an empty FAISS vectorstore with the correct embedding dimensions."""
    from faiss import IndexFlatL2
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    embed_dims = len(embedder.embed_query("test"))
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False,
    )


def aggregate_vstores(vectorstores, embedder):
    """Merge a list of FAISS vectorstores into one."""
    agg = default_FAISS(embedder)
    for vs in vectorstores:
        agg.merge_from(vs)
    return agg


# ---------------------------------------------------------------------------
# Gradio simulation helper
# ---------------------------------------------------------------------------

def queue_fake_streaming_gradio(chat_stream, history=None, max_questions=5):
    """
    Simulate a Gradio chat loop using Python's input().
    Useful for testing streaming chat functions without launching a UI.

    chat_stream: generator function(message, history, return_buffer=False)
    history: list of [human_msg, bot_msg] pairs
    """
    if history is None:
        history = []

    for human_msg, agent_msg in history:
        if human_msg:
            print(f"\n[ Human ]: {human_msg}")
        if agent_msg:
            print(f"\n[ Agent ]: {agent_msg}")

    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end="")
            history_entry[1] += token
        history.append(history_entry)
        print("\n")
