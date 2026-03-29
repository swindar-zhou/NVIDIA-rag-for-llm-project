## notebook 7: vector stores and RAG retrieval

this is where all previous pieces come together into a full RAG pipeline. we embed a corpus of documents, store them in a vector store, and wire retrieval into a chat system.

## learning objectives
1. build and query a FAISS vector store
2. implement a RAG retrieval chain with conversation memory
3. understand LongContextReorder and why it matters

## thinking questions
1. when should you pre-compute document embeddings vs. compute at query time?
2. how do you decide on chunk size given the retrieval model and LLM context window?
3. at what scale does a local FAISS store become insufficient vs. a managed vector DB?

## Part 1: building the vector store

```python
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

# Load and chunk multiple papers
docs = [ArxivLoader(query=paper_id).load() for paper_id in paper_ids]
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]

# Build FAISS stores per paper, then merge
vecstores = [FAISS.from_documents(chunks, embedder) for chunks in docs_chunks]
docstore = aggregate_vstores(vecstores)  # merges all FAISS indexes

# Save for reuse (avoid re-embedding on every run)
docstore.save_local("docstore_index")
```

```python
def default_FAISS():
    """Make an empty FAISS store with the right dimensions"""
    from faiss import IndexFlatL2
    from langchain_community.docstore.in_memory import InMemoryDocstore
    embed_dims = len(embedder.embed_query("test"))
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    agg = default_FAISS()
    for vs in vectorstores:
        agg.merge_from(vs)
    return agg
```

## Part 2: LongContextReorder

LLMs don't attend equally to all positions in the context window. they tend to focus more on the **beginning and end**, and miss content in the middle ("lost in the middle" problem).

`LongContextReorder` reorders retrieved documents so the most relevant are at the edges:

```python
from langchain.document_transformers import LongContextReorder
long_reorder = RunnableLambda(LongContextReorder().transform_documents)
```

takes list of docs (most relevant first from retriever) and reorders to: `[2nd, 4th, ..., 5th, 3rd, 1st]` — most relevant at positions 0 and -1.

## Part 3: the retrieval chain

this is the core RAG implementation:

```python
from operator import itemgetter

retrieval_chain = (
    {'input': (lambda x: x)}
    # retrieve conversation history from convstore (short-term memory)
    | RunnableAssign({'history': itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    # retrieve relevant document chunks from docstore (long-term knowledge)
    | RunnableAssign({'context': itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str})
)
```

**data flow:**
```
query (str)
    → {'input': query}
    → {'input': query, 'history': "<past conversation relevant to this>"}
    → {'input': query, 'history': "...", 'context': "<document chunks relevant to this>"}
```

`docs2str` converts a list of Document objects into a formatted string:

```python
def docs2str(docs, title="Document"):
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str
```

## Part 4: the full chat pipeline

```python
chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used.)"
), ('user', '{input}')])

stream_chain = chat_prompt | instruct_llm | StrOutputParser()

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    retrieval = retrieval_chain.invoke(message)         # step 1: retrieve
    for token in stream_chain.stream(retrieval):        # step 2: generate
        buffer += token
        yield buffer if return_buffer else token
    save_memory_and_get_output({'input': message, 'output': buffer}, convstore)  # step 3: save
```

## Part 5: conversation memory as a vector store

instead of a simple FIFO buffer, conversation history is also stored in a vector store (`convstore`):

```python
def save_memory_and_get_output(d, vstore):
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')
```

this means memory retrieval is **semantic** — past exchanges most relevant to the current query come back, not just the last N turns. much better for long conversations with topic shifts.

## RAG architecture summary

```
User Query
    ↓
[convstore retriever] → conversation history (semantic search)
[docstore retriever]  → relevant document chunks (semantic search)
    ↓
[LongContextReorder] → reorder for LLM attention
    ↓
[chat_prompt] → format all into a prompt
    ↓
[LLM] → generate grounded response
    ↓
[save to convstore] → update memory
```

everything is stateless per call (no global state like notebook 4) — retrieval provides the context instead.
