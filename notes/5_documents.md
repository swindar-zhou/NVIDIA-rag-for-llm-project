## notebook 5: handling large documents

this picks up from the state chain pattern in notebook 4 and applies it to documents. the problem: LLMs have limited context windows, but real documents can be hundreds of pages. how do you process large documents intelligently?

## learning objectives
1. load and chunk documents for LLM processing
2. build a progressive summarizer that refines a running summary across many chunks
3. understand the tradeoffs between different document processing strategies

## thinking questions
1. how do you handle malformed or meaningless chunks from the splitter?
2. when should you run re-summarization vs. just retrieving chunks directly?
3. what are the limitations of LLM-based progressive summarization at scale?

## Part 1: document loaders

different loaders for different source types:

```python
from langchain.document_loaders import UnstructuredFileLoader, ArxivLoader

# Generic loader - works for many formats, less structured
docs = UnstructuredFileLoader("path/to/file.pdf").load()

# Specialized loader - better metadata, optimized parsing for that format
docs = ArxivLoader(query="2404.16130").load()  # loads by arXiv paper ID
```

each loaded document has:
- `page_content`: the text
- `metadata`: title, source, etc.

specialized loaders extract better metadata automatically. for arXiv, you get title, abstract, authors built-in.

## Part 2: text splitting

you can't feed a 100-page paper into an LLM at once. you split it into chunks:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,       # max chars per chunk
    chunk_overlap=100,     # overlap to preserve context at boundaries
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],  # try these in order
)

docs_split = text_splitter.split_documents(documents)
```

**recursive splitting logic**: try to split at paragraph breaks first, then sentence ends, then word boundaries. preserves natural structure better than fixed-size slicing.

**chunk overlap**: a small overlap prevents losing context that spans a chunk boundary. 100 chars is typical.

## Part 3: the DocumentSummaryBase schema

same pattern as notebook 4's KnowledgeBase, but for document content:

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="Running description of the document. Do not override; only update!")
    main_ideas: List[str] = Field([], description="Most important information from the document (max 3)")
    loose_ends: List[str] = Field([], description="Open questions yet unknown (max 3)")
```

the prompt instructs the LLM to update (not replace) the running_summary. `loose_ends` captures things mentioned but not fully explained yet — those get resolved in later chunks.

## Part 4: RSummarizer — progressive refinement

same structure as RStateKeeper from notebook 4:

```python
def RSummarizer(knowledge, llm, prompt, verbose=False):
    def summarize_docs(docs):
        # use knowledge.__class__ so we get the Pydantic class (DocumentSummaryBase)
        parse_chain = RunnableAssign({'info_base': RExtract(knowledge.__class__, llm, prompt)})

        # start with the provided (empty) knowledge instance
        state = {'info_base': knowledge}

        for i, doc in enumerate(docs):
            # each iteration: feed current summary + new chunk → get updated summary
            state = parse_chain.invoke({**state, 'input': doc.page_content})

        return state['info_base']

    return RunnableLambda(summarize_docs)
```

**the key insight**: each call to `parse_chain.invoke` gives the LLM:
- the current `info_base` (what we know so far)
- the new `input` (current chunk)
- `format_instructions` (schema to fill)

the LLM returns an updated `DocumentSummaryBase` that incorporates both old knowledge and new content. it's like a rolling window of understanding.

```python
summarizer = RSummarizer(DocumentSummaryBase(), instruct_llm, summary_prompt, verbose=True)
summary = summarizer.invoke(docs_split[:15])
```

## Part 5: limitations and tradeoffs

| strategy | pros | cons |
|---|---|---|
| full document stuffing | simple, no info loss | only works on short docs |
| progressive summarization | handles large docs | info compression = potential loss |
| map-reduce | parallelizable | combines independently summarized chunks, may miss connections |
| map-rerank | all chunks scored | expensive, output selection not synthesis |

**practical note**: progressive summarization works well for getting a "gist" but may lose specific details from early chunks. for retrieval purposes, you often want the raw chunks in a vector store (next notebook) rather than compressed summaries.

## connection to notebook 4

this is essentially the same loop pattern:

notebook 4: `state = parse_chain.invoke({**state, 'input': user_message})`
notebook 5: `state = parse_chain.invoke({**state, 'input': doc_chunk})`

the difference is the schema (`DocumentSummaryBase` vs `KnowledgeBase`) and the prompt. the loop structure is identical. this is the power of the Runnable abstraction.
