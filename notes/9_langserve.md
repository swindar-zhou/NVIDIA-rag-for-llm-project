## notebook 9: serving RAG with LangServe (final assessment)

this is the capstone notebook. we take the RAG system from notebook 7 and deploy it as a REST API using **LangServe** + **FastAPI**, then wire it to the course's frontend for the final assessment.

## learning objectives
1. deploy LangChain runnables as HTTP endpoints with LangServe
2. implement `/retriever` and `/generator` endpoints separately
3. understand how a frontend consumes these endpoints to build a RAG pipeline

## Part 1: LangServe architecture

```
client (frontend)
    ↓ POST /retriever/invoke {"input": "query string"}
[FastAPI server]
    → calls retriever_chain.invoke("query string")
    → returns list of Document objects
    ↓
client gets documents, formats context string
    ↓ POST /generator/invoke {"input": {"input": "...", "context": "..."}}
[FastAPI server]
    → calls generator_chain.invoke({"input": "...", "context": "..."})
    → returns answer string
```

separating retrieval from generation allows:
- the frontend to display which documents were used
- independent scaling of retrieval vs. generation
- easy swapping of either component

## Part 2: server implementation (server_app.py)

```python
from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

app = FastAPI(title="LangChain Server", version="1.0")

# pre-assessment: basic chat endpoint (no RAG)
add_routes(app, instruct_llm, path="/basic_chat")

# load the vector store built in notebook 07
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
retriever = docstore.as_retriever()

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used.)"
    "\n\nUser Question: {input}"
)

# /retriever: takes a string query, returns list of Document objects
retriever_chain = RunnableLambda(lambda x: retriever.invoke(x))

# /generator: takes {'input': str, 'context': str}, returns answer string
generator_chain = chat_prompt | llm

add_routes(app, generator_chain, path="/generator")
add_routes(app, retriever_chain, path="/retriever")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
```

## Part 3: client-side RAG assembly

the frontend combines the two endpoints into a full RAG pipeline:

```python
from langserve import RemoteRunnable
from langchain.document_transformers import LongContextReorder

chains_dict = {
    'basic':     RemoteRunnable("http://lab:9012/basic_chat/"),
    'retriever': RemoteRunnable("http://lab:9012/retriever/"),
    'generator': RemoteRunnable("http://lab:9012/generator/"),
}

# step 1: retrieve documents from server
retrieval_chain = (
    {'input': (lambda x: x)}
    | RunnableAssign({'context':
        itemgetter('input')
        | chains_dict['retriever']   # returns list of docs from server
        | LongContextReorder().transform_documents
        | docs2str                   # format as context string
    })
)

# step 2: generate answer using context
output_chain = RunnableAssign({"output": chains_dict['generator']}) | output_puller

rag_chain = retrieval_chain | output_chain
```

**important**: the frontend's `rag_chain` composes the two remote calls. each is a separate HTTP request. the client handles the formatting glue between them.

## Part 4: running and testing the server

in the notebook, the server is started with:
```
!python server_app.py &   # background process
```

or via the cell that writes `server_app.py` then runs it via uvicorn.

to test directly:
```python
import requests

# test retriever
r = requests.post("http://localhost:9012/retriever/invoke", json={"input": "What is RAG?"})
docs = r.json()["output"]

# test generator
r = requests.post("http://localhost:9012/generator/invoke",
                  json={"input": {"input": "What is RAG?", "context": "..."}})
answer = r.json()["output"]
```

## Part 5: model selection tips

if the server throws timeout errors, the LLM is too slow. try smaller/faster models:

| model | speed | quality |
|---|---|---|
| `meta/llama-3.1-8b-instruct` | fast | good |
| `mistralai/mistral-7b-instruct-v0.3` | fast | good |
| `meta/llama3-8b-instruct` | fast | good |
| `mistralai/mixtral-8x7b-instruct-v0.1` | medium | better |

for the assessment, prioritize responsiveness over quality — the evaluator checks that the endpoints work, not that the answers are perfect.
