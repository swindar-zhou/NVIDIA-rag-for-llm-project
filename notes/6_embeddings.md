## notebook 6: embeddings and semantic reasoning

this notebook introduces embeddings — vector representations of text that capture semantic meaning. embeddings are the core technology behind retrieval in RAG systems.

## learning objectives
1. understand what embeddings are and how they encode meaning
2. use NVIDIA embedding models to embed queries and documents
3. measure semantic similarity via cosine similarity
4. use LLM-generated content to enrich documents before embedding

## thinking questions
1. why does matching a short query to a long document work at all given the length mismatch?
2. can you use embeddings to detect topics or languages in documents without reading them?
3. how do embeddings complement the running state chains from earlier notebooks?

## Part 1: what are embeddings?

a neural network trained on language builds internal representations called **embeddings** — dense vectors (typically 1024–4096 dimensions) where semantically similar content maps to nearby points in space.

```
"cat" ──────→ [0.12, -0.45, 0.78, ...]
"dog" ──────→ [0.15, -0.42, 0.81, ...]  ← nearby!
"car" ──────→ [-0.83, 0.24, -0.17, ...]  ← far away
```

**why this works for RAG**: instead of exact keyword matching, we can find documents whose *meaning* is similar to the query, even if the words differ.

## Part 2: encoder vs. decoder models

two types of transformer models:

| | Encoder | Decoder |
|---|---|---|
| reads | bidirectional (full context) | left-to-right only |
| good at | understanding, classification, embeddings | generation |
| examples | BERT, nv-embed | GPT, Llama, Mixtral |

embedding models are typically encoders. they read the full input simultaneously and produce a single fixed-size vector.

## Part 3: dual-encoder architecture

NVIDIA's `nv-embed-v1` uses a **dual-encoder** with two different paths:
- **query path** (`embed_query`): optimized for short questions
- **document path** (`embed_documents`): optimized for longer passages

```python
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# Short queries → query path
q_embeddings = [embedder.embed_query(query) for query in queries]

# Documents → document path
d_embeddings = embedder.embed_documents(documents)
```

**important**: the two paths are trained together to maximize dot-product similarity between matching pairs. this means the output vectors are in the same space but measured differently — intentional, not a bug.

## Part 4: cosine similarity

to compare embeddings, use cosine similarity (angle between vectors, ignoring magnitude):

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# rows = queries, cols = documents
cross_similarity = cosine_similarity(np.array(q_embeddings), np.array(d_embeddings))
```

a well-trained embedding model produces a near-diagonal similarity matrix: `cross_similarity[i][i]` (matching pairs) is high, off-diagonal is low.

## Part 5: document expansion with LLM (expound_chain)

short documents can be hard to embed meaningfully — too little signal. one technique: use an LLM to **expand** a short document into a richer story before embedding.

```python
expound_prompt = ChatPromptTemplate.from_template(
    "Generate part of a longer story that could reasonably answer all"
    " of these questions somewhere in its contents: {questions}\n"
    " Make sure the passage only answers the following concretely: {q1}."
    " Give it some weird formatting, and try not to answer the others."
    " Do not include any commentary like 'Here is your response'"
)

expound_chain = expound_prompt | instruct_llm | StrOutputParser()

longer_docs = []
for i, q in enumerate(queries):
    longer_doc = expound_chain.invoke({
        'questions': "\n".join(queries),  # all queries for plausible context
        'q1': q                           # this specific query to answer concretely
    })
    longer_docs += [longer_doc]
```

**why this works**: the expanded story has more semantic signal. the embedding has more to work with. retrieval performance typically improves for short/ambiguous documents.

**tradeoff**: adds latency (one LLM call per document). only worth it when documents are too short for reliable embedding.

## Part 6: embedding analysis

after embedding, visualize to verify quality:

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

q_embs = np.array(q_embeddings)
d_embs = np.array(d_embeddings)

pca = PCA(n_components=2)
all_embs = pca.fit_transform(np.vstack([q_embs, d_embs]))

plt.scatter(all_embs[:len(q_embs), 0], all_embs[:len(q_embs), 1], label='queries')
plt.scatter(all_embs[len(q_embs):, 0], all_embs[len(q_embs):, 1], label='documents')
```

if matching query-document pairs cluster together, the embedding model is working well for your domain.
