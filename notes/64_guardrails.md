## notebook 64: semantic guardrails

this notebook is an advanced/optional section. instead of letting the LLM decide whether to answer a question (which is slow and inconsistent), we train a fast embedding-based classifier to filter queries before they reach the LLM.

## learning objectives
1. generate synthetic good/bad query examples for a specific chatbot domain
2. build async embedding pipelines for speed
3. train a logistic regression classifier on embeddings
4. integrate the classifier as a semantic guardrail in a chat pipeline

## thinking questions
1. what are the tradeoffs between LLM-based filtering vs. embedding-based filtering?
2. how many synthetic examples do you need before accuracy plateaus?
3. could you reformulate rejected queries to be on-topic instead of just blocking them?

## Part 1: why embedding-based guardrails

**LLM filtering**: ask the LLM "is this query appropriate?" before answering
- pro: nuanced judgment, handles edge cases
- con: adds a full LLM call of latency for every message

**embedding-based filtering**: classify query embedding with a trained model
- pro: ~10ms vs ~500ms, no extra LLM call
- con: less nuanced, needs representative training data

for a production chatbot, the embedding approach is usually the right tradeoff.

## Part 2: generating synthetic training data

use an LLM to generate representative good and bad examples:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def EnumParser(*idxs):
    entry_parser = lambda v: v if ('. ' not in v) else v[v.index('. ')+2:]
    out_lambda = lambda x: [entry_parser(v).strip() for v in x.split("\n")]
    return StrOutputParser() | RunnableLambda(lambda x: itemgetter(*idxs)(out_lambda(x)))

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1") | EnumParser()

gen_prompt = {'input': lambda x: x} | ChatPromptTemplate.from_template(
    "Please generate 20 representative conversations that would be {input}."
    " Make sure all of the questions are very different. Do not respond to the questions; just list them."
    " Make sure all of your outputs are numbered."
)

# 4 categories of examples
good_responses = responses_1 + responses_2    # NVIDIA-relevant + general tech topics
poor_responses = responses_3 + responses_4    # irrelevant + harmful queries
```

generating from 4 categories:
1. NVIDIA domain questions → definitely answer
2. general tech questions → answer (within scope)
3. irrelevant questions (cooking, sports) → politely decline
4. harmful/offensive questions → politely decline

## Part 3: async embedding for speed

embedding 40-80 queries sequentially is slow. use `asyncio` with a semaphore to embed in parallel:

```python
import asyncio
from functools import partial

async def embed_with_semaphore(text, embed_fn, semaphore):
    async with semaphore:
        return await embed_fn(text)

# limit to 10 concurrent requests (found optimal in testing)
embed = partial(
    embed_with_semaphore,
    embed_fn=embedder.aembed_query,
    semaphore=asyncio.Semaphore(value=10)
)

with Timer():
    good_tasks = [embed(query) for query in good_responses]
    poor_tasks = [embed(query) for query in poor_responses]
    good_embeds = list(await asyncio.gather(*good_tasks))
    poor_embeds = list(await asyncio.gather(*poor_tasks))
```

**why semaphore**: without a limit, 80 concurrent requests can exceed the API rate limit. semaphore keeps at most N=10 in-flight at once.

**speedup**: ~8x faster than sequential embedding for 80 queries (measured with `Timer` class).

## Part 4: verify semantic clustering

before training, confirm that embeddings actually separate good from poor:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

embeddings = np.vstack([good_embeds, poor_embeds])
labels = np.array([0]*len(good_embeds) + [1]*len(poor_embeds))

# PCA: fast, linear
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# t-SNE: slower, better for clusters
tsne = TSNE(n_components=2, random_state=0)
embeddings_tsne = tsne.fit_transform(embeddings)
```

if the two categories form distinct clusters in the PCA/t-SNE plot, the embeddings have enough semantic density for a simple classifier to work.

## Part 5: training the classifier

logistic regression works well because embeddings are already high-quality representations:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_logistic_regression(class0, class1):
    x = class0 + class1
    y = [0] * len(class0) + [1] * len(class1)
    x0, x1, y0, y1 = train_test_split(x, y, test_size=0.5, random_state=42)
    model = LogisticRegression()
    model.fit(x0, y0)
    print("Training Results:", model.score(x0, y0))
    print("Testing Results:", model.score(x1, y1))
    return model

# class0 = poor (label 0), class1 = good (label 1)
model2 = train_logistic_regression(poor_embeds, good_embeds)
```

**note the class ordering**: `poor_embeds` is class0 (label=0) and `good_embeds` is class1 (label=1). this means `model2.predict_proba([emb])[0][1]` = P(query is good/relevant).

## Part 6: integrating into the chat pipeline

```python
def score_response(query):
    embedding = embedder.embed_query(query)
    # [0][0] = P(poor), [0][1] = P(good)
    score = model2.predict_proba([embedding])[0][1]
    return score

chat_chain = (
    {'input': (lambda x: x), 'score': score_response}
    | RunnableAssign(dict(
        system=RunnableBranch(
            # if score < 0.5 → query is probably irrelevant → use polite decline message
            ((lambda d: d['score'] < 0.5), RunnableLambda(lambda x: poor_sys_msg)),
            # default: score >= 0.5 → query is probably relevant → use helpful message
            RunnableLambda(lambda x: good_sys_msg)
        )
    ))
    | response_prompt
    | chat_llm
)
```

**key design**: we don't hard-block the query. we modify the **system prompt** based on the score:
- high score → "You are an NVIDIA chatbot, please help them"
- low score → "Their question is probably not relevant; politely explain why you can't help"

the LLM still generates a response, but with different instructions. this is softer than a hard block and allows the LLM to give a contextual explanation.

## architecture summary

```
user query
    ↓
[embedder.embed_query] (~10ms)
    ↓
[model2.predict_proba] (~1ms)
    ↓
score >= 0.5? → good_sys_msg → LLM responds helpfully
score <  0.5? → poor_sys_msg → LLM politely declines
```

total overhead for classification: ~11ms vs ~500ms for LLM-based filtering.
