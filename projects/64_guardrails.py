"""
Notebook 64: Semantic guardrails using embedding-based classification.

Demonstrates:
- Generating synthetic good/bad query examples
- Async embedding with asyncio + semaphore for throughput
- Training a logistic regression classifier on embeddings
- Integrating the classifier into a chat pipeline as a guardrail
"""

import asyncio
import time
from functools import partial
from typing import Callable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.runnables.passthrough import RunnableAssign

from utils import RPrint, queue_fake_streaming_gradio

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
chat_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1") | StrOutputParser()

# ---------------------------------------------------------------------------
# Async embedding utilities
# ---------------------------------------------------------------------------

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"\033[1mExecuted in {elapsed:.2f} seconds.\033[0m")


async def embed_with_semaphore(text: str, embed_fn: Callable, semaphore: asyncio.Semaphore):
    async with semaphore:
        return await embed_fn(text)


def make_embed_fn(max_concurrent: int = 10) -> Callable:
    """Return an async embed function limited to max_concurrent parallel requests."""
    return partial(
        embed_with_semaphore,
        embed_fn=embedder.aembed_query,
        semaphore=asyncio.Semaphore(value=max_concurrent),
    )


async def embed_all(texts: List[str], max_concurrent: int = 10) -> List[List[float]]:
    """Embed all texts concurrently with a semaphore limit."""
    embed = make_embed_fn(max_concurrent)
    tasks = [embed(t) for t in texts]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_queries(description: str, n: int = 20) -> List[str]:
    """Ask the LLM to generate n example queries matching a description."""
    gen_prompt = ChatPromptTemplate.from_template(
        "Please generate {n} representative questions that would be {description}."
        " Make sure all of the questions are very different in phrasing and content."
        " Do not respond to the questions; just list them."
        " Number each question: 1. ... 2. ... 3. ..."
    )
    raw = (gen_prompt | instruct_llm).invoke({"description": description, "n": n})
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    parsed = []
    for line in lines:
        if ". " in line:
            parsed.append(line[line.index(". ") + 2:].strip())
    return parsed[:n]


def build_training_data():
    """Generate good (in-scope) and poor (out-of-scope) query examples."""
    print("Generating NVIDIA-relevant queries...")
    good_1 = generate_queries(
        "reasonable for an NVIDIA document chatbot to answer"
        " — vary topics: deep learning, GPUs, research, gaming"
    )
    print("Generating general tech queries...")
    good_2 = generate_queries(
        "reasonable for a tech document chatbot to answer"
        " — vary topics: technology, research, graphics, language models"
    )
    print("Generating irrelevant queries...")
    poor_1 = generate_queries(
        "unreasonable for an NVIDIA chatbot to answer — irrelevant but not harmful"
    )
    print("Generating harmful queries...")
    poor_2 = generate_queries(
        "unreasonable for any chatbot to answer — insensitive or offensive"
    )

    good_responses = good_1 + good_2
    poor_responses = poor_1 + poor_2
    print(f"Generated {len(good_responses)} good + {len(poor_responses)} poor queries.")
    return good_responses, poor_responses


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_logistic_regression(poor_embeds: list, good_embeds: list) -> LogisticRegression:
    """
    Train a logistic regression classifier.

    class0 = poor_embeds (label 0 = irrelevant/harmful)
    class1 = good_embeds (label 1 = relevant)

    predict_proba(emb)[0][1] → P(relevant)
    """
    x = poor_embeds + good_embeds
    y = [0] * len(poor_embeds) + [1] * len(good_embeds)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    print(f"Training accuracy: {model.score(x_train, y_train):.3f}")
    print(f"Testing accuracy:  {model.score(x_test, y_test):.3f}")
    return model


# ---------------------------------------------------------------------------
# Guardrailed chat chain
# ---------------------------------------------------------------------------

good_sys_msg = (
    "You are an NVIDIA chatbot. Please answer their question while representing NVIDIA."
    " Please help them with their question if it is ethical and relevant."
)

poor_sys_msg = (
    "You are an NVIDIA chatbot. Please answer their question while representing NVIDIA."
    " Their question has been analyzed and labeled as 'probably not useful to answer"
    " as an NVIDIA Chatbot', so avoid answering if appropriate and explain your reasoning."
    " Make your response as short as possible."
)

response_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("user", "{input}"),
])


def build_guardrailed_chain(classifier, embedder_instance):
    """
    Build a chat chain that classifies queries before answering.

    score >= 0.5 → answer helpfully (good_sys_msg)
    score <  0.5 → politely decline (poor_sys_msg)
    """
    def score_response(query: str) -> float:
        embedding = embedder_instance.embed_query(query)
        return classifier.predict_proba([embedding])[0][1]  # P(relevant)

    return (
        {"input": (lambda x: x), "score": score_response}
        | RPrint("State: ")
        | RunnableAssign(dict(
            system=RunnableBranch(
                ((lambda d: d["score"] < 0.5), RunnableLambda(lambda x: poor_sys_msg)),
                RunnableLambda(lambda x: good_sys_msg),
            )
        ))
        | response_prompt
        | chat_llm
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    # 1. generate training data
    good_responses, poor_responses = build_training_data()

    # 2. embed all queries asynchronously
    print("\nEmbedding good queries...")
    with Timer():
        good_embeds = await embed_all(good_responses)

    print("Embedding poor queries...")
    with Timer():
        poor_embeds = await embed_all(poor_responses)

    print(f"Good embeds shape: {np.array(good_embeds).shape}")
    print(f"Poor embeds shape: {np.array(poor_embeds).shape}")

    # 3. train classifier
    print("\nTraining logistic regression classifier...")
    classifier = train_logistic_regression(poor_embeds, good_embeds)

    # 4. build and test the guardrailed chatbot
    print("\nStarting guardrailed chatbot...")
    chat_chain = build_guardrailed_chain(classifier, embedder)

    def chat_gen(message, history, return_buffer=True):
        buffer = ""
        for token in chat_chain.stream(message):
            buffer += token
            yield buffer if return_buffer else token

    history = [[None, "Hello! I'm your NVIDIA chat agent! Let me answer some questions!"]]
    queue_fake_streaming_gradio(chat_gen, history=history, max_questions=6)


if __name__ == "__main__":
    asyncio.run(main())
