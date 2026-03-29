"""
Notebook 3: LangChain LCEL patterns and the Rhyme Re-themer chatbot.

Demonstrates:
- RunnableLambda composition with the | operator
- Dict-based workflows (RInput, ROutput, itemgetter)
- Simple chat chain (prompt | llm | parser)
- Internal zero-shot classification
- Multi-turn streaming chatbot with chain branching
"""

from functools import partial
from operator import itemgetter

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign

from utils import RPrint, RInput, ROutput, queue_fake_streaming_gradio

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

chat_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")


# ---------------------------------------------------------------------------
# Example 1: simple rhyme chain
# ---------------------------------------------------------------------------

rhyme_prompt = ChatPromptTemplate.from_messages([
    ("system", "Only respond in rhymes"),
    ("user", "{input}"),
])

rhyme_chain = rhyme_prompt | chat_llm | StrOutputParser()


def demo_rhyme_chain():
    result = rhyme_chain.invoke({"input": "Tell me about birds!"})
    print(result)


# ---------------------------------------------------------------------------
# Example 2: internal zero-shot classification
# ---------------------------------------------------------------------------

ZSC_MODEL = "mistralai/mistral-7b-instruct-v0.3"

zsc_llm = ChatNVIDIA(model=ZSC_MODEL)

zsc_sys_msg = (
    "Choose the most likely topic classification given the sentence as context."
    " Only one word, no explanation.\n[Options : {options}]"
)

# One-shot prompt: give the model one labeled example so it learns the format
zsc_prompt = ChatPromptTemplate.from_messages([
    ("system", zsc_sys_msg),
    ("user", "[[The sea is awesome]]"),
    ("assistant", "boat"),
    ("user", "[[{input}]]"),
])

zsc_chain = zsc_prompt | zsc_llm | StrOutputParser()


def zsc_call(text: str, options=("car", "boat", "airplane", "bike")) -> str:
    """Classify text into one of the given options."""
    return zsc_chain.invoke({"input": text, "options": list(options)}).split()[0]


def demo_zsc():
    print(zsc_call("Should I take the next exit, or keep going?"))   # car
    print(zsc_call("I get seasick, so I'll skip the trip"))           # boat
    print(zsc_call("I'm scared of heights, so flying isn't for me"))  # airplane


# ---------------------------------------------------------------------------
# Example 3: rhyme re-themer chatbot (the notebook assessment)
# ---------------------------------------------------------------------------

prompt1 = ChatPromptTemplate.from_messages([
    ("user", "INSTRUCTION: Only respond in rhymes\n\nPROMPT: {input}"),
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user", (
        "INSTRUCTION: Only responding in rhyme, change the topic of the input poem"
        " to be about {topic}! Make it happy! Try to keep the same sentence structure,"
        " but make sure it's easy to recite! Try not to rhyme a word with itself."
        "\n\nOriginal Poem: {input}"
        "\n\nNew Topic: {topic}"
    )),
])

# chain1: poem from scratch (only needs 'input')
chain1 = prompt1 | instruct_llm | StrOutputParser()

# chain2: re-theme existing poem (needs 'input' = original poem, 'topic' = new subject)
chain2 = prompt2 | instruct_llm | StrOutputParser()


def rhyme_chat2_stream(message: str, history: list, return_buffer: bool = True):
    """
    Generator function for Gradio-style streaming.

    First message: generates a poem with chain1.
    Subsequent messages: re-themes the first poem with chain2.

    history format: [[user_msg_0, bot_msg_0], [user_msg_1, bot_msg_1], ...]
    """
    first_poem = None
    for entry in history:
        if entry[0] and entry[1]:
            first_poem = entry[1]  # first completed bot response
            break

    if first_poem is None:
        buffer = "Oh! I can make a wonderful poem about that! Let me think!\n\n"
        yield buffer

        inst_out = ""
        for token in chain1.stream({"input": message}):
            inst_out += token
            buffer += token
            yield buffer if return_buffer else token

        passage = "\n\nNow let me rewrite it with a different focus! What should the new focus be?"
        buffer += passage
        yield buffer if return_buffer else passage

    else:
        buffer = "Sure! Here you go!\n\n"
        yield buffer

        inst_out = ""
        # Pass the original poem as 'input' and the new subject as 'topic'
        for token in chain2.stream({"input": first_poem, "topic": message}):
            inst_out += token
            buffer += token
            yield buffer if return_buffer else token

        passage = "\n\nThis is fun! Give me another topic!"
        buffer += passage
        yield buffer if return_buffer else passage


def run_rhyme_chatbot():
    history = [[None, "Let me help you make a poem! What would you like for me to write?"]]
    queue_fake_streaming_gradio(rhyme_chat2_stream, history=history, max_questions=3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo: rhyme chain ===")
    demo_rhyme_chain()

    print("\n=== Demo: zero-shot classification ===")
    demo_zsc()

    print("\n=== Demo: rhyme re-themer chatbot ===")
    run_rhyme_chatbot()
