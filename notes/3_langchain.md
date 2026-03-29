## notebook 3: LangChain introduction

LangChain is an LLM orchestration library — it helps you wire together LLMs, prompts, memory, retrievers, and tools into coherent pipelines. this notebook introduces the modern LCEL (LangChain Expression Language) approach.

## learning objectives
1. orchestrate LLM systems using chains and runnables (LCEL)
2. understand external user-facing responses vs. internal reasoning chains
3. launch a simple Gradio chat interface from a notebook

## thinking questions
1. what tools are needed to propagate information through a workflow (preview of notebook 4)?
2. when you see a Gradio interface — have you seen this on HuggingFace Spaces?
3. if you want another microservice to receive a chain's output, what interface requirements does that create?

## Part 1: what is LangChain

LangChain is popular but moves fast — features appear and disappear rapidly. the key abstraction has evolved:

**classic chains (legacy)**: `LLMChain`, `ConversationChain`, `SequentialChain` — still work but verbose and less flexible.

**modern LCEL**: everything is a `Runnable`. you compose runnables with the `|` (pipe) operator, left to right. much more compact.

```
fn1 | fn2 | fn3
```
this creates a new Runnable. calling `.invoke(x)` on it passes `x` through each function in order.

## Part 2: runnables and the pipe operator

```python
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from functools import partial

# wrap any function as a Runnable
identity = RunnableLambda(lambda x: x)

def print_and_return(x, preface=""):
    print(f"{preface}{x}")
    return x

rprint0 = RunnableLambda(print_and_return)
rprint1 = RunnableLambda(partial(print_and_return, preface="1: "))

# factory pattern — cleaner than partial every time
def RPrint(preface=""):
    return RunnableLambda(partial(print_and_return, preface=preface))

# chain with |
chain1 = identity | rprint0
chain1.invoke("Hello World!")   # prints: Hello World!

# longer chain
output = (
    chain1         # prints "Hello World"
    | rprint1      # prints "1: Hello World"
    | RPrint("2: ") # prints "2: Hello World"
).invoke("Welcome Home!")
```

every step receives the output of the previous step and passes its own output forward.

## Part 3: dict-based workflows (the right default)

pass dicts between runnables — named fields prevent confusion as pipelines get complex.

```python
from operator import itemgetter

## make_dictionary coerces any value into a dict with the given key
def make_dictionary(v, key):
    if isinstance(v, dict): return v
    return {key: v}

def RInput(key='input'):
    return RunnableLambda(partial(make_dictionary, key=key))

def ROutput(key='output'):
    return RunnableLambda(partial(make_dictionary, key=key))

## implicit dict mapping: dict of functions → each key runs its function on the input
pipeline = (
    RInput()                     # ensure dict with 'input' key
    | itemgetter("input")        # pull the value
    | {                          # implicit map: run all lambdas, collect results
        'word1': (lambda x: x.split()[0]),
        'word2': (lambda x: x.split()[1]),
        'words': RunnablePassthrough(),
    }
    | RPrint("D: ")
    | itemgetter("word1")
    | RunnableLambda(str.upper)
    | ROutput()                  # wrap back in dict
)

pipeline.invoke({"input": "Hello World"})
# word1="HELLO", wrapped as {'output': 'HELLO'}
```

**why dicts?**
- LangChain prompt templates require dict inputs (`{"key": value}`)
- name-based tracking avoids position confusion
- implicit mapping lets you branch one input into multiple outputs

## Part 4: simple LLM chain

the classic prompt → LLM → parser pattern:

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Only respond in rhymes"),
    ("user", "{input}")
])

rhyme_chain = prompt | chat_llm | StrOutputParser()
print(rhyme_chain.invoke({"input": "Tell me about birds!"}))
```

**`StrOutputParser`**: extracts the text string from the LLM's `AIMessage` response object. without it, you get an object back, not a string.

**Gradio streaming interface**:
```python
import gradio as gr

def rhyme_chat_stream(message, history):
    buffer = ""
    for token in rhyme_chain.stream({"input": message}):
        buffer += token
        yield buffer  # gradio expects the full accumulated string each yield

gr.ChatInterface(rhyme_chat_stream).queue().launch(server_name="0.0.0.0", share=True, debug=True)
## IMPORTANT: click the square Stop button when done — leaves a port open otherwise
```

## Part 5: internal reasoning (zero-shot classification)

sometimes you want the LLM to reason internally before the user sees anything — e.g., route a query to the right handler. this is a **blocking** internal step, so model speed matters more than quality here.

```python
instruct_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3")

sys_msg = (
    "Choose the most likely topic classification given the sentence as context."
    " Only one word, no explanation.\n[Options : {options}]"
)

## one-shot prompt: show the model what format you want
zsc_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    ("user", "[[The sea is awesome]]"),
    ("assistant", "boat"),   # ← one-shot example
    ("user", "[[{input}]]"),
])

zsc_chain = zsc_prompt | instruct_llm | StrOutputParser()

def zsc_call(input, options=["car", "boat", "airplane", "bike"]):
    return zsc_chain.invoke({"input": input, "options": options}).split()[0]

print(zsc_call("Should I take the next exit, or keep going?"))   # → car
print(zsc_call("I get seasick on trips"))                        # → boat
```

**model selection tradeoffs for internal reasoning:**
- **stability**: does it follow the output format reliably across many inputs? (most important)
- **speed**: it's blocking — slow internal reasoning = slow user response
- **quality**: less critical here than for user-facing output

## Part 6: multi-component chains

building more complex systems by combining multiple steps:

```python
## given a zsc_chain that classifies topic, and separate generation chains per topic
route_chain = (
    {'input': RunnablePassthrough(), 'topic': zsc_chain}
    | RunnableBranch(
        ((lambda d: d['topic'] == 'boat'), boat_chain),
        ((lambda d: d['topic'] == 'car'),  car_chain),
        default_chain  # fallback
    )
)
```

LCEL `RunnableBranch` acts like a switch statement — first condition that returns `True` runs its chain, otherwise uses the default.

## Part 7: rhyme re-themer chatbot (the assessment exercise)

the full exercise: a chatbot that generates a poem on first message, then re-themes it on each subsequent message:

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

# chain1: generate a poem from scratch
prompt1 = ChatPromptTemplate.from_messages([
    ("user", "INSTRUCTION: Only respond in rhymes\n\nPROMPT: {input}")
])
chain1 = prompt1 | instruct_llm | StrOutputParser()

# chain2: re-theme an existing poem
prompt2 = ChatPromptTemplate.from_messages([("user", (
    "INSTRUCTION: Only responding in rhyme, change the topic of the input poem to be about {topic}!"
    " Make it happy! Try to keep the same sentence structure, but make sure it's easy to recite!"
    " Try not to rhyme a word with itself."
    "\n\nOriginal Poem: {input}"
    "\n\nNew Topic: {topic}"
))])
chain2 = prompt2 | instruct_llm | StrOutputParser()

def rhyme_chat2_stream(message, history, return_buffer=True):
    '''Generator function — each yield returns the next chunk'''

    # look for the first poem in history
    first_poem = None
    for entry in history:
        if entry[0] and entry[1]:
            first_poem = entry[1]  # first bot response = first poem
            break

    if first_poem is None:
        # first message: generate a new poem
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
        # subsequent messages: re-theme the existing poem
        buffer = "Sure! Here you go!\n\n"
        yield buffer
        inst_out = ""
        # key: chain2 needs the original poem as 'input' and the new topic as 'topic'
        for token in chain2.stream({"input": first_poem, "topic": message}):
            inst_out += token
            buffer += token
            yield buffer if return_buffer else token
        passage = "\n\nThis is fun! Give me another topic!"
        buffer += passage
        yield buffer if return_buffer else passage
```

**the key insight**: `history` in Gradio is a list of `[user_msg, bot_msg]` pairs. `first_poem` is the first bot response that followed a user message. `chain2` needs the *original poem* (not `message`) as its `input`, and the new topic as `topic`.
