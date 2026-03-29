## notebook 2: LLM services and AI foundation models

this notebook focuses on the `llm_client` microservice and how to interact with it at three levels of abstraction.

## learning objectives
1. understand the tradeoffs of local vs cloud LLM deployment
2. interact with NVIDIA AI Foundation Model endpoints: raw requests, OpenAI client, LangChain connector
3. pick the right model from the available endpoint pool

## thinking questions
1. what model access should LLM app developers get vs. end-users of an AI app?
2. when considering what devices to support, what are the computational tradeoffs?
   - provide a private LLM Jupyter interface for clients?
   - deploy locally in Jupyter Labs?
   - embedded devices like Jetson Nano?
3. you've deployed stable diffusion + mixtral + Llama-13B on shared GPU. stable diffusion is unused but your team experiments with the other two. do you need to delete stable diffusion?

## Part 1: LLM deployment tiers

three tiers depending on hardware:

| tier | hardware | pros | cons |
|---|---|---|---|
| data center | NVIDIA GPU clusters | scalable, ideal for multi-model, multi-user | inefficient per-user resource allocation |
| limited DC / prosumer | standard data center / consumer GPUs | good perf-per-user balance | still costly per user |
| consumer hardware | typical laptop/desktop | accessible everywhere | can't run multiple large models, heavy resource use |

the DLI course environment represents tier 3 (consumer-like CPU-only). we can't run an LLM locally, so instead we call out to NVIDIA's hosted services.

## Part 2: hosting options

**black-box hosted (e.g., OpenAI)**
- packaged, user-friendly
- limited customization, privacy concerns, high cost at scale

**self-deployed**
- full control over data and endpoints
- requires specialized infra engineers, limited support for independent devs

**NVIDIA NGC + AI Foundation Models** ← what we use
- pre-trained, optimized models available through `build.nvidia.com`
- uses **NIM (NVIDIA Inference Microservices)** — standardized OpenAI-compatible API, optimized for scalable inference on DGX Cloud

## Part 3: NIM architecture (how NVIDIA hosts models)

```
user request
    ↓
[API gateway] — OpenAI-compatible endpoint
    ↓
[Kubernetes cluster on DGX Cloud]
    ├── compute node 1 (GPU)  ← receives request if free
    ├── compute node 2 (GPU)  ← FIFO queue if node 1 busy
    └── compute node N (GPU)  ← auto-scales on demand
```

key behaviors:
- **in-flight batching**: up to 256 requests batched at a compute node before it's considered "full"
- **auto-scaling**: new nodes spin up when latency exceeds threshold
- **OpenAI-compatible format**: same JSON schema as OpenAI, so existing code mostly works

to get an API key: go to `build.nvidia.com`, find a model, click "Get API Key" → `nvapi-...`

```python
import os
os.environ["NVIDIA_API_KEY"] = "nvapi-..."
```

## Part 4.1: raw Python requests

lowest level — full control, most verbose:

```python
import requests, json

invoke_url = "http://llm_client:9000/v1/chat/completions"
headers = {"content-type": "application/json"}
payload = {
    "model": "mistralai/mixtral-8x7b-instruct-v0.1",
    "messages": [{"role": "user", "content": "Tell me hello in French"}],
    "temperature": 0.5,
    "max_tokens": 1024,
    "stream": True
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        data = json.loads(line[6:])  # strip "data: " prefix
        if data != "[DONE]":
            print(data["choices"][0]["delta"].get("content", ""), end="")
```

why `messages` with roles? it:
- enforces a conversation structure (system/user/assistant)
- enables custom system prompts
- lets the model optimize its reasoning per role

## Part 4.2: OpenAI client

same interface, cleaner code:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://llm_client:9000/v1",
    api_key="not-needed-for-local"
)

completion = client.chat.completions.create(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    messages=[{"role": "user", "content": "Hello World"}],
    temperature=1,
    top_p=1,
    max_tokens=1024,
    stream=True,
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

works because NIM uses the same API format as OpenAI — you just swap `base_url`.

## Part 4.3: ChatNVIDIA (LangChain connector)

highest level — integrates directly with LangChain:

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# single call
result = llm.invoke("Hello World")

# streaming
for token in llm.stream("Tell me about yourself! 2 sentences.", max_tokens=100):
    print(token.content, end="")

# see what request/response looks like under the hood
llm._client.last_inputs
llm._client.last_response.json()
```

**framework connector role**: adapts the raw API format to whatever shape the LangChain codebase expects. hides the HTTP boilerplate, exposes `invoke`, `stream`, `batch`, etc.

```python
# list available models, test each one
model_list = ChatNVIDIA.get_available_models(list_none=False)

for model_card in model_list:
    llm = ChatNVIDIA(model=model_card.id)
    print(f"TRIAL: {model_card.id}")
    try:
        for token in llm.stream("Tell me about yourself! 2 sentences.", max_tokens=100):
            print(token.content, end="")
    except Exception as e:
        print(f"EXCEPTION: {e}")
    print("\n" + "="*84)
```

## summary: three levels of abstraction

```
raw requests        → max control, max boilerplate, good for debugging
OpenAI client       → cleaner syntax, handles streaming/errors automatically
ChatNVIDIA/LangChain → fully integrated with LangChain ecosystem, use this for building apps
```

use raw requests when debugging API issues. use ChatNVIDIA for everything else in this course.
