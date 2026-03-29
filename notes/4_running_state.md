## notebook 4: running state

previous notebooks covered LLM services and LangChain basics. this one focuses on **how to maintain state across turns** — the foundation for building conversational agents that remember things.

## learning objectives
1. use `RunnableAssign` + Pydantic to build a running knowledge base
2. implement slot-filling extraction to update state from free-form conversation
3. wire retrieval (e.g., a flight database) into the state-driven pipeline

## thinking questions
1. if you had only one internal chain module (no environment input loop), where would you use it?
2. what types of JSON prediction failures can happen based on problem complexity or prompt format?
3. can you replace slot-filling prompt engineering with something else entirely (e.g., fine-tuning)?

## Part 1: why state matters

stateless chains process one input → output with no memory. this is fine for one-shot tasks, but real conversations need context. what did the user say before? what have we already figured out?

two approaches to memory:
- **buffer**: just append all messages — simple, but grows forever and can't focus
- **running summary / knowledge base**: use the LLM to maintain a structured, dense record — scales better, stays relevant

the key idea: treat knowledge as a Pydantic schema. at each turn, ask the LLM to fill/update the slots from context. this is **slot-filling**.

## Part 2: RExtract — the core building block

```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables.passthrough import RunnableAssign

def RExtract(pydantic_class, llm, prompt):
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions': lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = string.replace("\\_", "_").replace("\n", " ").replace("\]", "]").replace("\[", "[")
        return string
    return instruct_merge | prompt | llm | preparse | parser
```

**what it does:**
1. adds `format_instructions` to the dict (tells LLM what JSON to output)
2. formats the prompt with all state fields + new input
3. runs the LLM
4. parses output back to a Pydantic object

**usage:**
```python
extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
new_state = extractor.invoke({'know_base': old_kb, 'input': user_msg, 'output': agent_msg})
# new_state is a KnowledgeBase instance with updated fields
```

## Part 3: the full internal chain pattern

```python
class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="User's first name, 'unknown' if unknown")
    last_name: str = Field('unknown', description="User's last name, 'unknown' if unknown")
    confirmation: int = Field(-1, description="Flight confirmation number, -1 if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far")
    open_problems: list = Field([], description="Unresolved topics")
    current_goals: list = Field([], description="Current goals for the agent")
```

the internal chain runs before every external (user-facing) response:
1. **knowbase_getter**: `RExtract(KnowledgeBase, instruct_llm, parser_prompt)` → updates knowledge base from conversation
2. **database_getter**: uses updated KB to query a flight database
3. **external_chain**: generates user-facing response using KB + retrieved context

```python
internal_chain = (
    RunnableAssign({'know_base': knowbase_getter})  # update KB
    | RunnableAssign({'context': database_getter})  # look up flight info
)
```

state flows through:
```
{input, output, know_base}
    → internal_chain
    → {input, output, know_base (updated), context (flight info)}
    → external_chain (user sees this)
```

## Part 4: the database getter

once we have the KB, we can look up flight info:

```python
def database_getter(d):
    try:
        return get_flight_info(get_key_fn(d['know_base']))
    except Exception:
        return "Flight info not available yet. Please provide name and confirmation number."
```

`get_key_fn` converts a KnowledgeBase to `{first_name, last_name, confirmation}` dict that `get_flight_info` expects.

the try/except is important — if fields are still 'unknown'/missing, `get_flight_info` will fail. this becomes the agent's cue to ask for more info.

## key pattern: state as a dict

throughout this course, state is passed as a Python dict. each chain step reads from and writes to this dict:

```python
state = {'know_base': KnowledgeBase(), 'input': '', 'output': ''}

def chat_gen(message, history):
    global state
    state['input'] = message
    state['output'] = "" if not history else history[-1][1]
    state = internal_chain.invoke(state)  # updates know_base and context
    for token in external_chain.stream(state):
        yield token
```

this global state pattern is simple for a demo but would need proper session isolation in production.
