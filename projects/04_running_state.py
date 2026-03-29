"""
Notebook 4: Running state chains — SkyFlow airline chatbot.

Demonstrates:
- RExtract for slot-filling a Pydantic knowledge base
- Maintaining state across conversation turns
- Wiring a structured knowledge base to an external database
"""

from typing import Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.passthrough import RunnableAssign

from utils import RExtract, queue_fake_streaming_gradio

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

chat_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

# ---------------------------------------------------------------------------
# Mock flight database
# ---------------------------------------------------------------------------

_KEYS = ["first_name", "last_name", "confirmation", "departure", "destination",
         "departure_time", "arrival_time", "flight_day"]

_DB = [
    ["Jane", "Doe",      12345, "San Jose",     "New Orleans", "12:30 PM", "9:30 PM",  "tomorrow"],
    ["John", "Smith",    54321, "New York",      "Los Angeles", "8:00 AM",  "11:00 AM", "today"],
    ["Alice","Johnson",  98765, "Chicago",       "Miami",       "3:15 PM",  "7:45 PM",  "tomorrow"],
    ["Bob",  "Brown",    27494, "Seattle",       "Boston",      "6:00 AM",  "2:30 PM",  "today"],
    ["Carol","Williams", 36621, "San Francisco", "Denver",      "9:45 AM",  "1:15 PM",  "tomorrow"],
]


def get_flight_info(d: dict) -> str:
    """Look up flight info by first_name, last_name, confirmation."""
    req = ["first_name", "last_name", "confirmation"]
    assert all(k in d for k in req), f"Expected keys {req}, got {d}"

    for row in _DB:
        entry = dict(zip(_KEYS, row))
        if (entry["first_name"].lower() == str(d["first_name"]).lower()
                and entry["last_name"].lower() == str(d["last_name"]).lower()
                and str(entry["confirmation"]) == str(d["confirmation"])):
            return (
                f"{entry['first_name']} {entry['last_name']}'s flight from "
                f"{entry['departure']} to {entry['destination']} departs at "
                f"{entry['departure_time']} {entry['flight_day']} and lands at "
                f"{entry['arrival_time']}."
            )
    raise ValueError(f"No flight found for {d}")


def get_key_fn(base: BaseModel) -> dict:
    return {
        "first_name": base.first_name,
        "last_name": base.last_name,
        "confirmation": base.confirmation,
    }


# ---------------------------------------------------------------------------
# Knowledge base schema
# ---------------------------------------------------------------------------

class KnowledgeBase(BaseModel):
    first_name: str = Field("unknown", description="User's first name, 'unknown' if not provided")
    last_name: str = Field("unknown", description="User's last name, 'unknown' if not provided")
    confirmation: Optional[int] = Field(None, description="Flight confirmation number, None if unknown")
    discussion_summary: str = Field("", description="Summary of the conversation so far")
    open_problems: str = Field("", description="Topics not yet resolved")
    current_goals: str = Field("", description="What the agent is currently trying to accomplish")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant for SkyFlow Airlines tracking conversation info."
    " Fill in the schema based on the conversation."
    "\n\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nASSISTANT RESPONSE: {output}"
    "\n\nUSER MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE: "
)

external_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a chatbot for SkyFlow Airlines helping a customer."
        " Stay concise and clear!"
        " Your running knowledge base is: {know_base}. This is for you only; do not mention it!"
        "\nUsing that, we retrieved: {context}\n"
        " If retrieval failed, ask to confirm first/last name and confirmation number."
        " Do not ask for any other personal info."
    )),
    ("assistant", "{output}"),
    ("user", "{input}"),
])

# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------

external_chain = external_prompt | chat_llm

# RExtract: reads {know_base, output, input} → returns updated KnowledgeBase
knowbase_getter = RExtract(KnowledgeBase, instruct_llm, parser_prompt)


def database_getter(d: dict) -> str:
    """Use updated knowledge base to retrieve flight info."""
    try:
        return get_flight_info(get_key_fn(d["know_base"]))
    except Exception:
        return "Flight info not available yet. Please provide your name and confirmation number."


internal_chain = (
    RunnableAssign({"know_base": knowbase_getter})   # update KB from conversation
    | RunnableAssign({"context": database_getter})   # look up flight with updated KB
)

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

state = {"know_base": KnowledgeBase()}


def chat_gen(message: str, history: list, return_buffer: bool = True):
    global state
    state["input"] = message
    state["history"] = history
    state["output"] = "" if not history else history[-1][1]

    state = internal_chain.invoke(state)

    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token


def run_airline_chatbot():
    global state
    state = {"know_base": KnowledgeBase()}
    history = [[None, "Hello! I'm your SkyFlow agent! How can I help you?"]]
    queue_fake_streaming_gradio(chat_gen, history=history, max_questions=8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SkyFlow Airline Chatbot ===")
    print("Try: 'Can you tell me about my flight?'")
    print("Then provide: first name, last name, and confirmation number")
    print("Test data: Jane Doe #12345, Alice Johnson #98765\n")
    run_airline_chatbot()
