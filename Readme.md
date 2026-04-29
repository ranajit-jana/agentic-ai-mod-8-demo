# Gemini Research Agent

---

## HITL Comparison: AutoGen UserProxyAgent vs LangGraph interrupt

This project ships two parallel implementations of the same app to demonstrate the difference between the two most common HITL patterns in agentic AI.

### The core problem

In a Streamlit app, the human's turn is the UI itself — buttons, text inputs, chat. The framework needs a way to **pause agent execution**, hand control back to the UI, and **resume** when the human has decided. The two patterns solve this very differently.

---

### AutoGen — UserProxyAgent

`UserProxyAgent` was designed for **terminal-based** HITL. It sits between the human and the `AssistantAgent` and relays messages. In a web app, it becomes an awkward fit.

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful assistant.",
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",   # blocks waiting for terminal input
    max_consecutive_auto_reply=0,
    code_execution_config=False,
)

# Each call to initiate_chat starts a NEW run — no state is preserved
user_proxy.initiate_chat(assistant, message="Explain quantum computing")
```

**What happens in Streamlit:**

- `human_input_mode="ALWAYS"` blocks on terminal stdin — useless in a web app
- Setting `"NEVER"` + `max_consecutive_auto_reply=0` forces the run to terminate after one reply
- Every call to `initiate_chat()` **starts a fresh run** — you see `TERMINATING RUN (uuid)` in the logs
- State is not preserved between runs; history must be managed manually in `st.session_state`
- The "human in the loop" is faked — the loop terminates and Streamlit rebuilds it from scratch

**Verdict:** `UserProxyAgent` is not a HITL primitive for web apps. It is a terminal relay that terminates the run on every interaction.

---

### LangGraph — `interrupt()`

LangGraph's `interrupt()` is a **first-class HITL primitive**. It pauses the graph at any node, checkpoints the full state, and resumes from exactly the same point when the human responds.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

def plan_node(state):
    response = llm.invoke(build_messages(state))
    return {"outline": response.content}

def human_review_node(state):
    # Execution pauses HERE. State is saved to the checkpointer.
    # graph.invoke() returns immediately to the caller (Streamlit).
    # When Command(resume=value) is sent, execution continues from this line.
    decision = interrupt({"outline": state["outline"]})

    if decision["action"] == "approve":
        return {"feedback": None}
    else:
        return {"feedback": decision["feedback"]}

def generate_node(state):
    response = llm.invoke(build_messages(state))
    return {"full_answer": response.content}

builder = StateGraph(AgentState)
builder.add_node("plan", plan_node)
builder.add_node("human_review", human_review_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "human_review")
builder.add_conditional_edges("human_review", route_fn, {"revise": "revise", "generate": "generate"})
builder.add_edge("generate", END)

# MemorySaver persists state across Streamlit reruns — store in st.session_state
graph = builder.compile(checkpointer=MemorySaver())

# --- First call: graph runs plan_node, hits interrupt(), returns ---
graph.invoke({"user_input": "Explain quantum computing"}, config)

# --- Streamlit rerenders, human clicks Proceed ---
graph.invoke(Command(resume={"action": "approve"}), config)
# Graph resumes in human_review_node, routes to generate_node, streams answer
```

**What happens in Streamlit:**

- `graph.invoke()` returns immediately when `interrupt()` is hit — no blocking
- Full graph state (variables, position, history) is frozen in `MemorySaver`
- On next Streamlit rerun, `graph.get_state(config)` detects the interrupt
- `Command(resume=value)` resumes from exactly the paused line — same run, same state
- No run termination, no UUID noise, no manual state management

**Verdict:** `interrupt()` is the correct HITL primitive for web apps. The graph pauses and resumes without breaking the execution thread.

---

### Side-by-side comparison

| | AutoGen `UserProxyAgent` | LangGraph `interrupt()` |
|---|---|---|
| Designed for | Terminal / CLI | Web apps, async systems |
| Pause mechanism | Terminates the run | Checkpoints the run |
| State on pause | Lost — must rebuild manually | Fully preserved |
| Resume | New `initiate_chat()` call | `Command(resume=value)` |
| Streamlit rerun safe | No — new MemorySaver wipes state | Yes — state lives in checkpointer |
| Log noise | `TERMINATING RUN (uuid...)` | None |
| Multi-turn HITL | Each turn is a separate run | One continuous run with multiple interrupts |
| Streaming support | Requires workaround (google-genai direct) | Native via `stream_mode="messages"` |

---

## Overview

An AI-powered research assistant built with **Google Gemini 2.5 Flash**, demonstrating two HITL patterns:

- `streamlit_genai.py` — AutoGen `AssistantAgent` for planning + google-genai for streaming
- `streamlit_langgraph.py` — LangGraph `interrupt()` pattern with full state preservation

Both apps let you see the plan before the full answer, approve or redirect it, and stream the response token by token.

---

## Features

- Human-in-the-loop — see the plan before the full answer, approve or redirect it
- Streamed responses — full answer arrives token by token
- Automatic subtopic decomposition and one-click summarisation
- Full conversation context maintained across turns
- Download conversation as `.docx`, `.csv`, or `.pdf`
- Interactive Streamlit UI

---

## Tech Stack

- **Google Gemini 2.5 Flash** — language generation
- **AutoGen** — agent orchestration (`streamlit_genai.py`)
- **LangGraph** — graph-based HITL with interrupt (`streamlit_langgraph.py`)
- **google-genai** — streaming API
- **langchain-google-genai** — LangChain-wrapped Gemini for LangGraph streaming
- **Streamlit** — frontend UI
- **python-dotenv** — environment config
- **uv** — dependency and environment management

---

## Project Structure

```
├── streamlit_genai.py       # AutoGen + google-genai implementation
├── streamlit_langgraph.py   # LangGraph interrupt implementation
├── appinfo.py               # prints installed package versions
├── pyproject.toml           # project dependencies (uv)
├── .env                     # API key (not committed)
├── .gitignore
└── Readme.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gemini-research-agent.git
cd agentic-ai-mod-8-demo
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run the app

```bash
# AutoGen version
uv run streamlit run streamlit_genai.py

# LangGraph version
uv run streamlit run streamlit_langgraph.py
```

---

## Usage

1. Type a question in the chat input
2. The assistant shows a **2-3 bullet plan** of how it will answer
3. Click **Proceed** to stream the full answer, or type feedback and click **Update Plan** to redirect
4. Continue the conversation — the assistant has full context of all prior turns
5. Use **Generate Subtopics** or **Summarise** for quick follow-up actions (bypass the plan phase)
6. Download the full conversation as Word, CSV, or PDF
