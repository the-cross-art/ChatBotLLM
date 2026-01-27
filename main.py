# ============================================================
# Hybrid Memory Chat — Production Grade (Corrected)
# Multi-User • Multi-Thread • STM + LTM (Postgres + Vectors)
# ============================================================

import uuid
import psycopg2
import logging
import sys
from math import sqrt
from typing import List
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langchain_community.tools import DuckDuckGoSearchRun


from langchain_ollama import ChatOllama, OllamaEmbeddings

# ============================================================
# LOGGING SETUP (clean, quiet, debuggable)
# ============================================================


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence noisy libraries
    for noisy in [
        "psycopg",
        "psycopg2",
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "httpcore",
        "httpx",
        "urllib3",
        "asyncio",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


setup_logging()

log_memory = logging.getLogger("memory")
log_retrieve = logging.getLogger("memory.retrieve")
log_chat = logging.getLogger("chat")
log_tools = logging.getLogger("tools")

# ============================================================
# CONFIG
# ============================================================

DB_URI_USERS = "postgresql://imrannazir@localhost:5432/user_registry"
DB_URI_STM = "postgresql://imrannazir@localhost:5432/stm_persist"
DB_URI_LTM = "postgresql://imrannazir@localhost:5432/ltm_persistence"

MEMORY_SIM_THRESHOLD = 0.92
MAX_LTM_RESULTS = 5

# ============================================================
# MODELS
# ============================================================

chat_llm = ChatOllama(model="qwen2.5:7b")
memory_llm = ChatOllama(model="qwen2.5:7b")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

tools = [search_tool]

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """You are a helpful assistant with memory.

Context:
{context}

Rules:
- Use memory only if relevant
- Do not invent facts
"""

MEMORY_PROMPT = """You are a memory extraction system. Your job is to identify important facts about the user.

EXISTING MEMORIES:
{existing}

INSTRUCTIONS:
1. Read the user's message carefully
2. Extract ONLY factual information about the user (name, job, experience, projects, preferences)
3. Each fact should be ONE clear sentence
4. Compare with existing memories - only mark is_new=true if the fact is NOT already stored
5. If you find facts to store, set should_write=true
6. If the message has no factual information worth remembering, set should_write=false

EXAMPLES:
User: "My name is John and I'm a software engineer"
Response: should_write=true, memories=[
  {{"text": "User's name is John", "is_new": true}},
  {{"text": "User is a software engineer", "is_new": true}}
]

User: "What's the weather today?"
Response: should_write=false, memories=[]

Now process the user's latest message.
"""

# ============================================================
# DATA MODELS
# ============================================================


class MemoryItem(BaseModel):
    text: str
    type: str = "general"
    confidence: float = 0.85
    is_new: bool


class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


class ChatState(MessagesState):
    summary: str = ""
    retrieved_memories: List[str] = []
    merged_context: str = ""


memory_extractor = memory_llm.with_structured_output(MemoryDecision)

# ============================================================
# UTILS
# ============================================================


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sqrt(sum(x * x for x in a)) * sqrt(sum(x * x for x in b))) + 1e-8)


def ensure_user_thread(user_id: str, thread_id: str):
    conn = psycopg2.connect(DB_URI_USERS)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_threads (
            user_id TEXT,
            thread_id TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (user_id, thread_id)
        )
        """
    )
    cur.execute(
        """
        INSERT INTO user_threads (user_id, thread_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
        (user_id, thread_id),
    )
    conn.commit()
    cur.close()
    conn.close()


# ============================================================
# GRAPH NODES
# ============================================================


def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    if not state.get("messages"):
        return {}

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    try:
        existing_items = store.search(ns)
        existing_texts = [i.value["text"] for i in existing_items]

        decision = memory_extractor.invoke(
            [
                SystemMessage(
                    content=MEMORY_PROMPT.format(
                        existing="\n".join(existing_texts) or "(empty)"
                    )
                ),
                HumanMessage(content=state["messages"][-1].content),
            ]
        )

        log_memory.debug("Memory decision: %s", decision)

        if not decision.should_write:
            return {}

        for mem in decision.memories:
            if not mem.is_new:
                continue

            emb = embeddings.embed_query(mem.text)

            duplicate = False
            for it in existing_items:
                sim = cosine_similarity(emb, it.value["embedding"])
                if sim >= MEMORY_SIM_THRESHOLD:
                    duplicate = True
                    log_memory.debug("Skipped duplicate (sim=%.2f): %s", sim, mem.text)
                    break

            if duplicate:
                continue

            now = datetime.now(timezone.utc).isoformat()

            store.put(
                ns,
                str(uuid.uuid4()),
                {
                    "text": mem.text,
                    "type": mem.type,
                    "confidence": mem.confidence,
                    "created_at": now,
                    "last_confirmed": now,
                    "embedding": emb,
                },
            )

            log_memory.info("Stored memory: %s", mem.text)

    except Exception as e:
        log_memory.error("Memory write failed", exc_info=e)

    return {}


def retrieve_ltm(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    if not state.get("messages"):
        return {}

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    query = state["messages"][-1].content
    q_emb = embeddings.embed_query(query)

    scored = []
    for it in store.search(ns):
        score = cosine_similarity(q_emb, it.value["embedding"]) * it.value.get(
            "confidence", 1.0
        )
        scored.append((score, it.value["text"]))

    scored.sort(reverse=True)
    top = [t for _, t in scored[:MAX_LTM_RESULTS]]

    log_retrieve.debug("Retrieved memories: %s", top)

    return {"retrieved_memories": top}


def summarize_node(state: dict):
    messages = state.get("messages", [])

    log_chat.info(
        "Summarization triggered | messages_count=%d",
        len(messages),
    )

    response = chat_llm.invoke(
        messages + [HumanMessage(content="Summarize the conversation so far.")]
    )

    summary_text = response.content.strip()

    log_chat.debug("Generated summary:\n%s", summary_text)

    return {
        "summary": summary_text,
        "messages": [RemoveMessage(id=m.id) for m in messages[:-2]],
    }


def should_summarize(state: dict):
    msg_count = len(state.get("messages", []))
    decision = msg_count > 8

    log_chat.debug(
        "Summarization check | messages_count=%d | should_summarize=%s",
        msg_count,
        decision,
    )

    return decision


def merge_node(state: dict):
    blocks = []

    summary = state.get("summary")
    memories = state.get("retrieved_memories")

    if summary:
        log_chat.debug("Merging summary into context")
        blocks.append(f"Summary:\n{summary}")

    if memories:
        log_chat.debug(
            "Merging %d retrieved memories into context",
            len(memories),
        )
        blocks.append("Memory:\n" + "\n".join(memories))

    merged = "\n\n".join(blocks)

    log_chat.debug(
        "Final merged context length=%d chars",
        len(merged),
    )

    return {"merged_context": merged}


def chat_node(state: dict):
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(context=state.get("merged_context") or "(empty)")
    )
    chat_llm_with_tools = chat_llm.bind_tools(tools)
    messages = [system] + state["messages"]
    response = chat_llm_with_tools.invoke(messages)
    log_chat.debug("Assistant response generated")

    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        tool_map = {t.name: t for t in tools}
        tool_messages = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id")
            log_tools.info("Tool call: name=%s args=%s", name, args)

            tool = tool_map.get(name)
            if not tool:
                log_tools.error("Unknown tool requested: %s", name)
                continue

            # DuckDuckGoSearchRun expects a string query
            tool_input = args.get("query") if isinstance(args, dict) else args
            try:
                result = tool.invoke(tool_input)
                result_text = str(result)
                log_tools.info(
                    "Tool result (%s): %.200s", name, result_text.replace("\n", " ")
                )
                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=call_id, name=name)
                )
            except Exception as e:
                log_tools.error("Tool execution failed (%s)", name, exc_info=e)

        # Ask the model to produce a final answer using tool outputs
        final = chat_llm_with_tools.invoke(messages + [response] + tool_messages)
        return {"messages": [final]}

    return {"messages": [response]}


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(ChatState)

builder.add_node("remember", remember_node)
builder.add_node("retrieve_ltm", retrieve_ltm)
builder.add_node("summarize", summarize_node)
builder.add_node("merge", merge_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_conditional_edges(
    "remember", should_summarize, {True: "summarize", False: "retrieve_ltm"}
)
builder.add_edge("summarize", "retrieve_ltm")
builder.add_edge("retrieve_ltm", "merge")
builder.add_edge("merge", "chat")
builder.add_edge("chat", END)

# ============================================================
# MAIN
# ============================================================


def main():
    user_id = input("Enter user_id: ").strip()
    raw_thread_id = input("Enter thread_id: ").strip()
    thread_id = f"{user_id}:{raw_thread_id}"

    ensure_user_thread(user_id, thread_id)

    with (
        PostgresSaver.from_conn_string(DB_URI_STM) as checkpointer,
        PostgresStore.from_conn_string(DB_URI_LTM) as store,
    ):
        checkpointer.setup()
        store.setup()

        graph = builder.compile(checkpointer=checkpointer, store=store)

        config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

        print("\nChat started. Type 'exit' to quit.\n")

        while True:
            text = input("You: ")
            if text.lower() in {"exit", "quit"}:
                break

            out = graph.invoke(
                {"messages": [HumanMessage(content=text)]}, config=config
            )

            assistant_msg = out["messages"][-1]
            content = getattr(assistant_msg, "content", "") or ""
            if content.strip():
                print("\nAssistant:\n" + content.strip())
            else:
                print("\nAssistant:\n[No textual content; check logs for tool usage]")
            print("-" * 80)


if __name__ == "__main__":
    main()
