# ============================================================
# Hybrid Memory Chat — Production Grade (Corrected)
# Multi-User • Multi-Thread • STM + LTM (Postgres + Vectors)
# ============================================================

import uuid
import psycopg2
import logging
import sys
import os
import re
from math import sqrt, exp
from typing import List
from datetime import datetime, timezone
from rank_bm25 import BM25Okapi

from pydantic import BaseModel, Field

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    AIMessage,
)
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, tools_condition
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
        "langsmith",
        "langsmith.client",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Explicitly disable LangSmith tracing if env vars are set
    for k in [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
    ]:
        if os.environ.get(k):
            os.environ.pop(k, None)


setup_logging()

log_memory = logging.getLogger("memory")
log_retrieve = logging.getLogger("memory.retrieve")
log_chat = logging.getLogger("chat")
log_tools = logging.getLogger("tools")
log_rag = logging.getLogger("rag")
log_rag_retrieve = logging.getLogger("rag.retrieve")

# ============================================================
# RAG CACHE (BM25 + vectors)
# ============================================================

bm25_index = None
bm25_corpus = []  # list[list[str]] tokenized
rag_items_cache = []  # list[dict]: {chunk, embedding, source, doc_id, title, section}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def load_rag_cache():
    """Fetch all RAG items from dedicated store and prepare BM25 corpus."""
    global bm25_index, bm25_corpus, rag_items_cache
    from langgraph.store.postgres import PostgresStore

    rag_items_cache = []
    bm25_corpus = []

    try:
        with PostgresStore.from_conn_string(DB_URI_RAG) as rag_store:
            rag_store.setup()
            ns = ("rag", "global")
            for it in rag_store.search(ns):
                val = it.value
                rag_items_cache.append(val)
                bm25_corpus.append(_tokenize(val.get("chunk", "")))
        if rag_items_cache:
            bm25_index = BM25Okapi(bm25_corpus)
            log_rag.info("Loaded RAG cache: %d items", len(rag_items_cache))
        else:
            bm25_index = None
            log_rag.warning("RAG cache empty; ingestion required")
    except Exception as e:
        log_rag.error("Failed loading RAG cache", exc_info=e)
        bm25_index = None
        rag_items_cache = []
        bm25_corpus = []


# ============================================================
# CONFIG
# ============================================================

DB_URI_USERS = "postgresql://imrannazir@localhost:5432/user_registry"
DB_URI_STM = "postgresql://imrannazir@localhost:5432/stm_persist"
DB_URI_LTM = "postgresql://imrannazir@localhost:5432/ltm_persistence"
DB_URI_RAG = "postgresql://imrannazir@localhost:5432/rag_knowledge"

MAX_RAG_RESULTS = 4
RAG_SIM_THRESHOLD = 0.75

MEMORY_SIM_THRESHOLD = 0.92
MAX_LTM_RESULTS = 5

# Hybrid retrieval and memory tuning
HYBRID_ALPHA = 0.6  # weight for vector similarity
MEMORY_DECAY_LAMBDA = 0.05  # exponential decay per day
MEMORY_PROMOTION_BOOST = 0.1  # boost when reconfirmed
MIN_RAG_SCORE = 0.45  # minimum hybrid score to include a RAG chunk

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

SYSTEM_PROMPT = """You are a helpful assistant.

CONTEXT:
{context}

Context sections may include:
- Conversation Summary (what has been discussed)
- User Memory (facts about the user)
- Knowledge Context (retrieved documents)

Rules:
- Prefer Knowledge Context for factual answers and citations
- Use User Memory to answer personal questions about the user (e.g., name, role, experience)
- If the answer is not present in Knowledge Context, say you are unsure
- Do not fabricate details
- Do NOT use external tools unless the user explicitly asks
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
    rag_context: List[dict] = []
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

    def effective_confidence(item_val):
        c = float(item_val.get("confidence", 1.0))
        last = item_val.get("last_confirmed") or item_val.get("created_at")
        try:
            t_last = datetime.fromisoformat(last.replace("Z", "+00:00"))
            days = max(
                0.0, (datetime.now(timezone.utc) - t_last).total_seconds() / 86400.0
            )
            decay = exp(-MEMORY_DECAY_LAMBDA * days)
        except Exception:
            decay = 1.0
        return max(0.0, min(1.0, c * decay))

    scored = []
    for it in store.search(ns):
        sim = cosine_similarity(q_emb, it.value["embedding"])  # approx 0..1
        conf = effective_confidence(it.value)
        # promotion boost when strongly matched by current query
        if sim >= 0.9:
            conf = min(1.0, conf + MEMORY_PROMOTION_BOOST)
        score = sim * conf
        scored.append((score, it.value["text"]))

    scored.sort(reverse=True)
    top = [t for _, t in scored[:MAX_LTM_RESULTS]]

    # Log query and fetched LTM items
    log_retrieve.info("LTM query: %s", query)
    log_retrieve.info("LTM fetched: %s", top)

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


def route_retrieval_node(state: dict):
    """Fan-out marker to run LTM and RAG retrieval in parallel before merging."""
    log_chat.debug("Routing to parallel LTM and RAG retrieval")
    return {}


def retrieve_rag(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    if not state.get("messages"):
        return {}

    query = state["messages"][-1].content
    q_emb = embeddings.embed_query(query)
    try:
        log_rag.info("RAG query: %s", query)
        # Ensure cache is loaded from dedicated RAG store
        if bm25_index is None or not rag_items_cache:
            load_rag_cache()
        if not rag_items_cache:
            return {"rag_context": []}

        # Vector similarities
        vec_scores = []
        for idx, item in enumerate(rag_items_cache):
            sim = cosine_similarity(q_emb, item.get("embedding", []))
            vec_scores.append((idx, max(0.0, sim)))
        max_vec = max((s for _, s in vec_scores), default=1.0) or 1.0

        # BM25 scores
        q_tokens = [t for t in re.split(r"[^A-Za-z0-9]+", query.lower()) if t]
        bm_scores_list = (
            list(bm25_index.get_scores(q_tokens))
            if bm25_index
            else [0.0] * len(rag_items_cache)
        )
        max_bm = max(bm_scores_list) if bm_scores_list else 1.0
        if max_bm == 0.0:
            max_bm = 1.0

        # Combine scores and filter by minimum hybrid threshold
        combined = []
        for idx, vscore in vec_scores:
            v_norm = vscore / max_vec if max_vec > 0 else 0.0
            b_norm = (
                (bm_scores_list[idx] / max_bm) if idx < len(bm_scores_list) else 0.0
            )
            combo = HYBRID_ALPHA * v_norm + (1 - HYBRID_ALPHA) * b_norm
            if combo >= MIN_RAG_SCORE:
                combined.append((combo, idx))

        combined.sort(reverse=True)
        top_idxs = [i for _, i in combined[:MAX_RAG_RESULTS]]
        top_items = [rag_items_cache[i] for i in top_idxs]

        for i, val in enumerate(top_items):
            log_rag_retrieve.info(
                "RAG[%d]: src=%s title=%s section=%s",
                i,
                val.get("source"),
                val.get("title"),
                val.get("section"),
            )

        # Include metadata for citation grounding
        ctx = [
            {
                "text": v.get("chunk", ""),
                "source": v.get("source"),
                "doc_id": v.get("doc_id"),
                "title": v.get("title"),
                "section": v.get("section"),
            }
            for v in top_items
        ]
        return {"rag_context": ctx}
    except Exception as e:
        log_rag.error("RAG retrieval failed", exc_info=e)
        return {}


def merge_node(state: dict):
    blocks = []

    summary = state.get("summary")
    memories = state.get("retrieved_memories")
    rag_chunks = state.get("rag_context")

    if summary:
        log_chat.debug("Merging summary into context")
        log_chat.info("STM: summary present=%s", True)
        blocks.append(f"Conversation Summary:\n{summary}")

    if memories:
        log_chat.debug("Merging %d user memories", len(memories))
        log_chat.info("LTM: count=%d", len(memories))
        blocks.append("User Memory:\n" + "\n".join(memories))

    if rag_chunks:
        log_chat.debug("Merging %d RAG chunks", len(rag_chunks))
        log_chat.info("RAG: count=%d", len(rag_chunks))
        blocks.append(
            "Knowledge Context:\n" + "\n".join([c.get("text", "") for c in rag_chunks])
        )

    merged = "\n\n".join(blocks)

    log_chat.debug(
        "Final merged context | sections=%d | length=%d chars",
        len(blocks),
        len(merged),
    )

    return {"merged_context": merged}


def agent_node(state: dict):
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(context=state.get("merged_context") or "(empty)")
    )
    messages = [system] + state.get("messages", [])
    # Bind tools so the LLM can decide to call them
    response = chat_llm.bind_tools(tools).invoke(messages)
    log_chat.debug("Agent produced a message")
    return {"messages": [response]}


def finalize_node(state: dict):
    # Append citations to the last assistant message if any RAG context was used
    citations = state.get("rag_context") or []
    if not citations or not state.get("messages"):
        return {}
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "") or ""
    cite_lines = []
    for c in citations:
        src = c.get("source") or c.get("title") or c.get("doc_id")
        sect = c.get("section")
        if src and sect:
            cite_lines.append(f"- {src} ({sect})")
        elif src:
            cite_lines.append(f"- {src}")
    if not cite_lines:
        return {}
    new_content = content + "\n\nCitations:\n" + "\n".join(cite_lines)
    return {"messages": [RemoveMessage(id=last_msg.id), AIMessage(content=new_content)]}


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(ChatState)

builder.add_node("remember", remember_node)
builder.add_node("retrieve_ltm", retrieve_ltm)
builder.add_node("summarize", summarize_node)
builder.add_node("retrieve_rag", retrieve_rag)
builder.add_node("merge", merge_node)
builder.add_node("agent", agent_node)
builder.add_node("route_retrieval", route_retrieval_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "remember")
builder.add_conditional_edges(
    "remember", should_summarize, {True: "summarize", False: "route_retrieval"}
)
builder.add_edge("summarize", "route_retrieval")
builder.add_edge("route_retrieval", "retrieve_ltm")
builder.add_edge("route_retrieval", "retrieve_rag")
builder.add_edge("retrieve_ltm", "merge")
builder.add_edge("retrieve_rag", "merge")
builder.add_edge("merge", "agent")
builder.add_conditional_edges(
    "agent", tools_condition, {"tools": "tools", "__end__": "finalize"}
)
builder.add_edge("tools", "agent")
builder.add_edge("finalize", END)

# ============================================================
# MAIN
# ============================================================


def main():
    user_id = input("Enter user_id: ").strip()
    raw_thread_id = input("Enter thread_id: ").strip()
    thread_id = f"{user_id}:{raw_thread_id}"

    ensure_user_thread(user_id, thread_id)

    # Preload RAG cache (BM25 + vectors)
    load_rag_cache()

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
