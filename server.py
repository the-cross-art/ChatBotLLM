"""
FastAPI backend with WebSocket for React integration
Uses the same hybrid memory + RAG graph logic as main.py,
but exposes it via a WebSocket endpoint for a React client.

IMPROVEMENTS:
- Async/await throughout to prevent blocking
- Connection pooling for database connections
- Thread-safe RAG cache with locks
- Proper error handling and recovery
- Rate limiting per user
- Input validation
- Timeouts on LLM calls
- Health checks for dependencies
- Metrics/monitoring hooks
"""

import json
import asyncio
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from math import sqrt, exp
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import defaultdict

import psycopg2
from psycopg2 import pool
import asyncpg
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    AIMessage,
)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig

from rank_bm25 import BM25Okapi


# ============================================================
# LOGGING
# ============================================================


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in [
        "psycopg",
        "psycopg2",
        "sqlalchemy",
        "httpcore",
        "httpx",
        "urllib3",
        "asyncio",
        "langsmith",
        "langsmith.client",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Disable LangSmith tracing if present
    for k in [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
    ]:
        if os.environ.get(k):
            os.environ.pop(k, None)


setup_logging()

log_app = logging.getLogger("app")
log_rag = logging.getLogger("rag")
log_retrieve = logging.getLogger("memory.retrieve")
log_memory = logging.getLogger("memory")
log_chat = logging.getLogger("chat")
log_ws = logging.getLogger("ws")
log_health = logging.getLogger("health")


# ============================================================
# CONFIG & CONSTANTS
# ============================================================

# Database URIs
DB_URI_USERS = os.environ.get(
    "DB_URI_USERS", "postgresql://imrannazir@localhost:5432/user_registry"
)
DB_URI_STM = os.environ.get(
    "DB_URI_STM", "postgresql://imrannazir@localhost:5432/stm_persist"
)
DB_URI_LTM = os.environ.get(
    "DB_URI_LTM", "postgresql://imrannazir@localhost:5432/ltm_persistence"
)
DB_URI_RAG = os.environ.get(
    "DB_URI_RAG", "postgresql://imrannazir@localhost:5432/rag_knowledge"
)

# Model names (configurable)
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:7b")
MEMORY_MODEL = os.environ.get("MEMORY_MODEL", "qwen2.5:7b")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# RAG configuration
MAX_RAG_RESULTS = 4
RAG_SIM_THRESHOLD = 0.75
MIN_RAG_SCORE = 0.45
HYBRID_ALPHA = 0.6

# Memory configuration
MEMORY_SIM_THRESHOLD = 0.92
MAX_LTM_RESULTS = 5
MEMORY_DECAY_LAMBDA = 0.05
MEMORY_PROMOTION_BOOST = 0.1

# Message management
MAX_MESSAGES_BEFORE_SUMMARY = 8
MAX_MESSAGE_HISTORY = 50

# Rate limiting
RATE_LIMIT_MESSAGES = 30  # messages per window
RATE_LIMIT_WINDOW = 60  # seconds

# Timeouts
LLM_TIMEOUT = 30.0  # seconds
EMBEDDING_TIMEOUT = 10.0  # seconds
DB_TIMEOUT = 5.0  # seconds

# Validation limits
MAX_MESSAGE_LENGTH = 10000
MAX_USER_ID_LENGTH = 100
MAX_THREAD_ID_LENGTH = 100


# ============================================================
# VALIDATION MODELS
# ============================================================


class WebSocketInitMessage(BaseModel):
    init: bool = True
    user_id: str = Field(..., min_length=1, max_length=MAX_USER_ID_LENGTH)
    thread_id: str = Field(..., min_length=1, max_length=MAX_THREAD_ID_LENGTH)

    @validator("user_id", "thread_id")
    def validate_alphanumeric(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Must contain only alphanumeric characters, _ or -")
        return v


class WebSocketChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


# ============================================================
# CONNECTION POOLS
# ============================================================

user_db_pool: Optional[pool.ThreadedConnectionPool] = None
asyncpg_pool: Optional[asyncpg.Pool] = None


async def init_pools():
    """Initialize database connection pools"""
    global user_db_pool, asyncpg_pool

    try:
        # Sync pool for user registry (used in sync contexts)
        user_db_pool = pool.ThreadedConnectionPool(
            minconn=2, maxconn=10, dsn=DB_URI_USERS
        )
        log_app.info("User DB pool initialized")

        # Async pool for general async operations
        asyncpg_pool = await asyncpg.create_pool(
            DB_URI_USERS, min_size=2, max_size=10, command_timeout=DB_TIMEOUT
        )
        log_app.info("Asyncpg pool initialized")
    except Exception as e:
        log_app.error("Failed to initialize connection pools", exc_info=e)
        raise


async def close_pools():
    """Close database connection pools"""
    global user_db_pool, asyncpg_pool

    if user_db_pool:
        user_db_pool.closeall()
        log_app.info("User DB pool closed")

    if asyncpg_pool:
        await asyncpg_pool.close()
        log_app.info("Asyncpg pool closed")


# ============================================================
# MODELS
# ============================================================

# Initialize models with error handling
try:
    chat_llm_assistant = ChatOllama(model=CHAT_MODEL).with_config(tags=["assistant"])
    memory_llm = ChatOllama(model=MEMORY_MODEL).with_config(tags=["memory"])
    summary_llm = ChatOllama(model=SUMMARY_MODEL).with_config(tags=["summary"])
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    log_app.info("LLM models initialized successfully")
except Exception as e:
    log_app.error("Failed to initialize models - ensure Ollama is running", exc_info=e)
    # Models will be None and health check will fail
    chat_llm_assistant = None
    memory_llm = None
    summary_llm = None
    embeddings = None

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
# RAG CACHE (Thread-safe)
# ============================================================


class RAGCache:
    """Thread-safe RAG cache with locking"""

    def __init__(self):
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.items: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()
        self.last_load_attempt: Optional[datetime] = None
        self.load_failures = 0
        self.max_failures = 3
        self.backoff_seconds = 60

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]

    async def load(self):
        """Load RAG cache from database with failure tracking"""
        async with self.lock:
            # Implement backoff if we've had recent failures
            if self.last_load_attempt and self.load_failures >= self.max_failures:
                time_since_attempt = (
                    datetime.now() - self.last_load_attempt
                ).total_seconds()
                if time_since_attempt < self.backoff_seconds:
                    log_rag.warning(
                        f"Skipping cache reload due to recent failures "
                        f"({self.load_failures} failures, backing off)"
                    )
                    return

            self.last_load_attempt = datetime.now()
            items = []
            corpus = []

            try:
                # Use async store for non-blocking load
                async with AsyncPostgresStore.from_conn_string(DB_URI_RAG) as rag_store:
                    await rag_store.setup()
                    ns = ("rag", "global")
                    search_results = await rag_store.asearch(ns)
                    for it in search_results:
                        val = it.value
                        items.append(val)
                        corpus.append(self._tokenize(val.get("chunk", "")))

                if items:
                    # Atomically update all cache data
                    self.items = items
                    self.bm25_corpus = corpus
                    self.bm25_index = BM25Okapi(corpus)
                    self.load_failures = 0  # Reset on success
                    log_rag.info(f"RAG cache loaded: {len(items)} items")
                else:
                    self.bm25_index = None
                    self.items = []
                    self.bm25_corpus = []
                    log_rag.warning("RAG cache empty; ingestion required")
                    self.load_failures += 1

            except Exception as e:
                log_rag.error("Failed loading RAG cache", exc_info=e)
                self.load_failures += 1
                # Don't clear existing cache on failure - keep serving old data

    async def search(
        self, query: str, query_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Thread-safe search with hybrid scoring"""
        async with self.lock:
            if not self.items or self.bm25_index is None:
                return []

            # Vector similarity scores
            vec_scores = []
            for idx, item in enumerate(self.items):
                sim = cosine_similarity(query_embedding, item.get("embedding", []))
                vec_scores.append((idx, max(0.0, sim)))

            max_vec = max((s for _, s in vec_scores), default=1.0) or 1.0

            # BM25 scores
            q_tokens = self._tokenize(query)
            bm_scores_list = list(self.bm25_index.get_scores(q_tokens))
            max_bm = max(bm_scores_list) if bm_scores_list else 1.0
            if max_bm == 0.0:
                max_bm = 1.0

            # Hybrid scoring
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

            return [
                {
                    "text": self.items[i].get("chunk", ""),
                    "source": self.items[i].get("source"),
                    "doc_id": self.items[i].get("doc_id"),
                    "title": self.items[i].get("title"),
                    "section": self.items[i].get("section"),
                }
                for i in top_idxs
            ]


# Global RAG cache instance
rag_cache = RAGCache()


# ============================================================
# RATE LIMITING
# ============================================================


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, user_id: str) -> bool:
        """Returns True if request is allowed, False if rate limited"""
        async with self.lock:
            now = datetime.now().timestamp()
            cutoff = now - self.window_seconds

            # Remove old requests
            self.requests[user_id] = [
                ts for ts in self.requests[user_id] if ts > cutoff
            ]

            # Check limit
            if len(self.requests[user_id]) >= self.max_requests:
                return False

            # Add new request
            self.requests[user_id].append(now)
            return True


rate_limiter = RateLimiter(RATE_LIMIT_MESSAGES, RATE_LIMIT_WINDOW)


# ============================================================
# DATA MODELS
# ============================================================


class ChatState(MessagesState):
    summary: str = ""
    retrieved_memories: List[str] = []
    rag_context: List[dict] = []
    merged_context: str = ""


# ============================================================
# UTILS
# ============================================================


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity with proper zero vector handling"""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sqrt(sum(x * x for x in a))
    mag_b = sqrt(sum(x * x for x in b))

    # Handle zero vectors
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


async def ensure_user_thread(user_id: str, thread_id: str):
    """Ensure user-thread relationship exists in database (async with pool)"""
    if not asyncpg_pool:
        log_app.error("Asyncpg pool not initialized")
        raise RuntimeError("Database pool not available")

    async with asyncpg_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_threads (
                user_id TEXT,
                thread_id TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (user_id, thread_id)
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO user_threads (user_id, thread_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
            """,
            user_id,
            thread_id,
        )


async def call_with_timeout(coro, timeout: float, operation: str):
    """Execute async operation with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log_app.error(f"{operation} timed out after {timeout}s")
        raise
    except Exception as e:
        log_app.error(f"{operation} failed: {e}", exc_info=e)
        raise


# ============================================================
# GRAPH NODES (Async versions)
# ============================================================


from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    text: str
    type: str = "general"
    confidence: float = 0.85
    is_new: bool


class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


# Bind structured output
memory_extractor = (
    memory_llm.with_structured_output(MemoryDecision) if memory_llm else None
)


async def remember_node(state: ChatState, config, *, store: BaseStore):
    """Extract and store user memories (async)"""
    if not state.get("messages") or not memory_extractor:
        return {}

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    try:
        # Fetch existing memories
        search_results = await store.asearch(ns)
        existing_items = list(search_results)
        existing_texts = [i.value["text"] for i in existing_items]

        # Call LLM with timeout
        decision = await call_with_timeout(
            memory_extractor.ainvoke(
                [
                    SystemMessage(
                        content=MEMORY_PROMPT.format(
                            existing="\n".join(existing_texts) or "(empty)"
                        )
                    ),
                    HumanMessage(content=state["messages"][-1].content),
                ]
            ),
            LLM_TIMEOUT,
            "Memory extraction",
        )

        if not decision.should_write:
            return {}

        # Process new memories
        for mem in decision.memories:
            if not mem.is_new:
                continue

            # Compute embedding with timeout
            emb = await call_with_timeout(
                asyncio.to_thread(embeddings.embed_query, mem.text),
                EMBEDDING_TIMEOUT,
                "Memory embedding",
            )

            # Check for duplicates
            duplicate = False
            for it in existing_items:
                sim = cosine_similarity(emb, it.value.get("embedding", []))
                if sim >= MEMORY_SIM_THRESHOLD:
                    duplicate = True
                    break

            if duplicate:
                continue

            # Store new memory
            now = datetime.now(timezone.utc).isoformat()
            await store.aput(
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
            log_memory.info(f"Stored new memory for {user_id}: {mem.text}")

    except asyncio.TimeoutError:
        log_memory.error("Memory operation timed out")
        return {"error": "memory_timeout"}
    except Exception as e:
        log_memory.error("Memory write failed", exc_info=e)
        return {"error": "memory_failed"}

    return {}


async def retrieve_ltm(state: ChatState, config, *, store: BaseStore):
    """Retrieve long-term memories (async)"""
    if not state.get("messages"):
        return {}

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")
    query = state["messages"][-1].content

    try:
        # Compute query embedding with timeout
        q_emb = await call_with_timeout(
            asyncio.to_thread(embeddings.embed_query, query),
            EMBEDDING_TIMEOUT,
            "Query embedding for LTM",
        )

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

        # Search and score memories
        scored = []
        search_results = await store.asearch(ns)
        for it in search_results:
            sim = cosine_similarity(q_emb, it.value.get("embedding", []))
            conf = effective_confidence(it.value)
            if sim >= 0.9:
                conf = min(1.0, conf + MEMORY_PROMOTION_BOOST)
            score = sim * conf
            scored.append((score, it.value.get("text", "")))

        scored.sort(reverse=True)
        top = [t for _, t in scored[:MAX_LTM_RESULTS]]
        log_retrieve.info(f"LTM fetched {len(top)} memories")
        return {"retrieved_memories": top}

    except asyncio.TimeoutError:
        log_retrieve.error("LTM retrieval timed out")
        return {"retrieved_memories": []}
    except Exception as e:
        log_retrieve.error("LTM retrieval failed", exc_info=e)
        return {"retrieved_memories": []}


async def summarize_node(state: dict):
    """Summarize conversation and prune old messages (async)"""
    messages = state.get("messages", [])

    if not summary_llm:
        log_chat.warning("Summary LLM not available")
        return {}

    try:
        response = await call_with_timeout(
            summary_llm.ainvoke(
                messages + [HumanMessage(content="Summarize the conversation so far.")]
            ),
            LLM_TIMEOUT,
            "Conversation summary",
        )
        summary_text = response.content.strip()

        # Keep recent messages but remove older ones
        messages_to_remove = messages[:-2] if len(messages) > 2 else []

        return {
            "summary": summary_text,
            "messages": [RemoveMessage(id=m.id) for m in messages_to_remove],
        }
    except asyncio.TimeoutError:
        log_chat.error("Summarization timed out")
        return {}
    except Exception as e:
        log_chat.error("Summarization failed", exc_info=e)
        return {}


def should_summarize(state: dict):
    """Determine if conversation should be summarized"""
    msg_count = len(state.get("messages", []))
    return msg_count > MAX_MESSAGES_BEFORE_SUMMARY


async def route_retrieval_node(state: dict):
    """Routing node for parallel retrieval"""
    return {}


async def conditional_check(state: ChatState, config: RunnableConfig):
    """Check for easter eggs or special conditions (async)"""
    messages = state.get("messages", [])
    if not messages:
        return {}

    try:
        last = messages[-1]
        content = getattr(last, "content", "") or ""
        keywords = [
            "LangChain",
            "langchain",
            "Langchain",
            "LangGraph",
            "Langgraph",
            "langgraph",
        ]
        if any(k in content for k in keywords):
            await adispatch_custom_event("on_easter_egg", True, config=config)
    except Exception:
        pass

    return {}


async def retrieve_rag(state: ChatState, config, *, store: BaseStore):
    """Retrieve RAG context (async)"""
    if not state.get("messages"):
        return {}

    query = state["messages"][-1].content

    try:
        # Compute query embedding with timeout
        q_emb = await call_with_timeout(
            asyncio.to_thread(embeddings.embed_query, query),
            EMBEDDING_TIMEOUT,
            "Query embedding for RAG",
        )

        # Search RAG cache
        ctx = await rag_cache.search(query, q_emb)
        log_rag.info(f"RAG retrieved {len(ctx)} documents")
        return {"rag_context": ctx}

    except asyncio.TimeoutError:
        log_rag.error("RAG retrieval timed out")
        return {"rag_context": []}
    except Exception as e:
        log_rag.error("RAG retrieval failed", exc_info=e)
        return {"rag_context": []}


async def merge_node(state: dict):
    """Merge all context sources (async)"""
    blocks = []
    summary = state.get("summary")
    memories = state.get("retrieved_memories")
    rag_chunks = state.get("rag_context")

    if summary:
        blocks.append(f"Conversation Summary:\n{summary}")
    if memories:
        blocks.append("User Memory:\n" + "\n".join(memories))
    if rag_chunks:
        blocks.append(
            "Knowledge Context:\n" + "\n".join([c.get("text", "") for c in rag_chunks])
        )

    merged = "\n\n".join(blocks)
    return {"merged_context": merged}


async def agent_node(state: dict):
    """Main agent reasoning node (async)"""
    if not chat_llm_assistant:
        return {
            "messages": [
                AIMessage(
                    content="Chat model not available. Please check server configuration."
                )
            ]
        }

    system = SystemMessage(
        content=SYSTEM_PROMPT.format(context=state.get("merged_context") or "(empty)")
    )
    messages = [system] + state.get("messages", [])

    try:
        response = await call_with_timeout(
            chat_llm_assistant.bind_tools(tools).ainvoke(messages),
            LLM_TIMEOUT,
            "Agent response",
        )
        return {"messages": [response]}
    except asyncio.TimeoutError:
        log_chat.error("Agent response timed out")
        return {
            "messages": [
                AIMessage(
                    content="I apologize, but my response timed out. Please try again."
                )
            ]
        }
    except Exception as e:
        log_chat.error("Agent node failed", exc_info=e)
        return {
            "messages": [AIMessage(content="I encountered an error. Please try again.")]
        }


async def finalize_node(state: dict):
    """Finalize response (citations handled by WebSocket payload)"""
    return {}


def build_graph():
    """Build the LangGraph workflow"""
    builder = StateGraph(ChatState)

    # Add all nodes
    builder.add_node("conditional_check", conditional_check)
    builder.add_node("remember", remember_node)
    builder.add_node("retrieve_ltm", retrieve_ltm)
    builder.add_node("summarize", summarize_node)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("merge", merge_node)
    builder.add_node("agent", agent_node)
    builder.add_node("route_retrieval", route_retrieval_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("finalize", finalize_node)

    # Define edges
    builder.add_edge(START, "conditional_check")
    builder.add_edge("conditional_check", "remember")
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

    return builder


# ============================================================
# HEALTH CHECKS
# ============================================================


async def check_database_health() -> bool:
    """Check if databases are accessible"""
    try:
        if asyncpg_pool:
            async with asyncpg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        return False
    except Exception as e:
        log_health.error(f"Database health check failed: {e}")
        return False


async def check_ollama_health() -> bool:
    """Check if Ollama models are available"""
    try:
        if not embeddings:
            return False
        # Quick embedding test
        await asyncio.wait_for(
            asyncio.to_thread(embeddings.embed_query, "test"), timeout=5.0
        )
        return True
    except Exception as e:
        log_health.error(f"Ollama health check failed: {e}")
        return False


# ============================================================
# FASTAPI APP
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    log_app.info("Starting up application...")
    await init_pools()
    await rag_cache.load()
    log_app.info("Application startup complete")

    yield

    # Shutdown
    log_app.info("Shutting down application...")
    await close_pools()
    log_app.info("Application shutdown complete")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/health")
async def health():
    """Health check endpoint with dependency verification"""
    db_healthy = await check_database_health()
    ollama_healthy = await check_ollama_health()

    status = "healthy" if (db_healthy and ollama_healthy) else "degraded"

    return JSONResponse(
        {
            "status": status,
            "checks": {
                "database": "ok" if db_healthy else "fail",
                "ollama": "ok" if ollama_healthy else "fail",
                "rag_cache": "ok" if rag_cache.items else "empty",
            },
        }
    )


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return JSONResponse(
        {
            "rag_cache_size": len(rag_cache.items),
            "rag_load_failures": rag_cache.load_failures,
            "rate_limiter_users": len(rate_limiter.requests),
        }
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """WebSocket endpoint for chat"""
    log_ws.info("WebSocket connection attempt")
    await ws.accept()
    log_ws.info("WebSocket accepted")
    user_id: Optional[str] = None
    thread_id: Optional[str] = None

    try:
        # Expect an init message first
        log_ws.info("Waiting for init message...")
        raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
        log_ws.info(f"Received raw message: {raw}")

        try:
            payload = json.loads(raw)
            log_ws.info(f"Parsed payload: {payload}")
            init_msg = WebSocketInitMessage(**payload)
            user_id = init_msg.user_id
            thread_id = init_msg.thread_id
            log_ws.info(f"Initialized with user_id={user_id}, thread_id={thread_id}")
        except json.JSONDecodeError as e:
            log_ws.error(f"JSON decode error: {e}")
            await ws.send_text(json.dumps({"error": "invalid_json"}))
            return
        except ValueError as e:
            log_ws.error(f"Validation error: {e}")
            await ws.send_text(json.dumps({"error": f"validation_error: {str(e)}"}))
            return

        # Ensure user-thread relationship exists
        await ensure_user_thread(user_id, thread_id)

        # Open async checkpointer + store for this connection
        async with AsyncPostgresSaver.from_conn_string(DB_URI_STM) as checkpointer:
            await checkpointer.setup()

            async with AsyncPostgresStore.from_conn_string(DB_URI_LTM) as store:
                await store.setup()

                # Build graph
                builder = build_graph()
                graph_runnable = builder.compile(checkpointer=checkpointer, store=store)
                config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

                await ws.send_text(json.dumps({"ok": True, "message": "initialized"}))

                # Handle chat messages
                while True:
                    try:
                        raw = await ws.receive_text()

                        # Parse and validate message
                        try:
                            payload = json.loads(raw)
                            chat_msg = WebSocketChatMessage(**payload)
                            text = chat_msg.message
                        except json.JSONDecodeError:
                            await ws.send_text(json.dumps({"error": "invalid_json"}))
                            continue
                        except ValueError as e:
                            await ws.send_text(
                                json.dumps({"error": f"validation_error: {str(e)}"})
                            )
                            continue

                        # Check rate limit
                        if not await rate_limiter.check_rate_limit(user_id):
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "error": "rate_limited",
                                        "message": f"Rate limit exceeded. Max {RATE_LIMIT_MESSAGES} messages per {RATE_LIMIT_WINDOW}s.",
                                    }
                                )
                            )
                            continue

                        # Process message
                        final_text = ""
                        final_output = {}

                        try:
                            initial_input = {"messages": [HumanMessage(content=text)]}

                            async for event in graph_runnable.astream_events(
                                initial_input, config, version="v2"
                            ):
                                kind = event.get("event")
                                name = event.get("name")

                                if kind == "on_chat_model_stream" and name == "agent":
                                    chunk = event.get("data", {}).get("chunk")
                                    addition = getattr(chunk, "content", None)
                                    if addition:
                                        final_text += addition
                                        await ws.send_text(
                                            json.dumps(
                                                {"on_chat_model_stream": addition}
                                            )
                                        )

                                elif kind == "on_chat_model_end" and name == "agent":
                                    await ws.send_text(
                                        json.dumps({"on_chat_model_end": True})
                                    )

                                elif kind == "on_custom_event":
                                    # Forward custom events
                                    event_name = event.get("name")
                                    data = event.get("data")
                                    await ws.send_text(json.dumps({event_name: data}))

                                elif kind == "on_graph_end":
                                    data = event.get("data", {})
                                    output = data.get("output")
                                    if isinstance(output, dict):
                                        final_output = output

                        except asyncio.TimeoutError:
                            log_ws.error("Graph execution timed out")
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "error": "timeout",
                                        "message": "Request timed out. Please try again.",
                                    }
                                )
                            )
                            continue
                        except Exception as e:
                            log_ws.error("Graph execution error", exc_info=e)
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "error": "processing_failed",
                                        "message": "An error occurred processing your message.",
                                    }
                                )
                            )
                            continue

                        # Send final assembled message with citations
                        try:
                            content = final_text
                            citations = final_output.get("rag_context") or []
                            msgs = final_output.get("messages") or []

                            if msgs:
                                try:
                                    assistant_msg = msgs[-1]
                                    maybe_content = getattr(
                                        assistant_msg, "content", None
                                    )
                                    if (
                                        isinstance(maybe_content, str)
                                        and maybe_content.strip()
                                    ):
                                        content = maybe_content.strip()
                                except Exception:
                                    pass

                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "assistant_message",
                                        "content": content,
                                        "citations": citations,
                                    }
                                )
                            )
                        except Exception as e:
                            log_ws.error("Final message send failed", exc_info=e)

                    except WebSocketDisconnect:
                        log_ws.info(f"WebSocket disconnected for user {user_id}")
                        break
                    except Exception as e:
                        log_ws.error("Error in message loop", exc_info=e)
                        try:
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "error": "internal_error",
                                        "message": "An internal error occurred.",
                                    }
                                )
                            )
                        except Exception:
                            break

    except WebSocketDisconnect:
        log_ws.info("WebSocket disconnected before init")
    except asyncio.TimeoutError:
        log_ws.warning("WebSocket init timeout - no message received within 30s")
        try:
            await ws.send_text(json.dumps({"error": "init_timeout"}))
        except Exception:
            pass
    except Exception as e:
        log_ws.error(f"WebSocket error during init: {e}", exc_info=e)
        try:
            await ws.send_text(json.dumps({"error": "server_error"}))
        except Exception:
            pass
    finally:
        if user_id:
            log_ws.info(f"Connection closed for user {user_id}")
        else:
            log_ws.info("Connection closed before user initialization")


# Optionally serve built frontend
# @app.on_event("startup")
# async def mount_frontend():
#     """Mount static frontend if available"""
#     dist_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
#     if os.path.isdir(dist_path):
#         try:
#             app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")
#             log_app.info(f"Mounted static frontend at '/' from {dist_path}")
#         except Exception as e:
#             log_app.warning(f"Failed mounting static frontend: {e}")


# For local run: uvicorn server:app --reload
