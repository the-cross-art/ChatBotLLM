CREATE TABLE IF NOT EXISTS rag_chunks (
    id UUID PRIMARY KEY,
    doc_id TEXT,
    source TEXT,
    title TEXT,
    section TEXT,
    chunk TEXT,
    embedding FLOAT8[],
    created_at TIMESTAMP DEFAULT NOW()
);