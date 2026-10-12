"""pgvector / AlloyDB-compatible store (same SQL works on Cloud SQL Postgres + AlloyDB).
NOT exercised against a live database in this repo's results; tests cover SQL generation only."""
from __future__ import annotations

def schema_sql(dim: int, table: str = "chunks") -> str:
    return (f"CREATE EXTENSION IF NOT EXISTS vector;\n"
            f"CREATE TABLE IF NOT EXISTS {table} (id text PRIMARY KEY, source text, body text, embedding vector({dim}));\n"
            f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} USING hnsw (embedding vector_ip_ops);")

def query_sql(table: str = "chunks") -> str:
    # <#> is negative inner product in pgvector; ascending order = most similar first
    return f"SELECT id, source, body FROM {table} ORDER BY embedding <#> %s::vector LIMIT %s"

def to_pgvector(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

class PgVectorRetriever:
    def __init__(self, dsn: str, embedder, table: str = "chunks"):
        import psycopg  # optional dependency
        self.conn, self.embedder, self.table = psycopg.connect(dsn), embedder, table
    def upsert(self, chunks):
        vecs = self.embedder([c.text for c in chunks])
        with self.conn.cursor() as cur:
            cur.execute(schema_sql(len(vecs[0]), self.table))
            for c, v in zip(chunks, vecs):
                cur.execute(f"INSERT INTO {self.table} VALUES (%s,%s,%s,%s::vector) ON CONFLICT (id) DO NOTHING",
                            (c.id, c.source, c.text, to_pgvector(v)))
        self.conn.commit()
    def search_rows(self, q: str, k: int):
        with self.conn.cursor() as cur:
            cur.execute(query_sql(self.table), (to_pgvector(self.embedder([q])[0]), k))
            return cur.fetchall()
