# kapruka_agent

Mini project for crawling Kapruka and building a 3-tier cognitive memory stack.

## Parts

### Part 1: Web crawler

- `src/web_crawler.py`
- `catalog.json`
- `test_web_crawler.py`

The crawler builds a product catalog with:

- `name`
- `price`
- `description`
- `availability`
- `url`

### Part 2: Cognitive memory lab

The memory stack is implemented under `src/memory/`.

- Short-term memory: `src/memory/st_store.py`
  Uses Supabase/PostgreSQL when configured, with a local fallback for tests.
- Long-term catalog memory: `src/memory/lt_store.py`
  Uses Qdrant to store and search embedded `catalog.json` products.
- Semantic recipient profiles: `src/memory/semantic_store.py`
  Uses JSON storage for recipient preferences and notes.

High-level orchestration lives in:

- `src/memory/memory_ops.py`

Database helpers live in:

- `src/infastructure/db/supabase_client.py`
- `src/infastructure/db/sql_client.py`
- `src/infastructure/db/qdrant_client.py`

## Environment

Optional environment variables for the memory stack:

```bash
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_DB_URL=
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=kapruka_catalog
EMBEDDING_DIM=128
ST_MAX_TURNS=20
ST_TTL_SECONDS=86400
```

If Supabase or Qdrant are not configured yet, the project still imports and tests cleanly.

## Tests

```bash
./.venv/bin/python -m unittest test_web_crawler.py test_memory.py
```
