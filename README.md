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
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_CHAT_TEMPERATURE=0.2
OPENAI_CHAT_MAX_TOKENS=500
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
ST_MAX_TURNS=20
ST_TTL_SECONDS=86400
```

If Supabase or Qdrant are not configured yet, the project still imports and tests cleanly.

If you switch from the hash fallback to real OpenAI embeddings, make sure the
Qdrant collection dimension matches the embedding model. For
`text-embedding-3-small`, use `EMBEDDING_DIM=1536` and recreate the
`kapruka_catalog` collection if it was previously created with a different size.

## Tests

```bash
./.venv/bin/python -m unittest test_web_crawler.py test_memory.py
```

## FastAPI

Install dependencies:

```bash
./.venv/bin/pip install -r requirements.txt
```

Run the API from the project root:

```bash
./.venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "gift for wife",
    "user_id": "demo-user",
    "session_id": "demo-session",
    "top_k": 5
  }'
```

For a frontend, send user messages to `POST /chat`. The response includes the
assistant answer, selected route, progress messages, specialist output, and
timings for debug panels.

## End-to-End Demo

Run the three memory tiers together:

```bash
./.venv/bin/python -m memory.demo_stack \
  --query "gift for wife" \
  --turn "I need a romantic anniversary gift" \
  --preference "Loves dark chocolate" \
  --note "Prefers elegant packaging"
```

This will:

- store the latest turn in short-term memory
- save recipient preferences in the semantic profile store
- search the Qdrant-backed catalog using the combined memory context

## LLM + Memory

Run one query through all three memory layers and then send the assembled
context to the LLM:

```bash
./.venv/bin/python -m memory.ask_with_memory \
  --query "gift for wife" \
  --turn "I need a romantic anniversary gift" \
  --preference "Loves dark chocolate" \
  --note "Prefers elegant packaging" \
  --sync-catalog
```

This prints:

- the short-term conversation turns used
- the semantic recipient profile used
- the long-term Qdrant catalog matches used
- the final LLM answer

If you already ingested `catalog.json` into Qdrant, omit `--sync-catalog` so the
command does not re-embed and upsert the full catalog on every query.

## Part 3: Specialist orchestration

The specialist layer is implemented under `src/agents/`.

- Router: `src/agents/router.py`
- Catalog specialist: `src/agents/catalog_agent.py`
- Logistics specialist: `src/agents/logistics_agent.py`
- Orchestrator: `src/agents/orchestrator.py`

Run a single routed query:

```bash
./.venv/bin/python -m agents.demo_orchestrator \
  --query "gift for wife"
```

Run a logistics query:

```bash
./.venv/bin/python -m agents.demo_orchestrator \
  --query "Can you deliver this to Colombo today?"
```

Run a preference update:

```bash
./.venv/bin/python -m agents.demo_orchestrator \
  --query "Remember that my wife loves dark chocolate"
```

The demo prints:

- the route decision from the router
- the specialist output
- the final answer returned by the orchestrator
