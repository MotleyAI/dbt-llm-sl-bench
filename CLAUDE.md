# CLAUDE.md

## What is this?

LLM benchmark for comparing how well LLMs answer business questions using different strategies (raw SQL, dbt Semantic Layer, MCP, SLayer). See README.md for full details.

## Common Commands

```bash
# Install with SLayer extra (pulls motley-slayer from sibling ../slayer directory)
uv sync --extra slayer

# Reinstall slayer after editing its source code
uv sync --reinstall-package motley-slayer --extra slayer

# Full benchmark run: setup DB + run LLM queries + analyze
uv run python run_and_analyze.py --model openai:gpt-5.3-codex --effort medium --output benchmark_analysis.md

# Skip setup (reuse existing DB and models)
uv run python run_and_analyze.py --skip-setup --model openai:gpt-5.3-codex --effort medium

# Replay only (re-execute stored queries with updated SLayer code, no LLM calls)
uv run python run_and_analyze.py --skip-setup --skip-run --output benchmark_replay.md

# Run tests
uv run pytest tests/
```

## Key Files

- `run_and_analyze.py` — Main entry point: setup + benchmark + analysis in one script
- `run_bench.py` — Multi-strategy matrix benchmark (SQL vs SLayer, multiple models)
- `setup_slayer.py` — Loads CSVs into DuckDB, converts dbt models to SLayer format
- `benchmark_questions.ttl` — Benchmark questions and gold SQL queries (Turtle/RDF)
- `slayer_models/` — Auto-generated SLayer model definitions (YAML)
- `acme.duckdb` — Benchmark database (created by setup_slayer.py)
- `llm_bench.db` — Results database (appended by each run)

## Linting

Do NOT lint this repo. Only lint the SLayer repo (`../slayer`).
