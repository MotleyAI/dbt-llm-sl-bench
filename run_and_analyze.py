"""Run SLayer benchmark and produce a detailed analysis file.

Usage:
    python run_and_analyze.py [OPTIONS]

    --model MODEL        Model name (default: openai:gpt-5.3-codex)
    --effort EFFORT      Reasoning effort (default: medium)
    --iterations N       Number of iterations (default: 1)
    --no-bridges         Regenerate models without bridge models first
    --skip-setup         Skip setup_slayer.py (use existing models/db)
    --skip-run           Skip benchmark run (analyze latest results only)
    --output FILE        Output markdown file (default: benchmark_analysis.md)
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import duckdb
import nest_asyncio

nest_asyncio.apply()
warnings.filterwarnings("ignore")


def run_setup(no_bridges: bool) -> None:
    """Run setup_slayer.py to regenerate models and database."""
    cmd = [sys.executable, "setup_slayer.py"]
    if no_bridges:
        cmd.append("--no-bridges")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_benchmark(model: str, effort: str | None, iterations: int) -> None:
    """Run the SLayer benchmark."""
    from llm_bench.config import SLayerConfig
    from llm_bench.runners import run_single_benchmark

    kwargs = {"model_name": model, "number_of_iterations": iterations}
    if effort:
        kwargs["reasoning_effort"] = effort

    config = SLayerConfig(**kwargs)
    answers, _df = run_single_benchmark(config, parallel_challenges=True, max_workers=11)

    correct = sum(1 for a in answers if a.is_correct)
    print(f"\n=== Results: {correct}/{len(answers)} correct ===")
    for a in answers:
        mark = "[Y]" if a.is_correct else "[N]"
        print(f"  {mark} {a.challenge_text[:80]}")


def load_gold_queries() -> dict[str, str]:
    """Load gold SQL queries from benchmark questions file."""
    from llm_bench.utils.challenge_loader import load_challenges_from_ttl

    challenges = load_challenges_from_ttl("benchmark_questions.ttl", selected_challenges=None)
    return {row["challenge_text"]: row["gold_query_text"] for _, row in challenges.iterrows()}


def load_latest_answers(db_path: str = "llm_bench.db") -> list[tuple]:
    """Load answers from the latest benchmark batch."""
    conn = duckdb.connect(db_path, read_only=True)
    # Add full_response column if missing (older DB schemas)
    try:
        conn.execute("ALTER TABLE sql_answers ADD COLUMN IF NOT EXISTS full_response VARCHAR")
    except Exception:
        pass
    rows = conn.execute("""
        SELECT challenge_text, sql, is_correct, error, comparison_error, model, full_response
        FROM sql_answers
        WHERE batch_id = (SELECT MAX(batch_id) FROM sql_answers)
        ORDER BY challenge_text
    """).fetchall()
    conn.close()
    return rows


def get_generated_sql(slayer_json: str, models_dir: str = "slayer_models") -> str:
    """Replay a SLayer query to get the generated SQL."""
    from slayer.client.slayer_client import SlayerClient
    from slayer.storage.yaml_storage import YAMLStorage

    storage = YAMLStorage(base_dir=models_dir)
    client = SlayerClient(storage=storage)
    try:
        query_dict = json.loads(slayer_json)
        return client.sql_sync(query_dict)
    except Exception as e:
        return f"ERROR: {e}"


def _replay_and_compare(
    rows: list[tuple],
    gold: dict[str, str],
    models_dir: str = "slayer_models",
    db_path: str = "acme.duckdb",
) -> list[dict]:
    """Replay each stored query, execute gold SQL, compare, return full diagnostics."""
    import pandas as pd

    from slayer.client.slayer_client import SlayerClient
    from slayer.sql.client import _sync_engines
    from slayer.storage.yaml_storage import YAMLStorage

    from llm_bench.services.comparison import ComparisonService

    # Phase 1: execute all SLayer queries
    storage = YAMLStorage(base_dir=models_dir)
    client = SlayerClient(storage=storage)
    slayer_results: dict[str, dict] = {}
    for q_text, slayer_json, is_correct, err, comp_err, model_name, *rest in rows:
        entry: dict = {"slayer_df": None, "slayer_sql": None, "slayer_error": None}
        try:
            query_dict = json.loads(slayer_json)
            entry["slayer_sql"] = client.sql_sync(query_dict)
            result = client.query_sync(query_dict)
            entry["slayer_df"] = pd.DataFrame(result.data)
        except Exception as e:
            entry["slayer_error"] = str(e)
        slayer_results[q_text] = entry

    # Clean up SLayer's DuckDB connections
    for engine in _sync_engines.values():
        engine.dispose()
    _sync_engines.clear()
    del client, storage

    # Phase 2: execute gold queries via raw DuckDB
    gold_results: dict[str, dict] = {}
    conn = duckdb.connect(db_path, read_only=True)
    for q_text in [r[0] for r in rows]:
        gold_sql = gold.get(q_text, "")
        entry = {"gold_df": None, "gold_error": None}
        try:
            entry["gold_df"] = conn.execute(gold_sql).fetchdf()
        except Exception as e:
            entry["gold_error"] = str(e)
        gold_results[q_text] = entry
    conn.close()

    # Phase 3: compare and build diagnostics
    diagnostics = []
    for i, row in enumerate(rows, 1):
        q_text, slayer_json, bench_correct, bench_err, bench_comp_err, model_name = row[:6]
        full_response = row[6] if len(row) > 6 else None
        sr = slayer_results[q_text]
        gr = gold_results[q_text]
        diag: dict = {
            "num": i,
            "question": q_text,
            "slayer_query": slayer_json,
            "full_response": full_response,
            "bench_correct": bench_correct,
            "bench_error": bench_err,
            "bench_comp_error": bench_comp_err,
            "slayer_sql": sr["slayer_sql"],
            "slayer_error": sr["slayer_error"],
            "gold_sql": gold.get(q_text, "N/A"),
            "gold_error": gr["gold_error"],
            "replay_correct": None,
            "replay_comp_error": None,
            "slayer_data": None,
            "gold_data": None,
        }

        slayer_df = sr["slayer_df"]
        gold_df = gr["gold_df"]
        if slayer_df is not None:
            diag["slayer_data"] = slayer_df.head(5).to_string(index=False)
        if gold_df is not None:
            diag["gold_data"] = gold_df.head(5).to_string(index=False)

        if slayer_df is not None and gold_df is not None:
            cmp = ComparisonService.compare_query_results(gold_df, slayer_df)
            diag["replay_correct"] = cmp.is_equivalent
            diag["replay_comp_error"] = cmp.error
        elif sr["slayer_error"]:
            diag["replay_correct"] = False
        elif gr["gold_error"]:
            diag["replay_correct"] = None  # can't compare

        diagnostics.append(diag)
    return diagnostics


def write_analysis(
    rows: list[tuple],
    gold: dict[str, str],
    output_path: str,
    model: str,
    effort: str | None,
) -> None:
    """Replay queries, compare with gold, write full diagnostic analysis."""
    diagnostics = _replay_and_compare(rows, gold)

    bench_correct = sum(1 for d in diagnostics if d["bench_correct"])
    replay_correct = sum(1 for d in diagnostics if d["replay_correct"])
    total = len(diagnostics)
    model_label = f"{model} ({effort} effort)" if effort else model

    lines = [
        f"# Benchmark Analysis: {model_label}, {total} questions",
        "",
        f"**Benchmark score: {bench_correct}/{total}  |  Replay score: {replay_correct}/{total}**",
        "",
        "---",
        "",
    ]

    for d in diagnostics:
        bench_mark = "PASS" if d["bench_correct"] else "FAIL"
        replay_mark = "PASS" if d["replay_correct"] else ("FAIL" if d["replay_correct"] is False else "N/A")
        mismatch = " **MISMATCH**" if d["bench_correct"] != d["replay_correct"] and d["replay_correct"] is not None else ""
        lines.append(f"## Q{d['num']}: {d['question']}")
        lines.append(f"**Benchmark: {bench_mark} | Replay: {replay_mark}{mismatch}**")
        lines.append("")

        # LLM reasoning (full response before JSON extraction)
        if d.get("full_response"):
            lines.append(f"**LLM Reasoning:**\n\n{d['full_response']}\n")

        # Extracted SLayer query
        try:
            pretty_query = json.dumps(json.loads(d["slayer_query"]), indent=2)
        except (json.JSONDecodeError, TypeError):
            pretty_query = d["slayer_query"]
        lines.append(f"**Extracted SLayer Query:**\n```json\n{pretty_query}\n```\n")

        # Generated SQL
        if d["slayer_error"]:
            lines.append(f"**Generated SQL:**\n```\nERROR: {d['slayer_error']}\n```\n")
        elif d["slayer_sql"]:
            lines.append(f"**Generated SQL:**\n```sql\n{d['slayer_sql']}\n```\n")

        # Gold SQL
        lines.append(f"**Gold SQL:**\n```sql\n{d['gold_sql']}\n```\n")
        if d["gold_error"]:
            lines.append(f"**Gold SQL error:** {d['gold_error']}\n")

        # Data comparison
        if d["slayer_data"] or d["gold_data"]:
            lines.append("**Data comparison:**")
            if d["gold_data"]:
                lines.append(f"```\nGold (top 5):\n{d['gold_data']}\n```")
            if d["slayer_data"]:
                lines.append(f"```\nSLayer (top 5):\n{d['slayer_data']}\n```")
            lines.append("")

        if d["replay_comp_error"]:
            lines.append(f"**Replay comparison error:** {d['replay_comp_error']}\n")
        if d["bench_error"]:
            lines.append(f"**Benchmark error:** {d['bench_error']}\n")
        if d["bench_comp_error"]:
            lines.append(f"**Benchmark comparison error:** {d['bench_comp_error']}\n")

        lines.append("---\n")

    Path(output_path).write_text("\n".join(lines))
    print(f"\nAnalysis written to: {output_path}")
    print(f"Benchmark: {bench_correct}/{total}  |  Replay: {replay_correct}/{total}")
    if bench_correct != replay_correct:
        print("*** MISMATCH between benchmark and replay scores! ***")
        for d in diagnostics:
            if d["bench_correct"] != d["replay_correct"] and d["replay_correct"] is not None:
                print(f"  Q{d['num']}: bench={d['bench_correct']} replay={d['replay_correct']} | {d['question'][:55]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SLayer benchmark and produce analysis")
    parser.add_argument("--model", default="openai:gpt-5.3-codex", help="Model name")
    parser.add_argument("--effort", default="medium", help="Reasoning effort (or 'none')")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--no-bridges", action="store_true", help="Regenerate without bridge models")
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup_slayer.py")
    parser.add_argument("--skip-run", action="store_true", help="Skip benchmark run, analyze latest")
    parser.add_argument("--output", default="benchmark_analysis.md", help="Output file path")
    args = parser.parse_args()

    effort = None if args.effort == "none" else args.effort

    if not args.skip_setup:
        run_setup(no_bridges=args.no_bridges)

    if not args.skip_run:
        run_benchmark(model=args.model, effort=effort, iterations=args.iterations)

    gold = load_gold_queries()
    rows = load_latest_answers()
    write_analysis(rows, gold, output_path=args.output, model=args.model, effort=effort)


if __name__ == "__main__":
    main()
