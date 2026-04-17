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
    answers, _df = run_single_benchmark(config, parallel_challenges=False)

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
    rows = conn.execute("""
        SELECT challenge_text, sql, is_correct, error, comparison_error, model
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


def write_analysis(
    rows: list[tuple],
    gold: dict[str, str],
    output_path: str,
    model: str,
    effort: str | None,
) -> None:
    """Write the full analysis markdown file."""
    from slayer.sql.client import _sync_engines

    correct_count = sum(1 for r in rows if r[2])
    total = len(rows)
    model_label = model
    if effort:
        model_label += f" ({effort} effort)"

    lines = [
        f"# Benchmark Analysis: {model_label}, {total} questions",
        f"",
        f"**Result: {correct_count}/{total} correct ({100 * correct_count // total}%)**",
        f"",
        f"---",
        f"",
    ]

    for i, (q_text, slayer_json, is_correct, err, comp_err, model_name) in enumerate(rows, 1):
        status = "CORRECT" if is_correct else "WRONG"
        lines.append(f"## Q{i}: {q_text} -- {status}")
        lines.append("")

        # SLayer query
        try:
            pretty_query = json.dumps(json.loads(slayer_json), indent=2)
        except (json.JSONDecodeError, TypeError):
            pretty_query = slayer_json
        lines.append("**SLayer Query:**")
        lines.append(f"```json\n{pretty_query}\n```")
        lines.append("")

        # Generated SQL
        gen_sql = get_generated_sql(slayer_json)
        lines.append("**Generated SQL:**")
        if gen_sql.startswith("ERROR:"):
            lines.append(f"```\n{gen_sql}\n```")
        else:
            lines.append(f"```sql\n{gen_sql}\n```")
        lines.append("")

        # Gold SQL
        gold_sql = gold.get(q_text, "N/A")
        lines.append("**Gold SQL:**")
        lines.append(f"```sql\n{gold_sql}\n```")
        lines.append("")

        # Error info
        if err:
            lines.append(f"**Error:** {err}")
            lines.append("")
        if comp_err:
            lines.append(f"**Comparison error:** {comp_err}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Clean up SQLAlchemy engines
    for engine in _sync_engines.values():
        engine.dispose()
    _sync_engines.clear()

    Path(output_path).write_text("\n".join(lines))
    print(f"\nAnalysis written to: {output_path}")


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
