"""Unit tests for setup_slayer.py configuration and CSV loading.

Doesn't touch the external semantic-layer-llm-benchmarking repo — argparse
parsing happens entirely in-process, and the DuckDB CSV loader is exercised
against a tiny tmp_path fixture. The branch-switching logic (`_ensure_branch`)
is NOT covered here because it mutates a sibling git repo; regenerating
slayer_models/ with each of the two flag values is the integration test for
that path.
"""

import sys
from pathlib import Path

import duckdb
import pytest

# setup_slayer.py is at the repo root, not on the import path by default.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import setup_slayer  # noqa: E402


class TestNoBridgesFlag:
    def test_default_selects_refresh_branch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Without --no-bridges the `refresh-2025-additional-models` branch is picked."""
        captured: dict[str, str] = {}

        def fake_convert(dbt_project_path, models_dir, db_path, dbt_branch):
            captured["dbt_branch"] = dbt_branch

        def fake_load(csv_dir, db_path):
            return None

        def fake_verify(db_path):
            return None

        # Point --dbt-project-path at tmp_path so the CSV-existence check passes.
        csv_dir = tmp_path / "ACME_Insurance" / "data"
        csv_dir.mkdir(parents=True)
        (csv_dir / "Claim.csv").write_text("Claim_Identifier\n1\n")

        monkeypatch.setattr(setup_slayer, "convert_dbt_to_slayer", fake_convert)
        monkeypatch.setattr(setup_slayer, "load_csvs_into_duckdb", fake_load)
        monkeypatch.setattr(setup_slayer, "verify_gold_queries", fake_verify)
        monkeypatch.setattr(
            sys, "argv",
            ["setup_slayer.py", "--dbt-project-path", str(tmp_path), "--db-path", str(tmp_path / "x.duckdb")],
        )

        setup_slayer.main()

        assert captured["dbt_branch"] == "refresh-2025-additional-models"

    def test_no_bridges_selects_main_branch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured: dict[str, str] = {}

        def fake_convert(dbt_project_path, models_dir, db_path, dbt_branch):
            captured["dbt_branch"] = dbt_branch

        csv_dir = tmp_path / "ACME_Insurance" / "data"
        csv_dir.mkdir(parents=True)
        (csv_dir / "Claim.csv").write_text("Claim_Identifier\n1\n")

        monkeypatch.setattr(setup_slayer, "convert_dbt_to_slayer", fake_convert)
        monkeypatch.setattr(setup_slayer, "load_csvs_into_duckdb", lambda *a, **k: None)
        monkeypatch.setattr(setup_slayer, "verify_gold_queries", lambda *a, **k: None)
        monkeypatch.setattr(
            sys, "argv",
            [
                "setup_slayer.py",
                "--dbt-project-path", str(tmp_path),
                "--db-path", str(tmp_path / "x.duckdb"),
                "--no-bridges",
            ],
        )

        setup_slayer.main()

        assert captured["dbt_branch"] == "main"


class TestLoadCsvsIntoDuckdbProducesNoViews:
    """The inline-SQL feature in SLayer replaced DuckDB views for bridges.
    Regardless of branch, setup must only produce plain tables — no views.
    """

    def test_only_base_tables_created(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csvs"
        csv_dir.mkdir()
        (csv_dir / "Alpha.csv").write_text("id,value\n1,10\n2,20\n")
        (csv_dir / "Beta.csv").write_text("id,note\n1,a\n2,b\n")

        db_path = tmp_path / "out.duckdb"
        setup_slayer.load_csvs_into_duckdb(csv_dir, db_path)

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
            views = conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        finally:
            conn.close()

        assert tables == {"Alpha", "Beta"}
        assert views == []

    def test_raises_when_csv_dir_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        db_path = tmp_path / "out.duckdb"
        with pytest.raises(FileNotFoundError):
            setup_slayer.load_csvs_into_duckdb(empty, db_path)
