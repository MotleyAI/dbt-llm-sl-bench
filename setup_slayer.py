"""Setup script for SLayer benchmark strategy.

One-time script that:
1. Loads ACME Insurance CSV data into a DuckDB database.
2. Converts dbt semantic models to SLayer models via dbt ingestion. When
   the chosen dbt branch includes regular dbt models (SQL bridge views),
   SLayer inlines their resolved SQL into ``SlayerModel.sql`` directly —
   no DuckDB views are created as part of setup.

Usage:
    python setup_slayer.py [--dbt-project-path PATH] [--db-path PATH]
                           [--models-dir PATH] [--no-bridges]

    --no-bridges: use the `main` branch of the dbt project (25 base
    semantic models, no `policy_holder_policy`/`claim_policy_bridge`/
    `policy_premium_detail` bridges). Default uses the
    `refresh-2025-additional-models` branch which adds those 3 bridge
    models as pure SLayer models (ingested via SLayer's dbt inline-SQL
    support).
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import duckdb

from slayer.async_utils import run_sync
from slayer.core.models import DatasourceConfig
from slayer.dbt.converter import DbtToSlayerConverter
from slayer.dbt.parser import parse_dbt_project
from slayer.storage.yaml_storage import YAMLStorage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default paths relative to this script
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_DBT_PROJECT = _SCRIPT_DIR.parent / "semantic-layer-llm-benchmarking"
_DEFAULT_DB_PATH = _SCRIPT_DIR / "acme.duckdb"
_DEFAULT_MODELS_DIR = _SCRIPT_DIR / "slayer_models"
_CSV_DIR_NAME = "ACME_Insurance/data"
_DEFAULT_DBT_BRANCH = "refresh-2025-additional-models"
_NO_BRIDGES_DBT_BRANCH = "main"


def load_csvs_into_duckdb(csv_dir: Path, db_path: Path) -> None:
    """Load all ACME Insurance CSVs into DuckDB tables.

    Only the raw CSV-backed tables are created. Bridge models — when
    present on the selected dbt branch — come through as SLayer models
    with inline SQL, not as DuckDB views.
    """
    if db_path.exists():
        db_path.unlink()
        logger.info(f"Removed existing database: {db_path}")

    conn = duckdb.connect(str(db_path))
    try:
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        for csv_file in csv_files:
            # Table name = CSV filename without extension (e.g., Claim.csv -> Claim)
            table_name = csv_file.stem
            # Skip non-data files
            if table_name.endswith(".r2rml"):
                continue
            logger.info(f"  Loading {csv_file.name} -> table '{table_name}'")
            conn.execute(
                f"CREATE TABLE \"{table_name}\" AS SELECT * FROM read_csv_auto('{csv_file}', header=true)"
            )

        # Fix columns that DuckDB inferred as VARCHAR due to all-NULL CSV data
        conn.execute("ALTER TABLE FireClaim ALTER Premium TYPE DOUBLE")

        # Verify tables
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"  Created {len(tables)} tables: {[t[0] for t in tables]}")

    finally:
        conn.close()


def _ensure_branch(dbt_project_path: Path, branch: str) -> str:
    """Ensure the dbt project is on the specified branch. Returns the previous branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=dbt_project_path, capture_output=True, text=True, check=True,
    )
    current_branch = result.stdout.strip()
    if current_branch != branch:
        logger.info(f"  Switching dbt project from '{current_branch}' to '{branch}'")
        subprocess.run(
            ["git", "checkout", branch],
            cwd=dbt_project_path, capture_output=True, text=True, check=True,
        )
    return current_branch


def convert_dbt_to_slayer(
    dbt_project_path: Path,
    models_dir: Path,
    db_path: Path,
    dbt_branch: str = _DEFAULT_DBT_BRANCH,
) -> None:
    """Convert dbt semantic models to SLayer models and save to YAML storage."""
    # Clean up existing models directory
    if models_dir.exists():
        shutil.rmtree(models_dir)
        logger.info(f"Removed existing models directory: {models_dir}")

    storage = YAMLStorage(base_dir=str(models_dir))

    # Ensure we're on the right branch (with additional models)
    original_branch = _ensure_branch(dbt_project_path, dbt_branch)

    # Parse dbt project (semantic models + metrics from YAML files)
    semantics_path = dbt_project_path / "models"
    logger.info(f"Parsing dbt project from: {semantics_path} (branch: {dbt_branch})")
    project = parse_dbt_project(str(semantics_path))
    logger.info(
        f"  Found {len(project.semantic_models)} semantic models, "
        f"{len(project.metrics)} metrics"
    )

    # Convert to SLayer models
    datasource_name = "acme_duckdb"
    converter = DbtToSlayerConverter(
        project=project,
        data_source=datasource_name,
        strict_aggregations=True,
    )
    result = converter.convert()
    logger.info(
        f"  Converted to {len(result.models)} SLayer models, "
        f"{len(result.queries)} query definitions"
    )

    # Log warnings
    for warning in result.warnings:
        logger.warning(f"  Conversion: {warning.message}")

    # Save models — YAMLStorage methods are async, bridge with run_sync.
    for model in result.models:
        run_sync(storage.save_model(model))
        logger.info(f"  Saved model: {model.name}")

    # Save datasource config
    ds_config = DatasourceConfig(
        name=datasource_name,
        type="duckdb",
        database=str(db_path.resolve()),
    )
    run_sync(storage.save_datasource(ds_config))
    logger.info(f"  Saved datasource config: {datasource_name}")

    # Restore original branch
    if original_branch != dbt_branch:
        _ensure_branch(dbt_project_path, original_branch)

    # Log summary
    logger.info(f"\nSLayer models saved to: {models_dir}")
    logger.info(f"Models: {run_sync(storage.list_models())}")


def verify_gold_queries(db_path: Path) -> None:
    """Verify that the 11 benchmark gold queries execute against DuckDB."""
    gold_queries = [
        ("Count claims", "SELECT COUNT(*) AS NoOfClaims FROM claim"),
        ("Count policies", "SELECT COUNT(*) AS NoOfPolicies FROM policy"),
    ]
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        for name, sql in gold_queries:
            try:
                result = conn.execute(sql).fetchdf()
                logger.info(f"  {name}: {result.iloc[0, 0]}")
            except Exception as e:
                logger.error(f"  {name}: FAILED - {e}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup SLayer benchmark environment")
    parser.add_argument(
        "--dbt-project-path",
        type=Path,
        default=_DEFAULT_DBT_PROJECT,
        help="Path to dbt semantic-layer-llm-benchmarking project",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=_DEFAULT_DB_PATH,
        help="Path to output DuckDB database file",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_DEFAULT_MODELS_DIR,
        help="Path to output SLayer models directory",
    )
    parser.add_argument(
        "--no-bridges",
        action="store_true",
        help=(
            "Use the 'main' branch of the dbt project (25 base semantic "
            "models, no bridge models). Default uses "
            "'refresh-2025-additional-models' which adds 3 bridge "
            "semantic models whose SQL is inlined by SLayer."
        ),
    )
    args = parser.parse_args()

    dbt_branch = _NO_BRIDGES_DBT_BRANCH if args.no_bridges else _DEFAULT_DBT_BRANCH

    csv_dir = args.dbt_project_path / _CSV_DIR_NAME
    if not csv_dir.exists():
        logger.error(f"CSV data directory not found: {csv_dir}")
        logger.error("Make sure --dbt-project-path points to the semantic-layer-llm-benchmarking repo")
        raise SystemExit(1)

    logger.info("=" * 60)
    logger.info("SLayer Benchmark Setup")
    logger.info(f"  dbt branch: {dbt_branch}")
    logger.info("=" * 60)

    logger.info(f"\n1. Loading CSVs from {csv_dir} into DuckDB...")
    load_csvs_into_duckdb(csv_dir, args.db_path)

    logger.info("\n2. Converting dbt semantic models to SLayer...")
    convert_dbt_to_slayer(args.dbt_project_path, args.models_dir, args.db_path, dbt_branch=dbt_branch)

    logger.info("\n3. Verifying gold queries against DuckDB...")
    verify_gold_queries(args.db_path)

    logger.info("\n" + "=" * 60)
    logger.info("Setup complete!")
    logger.info(f"  DuckDB database: {args.db_path}")
    logger.info(f"  SLayer models:   {args.models_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
