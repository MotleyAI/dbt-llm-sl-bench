"""Database service for SLayer strategy — handles both SLayer JSON queries and raw SQL."""

import json
from typing import Any

import duckdb
import pandas as pd
from loguru import logger

from llm_bench.models.results import DatabaseExecutionResult


class SLayerDatabaseService:
    """Service for executing queries via SLayer (JSON) or raw SQL (gold queries).

    Routes based on input format:
    - JSON (starts with '{') → parsed as SlayerQuery dict, executed via SlayerClient
    - SQL → executed directly against DuckDB
    """

    def __init__(self, slayer_models_dir: str, slayer_db_path: str) -> None:
        self.slayer_models_dir = slayer_models_dir
        self.slayer_db_path = slayer_db_path
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-init SLayer client in local mode."""
        if self._client is None:
            from slayer.client.slayer_client import SlayerClient
            from slayer.storage.yaml_storage import YAMLStorage

            storage = YAMLStorage(base_dir=self.slayer_models_dir)
            self._client = SlayerClient(storage=storage)
        return self._client

    def execute_query(
        self,
        query: str,
        db_opts: dict[str, Any] | None = None,
        print_result: bool = False,
    ) -> DatabaseExecutionResult:
        """Execute a query — auto-detects SLayer JSON vs raw SQL."""
        query_stripped = query.strip()
        if query_stripped.startswith("{"):
            return self._execute_slayer_query(query_stripped, print_result)
        return self._execute_sql(query_stripped, print_result)

    def _execute_slayer_query(self, query_json: str, print_result: bool = False) -> DatabaseExecutionResult:
        """Execute a SLayer query from JSON."""
        try:
            query_dict = json.loads(query_json)
            client = self._get_client()
            response = client.query_sync(query_dict)
            df = pd.DataFrame(response.data)
            if print_result:
                logger.info(f"SLayer result:\n{df}")
            return DatabaseExecutionResult.success_result(df)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid SLayer query JSON: {e}"
            logger.error(error_msg)
            return DatabaseExecutionResult.error_result(error_msg)
        except Exception as e:
            error_msg = f"SLayer query error: {e}"
            logger.error(error_msg)
            return DatabaseExecutionResult.error_result(error_msg)
        finally:
            # Release DuckDB file lock so subsequent gold queries via raw
            # duckdb.connect() don't fail with "different configuration".
            from slayer.sql.client import _sync_engines

            for engine in _sync_engines.values():
                engine.dispose()
            _sync_engines.clear()

    def _execute_sql(self, sql: str, print_result: bool = False) -> DatabaseExecutionResult:
        """Execute raw SQL against DuckDB (for gold queries)."""
        try:
            conn = duckdb.connect(self.slayer_db_path)
            try:
                df = conn.execute(sql).fetchdf()
                if print_result:
                    logger.info(f"SQL result:\n{df}")
                return DatabaseExecutionResult.success_result(df)
            finally:
                conn.close()
        except Exception as e:
            error_msg = f"DuckDB query error: {e}"
            logger.error(error_msg)
            return DatabaseExecutionResult.error_result(error_msg)
