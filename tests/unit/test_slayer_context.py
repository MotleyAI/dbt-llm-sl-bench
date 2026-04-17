"""Tests for SLayer context building from MCP server tools."""

import duckdb
import pytest

from llm_bench.config.strategies import SLayerConfig
from llm_bench.models.requests import QueryRequest
from llm_bench.runners.benchmark import _HELP_TOPICS, _build_slayer_context
from llm_bench.services.query_generation import SLayerQueryStrategy

from slayer.async_utils import run_sync
from slayer.core.enums import DataType
from slayer.core.models import (
    DatasourceConfig,
    Dimension,
    Measure,
    ModelJoin,
    SlayerModel,
)
from slayer.storage.yaml_storage import YAMLStorage


@pytest.fixture
def slayer_storage(tmp_path):
    """Create a YAML storage with a DuckDB datasource and two models."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE orders (id INT, customer_id INT, status VARCHAR, amount DECIMAL, created_at DATE)"
    )
    conn.execute(
        "INSERT INTO orders VALUES (1, 10, 'completed', 100.0, '2025-01-15'), "
        "(2, 20, 'pending', 200.0, '2025-02-20')"
    )
    conn.execute("CREATE TABLE customers (id INT, name VARCHAR, region VARCHAR)")
    conn.execute(
        "INSERT INTO customers VALUES (10, 'Acme', 'US'), (20, 'Globex', 'EU')"
    )
    conn.close()

    storage = YAMLStorage(base_dir=str(tmp_path / "models"))

    run_sync(
        storage.save_datasource(
            DatasourceConfig(name="testdb", type="duckdb", database=str(db_path))
        )
    )

    run_sync(
        storage.save_model(
            SlayerModel(
                name="orders",
                sql_table="orders",
                data_source="testdb",
                description="Order records",
                dimensions=[
                    Dimension(name="id", type=DataType.NUMBER, primary_key=True),
                    Dimension(name="customer_id", type=DataType.NUMBER),
                    Dimension(name="status", type=DataType.STRING),
                    Dimension(name="created_at", type=DataType.DATE),
                ],
                measures=[
                    Measure(name="amount", sql="amount"),
                ],
                joins=[
                    ModelJoin(
                        target_model="customers",
                        join_pairs=[["customer_id", "id"]],
                    )
                ],
            )
        )
    )

    run_sync(
        storage.save_model(
            SlayerModel(
                name="customers",
                sql_table="customers",
                data_source="testdb",
                description="Customer records",
                dimensions=[
                    Dimension(name="id", type=DataType.NUMBER, primary_key=True),
                    Dimension(name="name", type=DataType.STRING),
                    Dimension(name="region", type=DataType.STRING),
                ],
                measures=[],
            )
        )
    )

    # Also save a hidden model that should NOT appear
    run_sync(
        storage.save_model(
            SlayerModel(
                name="internal",
                sql_table="orders",
                data_source="testdb",
                hidden=True,
            )
        )
    )

    return str(tmp_path / "models")


class TestBuildSlayerContext:
    def test_returns_expected_keys(self, slayer_storage):
        context = _build_slayer_context(slayer_storage)
        assert "help" in context
        assert "model_inspections" in context
        assert len(context) == 2

    def test_help_contains_intro_and_topics(self, slayer_storage):
        context = _build_slayer_context(slayer_storage)
        help_text = context["help"]
        # Intro section
        assert "SLayer" in help_text
        assert "semantic layer" in help_text.lower()
        # Each topic should be present
        for topic in _HELP_TOPICS:
            assert topic.capitalize() in help_text or f"# {topic.capitalize()}" in help_text.lower() or topic in help_text.lower()

    def test_model_inspections_contains_all_visible_models(self, slayer_storage):
        context = _build_slayer_context(slayer_storage)
        inspections = context["model_inspections"]
        assert "orders" in inspections
        assert "customers" in inspections
        # Hidden model should not appear
        assert "internal" not in inspections

    def test_model_inspections_contains_schema_details(self, slayer_storage):
        context = _build_slayer_context(slayer_storage)
        inspections = context["model_inspections"]
        # Dimensions and measures from the orders model
        assert "status" in inspections
        assert "amount" in inspections
        assert "customer_id" in inspections
        # Join info
        assert "customers" in inspections

    def test_model_inspections_contains_sample_data(self, slayer_storage):
        context = _build_slayer_context(slayer_storage)
        inspections = context["model_inspections"]
        # Sample data should include values from the test database
        assert "Acme" in inspections or "completed" in inspections


class TestSLayerQueryStrategyContext:
    def test_rejects_missing_context(self):
        config = SLayerConfig(model_name="openai:gpt-5")
        strategy = SLayerQueryStrategy(config)

        result = strategy.generate_query(QueryRequest("test question", context=None))
        assert not result.success
        assert result.error is not None

    def test_rejects_old_context_key(self):
        config = SLayerConfig(model_name="openai:gpt-5")
        strategy = SLayerQueryStrategy(config)

        old_context = {"slayer_model_summaries": "[]"}
        result = strategy.generate_query(
            QueryRequest("test question", context=old_context)
        )
        assert not result.success
        assert result.error is not None

    def test_rejects_partial_context(self):
        config = SLayerConfig(model_name="openai:gpt-5")
        strategy = SLayerQueryStrategy(config)

        partial_context = {"help": "some help"}
        result = strategy.generate_query(
            QueryRequest("test question", context=partial_context)
        )
        assert not result.success
        assert result.error is not None
