"""Strategy-specific configuration classes."""

from dataclasses import dataclass, field

from llm_bench.config.base import BaseConfig
from llm_bench.config.settings import settings


@dataclass
class SemanticLayerConfig(BaseConfig):
    """Configuration for semantic layer strategy"""

    strategy: str = "semantic_layer"


@dataclass
class MCPConfig(BaseConfig):
    """Configuration for MCP strategy"""

    strategy: str = "mcp"


@dataclass
class SQLConfig(BaseConfig):
    """Configuration for SQL strategy"""

    strategy: str = "sql"


@dataclass
class SLayerConfig(BaseConfig):
    """Configuration for SLayer strategy"""

    strategy: str = "slayer"
    slayer_models_dir: str = field(default_factory=lambda: settings.slayer_models_dir)
    slayer_db_path: str = field(default_factory=lambda: settings.slayer_db_path)
