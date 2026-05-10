
import os
import json
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


load_dotenv()

# ── Logging setup 
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Groq
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

# Qdrant
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

# Langfuse
LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

# Embedding model
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ── Chunking strategy 
CHUNK_SIZE: int =200
CHUNK_OVERLAP: int = 30

# Retrieval

TOP_K_RETRIEVAL: int = 20
RERANK_TOP_K: int = 3

# Paths
ROOT_DIR: Path = Path(__file__).parent.parent.resolve()
DATA_DIR: Path = ROOT_DIR / "data"

# Tenants
TENANTS: list[str] = [
    "surgery_greenfield",
    "surgery_riverside",
]


def load_tenant_config(tenant_id: str) -> dict:
    """
    Load and validate a surgery's JSON config file.

    Args:
        tenant_id: e.g. "surgery_greenfield"

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: if config.json doesn't exist for the tenant.
        json.JSONDecodeError: if the config is malformed JSON.
        ValueError: if required fields are missing.
    """
    config_path = DATA_DIR / tenant_id / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found for tenant '{tenant_id}': {config_path}"
        )

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Validate required fields
    required_fields = [
        "tenant_id", "surgery_name", "emergency_number",
        "urgent_number", "system_prompt_extras", "escalation_message",
    ]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(
            f"Tenant config for '{tenant_id}' is missing fields: {missing}"
        )

    logger.debug("Loaded config for tenant '%s'", tenant_id)
    return config


def get_qdrant_collection_name(tenant_id: str) -> str:
    """
    Returns the isolated Qdrant collection name for a given tenant.

    Convention: 'emma_{tenant_id}' — keeps all EMMA collections grouped
    with a prefix while ensuring surgical isolation between surgeries.
    """
    return f"emma_{tenant_id}"
