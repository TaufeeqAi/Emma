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

# ── STT 

STT_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("STT_CONFIDENCE_THRESHOLD", "-0.5")
)

# Kept for backward compatibility — Phase 2 used CONFIDENCE_THRESHOLD
CONFIDENCE_THRESHOLD: float = STT_CONFIDENCE_THRESHOLD

# ── Audio 
# WebRTC VAD requires: 8000, 16000, 32000, or 48000 Hz
AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

# VAD aggressiveness: 0–3. Mode 3 filters most background noise.
VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "3"))

# Minimum audio buffered before we attempt STT (prevents cutting off mid-word)
MIN_SPEECH_DURATION_SECONDS: float = float(
    os.getenv("MIN_SPEECH_DURATION_SECONDS", "0.8")
)

# Silence after speech ends before we process the utterance
END_OF_SPEECH_SILENCE_SECONDS: float = float(
    os.getenv("END_OF_SPEECH_SILENCE_SECONDS", "0.8")
)

# Inactivity timeout before "are you still there?" prompt
MAX_SILENCE_BEFORE_PROMPT_SECONDS: float = float(
    os.getenv("MAX_SILENCE_BEFORE_PROMPT_SECONDS", "8.0")
)

# TTS speaking rate
TTS_SPEAKING_RATE: float = float(os.getenv("TTS_SPEAKING_RATE", "0.92"))

# ── FastAPI 
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000"
).split(",")

# ── Langfuse
LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

# Langfuse is optional — pipeline runs without it (graceful degradation).
# Set to False to disable tracing entirely (e.g. during local unit tests).
LANGFUSE_ENABLED: bool = bool(
    LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
)

# ── Evaluation Thresholds 
# CI pipeline fails if RAGAS scores drop below these values.
# Tuned to reflect baseline on 2 mock surgery documents.
RAGAS_FAITHFULNESS_THRESHOLD: float = float(
    os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.80")
)
RAGAS_ANSWER_RELEVANCY_THRESHOLD: float = float(
    os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD", "0.75")
)
RAGAS_CONTEXT_PRECISION_THRESHOLD: float = float(
    os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD", "0.70")
)
RAGAS_CONTEXT_RECALL_THRESHOLD: float = float(
    os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD", "0.70")
)

# ── Evaluation result output directory 
EVAL_RESULTS_DIR: Path = ROOT_DIR / "eval_results"
EVAL_RESULTS_DIR.mkdir(exist_ok=True)


# ── LiveKit Server 
LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_WS_URL: str = os.getenv("LIVEKIT_WS_URL", LIVEKIT_URL)
LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "secret")

# ── LiveKit SIP 
LIVEKIT_SIP_DOMAIN: str = os.getenv("LIVEKIT_SIP_DOMAIN", "sip.emma-local:5060")
SIP_AUTH_USERNAME: str = os.getenv("SIP_AUTH_USERNAME", "emma")
SIP_AUTH_PASSWORD: str = os.getenv("SIP_AUTH_PASSWORD", "change_me_in_production")

# ── Webhook Security 
# LiveKit signs webhook requests with HMAC-SHA256 over the raw body.
# The shared secret is the API secret (or a dedicated webhook secret).
LIVEKIT_WEBHOOK_SECRET: str = os.getenv("LIVEKIT_WEBHOOK_SECRET", LIVEKIT_API_SECRET)

# ── Agent Worker 
# Unique name for this EMMA agent type (used in dispatch rules targeting).
EMMA_AGENT_NAME: str = os.getenv("EMMA_AGENT_NAME", "emma-voice-agent")

# Max concurrent calls per agent worker process (soft limit; worker will queue beyond this)
EMMA_MAX_CONCURRENT_CALLS: int = int(os.getenv("EMMA_MAX_CONCURRENT_CALLS", "20"))

# ── Tenant Configuration 
TENANTS: list[str] = [
    "surgery_greenfield",
    "surgery_riverside",
]

DEFAULT_TENANT: str = "surgery_greenfield"  

# ── App Environment 
APP_ENV: str = os.getenv("APP_ENV", "development")  

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
