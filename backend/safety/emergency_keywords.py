import logging
import re
from typing import Final, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Medical slang expansion 

_MEDICAL_SYNONYMS: Final[dict[str, str]] = {
    # Cardiac slang
    "ticker": "heart",

    # Respiratory slang
    "windpipe": "throat",
    "gasping": "cannot breathe",

    # Mental health slang
    "off myself": "kill myself",
    "top myself": "kill myself",
    "do myself in": "kill myself",

    # Paediatric / general
    "kiddo": "child",
    "little one": "baby",
    "bub": "baby",
}

# Layer 1 & 1.5: Keyword triggers 
# Mix of single-word and multi-word triggers.
# Multi-word triggers are checked by both exact and gap-tolerant matching.

EMERGENCY_KEYWORDS: Final[List[str]] = [
    #  Cardiac 
    "chest pain",
    "chest tightness",
    "chest pressure",
    "chest feels tight",
    "heart attack",
    "left arm pain",
    "jaw pain",
    "heart racing",
    "heart pounding",
    "severe palpitations",
    "my heart feels like it is going to burst",
    "heart feels like it is going to burst",
    "my ticker feels like it's going to burst",
    "ticker feels like it's going to burst",

    #  Respiratory 
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "shortness of breath",
    "not breathing",
    "stopped breathing",
    "choking on",

    # Stroke (FAST criteria) 
    "having a stroke",
    "think i had a stroke",
    "face drooping",
    "face is drooping",
    "arm weakness",
    "sudden confusion",
    "speech problems",
    "slurred speech",
    "sudden severe headache",
    "can't speak",
    "vision went black",

    # Loss of consciousness 
    "unconscious",
    "passed out",
    "not responding",
    "won't wake up",
    "fainted and",
    "collapsed and",
    "unresponsive after",

    # Severe bleeding / trauma 
    "severe bleeding",
    "bleeding heavily",
    "bleeding severely",
    "blood won't stop",
    "lot of blood",

    # Overdose / poisoning 
    "overdose",
    "taken too many tablets",
    "taken too many pills",
    "swallowed something",
    "swallowed the entire bottle",
    "poisoning",
    "ingested something",

    # Mental health crisis 
    "suicidal",
    "want to die",
    "end my life",
    "ending my life",
    "kill myself",
    "self harm",
    "hurt myself",
    "harm myself",
    "going to hurt myself",

    # ── Paediatric emergencies ────────────────────────────────────────────────
    "my baby is not breathing",
    "child collapsed",
    "baby has a seizure",
    "child won't wake",
]

# ── Layer 2: Semantic reference sentences 


_EMERGENCY_REFERENCE_SENTENCES: Final[List[str]] = [
    "I am having a heart attack and need help immediately",
    "My heart feels like it is going to burst",
    "My ticker feels like it is going to burst",
    "I have a crushing pain in my chest",
    "I feel like I am dying",
    "I cannot breathe and I need emergency assistance",
    "Someone is having a stroke right now",
    "My vision went black and I fell down",
    "There is a life-threatening medical emergency happening",
    "I want to hurt myself or end my life",
    "I have taken an overdose of medication",
    "I swallowed too many pills",
    "Someone has collapsed and is not responding",
    "Someone fell and is unresponsive",
    "A child is not breathing and needs help",
    "There is severe uncontrollable bleeding",
    "I am bleeding heavily and cannot stop it",
    "I am losing consciousness and feel very unwell",
]


# ── Text normalization helpers ───────────────────────────────────────────────

def _expand_slang(text: str) -> str:
    """
    Expand known medical slang to standard clinical vocabulary.

    Uses whole-word replacement to avoid partial-word corruption.
    """
    text_lower = text.lower()
    for slang, standard in _MEDICAL_SYNONYMS.items():
        pattern = r"\b" + re.escape(slang) + r"\b"
        text_lower = re.sub(pattern, standard, text_lower)
    return text_lower


def _normalize(text: str) -> str:
    """
    Lowercase, expand slang, strip punctuation, collapse whitespace.
    """
    text = _expand_slang(text)
    text = text.lower().replace("'", "")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Normalize then split to tokens."""
    return _normalize(text).split()


# Layer 1.5: Gap-tolerant matching 

def _ordered_gap_match(query_tokens: List[str], keyword: str, max_gap: int = 5) -> bool:
    """
    Layer 1.5: Check whether keyword tokens appear in order within query_tokens,
    allowing up to `max_gap` tokens between consecutive matched keyword tokens.

    Uses exact token equality to avoid partial-word false positives.
    """
    kw_tokens = _tokenize(keyword)
    if len(kw_tokens) < 2:
        return False

    for start in range(len(query_tokens)):
        if query_tokens[start] != kw_tokens[0]:
            continue

        pos = start + 1
        matched_all = True

        for kw_tok in kw_tokens[1:]:
            found = False
            end = min(pos + max_gap + 1, len(query_tokens))
            for j in range(pos, end):
                if query_tokens[j] == kw_tok:
                    found = True
                    pos = j + 1
                    break
            if not found:
                matched_all = False
                break

        if matched_all:
            return True

    return False


def _check_keyword_layers(query_lower: str, query_tokens: List[str]) -> Tuple[bool, str]:
    """
    Shared helper: runs Layer 1 (normalized exact phrase match) and Layer 1.5
    (gap-tolerant ordered token match).
    """
    # Layer 1: Exact normalized phrase match
    for keyword in EMERGENCY_KEYWORDS:
        if _normalize(keyword) in query_lower:
            return True, f"Emergency keyword detected: '{keyword}'"

    # Layer 1.5: Ordered gap-tolerant match
    for keyword in EMERGENCY_KEYWORDS:
        if _ordered_gap_match(query_tokens, keyword, max_gap=5):
            return True, f"Emergency keyword detected (gap-tolerant): '{keyword}'"

    return False, ""


# EmergencyDetector class 

class EmergencyDetector:
    """
    Three-layer emergency detector with optional shared embedding model.

    Pass `embedder` to reuse an existing SentenceTransformer instance.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.72,
        model_name: str = "all-MiniLM-L6-v2",
        embedder: Optional[SentenceTransformer] = None,
    ) -> None:
        self.threshold = semantic_threshold
        self._owns_embedder = embedder is None

        if embedder is not None:
            logger.info(
                "Loading EmergencyDetector | shared embedder | threshold=%.2f",
                semantic_threshold,
            )
            self._embedder = embedder
        else:
            logger.info(
                "Loading EmergencyDetector | model=%s threshold=%.2f",
                model_name,
                semantic_threshold,
            )
            self._embedder = SentenceTransformer(model_name)

        self.dimension = self._embedder.get_embedding_dimension()

        # Pre-compute reference embeddings at startup — not at detection time.
        self._reference_embeddings: np.ndarray = self._embedder.encode(
            _EMERGENCY_REFERENCE_SENTENCES,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info(
            "EmergencyDetector ready | %d keywords | %d semantic references | dim=%d | owns_embedder=%s",
            len(EMERGENCY_KEYWORDS),
            len(_EMERGENCY_REFERENCE_SENTENCES),
            self.dimension,
            self._owns_embedder,
        )

    def is_emergency(self, query: str) -> Tuple[bool, str]:
        """
        Public entrypoint. Never raises — fails closed.
        """
        if not query or not query.strip():
            logger.warning("EmergencyDetector received empty query — failing closed.")
            return True, "Empty query — failing closed for safety."

        try:
            return self._detect(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception("EmergencyDetector error for query '%s': %s", query, exc)
            return True, f"Detection error — failing closed: {exc}"

    def _detect(self, query: str) -> Tuple[bool, str]:
        query_lower = _normalize(query)
        query_tokens = query_lower.split()

        # Layers 1 & 1.5
        matched, reason = _check_keyword_layers(query_lower, query_tokens)
        if matched:
            logger.warning("EMERGENCY — %s | query='%.80s'", reason, query)
            return True, reason

        # Layer 2: Semantic similarity
        query_embedding: np.ndarray = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities: np.ndarray = cosine_similarity(
            query_embedding, self._reference_embeddings
        )[0]
        max_score: float = float(similarities.max())
        max_idx: int = int(similarities.argmax())

        logger.debug(
            "Semantic similarity | max=%.3f idx=%d threshold=%.2f | query='%.80s'",
            max_score,
            max_idx,
            self.threshold,
            query,
        )

        if max_score >= self.threshold:
            reference = _EMERGENCY_REFERENCE_SENTENCES[max_idx]
            reason = (
                f"Semantic emergency match (score={max_score:.3f}): "
                f"similar to '{reference}'"
            )
            logger.warning("EMERGENCY — Layer 2 | %s | query='%.80s'", reason, query)
            return True, reason

        return False, ""

    def batch_check(self, queries: List[str]) -> List[Tuple[bool, str]]:
        """
        Check multiple queries at once.
        Layers 1 & 1.5 are sequential per query; Layer 2 is batched.
        """
        results: List[Tuple[bool, str]] = []
        layer2_indices: List[int] = []

        for i, query in enumerate(queries):
            query_lower = _normalize(query)
            query_tokens = query_lower.split()

            matched, reason = _check_keyword_layers(query_lower, query_tokens)
            if matched:
                results.append((True, reason))
            else:
                results.append((False, ""))  # placeholder
                layer2_indices.append(i)

        if not layer2_indices:
            return results

        layer2_queries = [queries[i] for i in layer2_indices]
        embeddings = self._embedder.encode(
            layer2_queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = cosine_similarity(embeddings, self._reference_embeddings)

        for j, i in enumerate(layer2_indices):
            max_score = float(sims[j].max())
            max_idx = int(sims[j].argmax())
            if max_score >= self.threshold:
                ref = _EMERGENCY_REFERENCE_SENTENCES[max_idx]
                results[i] = (True, f"Semantic (score={max_score:.3f}): '{ref}'")

        return results