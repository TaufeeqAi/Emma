import logging
import re
from typing import List, Optional, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Hybrid section-aware chunker for GP surgery guidelines.

    Strategy:
      1. Detect ALL CAPS section headers (hard boundaries).
      2. Each section becomes its own chunk — topic-pure.
      3. If a section exceeds chunk_size, split it recursively with overlap.
      4. Attach section_name metadata to every chunk for traceability.

    Why this beats pure RecursiveCharacterTextSplitter:
      - Prevents topic mixing (e.g., OPENING HOURS + APPOINTMENT BOOKING).
      - Preserves clinical protocol integrity within each chunk.
      - Cross-encoder sees coherent, single-topic chunks → higher relevance scores.
    """

    # Pattern: ALL CAPS headers, ≥3 chars, spaces allowed, no lowercase, no digits
    _HEADER_PATTERN = re.compile(
        r'^[A-Z][A-Z\s/&\-]{2,}(?:\n|$)',
        re.MULTILINE
    )

    # Internal splitter for oversized sections
    _FALLBACK_SEPARATORS = ["\n", ". ", ", ", " "]

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 30,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Fallback splitter for oversized sections
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._FALLBACK_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            keep_separator=False,
        )

        logger.info(
            "SemanticChunker (hybrid) initialised | chunk_size=%d overlap=%d",
            chunk_size, chunk_overlap,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[dict]:
        """
        Chunk text using section-first, recursive-fallback strategy.

        Args:
            text:     Raw document text.
            metadata: Base metadata attached to every chunk (e.g., tenant_id).

        Returns:
            List of chunk dicts with keys: text, chunk_index, char_start,
            metadata (including section_name).
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty document text.")

        base_meta = metadata or {}

        # Step 1: Split into sections by headers
        sections = self._split_into_sections(text)
        logger.info(
            "Detected %d sections in document",
            len(sections),
        )

        # Step 2: Chunk each section
        all_chunks: List[dict] = []
        for sec_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(section, sec_idx, base_meta)
            all_chunks.extend(section_chunks)

        # Step 3: Re-index globally
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = i

        logger.info(
            "Produced %d final chunks from %d sections",
            len(all_chunks), len(sections),
        )
        return all_chunks

    def chunk_stats(self, chunks: List[dict]) -> dict:
        lengths = [len(c["text"]) for c in chunks]
        if not lengths:
            return {}
        return {
            "count": len(lengths),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
        }

    # ── Private: Section Detection ────────────────────────────────────────────

    def _split_into_sections(self, text: str) -> List[Dict]:
        """
        Split document into sections using header detection.

        Returns:
            List of dicts: {"header": str, "body": str, "start": int}
        """
        matches = list(self._HEADER_PATTERN.finditer(text))

        if not matches:
            # No headers found — treat entire text as one section
            return [{"header": "", "body": text.strip(), "start": 0}]

        sections: List[Dict] = []

        # Handle preamble before first header
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append({
                    "header": "PREAMBLE",
                    "body": preamble,
                    "start": 0,
                })

        # Extract each section
        for i, match in enumerate(matches):
            header = match.group().strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            if body:  # Skip empty sections
                sections.append({
                    "header": header,
                    "body": body,
                    "start": match.start(),
                })

        return sections

    # ── Private: Section Chunking ───────────────────────────────────────────

    def _chunk_section(
        self,
        section: Dict,
        sec_idx: int,
        base_meta: dict,
    ) -> List[dict]:
        """
        Chunk a single section. Keep it whole if it fits, else split recursively.
        """
        header = section["header"]
        body = section["body"]
        start_pos = section["start"]

        # Reconstruct full section text (header + body)
        full_text = f"{header}\n{body}".strip() if header else body

        section_meta = {
            **base_meta,
            "section_name": header or "NO_HEADER",
        }

        # Case 1: Section fits in one chunk — keep it whole
        if len(full_text) <= self.chunk_size:
            return [{
                "text": full_text,
                "chunk_index": sec_idx,  # Will be overwritten globally
                "char_start": start_pos,
                "metadata": section_meta,
            }]

        # Case 2: Section too big — split with fallback splitter
        logger.info(
            "Section '%s' oversized (%d chars > %d). Splitting recursively.",
            header, len(full_text), self.chunk_size,
        )

        sub_texts = self._fallback_splitter.split_text(full_text)

        return [
            {
                "text": t.strip(),
                "chunk_index": f"{sec_idx}-{j}",  # Will be overwritten globally
                "char_start": start_pos,  # Approximate
                "metadata": {
                    **section_meta,
                    "section_part": j,
                    "is_sub_chunk": True,
                },
            }
            for j, t in enumerate(sub_texts)
            if t.strip()
        ]