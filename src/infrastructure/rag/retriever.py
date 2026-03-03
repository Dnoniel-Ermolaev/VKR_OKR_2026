from __future__ import annotations

from pathlib import Path
from typing import List


class GuidelinesRetriever:
    """
    Lightweight local retriever over .txt files.

    This keeps console-first setup simple; can be replaced with Chroma later.
    """

    def __init__(self, guidelines_dir: Path) -> None:
        self.guidelines_dir = guidelines_dir
        self.guidelines_dir.mkdir(parents=True, exist_ok=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        documents = list(self.guidelines_dir.glob("*.txt"))
        if not documents:
            return ["No local guideline snippets found. Add .txt files to data/guidelines."]

        query_tokens = set(query.lower().split())
        scored: List[tuple[int, str]] = []
        for doc in documents:
            text = doc.read_text(encoding="utf-8", errors="ignore")
            score = sum(1 for token in query_tokens if token in text.lower())
            scored.append((score, text[:1200]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [snippet for _, snippet in scored[:top_k]]
