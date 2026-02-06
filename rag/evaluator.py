from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rag.retriever import RetrievedChunk


@dataclass
class EvalResult:
    has_sources: bool
    used_context: bool
    notes: str = ""


class Evaluator:

    def evaluate(self, answer_text: str, retrieved: List[RetrievedChunk]) -> EvalResult:
        text = (answer_text or "").strip()

        has_sources = "Źródła" in text or "Zrodla" in text or "Sources" in text
        used_context = len(retrieved) > 0 and len(text) > 0

        notes = ""
        if not text:
            notes = "Pusta odpowiedź."
        elif not has_sources:
            notes = "Brak sekcji źródeł w odpowiedzi."

        return EvalResult(has_sources=has_sources, used_context=used_context, notes=notes)
    
    def evaluate_with_keywords(
        self,
        answer_text: str,
        retrieved: List[RetrievedChunk],
        expected_keywords: List[str],
    ) -> EvalResult:
        base = self.evaluate(answer_text, retrieved)

        text = (answer_text or "").lower()
        hits = [kw for kw in expected_keywords if (kw or "").lower() in text]

        if expected_keywords:
            base.notes = (base.notes + " | " if base.notes else "") + f"keywords: {len(hits)}/{len(expected_keywords)}"

        return base
