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
    """
    Minimalny evaluator pod zaliczenie:
    - sprawdza czy odpowiedź ma sekcję Źródła
    - sprawdza czy odpowiedź nie jest pusta
    - (opcjonalnie) można rozszerzyć o LLM-as-judge / faithfulness
    """

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
