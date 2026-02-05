from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag.retriever import RetrievedChunk

DEFAULT_LLM_MODEL = "gpt-4o-mini"


FEW_SHOT_QA = """
PRZYKŁAD:
KONTEKST:
[SOURCE: wykład.pdf | PAGE: 2 | CHUNK: 10]
Korelacja Pearsona r mierzy siłę i kierunek liniowej zależności dwóch zmiennych. r ∈ [-1, 1].

PYTANIE:
Co oznacza r = -0.8?

ODPOWIEDŹ:
r = -0.8 oznacza silną ujemną zależność liniową: gdy jedna zmienna rośnie, druga ma tendencję spadać.
Źródła: wykład.pdf str.2 chunk 10
"""

FEW_SHOT_STUDY = """
PRZYKŁAD:
KONTEKST:
[SOURCE: wykład.pdf | PAGE: 5 | CHUNK: 21]
Regresja liniowa: y = a + bx. Współczynnik b określa zmianę y przy jednostkowej zmianie x.

TEMAT/PYTANIE:
Wyjaśnij współczynnik b w regresji liniowej.

NOTATKA:
- b to nachylenie prostej regresji (slope)
- mówi o tym, o ile średnio zmieni się y, gdy x wzrośnie o 1
- znak b: dodatni = rosnąca zależność, ujemny = malejąca
Pułapki / na co uważać:
- interpretacja b ma sens w zakresie danych (ekstrapolacja bywa myląca)
Źródła: wykład.pdf str.5 chunk 21
"""


@dataclass
class PromptConfig:
    mode: str = "qa"          # "qa" albo "study"
    use_few_shot: bool = True


class PromptBuilder:
    """
    Składa prompt do LLM (QA / Study) + kontekst z retrieved chunków.
    """

    def build_context(self, retrieved: List[RetrievedChunk]) -> str:
        blocks = []
        for r in retrieved:
            m = r.meta
            blocks.append(
                f"[SOURCE: {m['source']} | PAGE: {m['page']} | CHUNK: {m['id']}]\n{m['text'].strip()}\n"
            )
        return "\n---\n".join(blocks)

    def build_prompt(self, question: str, retrieved: List[RetrievedChunk], cfg: PromptConfig) -> str:
        ctx = self.build_context(retrieved)

        few_shot_block = ""
        if cfg.use_few_shot:
            few_shot_block = FEW_SHOT_QA if cfg.mode == "qa" else FEW_SHOT_STUDY

        if cfg.mode == "qa":
            return f"""Odpowiedz WYŁĄCZNIE na podstawie kontekstu. Jeśli brak odpowiedzi w kontekście, powiedz: "Nie mam informacji w PDF-ach."
{few_shot_block}

KONTEKST:
{ctx}

PYTANIE:
{question}

WYMAGANIA:
- krótko i konkretnie po polsku
- na końcu "Źródła:" (plik + strona + chunk)
"""
        else:
            return f"""Zrób notatkę do nauki WYŁĄCZNIE na podstawie kontekstu. Jeśli brak odpowiedzi w kontekście, powiedz: "Nie mam informacji w PDF-ach."
{few_shot_block}

KONTEKST:
{ctx}

TEMAT/PYTANIE:
{question}

FORMAT:
- 5–10 punktów najważniejszych
- definicje/wzory jeśli są
- "Pułapki / na co uważać" (1–3)
- "Źródła:" (plik + strona + chunk)
"""
