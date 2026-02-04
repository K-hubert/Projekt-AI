import os
import re
import json
import random
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from rag.pdf_loader import extract_pages_from_pdf_bytes
from rag.chunking import Chunk, chunk_text

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


# =========================
# Few-shot examples
# =========================
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


class RagIndex:
    """
    Index dokumentów (PDF -> tekst -> chunki -> embedding -> FAISS).
    Przechowuje:
    - self.index: FAISS index
    - self.meta:  lista metadanych chunków (id/source/page/text)
    - self.vectors: embeddingi chunków (n_chunks, dim) do MMR/cosine
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index: Optional[faiss.IndexFlatL2] = None
        self.meta: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None  # (n, d) float32

    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def build_from_uploaded_pdfs(
        self,
        files: List[Tuple[str, bytes]],
        chunk_size: int = 1200,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> None:
        pages = []
        for filename, b in files:
            pages.extend(extract_pages_from_pdf_bytes(b, filename))

        if not pages:
            raise RuntimeError("Nie udało się wyciągnąć tekstu z PDF. Jeśli to skan, potrzebujesz OCR.")

        chunks: List[Chunk] = []
        cid = 0
        for p in pages:
            parts = chunk_text(p.text, chunk_size=chunk_size, overlap=overlap)
            for part in parts:
                chunks.append(Chunk(id=cid, source=p.source, page=p.page, text=part))
                cid += 1

        if not chunks:
            raise RuntimeError("Brak chunków po chunkingu — PDF może być pusty albo nietekstowy.")

        vecs_all = []
        meta_all: List[Dict[str, Any]] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = self._embed([c.text for c in batch])
            vecs_all.append(vecs)

            for c in batch:
                meta_all.append({"id": c.id, "source": c.source, "page": c.page, "text": c.text})

        vectors = np.vstack(vecs_all).astype(np.float32)
        dim = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        self.meta = meta_all
        self.vectors = vectors

    # -------- Retrieval: similarity --------
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None or not self.meta:
            raise RuntimeError("Index not built.")

        qvec = self._embed([query])
        distances, ids = self.index.search(qvec, top_k)

        out = []
        for rank, idx in enumerate(ids[0]):
            if idx == -1:
                continue
            out.append((self.meta[int(idx)], float(distances[0][rank])))
        return out

    # -------- Retrieval: MMR --------
    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        eps = 1e-8
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + eps)

    def retrieve_mmr(
        self,
        query: str,
        top_k: int = 4,
        fetch_k: int = 30,
        lambda_: float = 0.6,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None or self.vectors is None or not self.meta:
            return self.retrieve(query, top_k=top_k)

        qvec = self._embed([query])
        distances, ids = self.index.search(qvec, fetch_k)
        cand_ids = [int(i) for i in ids[0] if i != -1]

        if not cand_ids:
            return []

        cand_vecs = self.vectors[cand_ids]
        cand_vecs_n = self._normalize(cand_vecs)
        q_n = self._normalize(qvec)[0:1]

        sim_to_query = (cand_vecs_n @ q_n.T).reshape(-1)

        selected_local: List[int] = []

        while len(selected_local) < min(top_k, len(cand_ids)):
            if not selected_local:
                selected_local.append(int(sim_to_query.argmax()))
                continue

            sel_vecs = cand_vecs_n[selected_local]
            sim_to_sel = cand_vecs_n @ sel_vecs.T
            max_sim_sel = sim_to_sel.max(axis=1)

            mmr_score = lambda_ * sim_to_query - (1 - lambda_) * max_sim_sel

            for idx in selected_local:
                mmr_score[idx] = -1e9

            selected_local.append(int(mmr_score.argmax()))

        dist_map = {int(ids[0][i]): float(distances[0][i]) for i in range(len(ids[0])) if int(ids[0][i]) != -1}

        out = []
        for local_idx in selected_local:
            global_id = cand_ids[local_idx]
            out.append((self.meta[global_id], dist_map.get(global_id, 0.0)))

        return out


def _build_context(retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
    blocks = []
    for m, _dist in retrieved:
        blocks.append(
            f"[SOURCE: {m['source']} | PAGE: {m['page']} | CHUNK: {m['id']}]\n{m['text'].strip()}\n"
        )
    return "\n---\n".join(blocks)


def answer(
    question: str,
    rag: RagIndex,
    mode: str = "qa",
    top_k: int = 4,
    llm_model: str = DEFAULT_LLM_MODEL,
    retrieval_method: str = "similarity",
    mmr_lambda: float = 0.6,
    fetch_k: int = 30,
    use_few_shot: bool = True,
) -> str:
    if retrieval_method == "mmr":
        retrieved = rag.retrieve_mmr(question, top_k=top_k, fetch_k=fetch_k, lambda_=mmr_lambda)
    else:
        retrieved = rag.retrieve(question, top_k=top_k)

    ctx = _build_context(retrieved)

    few_shot_block = ""
    if use_few_shot:
        few_shot_block = FEW_SHOT_QA if mode == "qa" else FEW_SHOT_STUDY

    if mode == "qa":
        prompt = f"""Odpowiedz WYŁĄCZNIE na podstawie kontekstu. Jeśli brak odpowiedzi w kontekście, powiedz: "Nie mam informacji w PDF-ach."
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
        prompt = f"""Zrób notatkę do nauki WYŁĄCZNIE na podstawie kontekstu. Jeśli brak odpowiedzi w kontekście, powiedz: "Nie mam informacji w PDF-ach."
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

    resp = rag.client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "Jesteś asystentem RAG. Nie zmyślaj. Nie używaj wiedzy spoza kontekstu. Trzymaj format."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def generate_flashcards(
    rag: RagIndex,
    n_cards: int = 20,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> List[Dict[str, Any]]:
    if rag.index is None or not rag.meta:
        raise RuntimeError("Index not ready.")

    sample_size = min(max(n_cards * 3, 20), len(rag.meta))
    sampled = random.sample(rag.meta, sample_size)

    context_blocks = []
    for m in sampled:
        context_blocks.append(
            f"[SOURCE: {m['source']} | PAGE: {m['page']} | CHUNK: {m['id']}]\n{m['text'].strip()}\n"
        )
    context = "\n---\n".join(context_blocks)

    prompt = f"""Na podstawie kontekstu wygeneruj {n_cards} fiszek do nauki na egzamin.
Fiszka = pytanie + krótka odpowiedź, wyłącznie z kontekstu.
Unikaj ogólników. Preferuj definicje, porównania, kroki, interpretacje, typowe pułapki.

Zwróć wynik jako poprawny JSON (bez dodatkowego tekstu), lista obiektów:
[
  {{
    "question": "...",
    "answer": "...",
    "sources": ["plik | strona:X | chunk:Y", ...]
  }}
]

KONTEKST:
{context}
"""

    resp = rag.client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "Jesteś asystentem RAG. Nie zmyślaj. JSON ma być poprawny."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    raw = resp.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def compute_text_stats(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [m.get("text", "") or "" for m in meta]
    sources = set([m.get("source", "") or "" for m in meta])

    total_chars = sum(len(t) for t in texts)

    stop = set("""
    i a o u w z ze na do od po za pod nad oraz lub czy że to jest są był była było
    the a an and or to of in on for with without is are was were be as by from this that
    """.split())

    tokens: List[str] = []
    words_per_chunk: List[int] = []

    for t in texts:
        ws = re.findall(r"[A-Za-zÀ-ÿĄąĆćĘęŁłŃńÓóŚśŹźŻż0-9_]+", t.lower())
        ws = [w for w in ws if len(w) >= 3 and w not in stop]
        tokens.extend(ws)
        words_per_chunk.append(len(ws))

    total_words = sum(words_per_chunk) if words_per_chunk else 0
    top_terms = Counter(tokens).most_common(30)

    n_sources = len([s for s in sources if s])

    return {
        "n_sources": n_sources,
        "n_chunks": len(meta),
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_words_per_chunk": int(total_words / len(words_per_chunk)) if words_per_chunk else 0,
        "min_words_per_chunk": min(words_per_chunk) if words_per_chunk else 0,
        "max_words_per_chunk": max(words_per_chunk) if words_per_chunk else 0,
        "top_terms": top_terms,
    }
