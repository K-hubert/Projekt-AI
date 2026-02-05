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

from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder, PromptConfig
from rag.evaluator import Evaluator

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


class RagIndex:
    """
    Index dokumentów (PDF -> tekst -> chunki -> embedding -> FAISS).
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index: Optional[faiss.IndexFlatL2] = None
        self.meta: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

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

        chunks: List[Chunk] = []
        cid = 0

        for p in pages:
            parts = chunk_text(p.text, chunk_size=chunk_size, overlap=overlap)
            for part in parts:
                chunks.append(Chunk(id=cid, source=p.source, page=p.page, text=part))
                cid += 1

        vecs_all = []
        meta_all: List[Dict[str, Any]] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vecs = self._embed([c.text for c in batch])
            vecs_all.append(vecs)

            for c in batch:
                meta_all.append({
                    "id": c.id,
                    "source": c.source,
                    "page": c.page,
                    "text": c.text
                })

        vectors = np.vstack(vecs_all).astype(np.float32)
        dim = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        self.meta = meta_all
        self.vectors = vectors


# =====================
# ANSWER
# =====================

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
    profile: str = "concise",
    prompt_template: str = "Standard",
) -> str:

    retriever = Retriever(rag)

    if retrieval_method == "mmr":
        retrieved = retriever.mmr(
            question,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_=mmr_lambda
        )
    else:
        retrieved = retriever.similarity(question, top_k=top_k)
    
    rag.last_retrieved = retrieved

    prompt_builder = PromptBuilder()

    prompt = prompt_builder.build_prompt(
        question=question,
        retrieved=retrieved,
        cfg=PromptConfig(
            mode=mode,
            use_few_shot=use_few_shot,
            profile=profile,
            template=prompt_template
        )
    )

    resp = rag.client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": "Jesteś asystentem RAG. Nie zmyślaj. Nie używaj wiedzy spoza kontekstu. Trzymaj format."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    out = resp.choices[0].message.content

    # evaluator (na razie tylko do testów)
    Evaluator().evaluate(out, retrieved)

    return out


# =====================
# FLASHCARDS
# =====================

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

Zwróć wynik jako poprawny JSON:
[{{"question":"...","answer":"...","sources":[...]}}]

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
        return json.loads(raw[start:end + 1])


# =====================
# ANALIZA
# =====================

def compute_text_stats(meta: List[Dict[str, Any]]) -> Dict[str, Any]:

    texts = [m.get("text", "") or "" for m in meta]
    sources = set([m.get("source", "") or "" for m in meta])

    total_chars = sum(len(t) for t in texts)

    stop = set("""
    i a o u w z ze na do od po za pod nad oraz lub czy że to jest są był była było
    the a an and or to of in on for with without is are was were be as by from this that
    """.split())

    tokens = []
    words_per_chunk = []

    for t in texts:
        ws = re.findall(r"[A-Za-zÀ-ÿĄąĆćĘęŁłŃńÓóŚśŹźŻż0-9_]+", t.lower())
        ws = [w for w in ws if len(w) >= 3 and w not in stop]
        tokens.extend(ws)
        words_per_chunk.append(len(ws))

    total_words = sum(words_per_chunk)
    top_terms = Counter(tokens).most_common(30)

    return {
        "n_sources": len([s for s in sources if s]),
        "n_chunks": len(meta),
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_words_per_chunk": int(total_words / len(words_per_chunk)) if words_per_chunk else 0,
        "min_words_per_chunk": min(words_per_chunk) if words_per_chunk else 0,
        "max_words_per_chunk": max(words_per_chunk) if words_per_chunk else 0,
        "top_terms": top_terms,
    }
