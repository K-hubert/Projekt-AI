import os
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from rag.pdf_loader import extract_pages_from_pdf_bytes
from rag.chunking import Chunk, chunk_text

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

class RagIndex:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = None
        self.meta: List[Dict[str, Any]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def build_from_uploaded_pdfs(self, files: List[Tuple[str, bytes]], chunk_size=1200, overlap=200):
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

        # embeddings w batchach
        batch_size = 64
        vecs_all = []
        meta_all = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vecs = self._embed([c.text for c in batch])
            vecs_all.append(vecs)
            for c in batch:
                meta_all.append({"id": c.id, "source": c.source, "page": c.page, "text": c.text})

        vectors = np.vstack(vecs_all)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.meta = meta_all

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        qvec = self._embed([query])
        distances, ids = self.index.search(qvec, top_k)
        out = []
        for r, idx in enumerate(ids[0]):
            if idx == -1:
                continue
            m = self.meta[int(idx)]
            out.append((m, float(distances[0][r])))
        return out

def _build_context(retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
    blocks = []
    for m, _dist in retrieved:
        blocks.append(
            f"[SOURCE: {m['source']} | PAGE: {m['page']} | CHUNK: {m['id']}]\n{m['text'].strip()}\n"
        )
    return "\n---\n".join(blocks)

def answer(question: str, rag: RagIndex, mode: str = "qa", top_k: int = 4, llm_model: str = DEFAULT_LLM_MODEL) -> str:
    ctx = _build_context(rag.retrieve(question, top_k=top_k))

    if mode == "qa":
        prompt = f"""Odpowiedz WYŁĄCZNIE na podstawie kontekstu. Jeśli brak odpowiedzi w kontekście, powiedz: "Nie mam informacji w PDF-ach."

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

    client = rag.client
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "Jesteś asystentem RAG. Nie zmyślaj. Nie używaj wiedzy spoza kontekstu."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
