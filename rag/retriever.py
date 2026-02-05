from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.rag_core import RagIndex


@dataclass
class RetrievedChunk:
    meta: Dict[str, Any]
    score: float  


class Retriever:

    def __init__(self, rag_index: "RagIndex"):
        self.rag = rag_index

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        eps = 1e-8
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + eps)

    def similarity(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.rag.index is None or not self.rag.meta:
            raise RuntimeError("Index not built.")

        qvec = self.rag._embed([query])
        distances, ids = self.rag.index.search(qvec, top_k)

        out: List[RetrievedChunk] = []
        for rank, idx in enumerate(ids[0]):
            if idx == -1:
                continue
            out.append(RetrievedChunk(meta=self.rag.meta[int(idx)], score=float(distances[0][rank])))
        return out

    def mmr(
        self,
        query: str,
        top_k: int = 4,
        fetch_k: int = 30,
        lambda_: float = 0.6,
    ) -> List[RetrievedChunk]:
        
        if self.rag.index is None or self.rag.vectors is None or not self.rag.meta:
            return self.similarity(query, top_k=top_k)

        qvec = self.rag._embed([query])
        distances, ids = self.rag.index.search(qvec, fetch_k)
        cand_ids = [int(i) for i in ids[0] if i != -1]
        if not cand_ids:
            return []

        cand_vecs = self.rag.vectors[cand_ids]
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

        out: List[RetrievedChunk] = []
        for local_idx in selected_local:
            global_id = cand_ids[local_idx]
            out.append(RetrievedChunk(meta=self.rag.meta[global_id], score=dist_map.get(global_id, 0.0)))
        return out
