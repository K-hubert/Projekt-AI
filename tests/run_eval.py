from __future__ import annotations

from pathlib import Path
import json
import csv

from rag.evaluator import Evaluator
from rag.rag_core import RagIndex, answer


def main() -> None:
    # Ścieżki względem folderu tests/
    base_dir = Path(__file__).resolve().parent
    questions_path = base_dir / "eval_questions.json"

    # 1) Wczytaj pytania testowe
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    # 2) Zbuduj indeks z PDF-ów z dysku (tak jak w app.py, tylko bez uploadu)
    project_root = base_dir.parent
    pdf_dir = project_root / "data" / "pdfs"  # <- TU mają leżeć PDF-y do testów

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Brak PDF-ów do testów w: {pdf_dir}")

    files = [(p.name, p.read_bytes()) for p in pdf_paths]

    rag = RagIndex()
    rag.build_from_uploaded_pdfs(files)

    evaluator = Evaluator()
    results = []

    # 3) Odpal testy
    methods = ["similarity", "mmr"]

    for method in methods:
        for t in questions:
            resp = answer(
                question=t["question"],
                rag=rag,
                retrieval_method=method,
                top_k=4,
                fetch_k=30,
                mmr_lambda=0.6,
                use_few_shot=True,
            )

            retrieved = getattr(rag, "last_retrieved", [])
            eval_res = evaluator.evaluate(resp, retrieved)

            expected_keywords = t.get("expected_keywords", [])
            keyword_hits = [
                kw for kw in expected_keywords
                if kw.lower() in resp.lower()
            ]

            results.append(
                {
                    "id": t.get("id", "?"),
                    "question": t.get("question", ""),
                    "retrieval_method": method,
                    "has_sources": eval_res.has_sources,
                    "used_context": eval_res.used_context,
                    "keywords_hit": len(keyword_hits),
                    "keywords_total": len(expected_keywords),
                    "notes": eval_res.notes,
                }
            )

    # 4) Raport w konsoli
    print("\n=== WYNIKI EWALUACJI ===")
    for r in results:
        print(
            f"[{r['id']}] ({r['retrieval_method']}) {r['question']}\n"
            f"  - źródła: {'OK' if r['has_sources'] else 'BRAK'}\n"
            f"  - kontekst: {'OK' if r['used_context'] else 'BRAK'}\n"
            f"  - keywords: {r['keywords_hit']}/{r['keywords_total']}\n"
            f"  - notes: {r['notes']}\n"
        )

    # 5) Zapis do CSV
    out_path = base_dir / "eval_report.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nZapisano raport CSV: {out_path}")


if __name__ == "__main__":
    main()
