import json
from typing import List

from rag.evaluator import Evaluator, EvalCaseResult
from rag.rag_core import answer, RagIndex


def run_eval_set(
    rag: RagIndex,
    eval_path: str,
    llm_model: str,
    retrieval_method: str,
    profile: str,
) -> List[EvalCaseResult]:

    with open(eval_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    evaluator = Evaluator()
    results: List[EvalCaseResult] = []

    for c in cases:
        resp = answer(
            question=c["question"],
            rag=rag,
            retrieval_method=retrieval_method,
            llm_model=llm_model,
            profile=profile,
        )

        # retrieved masz w rag.last_retrieved (albo dodasz)
        retrieved = rag.last_retrieved

        r = evaluator.evaluate_with_keywords(
            answer_text=resp,
            retrieved=retrieved,
            expected_keywords=c.get("expected_keywords", []),
        )
        r.question_id = c["id"]
        results.append(r)

    return results
