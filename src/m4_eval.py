"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    import pandas as pd

    # Create LLM and embeddings objects
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # 2. Run evaluation with LLM and embeddings
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    # 3. Convert to EvalResult list
    df = result.to_pandas()
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row.get("user_input", row.get("question", "")),
            answer=row.get("response", row.get("answer", "")),
            contexts=row.get("retrieved_contexts", row.get("contexts", [])),
            ground_truth=row.get("reference", row.get("ground_truth", "")),
            faithfulness=float(row.get("faithfulness", 0.0)),
            answer_relevancy=float(row.get("answer_relevancy", 0.0)),
            context_precision=float(row.get("context_precision", 0.0)),
            context_recall=float(row.get("context_recall", 0.0))
        ))

    # 4. Return results dictionary
    # Ragas result indexing returns a list of scores per row. We need the mean.
    def get_metric_mean(res, metric_name):
        scores = res[metric_name]
        if isinstance(scores, (list, pd.Series)):
            # Convert to series to use mean() which handles NaN better or just sum/len
            s = pd.Series(scores).fillna(0.0)
            return float(s.mean())
        return float(scores) if pd.notnull(scores) else 0.0

    return {
        "faithfulness": get_metric_mean(result, "faithfulness"),
        "answer_relevancy": get_metric_mean(result, "answer_relevancy"),
        "context_precision": get_metric_mean(result, "context_precision"),
        "context_recall": get_metric_mean(result, "context_recall"),
        "per_question": per_question
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    processed = []
    for res in eval_results:
        # Calculate avg score for sorting
        scores = [res.faithfulness, res.answer_relevancy, res.context_precision, res.context_recall]
        avg_score = sum(scores) / 4.0
        
        # Default values
        diagnosis = "Passing thresholds"
        fix = "None needed"
        worst_metric = ""
        score = 0.0
        
        # Map to diagnosis using Diagnostic Tree thresholds
        if res.faithfulness < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature"
            worst_metric = "faithfulness"
            score = res.faithfulness
        elif res.context_recall < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking or add BM25"
            worst_metric = "context_recall"
            score = res.context_recall
        elif res.context_precision < 0.75:
            diagnosis = "Too many irrelevant chunks"
            fix = "Add reranking or metadata filter"
            worst_metric = "context_precision"
            score = res.context_precision
        elif res.answer_relevancy < 0.80:
            diagnosis = "Answer doesn't match question"
            fix = "Improve prompt template"
            worst_metric = "answer_relevancy"
            score = res.answer_relevancy
        else:
            # If all metrics pass thresholds, still identify the lowest one for reporting
            metrics = {
                "faithfulness": res.faithfulness,
                "answer_relevancy": res.answer_relevancy,
                "context_precision": res.context_precision,
                "context_recall": res.context_recall
            }
            worst_metric = min(metrics, key=metrics.get)
            score = metrics[worst_metric]

        processed.append({
            "question": res.question,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": score,
            "diagnosis": diagnosis,
            "suggested_fix": fix
        })
    
    # Sort by avg_score ascending and take bottom_n
    processed.sort(key=lambda x: x["avg_score"])
    return processed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
