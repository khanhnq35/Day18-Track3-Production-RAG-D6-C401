"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/3] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    parent_lookup = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_lookup[parent.parent_id] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")
    print(f"  {len(parent_lookup)} parent chunks for hierarchical context")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    if enriched:
        # Use enriched text for indexing
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker, parent_lookup


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              parent_lookup: dict[str, str]) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    source_results = reranked if reranked else results[:3]
    contexts = []
    seen_parent_ids = set()
    
    for result in source_results:
        parent_id = result.metadata.get("parent_id")
        if parent_id and parent_id in parent_lookup:
            if parent_id in seen_parent_ids:
                continue
            contexts.append(parent_lookup[parent_id])
            seen_parent_ids.add(parent_id)
        else:
            contexts.append(result.text)

    # TODO (nhóm): Replace with LLM generation for better scores
    try:
        from openai import OpenAI
        client = OpenAI()
        context_str = "\n\n".join(contexts)
        
        sys_prompt = (
            "Bạn là trợ lý RAG trả lời câu hỏi tiếng Việt dựa trên tài liệu được cung cấp. "
            "Chỉ sử dụng thông tin xuất hiện trực tiếp trong phần Context. "
            "Không sử dụng kiến thức bên ngoài, không suy luận ngoài context, không bịa thêm chi tiết. "
            "Trả lời trực tiếp, ngắn gọn, đúng trọng tâm câu hỏi. "
            "Nếu context không đủ thông tin để trả lời chắc chắn, chỉ trả lời đúng câu sau: "
            "'Không tìm thấy thông tin trong tài liệu.'"
        )
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ]
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        answer = contexts[0] if contexts else "Không tìm thấy thông tin trong tài liệu."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker,
                      parent_lookup: dict[str, str]):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, parent_lookup)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_lookup = build_pipeline()
    evaluate_pipeline(search, reranker, parent_lookup)
    print(f"\nTotal: {time.time() - start:.1f}s")
