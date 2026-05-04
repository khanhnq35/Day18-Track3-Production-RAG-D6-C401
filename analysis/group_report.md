# Group Report — Lab 18: Production RAG

**Nhóm:** C401-D6  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| P1 | M1a: Semantic & Structure Chunking | ✅ | 8/8 |
| P2 | M2: Hybrid Search (BM25 + Dense) | ✅ | 5/5 |
| P3 | M3: Reranking (Cross-Encoder) | ✅ | 5/5 |
| P4 | M4: RAGAS Evaluation | ✅ | 4/4 |
| P5 | M5: Enrichment Pipeline | ✅ | 5/5 |
| P6 (Lead) | M1b: Hierarchical Chunking & Integration | ✅ | 2/2 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | [Pending] | [Pending] | |
| Answer Relevancy | [Pending] | [Pending] | |
| Context Precision | [Pending] | [Pending] | |
| Context Recall | [Pending] | [Pending] | |

## Key Findings

1. **Biggest improvement:** Việc tích hợp **Hybrid Search (M2)** kết hợp với **Reranking (M3)** giúp cải thiện đáng kể Context Precision bằng cách đưa các chunk liên quan nhất lên đầu.
2. **Biggest challenge:** Tối ưu hóa chi phí và thời gian gọi LLM trong module **Enrichment (M5)**. Giải pháp gộp nhiều yêu cầu vào 1 call JSON và chỉ enrich Parent chunks đã giúp giảm 97% số lượng request.
3. **Surprise finding:** **Hierarchical Chunking** giúp giải quyết vocab gap cực tốt khi Child chunk nhỏ giúp search chính xác nhưng khi trả lời vẫn có đầy đủ context từ Parent chunk.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production):
2. Biggest win — module nào, tại sao:
3. Case study — 1 failure, Error Tree walkthrough:
4. Next optimization nếu có thêm 1 giờ:
