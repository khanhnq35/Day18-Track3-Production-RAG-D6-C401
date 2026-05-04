# Individual Reflection — Lab 18

**Tên:** [Họ tên]
**Module phụ trách:** M3 — Reranking + Latency Benchmark

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** `src/m3_rerank.py` — Module 3: Reranking
- **Các hàm/class chính đã viết:**

| Hàm / Class | Mô tả |
|-------------|-------|
| `CrossEncoderReranker._load_model()` | Load `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)` — Vietnamese-compatible multilingual cross-encoder. Lazy-load để tránh tốn RAM khi chưa dùng. |
| `CrossEncoderReranker.rerank()` | Tạo cặp (query, document_text) → compute_score bằng cross-encoder → sort descending theo rerank_score → trả về top-k `RerankResult`. Xử lý edge case: empty documents, single document. |
| `FlashrankReranker._load_model()` & `.rerank()` | Lightweight alternative reranker dùng FlashRank. Có try/except ImportError để fallback gracefully nếu flashrank chưa cài. |
| `benchmark_reranker()` | Đo latency qua `n_runs` lần bằng `time.perf_counter()` (độ phân giải cao). Trả về `{avg_ms, min_ms, max_ms}` với `statistics.mean()`. |

- **Số tests pass:** Chờ kết quả `pytest tests/test_m3.py -v` (5 tests)

---

## 2. Kiến thức học được

### Khái niệm mới nhất
1. **Cross-encoder reranking khác bi-encoder:**
   - Bi-encoder (như bge-m3): encode query và document riêng biệt → so sánh cosine similarity. Nhanh nhưng kém chính xác hơn vì query và doc không "nhìn thấy" nhau khi encode.
   - Cross-encoder (bge-reranker-v2-m3): nhận cả cặp (query, doc) làm input → output một relevance score duy nhất. Chính xác hơn vì model có thể học tương tác giữa query và doc, nhưng chậm hơn → phù hợp cho bước rerank top-20 → top-3.

2. **bge-reranker-v2-m3 là multilingual:** Hỗ trợ tiếng Việt, đáp ứng yêu cầu A3 (Vietnamese-specific handling). Đây là cross-encoder được fine-tune từ BGE-M3, tối ưu cho reranking đa ngôn ngữ.

3. **Latency profiling với `time.perf_counter()`:** Khác với `time.time()`, `perf_counter()` có độ phân giải cao hơn (microseconds) và không bị ảnh hưởng bởi system clock adjustments. Phù hợp để benchmark các operation nhỏ như rerank.

### Điều bất ngờ nhất
- Cross-encoder load model lần đầu rất chậm (có thể >5 giây do tải model từ HuggingFace hub), nhưng các lần rerank sau rất nhanh (~50-200ms cho 20 documents). Điều này giải thích tại sao test `test_rerank_*` cho phép first load chậm nhưng latency sau đó phải < 5 giây.

### Kết nối với bài giảng
- **Slide về RAG Pipeline:** Reranking là bước nằm giữa Retrieval và Generation. Hybrid Search trả về top-20 → Reranker chọn top-3 relevant nhất → đưa vào LLM context.
- **Slide về Evaluation:** Latency benchmark là một phần của production monitoring — cần đo được thời gian từng bước để tối ưu bottleneck.

---

## 3. Khó khăn & Cách giải quyết

### Khó khăn lớn nhất
1. **Cài đặt FlagEmbedding:** Package kéo theo nhiều dependency nặng (torch ~2GB, transformers, scipy). Mất thời gian tải và cài đặt.
2. **Xử lý output của `compute_score()`:** Với 1 cặp duy nhất, `compute_score()` trả về `float` thay vì `list[float]` → cần xử lý edge case để `zip(scores, documents)` không lỗi.

### Cách giải quyết
- Kiểm tra `len(documents) == 1` → wrap score vào list trước khi zip.
- Đọc kỹ FlagEmbedding documentation để hiểu `compute_score()` API.

### Thời gian debug
- Khoảng 15 phút cho edge case single document và đảm bảo sort descending.

---

## 4. Nếu làm lại

### Sẽ làm khác điều gì
- **So sánh nhiều reranker model hơn:** Ngoài bge-reranker-v2-m3, có thể thử `BAAI/bge-reranker-v2-minicpm-layerwise` (nhẹ hơn) hoặc Cohere Rerank API để so sánh quality vs latency trade-off.
- **Thêm warm-up run trong benchmark:** Lần rerank đầu tiên luôn chậm hơn do model loading. Nên thêm 1 warm-up run không tính vào stats để kết quả benchmark chính xác hơn.

### Module nào muốn thử tiếp
- M4 (RAGAS Evaluation): Muốn hiểu cách đánh giá pipeline end-to-end và Error Tree analysis.
- M2 (Hybrid Search): Muốn thử BM25 + Dense fusion với RRF.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |

**Giải thích:**
- **Hiểu bài giảng (4/5):** Nắm vững cross-encoder reranking và vai trò trong RAG pipeline. Còn muốn tìm hiểu thêm về các fusion strategy khác ngoài RRF.
- **Code quality (5/5):** Code có đầy đủ Google-style docstrings, type hints, xử lý edge cases (empty docs, single doc, import error cho flashrank). Tuân thủ PEP 8.
- **Teamwork (4/5):** Module M3 hoạt động độc lập, sẵn sàng tích hợp vào pipeline với interface rõ ràng (`rerank(query, documents, top_k) → list[RerankResult]`).
- **Problem solving (4/5):** Tự xử lý edge case single document và graceful fallback cho flashrank.

---

## Bonus: Latency Breakdown (Pipeline Production)

Dưới đây là bảng latency breakdown ước tính cho từng bước trong production RAG pipeline:

| Bước | Module | Thời gian ước tính (ms) | Ghi chú |
|------|--------|-------------------------|---------|
| 1. Chunking | M1 | ~500ms | Semantic/Structure-Aware chunking (offline, 1 lần) |
| 2. BM25 Search | M2 | ~10ms | BM25Okapi trên text đã segment |
| 3. Dense Search | M2 | ~50ms | bge-m3 embedding + Qdrant search |
| 4. RRF Fusion | M2 | ~1ms | Merge 2 rankings |
| 5. **Reranking (M3)** | **M3** | **~100ms** | **bge-reranker-v2-m3 top-20 → top-3** |
| 6. Enrichment | M5 | ~500ms | LLM contextual prepend + HyQA (offline) |
| 7. LLM Generation | Pipeline | ~1000ms | gpt-4o-mini generate answer |

**Tổng latency online (per query):** ~1,161ms (≈ 1.2 giây)
**Bottleneck:** LLM Generation (chiếm ~86% thời gian online). Reranking chỉ chiếm ~9%.

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-04 | [Họ tên] | Initial version |