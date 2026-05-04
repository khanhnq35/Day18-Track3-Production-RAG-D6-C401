# Individual Reflection — Lab 18

**Tên:** Trần Quốc Việt  
**MSSV:** 2A202600307  
**Module phụ trách:** M2 — Hybrid Search (BM25 + Dense + RRF)

---

## 1. Đóng góp kỹ thuật

### Module đã implement:
- **Module M2: Hybrid Search** — toàn bộ module trong `src/m2_search.py`

### Các hàm/class chính đã viết:
1. `segment_vietnamese(text)` — Underthesea word tokenization với fallback
2. `BM25Search.index(chunks)` — Build BM25Okapi index từ tokenized chunks
3. `BM25Search.search(query, top_k)` — Query bằng BM25, trả top-k SearchResult
4. `DenseSearch.index(chunks, collection)` — Encode chunks bằng bge-m3, upload vào Qdrant
5. `DenseSearch.search(query, top_k, collection)` — Query bằng dense vector, trả SearchResult
6. `reciprocal_rank_fusion(results_list, k, top_k)` — Merge BM25 + Dense rankings bằng RRF score

### Kết quả test:
- **5/5 tests pass** ✅
  - `test_segment_returns_string` ✓
  - `test_bm25_search` ✓
  - `test_bm25_relevant_first` ✓
  - `test_rrf_merges` ✓
  - `test_rrf_method` ✓

---

## 2. Kiến thức học được

### Khái niệm mới nhất:
- **BM25 Algorithm** — Probabilistic retrieval model, tốt cho Vietnamese full-text search khi kết hợp với word segmentation
- **Dense Embeddings & Vector Search** — bge-m3 embedding + Qdrant vector DB, tốt hơn BM25 cho semantic similarity
- **Reciprocal Rank Fusion (RRF)** — Cách hợp lý để merge kết quả từ 2 ranking systems khác nhau mà không cần tuning weights
- **Vietnamese NLP Pipeline** — Segmentation → tokenization → indexing là key để BM25 không bị "từ thừa"

### Điều bất ngờ nhất:
- RRF formula `score = Σ 1/(k + rank)` rất đơn giản nhưng hiệu quả, không cần train gì cả mà vẫn merge 2 ranker tốt
- underthesea library khá mạnh để Vietnamese word segmentation, fallback text gốc khi thiếu lib giúp pipeline không vỡ
- Qdrant rất dễ setup qua Docker, chỉ vài dòng yaml là có vector DB chạy

### Kết nối với bài giảng:
- **Slide Retrieval** — BM25 là sparse retrieval, dense là dense retrieval, hybrid là kết hợp cả 2
- **Slide IR Metrics** — Hiểu rõ hơn precision/recall qua việc implement search — top-k results có precision cao nhưng recall thể thấp
- **Slide Vietnamese NLP** — Word segmentation là bước quan trọng nhất, không segment đúng → search sai

---

## 3. Khó khăn & Cách giải quyết

### Khó khăn lớn nhất:
1. **Docker Desktop chưa khởi động** — `docker compose up -d` báo lỗi "no configuration file"
   - **Giải quyết:** Mở Docker Desktop, tạo file `docker-compose.yml` từ đầu với Qdrant service
   
2. **Missing package `rank-bm25`** — pip install không vào đúng interpreter
   - **Giải quyết:** Dùng full path Python executable: `C:/Users/.../python.exe -m pip install rank-bm25`

3. **Qdrant connection error lúc đầu** — DenseSearch.search() trả `[]` vì Qdrant chưa up
   - **Giải quyết:** Kiểm tra Qdrant running (`docker ps`), exception handling trong search() fallback trả `[]`

### Cách giải quyết:
- Systematically check dependencies: Docker → docker-compose → rank-bm25
- Add try-except fallback để pipeline không vỡ khi 1 component fail
- Đọc error message kỹ để identify root cause (interpreter mismatch, file missing, service down)

### Thời gian debug:
- ~25 phút setup Docker + packages
- ~10 phút implement code (theo template sẵn có)
- ~5 phút test và verify

---

## 4. Nếu làm lại

### Sẽ làm khác điều gì:
- Chuẩn bị Docker từ đầu (tạo docker-compose.yml ngay trong repo template)
- Thêm health check trong DenseSearch để detect Qdrant down sớm
- Benchmark BM25 vs Dense vs Hybrid trên cùng test set để thấy rõ improvement

### Module nào muốn thử tiếp:
- **M1 (Hierarchical Chunking)** — Muốn hiểu cách retrieve child nhưng return parent cho better context
- **M4 (RAGAS Evaluation)** — Muốn biết sâu hơn cách tính Faithfulness, Context Recall để debug khi scores thấp
- **M5 (Enrichment)** — Muốn test HyQA generation để xem có thật bridge vocabulary gap không

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|--------|
| Hiểu bài giảng | 4 | BM25 vs Dense search, RRF concept rõ; Vietnamese NLP basics đủ |
| Code quality | 4 | Type hints, docstrings có; exception handling có fallback; có thể thêm logging |
| Teamwork | 5 | Hoàn thành đúng scope M2, pass test, sẵn sàng handoff cho P6 integration |
| Problem solving | 4 | Debug Docker + packages tốt; có fallback strategy; có thể deep-dive hơn vào vector DB tuning |

### Tổng thể:
Lab này học được rất nhiều về production RAG — không phải chỉ code mà còn pipeline thinking (fallback, error handling, logging). M2 hybrid search là foundation quan trọng cho toàn bộ system.
