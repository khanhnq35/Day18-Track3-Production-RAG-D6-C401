# Individual Reflection — Lab 18

**Tên:** Ngô Anh Tú
**MSSV:** 2A202600128
**Module phụ trách:** M4 — RAGAS Evaluation (4 metrics + Failure Analysis)

---

## 1. Đóng góp kỹ thuật

### Module đã implement:
- **Module M4: RAGAS Evaluation** — toàn bộ module trong `src/m4_eval.py`

### Các hàm/class chính đã viết:
1. `VertexRagasLLM` (class) — Custom LLM wrapper để tích hợp Vertex AI (Gemini) với RAGAS framework
2. `VertexRagasEmbeddings` (class) — Custom Embeddings wrapper cho RAGAS sử dụng bge-m3
3. `evaluate_ragas(questions, answers, contexts, ground_truths)` — Chạy RAGAS evaluation với 4 metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
4. `failure_analysis(eval_results, bottom_n)` — Phân tích bottom-N câu hỏi tệ nhất, áp dụng Diagnostic Tree để chẩn đoán lỗi
5. `save_report(results, failures, path)` — Lưu báo cáo JSON kèm aggregate scores và failure analysis

### Kết quả test:
- **4/4 tests pass** ✅
  - `test_evaluate_ragas` ✓
  - `test_scores_numeric` ✓
  - `test_failure_analysis` ✓
  - `test_load_test_set` ✓

---

## 2. Kiến thức học được

### Khái niệm mới nhất:
- **RAGAS Framework** — Bộ metrics đánh giá RAG toàn diện: Faithfulness (độ trung thực), Answer Relevancy (độ liên quan), Context Precision (độ chính xác context), Context Recall (độ bao phủ context)
- **LLM-as-a-Judge** — Dùng LLM để chấm điểm câu trả lời thay vì rule-based metrics, linh hoạt nhưng có điểm mù
- **Diagnostic Tree** — Cây chẩn đoán lỗi: Faithfulness thấp → LLM hallucinating; Context Recall thấp → Missing chunks; Context Precision thấp → Irrelevant chunks
- **RAGAS Wrappers** — Cần custom LLM và Embeddings wrapper để tích hợp được các LLM provider khác nhau (Vertex AI, OpenAI...) vào RAGAS

### Điều bất ngờ nhất:
- RAGAS đánh giá rất khắt khe: LLM judge có thể chấm điểm thấp cho câu trả lời đúng nếu prompt quá cứng nhắc (ép "Không tìm thấy thông tin")
- Việc làm sạch JSON response từ LLM rất quan trọng — LLM thường trả về kèm markdown code blocks hoặc text thừa, cần regex để extract JSON chuẩn
- `RunConfig(max_workers=4)` của RAGAS giúp kiểm soát số lượng threads, tránh bị Rate Limit (HTTP 429) khi gọi LLM nhiều lần

### Kết nối với bài giảng:
- **Slide Evaluation Metrics** — Hiểu sâu hơn về Faithfulness, Answer Relevancy, Context Precision, Context Recall qua việc implement thực tế
- **Slide Failure Analysis** — Diagnostic Tree chính là công cụ để map từ low scores sang root cause và suggested fix
- **Slide LLM Integration** — Custom wrapper pattern để adapt bất kỳ LLM API nào vào framework có sẵn

---

## 3. Khó khăn & Cách giải quyết

### Khó khăn lớn nhất:
1. **LLM response format không consistent** — Gemini trả về JSON kèm markdown ```json...``` hoặc text giải thích, làm RAGAS parse lỗi
   - **Giải quyết:** Dùng regex `re.search(r"(\{.*\})", res, re.DOTALL)` để extract JSON object từ response bất kể format

2. **Rate Limit (HTTP 429) khi chạy RAGAS** — Đánh giá 20 questions × 4 metrics = 80 LLM calls, dễ bị quota exceeded
   - **Giải quyết:** Cấu hình `RunConfig(max_workers=4, timeout=120)` để giới hạn số lượng concurrent calls, kết hợp với model JUDGE_LLM nhẹ hơn

3. **RAGAS dataset column names** — RAGAS expect columns `user_input`, `response`, `retrieved_contexts`, `reference` nhưng dataset lại dùng `question`, `answer`, `contexts`, `ground_truth`
   - **Giải quyết:** Thêm logic map column names trong hàm `evaluate_ragas()` để tương thích linh hoạt cả 2 format

### Cách giải quyết:
- Thêm debug logging vào `logs/ragas_debug.txt` để trace prompt và response khi có lỗi
- Dùng `raise_exceptions=False` trong `evaluate()` để pipeline không vỡ khi 1 câu hỏi fail
- Exception handling trong toàn bộ wrapper methods

### Thời gian debug:
- ~30 phút xử lý LLM response parsing và column mapping
- ~20 phút cấu hình RunConfig để tránh Rate Limit
- ~10 phút implement core functions (theo template sẵn có)
- ~10 phút test và verify

---

## 4. Nếu làm lại

### Sẽ làm khác điều gì:
- Thiết kế sẵn retry mechanism với exponential backoff cho LLM calls thay vì chỉ dựa vào `max_workers`
- Tạo visualization (matplotlib/seaborn) cho RAGAS scores thay vì chỉ JSON text — dễ nhìn hơn cho presentation
- Thêm custom metrics riêng cho tiếng Việt (ví dụ: kiểm tra từ khóa tiếng Việt có trong answer hay không)

### Module nào muốn thử tiếp:
- **M5 (Enrichment Pipeline)** — Muốn tìm hiểu cách HyQA và Contextual Prepend giúp bridge vocabulary gap, cải thiện Context Recall
- **M3 (Reranking)** — Muốn hiểu sâu hơn Cross-Encoder hoạt động thế nào để boost Context Precision
- **Pipeline Integration** — Muốn tham gia tích hợp end-to-end để thấy tác động của M4 evaluation lên toàn bộ system

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|--------|
| Hiểu bài giảng | 5 | Nắm rõ 4 RAGAS metrics, Diagnostic Tree, LLM-as-a-Judge concept |
| Code quality | 4 | Có type hints, docstrings, debug logging; có thể thêm unit tests cho wrapper classes |
| Teamwork | 5 | Hoàn thành đúng scope M4, cung cấp output format chuẩn cho P6 integration |
| Problem solving | 4 | Xử lý tốt LLM response parsing và Rate Limit; có thể proactive hơn về retry strategy |

### Tổng thể:
Module M4 đóng vai trò "thước đo" cho toàn bộ RAG pipeline. Qua việc implement và debug RAGAS, hiểu rõ được điểm mạnh/yếu của từng module trong pipeline. Đặc biệt, failure analysis giúp chỉ ra rằng Faithfulness có thể bị ảnh hưởng bởi prompt design — bài học quan trọng về sự đánh đổi giữa chống ảo giác và điểm số tự động.
