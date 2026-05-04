# Individual Reflection — Lab 18

**Tên:** Nguyễn Quốc Khánh (khanhnq)
**Module phụ trách:** Integration Lead & P6 (Hierarchical Chunking + Pipeline Integration)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1b (Hierarchical Chunking) & Pipeline End-to-End.
- Các hàm/class chính đã viết: 
  - `chunk_hierarchical()`: Cắt văn bản theo cấu trúc cha-con để giải quyết Vocab Gap.
  - Tích hợp `pipeline.py`: Xây dựng luồng chạy đa luồng bằng `ThreadPoolExecutor` (max_workers=10) cho LLM generation.
  - Cấu hình đa luồng `RunConfig(max_workers=4)` cho Ragas để tối ưu tốc độ đánh giá mà không dính Rate Limit (HTTP 429).
  - Debug và fix nóng lỗi chí mạng của `qdrant-client` 1.17 (đổi tên hàm `search` thành `query_points` gây lỗi trả về mảng rỗng làm sập điểm số toàn hệ thống).
- Số tests pass: 100%

## 2. Kiến thức học được

- Khái niệm mới nhất: Tầm quan trọng của Cross-encoder Reranker và kỹ thuật Hierarchical Chunking giúp cải thiện đáng kể Context Precision.
- Điều bất ngờ nhất: RAG Pipeline có thể "tạch" do các nguyên nhân rất cơ bản (như lỗi thư viện vector DB), dẫn đến LLM bị ảo giác (hallucination) do không nhận được context, điểm Ragas rớt xuống 0 mặc dù prompt rất chuẩn.
- Kết nối với bài giảng: Phần Hybrid Search (kết hợp Dense và BM25) thực sự cứu cánh khi một trong hai phương pháp bị fail (như đã thấy khi DenseSearch bị lỗi).

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Quản lý Rate Limit (Quotas) của LLM (Gemini 2.5 Pro) khi chạy đánh giá RAGAS. Ban đầu chạy tuần tự thì quá chậm, bung đa luồng không kiểm soát thì bị 503 Timeout và 429 Too Many Requests. Ngoài ra, việc hợp nhất code từ các thành viên khác gây ra lỗi schema (ví dụ: `SearchResult` dùng `score` nhưng `RerankResult` lại dùng `rerank_score`).
- Cách giải quyết: 
  - Thiết kế lại kiến trúc đa luồng: Chia pha Search (nhanh, rẻ) chạy song song nhiều luồng, pha Judge (chậm, dễ tạch) chạy với số lượng worker hạn chế (`max_workers=4`).
  - Dùng `getattr(r, "rerank_score", getattr(r, "score", 0.0))` để tương thích linh hoạt mọi kiểu kết quả trả về.
- Thời gian debug: Khoảng 1.5 giờ.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Sẽ thiết lập Diagnostic Trace Logging (in vết từng bước Search, Rerank, Prompt, Context) ngay từ đầu dự án thay vì đến lúc bị điểm 0 mới cuống cuồng đi dò lỗi rỗng context.
- Module nào muốn thử tiếp: RAG Evaluation (M4). Muốn tự xây dựng các custom metrics thay vì phụ thuộc hoàn toàn vào Ragas, đặc biệt là các metric tính điểm tiếng Việt thuần túy.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
