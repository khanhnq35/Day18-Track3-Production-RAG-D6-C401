# Failure Analysis — Lab 18: Production RAG

**Nhóm:** C401-D6  
**Thành viên:** Trần Long Hải (P1) · Trần Quốc Việt (P2) · Nguyễn Bình Minh (P3) · Ngô Anh Tú (P4) · Phan Văn Tấn (P5) · Nguyễn Quốc Khánh (P6)

---

## RAGAS Scores

| Metric | Basic | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.5583 | **0.9500** | +0.3917 |
| Answer Relevancy | 0.7805 | **0.8640** | +0.0835 |
| Context Precision | 0.1250 | **0.9500** | +0.8250 |
| Context Recall | 0.0500 | **0.9000** | +0.8500 |

## Analysis of Improvements

Sự nhảy vọt về các chỉ số **Context Precision (+0.825)** và **Context Recall (+0.85)** cho thấy các kỹ thuật nâng cao đã giải quyết triệt để bài toán truy xuất thông tin trong văn bản pháp luật và báo cáo tài chính phức tạp:

1.  **Hierarchical Chunking**: Việc chia nhỏ văn bản (Child chunks) để nhúng nhưng trả về ngữ cảnh rộng (Parent chunks) giúp LLM có đủ thông tin dẫn dắt để trả lời chính xác, thay vì chỉ nhận được các đoạn vụn vặt như bản Basic.
2.  **Hybrid Search (Dense + BM25)**: Sự kết hợp này cực kỳ hiệu quả với các truy vấn chứa từ khóa chuyên ngành (nhu "01/GTGT", "DHA Surfaces"). BM25 bắt chính xác các định danh này, trong khi Dense embedding xử lý tốt các câu hỏi về ngữ nghĩa.
3.  **Reranking (Cohere)**: Đóng vai trò là "màng lọc cuối", giúp đẩy các đoạn chứa số liệu thực tế (vốn có điểm embedding không quá cao) lên top đầu, trực tiếp giúp tăng Faithfulness và Answer Relevancy.

## Remaining Failures & Challenges

Mặc dù điểm số đã rất cao, hệ thống vẫn đối mặt với một số thách thức nhỏ (giảm 5% Faithfulness so với bản demo trước đó):

### #1: Vấn đề "Hallucination" nhẹ khi LLM cố gắng giải thích bảng biểu
- **Hiện tượng**: LLM lấy đúng số liệu từ bảng (trong Context) nhưng đôi khi thêm các từ ngữ diễn đạt không có trong tài liệu gốc để câu trả lời trơn tru hơn.
- **Root cause**: System Prompt hiện tại đang khuyến khích trả lời chi tiết, dẫn đến việc LLM "sáng tạo" thêm các liên từ hoặc bối cảnh không có thực (Faithfulness < 1.0).

### #2: Trễ thời gian (Latency)
- **Vấn đề**: Tổng thời gian chạy tăng lên đáng kể (hơn 4700s) do phải thực hiện nhiều bước: HyQA Enrichment, Hybrid Search, và Reranking.
- **Trade-off**: Đây là sự đánh đổi cần thiết để đạt được độ chính xác (Recall) tiệm cận tuyệt đối trong môi trường sản xuất.

## Error Tree Walkthrough: Case "Thuế GTGT phải nộp..."

1.  **Output đúng?** -> Có. Hệ thống đã trích xuất đúng con số `52.133.830` từ bảng.
2.  **Context đúng?** -> Có. Nhờ Reranking, đoạn text chứa bảng kê khai thuế đã nằm ở Top 1.
3.  **Faithfulness?** -> Đạt điểm cao vì số liệu khớp hoàn toàn với Context.
4.  **Bài học**: Đối với dữ liệu tài chính dạng bảng, **Context Enrichment** (thêm ngữ cảnh "Đây là bảng kê khai thuế") là chìa khóa để LLM không bị lạc lối giữa các hàng số liệu.

## Nếu có thêm 1 giờ, sẽ optimize:
- **Table-to-Markdown Optimization**: Sử dụng các tool chuyên dụng để parse bảng từ PDF sang Markdown sạch hơn, tránh việc LLM đọc nhầm cột/hàng (unrolled rows).
- **Asynchronous Execution**: Song song hóa các bước truy vấn và làm giàu dữ liệu để giảm Latency từ 4700s xuống dưới 1000s.
- **Query Expansion**: Sử dụng HyDE (Hypothetical Document Embeddings) để cải thiện hơn nữa khả năng tìm kiếm cho các câu hỏi mang tính khái niệm cao.
