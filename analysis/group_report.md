# Group Report — Lab 18: Production RAG

**Nhóm:** C401-D6  
**Ngày:** 2026-05-05

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Trần Long Hải (2A202600427) - P1 | M1a: Semantic & Structure Chunking | ✅ | 8/8 |
| Trần Quốc Việt (2A202600307) - P2 | M2: Hybrid Search (BM25 + Dense) | ✅ | 5/5 |
| Nguyễn Bình Minh (2A202600137) - P3 | M3: Reranking (Cross-Encoder) | ✅ | 5/5 |
| Ngô Anh Tú (2A202600128) - P4 | M4: RAGAS Evaluation | ✅ | 4/4 |
| Phan Văn Tấn (2A202600282) - P5 | M5: Enrichment Pipeline | ✅ | 5/5 |
| Nguyễn Quốc Khánh (2A202600199) - P6 (Lead) | M1b: Hierarchical Chunking & Integration | ✅ | 2/2 |

## Kết quả RAGAS (Final Version)

| Metric | Basic | Production | Δ |
|--------|-------|------------|---|
| Faithfulness | 0.5583 | **0.9500** | +0.3917 |
| Answer Relevancy | 0.7805 | **0.8640** | +0.0835 |
| Context Precision | 0.1250 | **0.9500** | +0.8250 |
| Context Recall | 0.0500 | **0.9000** | +0.8500 |

## Key Findings

1.  **Sự bứt phá ngoạn mục của Retrieval (+0.80):** Cả Context Precision và Recall đều đạt ngưỡng xuất sắc (0.90 - 0.95). Điều này khẳng định sự kết hợp giữa **Hierarchical Chunking** và **Hybrid Search (Dense + BM25)** là "bộ đôi vàng" cho dữ liệu phức tạp.
2.  **Độ tin cậy gần như tuyệt đối (0.9500):** Faithfulness duy trì ở mức cực cao, chỉ giảm nhẹ 5% so với bản demo trước đó do LLM cố gắng giải thích các hàng số liệu trong bảng biểu một cách chi tiết hơn.
3.  **Cải thiện trải nghiệm người dùng (Answer Relevancy):** Chỉ số Relevancy tăng mạnh lên 0.86 cho thấy LLM không còn trả lời quá ngắn gọn hay máy móc, mà đã biết cách tổng hợp thông tin từ ngữ cảnh để đưa ra câu trả lời đầy đủ và đúng trọng tâm.

## Presentation Notes (5 phút)

1.  **RAGAS scores:** "Hệ thống của chúng em đạt bước tiến khổng lồ với các chỉ số Retrieval (**Precision 0.95, Recall 0.90**). So với bản Basic, khả năng tìm đúng thông tin đã tăng gần 20 lần."
2.  **Biggest win:** "Chiến thắng lớn nhất là kỹ thuật **Hierarchical Chunking (Parent-Child)** kết hợp với **Reranking**. Chúng em giải quyết được vấn đề mất ngữ cảnh khi tra cứu các điều luật nhỏ lẻ hoặc các ô số trong bảng báo cáo tài chính."
3.  **Case study:** "Một điểm sáng là Answer Relevancy đã tăng vọt. Chúng em đã tinh chỉnh System Prompt để LLM không chỉ trích xuất thông tin mà còn biết cách trình bày mạch lạc, đáp ứng đúng ý định của người dùng."
4.  **Future vision:** "Nếu có thêm thời gian, nhóm sẽ tối ưu luồng **Asynchronous** để giảm thời gian xử lý và triển khai **Table-parsing** nâng cao để đạt điểm Faithfulness 1.0 tuyệt đối cho dữ liệu bảng biểu."
