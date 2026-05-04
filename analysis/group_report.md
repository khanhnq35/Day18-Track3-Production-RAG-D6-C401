# Group Report — Lab 18: Production RAG

**Nhóm:** C401-D6  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Trần Long Hải (2A202600427) - P1 | M1a: Semantic & Structure Chunking | ✅ | 8/8 |
| Trần Quốc Việt (2A202600307) - P2 | M2: Hybrid Search (BM25 + Dense) | ✅ | 5/5 |
| Nguyễn Bình Minh (2A202600137) - P3 | M3: Reranking (Cross-Encoder) | ✅ | 5/5 |
| Ngô Anh Tú (2A202600128) - P4 | M4: RAGAS Evaluation | ✅ | 4/4 |
| Phan Văn Tấn (2A202600282) - P5 | M5: Enrichment Pipeline | ✅ | 5/5 |
| Nguyễn Quốc Khánh (2A202600199) - P6 (Lead) | M1b: Hierarchical Chunking & Integration | ✅ | 2/2 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.6833 | 0.5000 | -0.1833 |
| Answer Relevancy | 0.8109 | 0.4134 | -0.3975 |
| Context Precision | 0.1250 | 0.4792 | +0.3542 |
| Context Recall | 0.0500 | 0.4250 | +0.3750 |

## Key Findings

1. **Biggest improvement:** Việc tích hợp **Hybrid Search (M2)** kết hợp với **Reranking (M3)** giúp cải thiện đáng kể Context Precision bằng cách đưa các chunk liên quan nhất lên đầu.
2. **Biggest challenge:** Tối ưu hóa chi phí và thời gian gọi LLM trong module **Enrichment (M5)**. Giải pháp gộp nhiều yêu cầu vào 1 call JSON và chỉ enrich Parent chunks đã giúp giảm 97% số lượng request.
3. **Surprise finding:** **Hierarchical Chunking** giúp giải quyết vocab gap cực tốt khi Child chunk nhỏ giúp search chính xác nhưng khi trả lời vẫn có đầy đủ context từ Parent chunk. Tuy nhiên, Faithfulness bị giảm do LLM từ chối trả lời ("Không tìm thấy thông tin") khi Prompt được set quá khắt khe, chứng minh việc đánh giá tự động (LLM-as-a-judge) có điểm mù.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): "Như thầy cô thấy, Context Precision và Recall tăng vọt (Precision từ 0.12 lên 0.47, Recall từ 0.05 lên 0.42). Tuy nhiên, Faithfulness và Relevancy giảm do nhóm ép LLM tuân thủ nguyên tắc tuyệt đối 'Không tìm thấy thì phải nói không tìm thấy' để chống ảo giác (Hallucination), dẫn đến RAGAS chấm điểm rất khắt khe."
2. Biggest win — module nào, tại sao: "Thắng lợi lớn nhất là tích hợp thành công Hybrid Search và Hierarchical Chunking (M1b + M2). Việc tìm kiếm bằng cả Dense và BM25 trên các chunk con, sau đó trả về chunk cha giúp bảo toàn trọn vẹn ngữ cảnh ngữ nghĩa."
3. Case study — 1 failure, Error Tree walkthrough: "Mời thầy cô xem lỗi điển hình: Câu hỏi về 'Tên người nộp thuế'. Hệ thống tìm thấy thông tin ở chunk đầu tiên (Context đúng), nhưng LLM lại kết luận 'Không tìm thấy'. Nguyên nhân là do đoạn Enrichment của Parent chunk quá dài khiến LLM bị 'Lost in the middle'. Lỗi này xuất phát từ khâu LLM Generation (System Prompt)."
4. Next optimization nếu có thêm 1 giờ: "Nếu có thêm 1 giờ, nhóm sẽ áp dụng thư viện phân tích cấu trúc bảng biểu (Table Extraction chuyên dụng) cho BCTC, đồng thời dùng LLM lớn hơn (như Gemini 1.5 Pro) để trả lời thay vì Flash để tránh mất mát thông tin."
