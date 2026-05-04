# Failure Analysis — Lab 18: Production RAG

**Nhóm:** C401-D6  
**Thành viên:** Trần Long Hải (P1) · Trần Quốc Việt (P2) · Nguyễn Bình Minh (P3) · Ngô Anh Tú (P4) · Phan Văn Tấn (P5) · Nguyễn Quốc Khánh (P6)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6833 | 0.5000 | -0.1833 |
| Answer Relevancy | 0.8109 | 0.4134 | -0.3975 |
| Context Precision | 0.1250 | 0.4792 | +0.3542 |
| Context Recall | 0.0500 | 0.4250 | +0.3750 |

## Bottom-5 Failures

### #1
- **Question:** Tên người nộp thuế và mã số thuế trong tờ khai thuế GTGT Mẫu số 01/GTGT là gì?
- **Expected:** CÔNG TY CỔ PHẦN DHA SURFACES và mã số thuế là 0106769437.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? (Có thể) → LLM từ chối trả lời do Prompt cứng nhắc.
- **Root cause:** Prompt quá gắt yêu cầu tuyệt đối, hoặc Context quá dài khiến LLM bỏ sót thông tin.
- **Suggested fix:** Nới lỏng system prompt hoặc giảm bớt độ nhiễu của Enrichment metadata.

### #2
- **Question:** Thuế giá trị gia tăng phải nộp của hoạt động sản xuất kinh doanh trong kỳ 4 năm 2024 là bao nhiêu?
- **Expected:** 52.133.830 đồng.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context sai? → Search không tìm thấy bảng biểu.
- **Root cause:** OCR không đọc tốt định dạng bảng biểu, dẫn đến Semantic Search không bám được vào các con số tài chính.
- **Suggested fix:** Tích hợp Document Parsing chuyên dụng cho bảng (ví dụ: Unstructured.io) thay vì OCR text thuần.

### #3
- **Question:** Kỳ tính thuế của Tờ khai thuế giá trị gia tăng Mẫu số 01/GTGT của công ty DHA Surfaces là khi nào?
- **Expected:** Quý 4 năm 2024.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? → LLM hallucinating.
- **Root cause:** Câu trả lời có thể nằm trong khoảng cách xa giữa các chunk (do BCTC chia làm nhiều trang).
- **Suggested fix:** Tăng kích thước cửa sổ trượt (sliding window) khi cắt chunk.

### #4
- **Question:** Việc mua bán dữ liệu cá nhân có được pháp luật cho phép không?
- **Expected:** Không (bị nghiêm cấm).
- **Got:** Không tìm thấy thông tin cụ thể (hoặc trả lời sai lệch).
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? (Có) → Suy luận của LLM sai.
- **Root cause:** LLM gặp khó khăn với các từ phủ định hoặc câu mang tính chất pháp lý phức tạp khi bị nhiễu bởi các điều khoản lân cận.
- **Suggested fix:** Sử dụng mô hình LLM lớn hơn (như Gemini Pro thay vì Flash) cho khâu trả lời.

### #5
- **Question:** Hàng hóa, dịch vụ mua vào chịu thuế suất 10% có giá trị chưa thuế là bao nhiêu?
- **Expected:** 2.405.743.241 đồng.
- **Got:** Không tìm thấy thông tin.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context sai? → Reranker đẩy chunk bảng biểu xuống dưới.
- **Root cause:** Cross-Encoder reranker có thể chưa được fine-tune tốt cho domain số liệu tài chính tiếng Việt.
- **Suggested fix:** Fine-tune lại Reranker hoặc dùng hybrid score kết hợp với BM25 tốt hơn cho các con số.

## Case Study (cho presentation)

**Question chọn phân tích:** Tên người nộp thuế và mã số thuế... (Fail #1)

**Error Tree walkthrough:**
1. Output đúng? → Sai (Bảo không tìm thấy).
2. Context đúng? → Có chứa thông tin trong chunk đầu tiên (do Hybrid Search tìm rất tốt phần header của BCTC).
3. LLM xử lý đúng? → Sai, do bị nhiễu bởi phần Enrichment (Parent chunk lồng vào quá dài) làm LLM "Lost in the middle", và bị ép bởi System Prompt "Không suy luận".
4. Fix ở bước: LLM Generation (System Prompt).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tối ưu hóa lại độ dài của Parent Chunk trong `m1_chunking.py`.
- Tinh chỉnh System Prompt mềm dẻo hơn.
- Áp dụng các kỹ thuật parse bảng biểu (Table Extraction) chuyên dụng cho BCTC.
