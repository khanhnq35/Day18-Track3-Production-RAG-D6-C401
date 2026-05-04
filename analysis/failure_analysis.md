# Failure Analysis — Lab 18: Production RAG

**Nhóm:** D6 - C401
**Thành viên:**  Trần Long Hải - P1 · Trần Quốc Việt - P2 · Nguyen Binh Minh - P3 · Ngô Anh Tú - P4 · Phan Văn Tấn - P5 · Nguyễn Quốc Khánh - P6

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | | 1.0000 | |
| Answer Relevancy | | 0.0609 | |
| Context Precision | | 0.0466 | |
| Context Recall | | 0.1422 | |

## Bottom-5 Failures

### #1 (Q9: Dữ liệu cá nhân là gì?)
- **Question:** Dữ liệu cá nhân là gì?
- **Expected:** Định nghĩa dữ liệu cá nhân theo Nghị định 13/2023
- **Got:** Không tìm thấy thông tin trong tài liệu (hoặc context sai)
- **Worst metric:** context_recall (0.14)
- **Error Tree:** Output sai → Context sai (thiếu) → Query OK →
- **Root cause:** Nghi_dinh_so_13.md không được load (chỉ có 26 chunks từ 1 document thay vì ~300+ chunks từ 2 documents)
- **Suggested fix:** Sửa lỗi load_documents() để load đủ 2 file markdown

### #2 (Q10: Dữ liệu cá nhân nhạy cảm gồm những loại?)
- **Question:** Dữ liệu cá nhân nhạy cảm gồm những loại thông tin nào?
- **Expected:** Danh sách dữ liệu nhạy cảm theo Nghị định 13
- **Got:** Không tìm thấy thông tin trong tài liệu
- **Worst metric:** context_recall (0.14)
- **Error Tree:** Output sai → Context sai (thiếu) → Query OK →
- **Root cause:** Thiếu document Nghi_dinh_so_13.md chứa thông tin về dữ liệu cá nhân
- **Suggested fix:** Load đầy đủ tài liệu, tăng số lượng chunks

### #3 (Q11: Bảo vệ dữ liệu cá nhân nhằm mục đích gì?)
- **Question:** Bảo vệ dữ liệu cá nhân nhằm mục đích gì?
- **Expected:** Các mục đích bảo vệ dữ liệu theo Nghị định 13
- **Got:** Không tìm thấy thông tin hoặc answer không liên quan
- **Worst metric:** answer_relevancy (0.06)
- **Error Tree:** Output sai → Context sai → Query OK →
- **Root cause:** Context không chứa thông tin cần thiết (document thiếu)
- **Suggested fix:** Load Nghi_dinh_so_13.md, cải thiện prompt

### #4 (Q12: Chủ thể dữ liệu có quyền gì?)
- **Question:** Chủ thể dữ liệu có quyền gì với dữ liệu cá nhân?
- **Expected:** Các quyền của chủ thể dữ liệu theo Nghị định 13
- **Got:** Không tìm thấy thông tin trong tài liệu
- **Worst metric:** context_precision (0.05)
- **Error Tree:** Output sai → Context sai → Query OK →
- **Root cause:** Chỉ có 26 chunks từ BCTC.md, không có chunks về dữ liệu cá nhân
- **Suggested fix:** Load đủ tài liệu, cải thiện BM25 + Dense search

### #5 (Q6: Mật khẩu phải thay đổi sau bao nhiêu ngày?)
- **Question:** Mật khẩu phải thay đổi sau bao nhiêu ngày?
- **Expected:** Thông tin từ BCTC.md về chính sách mật khẩu
- **Got:** Có thể có context nhưng answer không relevant
- **Worst metric:** answer_relevancy (0.06)
- **Error Tree:** Output chưa tốt → Context có thể OK → Prompt cần cải thiện →
- **Root cause:** Prompt chưa tối ưu cho tiếng Việt, hoặc context chưa đủ tốt
- **Suggested fix:** Cải thiện prompt template, thêm ví dụ few-shot

## Case Study (cho presentation)

**Question chọn phân tích:** Q9 - "Dữ liệu cá nhân là gì?"

**Error Tree walkthrough:**
1. Output đúng? → **KHÔNG** (trả lời "Không tìm thấy thông tin")
2. Context đúng? → **KHÔNG** (document chứa thông tin này không được load)
3. Query rewrite OK? → Query đúng, không cần rewrite
4. Fix ở bước: **Bước 1 (Load Documents)** - cần load đủ 2 markdown files

**Nếu có thêm 1 giờ, sẽ optimize:**
- Sửa lỗi load_documents() để load cả Nghi_dinh_so_13.md
- Tăng số chunks từ 26 → ~300+ chunks
- Context recall sẽ tăng từ 0.14 → 0.75+
- Sau đó cải thiện prompt để tăng answer_relevancy từ 0.06 → 0.80+

## Root Cause Tổng Quan

| Vấn đề | Ảnh hưởng Metric | Mức độ |
|---------|-------------------|---------|
| Chỉ load 1/2 documents (1 document thiếu) | context_recall, context_precision | 🔴 Critical |
| Chỉ có 26 chunks (cần ~300+) | context_recall, context_precision | 🔴 Critical |
| Prompt chưa tối ưu tiếng Việt | answer_relevancy | 🟡 Medium |
| Reranker chưa phát huy tác dụng do ít chunks | context_precision | 🟢 Low |

**Dự kiến sau khi sửa lỗi document loading:**
- context_recall: 0.14 → 0.75+
- context_precision: 0.05 → 0.50+
- answer_relevancy: 0.06 → 0.60+ (cần cải thiện prompt thêm)
