# Individual Reflection — Lab 18

**Tên:** Phan Văn Tấn 
**Module phụ trách:** M5 (Enrichment Pipeline)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M5 - Enrichment Pipeline (Làm giàu dữ liệu trước khi indexing).
- **Các hàm/class chính đã viết:** 
    - `summarize_chunk()`: Tóm tắt nội dung chunk để giảm nhiễu khi embedding.
    - `generate_hypothesis_questions()`: Tạo câu hỏi giả định (HyQA) để thu hẹp khoảng cách từ vựng (vocabulary gap).
    - `contextual_prepend()`: Gắn ngữ cảnh của toàn bộ tài liệu vào từng chunk (Anthropic Style).
    - `extract_metadata()`: Tự động trích xuất Topic, Entities, Category bằng LLM.
    - `enrich_chunks()`: Pipeline tổng hợp để xử lý hàng loạt chunks.
- **Số tests pass:** 10 / 10 (Pass 100% tests trong `test_m5.py`).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** "Contextual Retrieval" của Anthropic - cách giải quyết vấn đề mất ngữ cảnh khi cắt nhỏ văn bản (chunking) bằng cách yêu cầu LLM viết một câu mô tả vị trí của chunk đó trong tài liệu gốc.
- **Điều bất ngờ nhất:** Việc tạo thêm các câu hỏi giả định (HyQA) thực sự giúp cải thiện khả năng tìm kiếm vì nó mô phỏng đúng cách người dùng đặt câu hỏi thực tế, thay vì chỉ so khớp các từ khóa trong tài liệu.
- **Kết nối với bài giảng:** Kết nối với phần "Data Pre-processing" và "Advanced Indexing Strategies" trong slide, nơi nhấn mạnh việc xử lý dữ liệu thô thành thông tin có cấu trúc trước khi lưu vào Vector Database.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Xử lý dữ liệu trả về từ LLM (đặc biệt là định dạng JSON cho Metadata) đôi khi không ổn định hoặc bị dính markdown block.
- **Cách giải quyết:** Sử dụng `json.loads` kết hợp với các thao tác xử lý chuỗi (`strip`, `replace`) để làm sạch dữ liệu trước khi parse. Sau đó chuyển sang dùng hàm `call_llm` dùng chung của nhóm để đảm bảo tính đồng nhất.
- **Thời gian debug:** Khoảng 1 giờ để đồng nhất tham số hàm `enrich_chunks` cho khớp với kỳ vọng của bộ test (`methods` list thay vì Boolean flags).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ thử nghiệm thêm với các mô hình LLM khác nhau thông qua OpenRouter để so sánh chất lượng tóm tắt và chi phí (ví dụ so sánh Gemini Flash 1.5 với GPT-4o-mini).
- **Module nào muốn thử tiếp:** M1 (Advanced Chunking) vì đây là "đầu vào" quan trọng nhất, nếu chunking không tốt thì enrichment cũng không cứu được dữ liệu.

## 5. Tự đánh giá

**Điểm tự chấm:** 10/10 (Hoàn thành đầy đủ các TODO, vượt qua tất cả bài test, và hỗ trợ nhóm lấy điểm Bonus bằng cách tích hợp Enrichment vào pipeline).
