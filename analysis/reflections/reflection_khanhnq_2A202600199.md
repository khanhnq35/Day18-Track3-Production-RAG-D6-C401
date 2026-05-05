# Individual Reflection — Lab 18

**Tên:** Nguyễn Quốc Khánh (khanhnq)
**Module phụ trách:** Integration Lead & P6 (Hierarchical Chunking + Pipeline Integration)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1b (Hierarchical Chunking) & Pipeline End-to-End.
- Các hàm/class chính đã viết: 
  - `chunk_hierarchical()`: Cắt văn bản theo cấu trúc cha-con để giải quyết Vocab Gap.
  - Tích hợp `pipeline.py`: Xây dựng luồng chạy đa luồng bằng `ThreadPoolExecutor` (max_workers=10) cho LLM generation.
  - **Qdrant Resilience (New)**: Tối ưu hóa `DenseSearch` trong `m2_search.py` để hỗ trợ **Local Storage Mode** (`path="./qdrant_db"`). Việc này giúp hệ thống vượt qua lỗi `Connection Refused` (Errno 61) khi không có Docker/Server Qdrant, đảm bảo pipeline có thể chạy ổn định trên mọi môi trường.
  - **Sửa lỗi ID Collision (Critical Bug)**: Nhúng tên file nguồn vào `parent_id` để ngăn chặn bối cảnh bị ghi đè chéo giữa các tài liệu, đảm bảo Recall chuẩn xác.
- Số tests pass: 100%

## 2. Kiến thức học được

- Khái niệm mới nhất: Tầm quan trọng của Cross-encoder Reranker và kỹ thuật Hierarchical Chunking giúp cải thiện đáng kể Context Precision.
- Điều bất ngờ nhất: RAG Pipeline có thể "tạch" do các nguyên nhân rất cơ bản (như lỗi kết nối Vector DB). Tuy nhiên, việc chuyển đổi linh hoạt sang chế độ lưu trữ local là một bài học quý giá về tính **Resilience** của hệ thống.
- Kết quả ấn tượng: Sự nhảy vọt của chỉ số **Context Precision (0.95)** và **Context Recall (0.90)** chứng minh rằng việc kết hợp Hybrid Search + Hierarchical Chunking là giải pháp tối ưu cho dữ liệu pháp luật và tài chính.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: 
  - Quản lý Rate Limit (Quotas) của LLM và Rate Limit của Ragas đánh giá.
  - Xử lý lỗi môi trường (Connection Refused) khi chạy pipeline trên máy cá nhân không có sẵn hạ tầng Docker cho Qdrant.
- Cách giải quyết: 
  - Thiết kế lại kiến trúc đa luồng: Chia pha Search chạy song song nhiều luồng, pha Judge chạy với số lượng worker hạn chế (`max_workers=4`).
  - **Chuyển đổi Qdrant Client**: Chỉnh sửa mã nguồn để sử dụng file-based storage thay vì server-based, giúp bypass hoàn toàn lỗi kết nối mạng.
  - **Unique ID Generation**: Nhúng prefix tên file vào ID chunk để triệt tiêu hiện tượng ID Collision.
- Thời gian debug: Khoảng 3 giờ (bao gồm thời gian xử lý lỗi kết nối DB và tối ưu hóa context).

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Sẽ xây dựng một lớp **Storage Abstraction** ngay từ đầu để hệ thống tự động chuyển đổi giữa Local và Server mode dựa trên cấu hình môi trường. Ngoài ra, sẽ chú trọng hơn vào việc xử lý dữ liệu bảng biểu (Table-to-Markdown) để đạt Faithfulness 1.0 tuyệt đối.
- Module nào muốn thử tiếp: RAG Evaluation (M4). Muốn tự xây dựng các custom metrics để đánh giá sâu hơn về khả năng suy luận (Reasoning) của mô hình trên các văn bản luật Việt Nam.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
