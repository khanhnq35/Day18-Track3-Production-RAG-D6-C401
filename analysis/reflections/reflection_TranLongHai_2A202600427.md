# Individual Reflection — Lab 18

**Tên:** Trần Long Hải  
**Module phụ trách:** M1a (Semantic + Structure-Aware Chunking)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** 
  - Hoàn thiện xử lý trích xuất văn bản (OCR) từ các file ảnh scan (`BCTC.pdf`, `Nghi_dinh_so_13.pdf`).
  - Lập trình thuật toán cắt văn bản theo Semantic Chunking và Structure-Aware Chunking.
- **Các hàm/class chính đã viết:**
  - Viết script sử dụng GPT-4o Vision API để giải quyết vấn đề dữ liệu đầu vào.
  - Viết hàm `chunk_semantic()` dùng `SentenceTransformer` tính Cosine Similarity để ghép câu theo ngữ nghĩa.
  - Viết hàm `chunk_structure_aware()` dùng regex bóc tách cấu trúc Header Markdown.
  - Viết hàm `compare_strategies()` chạy benchmark để đối chiếu thông số giữa các cách cắt chunk khác nhau.
- **Số tests pass:** 11 / 11 

## 2. Kiến thức học được

- **Khái niệm mới nhất:** 
  - Khái niệm Semantic Chunking. Trước đây chỉ biết cắt chữ theo độ dài cố định (đếm ký tự), nay biết cách dùng Vector Embedding để đo lường độ tương đồng ngữ nghĩa giữa các câu, giúp bảo toàn ý nghĩa nguyên vẹn.
- **Điều bất ngờ nhất:**
  - Việc dữ liệu đầu vào (data ingestion) quan trọng thế nào đối với Chunking. Các thư viện OCR phổ thông như `EasyOCR` lại làm hỏng định dạng tiếng Việt. Phải chuyển sang mô hình MLLM (GPT-4o Vision) để giữ được các bảng biểu và bố cục mới có thể áp dụng Structure-Aware Chunking hiệu quả.
- **Kết nối với bài giảng:** 
  - Liên kết trực tiếp tới nội dung Failure Mode liên quan tới "Missing Context" do cắt văn bản bị gãy. Việc áp dụng các kỹ thuật Chunking cấu trúc (Structure) đã giải quyết được lỗi này.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Dữ liệu được cung cấp hoàn toàn là bản chụp (Scan), không có layer text.
- **Cách giải quyết:** Đã tự chủ động thử nghiệm các tools OCR. Ban đầu thử EasyOCR nhưng nhận dạng chữ tiếng Việt quá kém. Cuối cùng quyết định dùng PyMuPDF cắt ảnh và bắn qua API GPT-4o-mini để trích xuất chuẩn Markdown.
- **Thời gian debug:** Khoảng 1 tiếng (chủ yếu là chờ OCR 39 trang Nghị định và test thử độ lệch của threshold trong Semantic chunking).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Nếu có thời gian sẽ viết thêm cache cho phần sinh Embedding trong Semantic Chunking để mỗi lần test chạy nhanh hơn, thay vì encode lại toàn bộ list sentences từ đầu.
- **Module nào muốn thử tiếp:** Muốn thử làm phần M3 (Rerank) để xem cách thuật toán sắp xếp lại các Chunk sau khi tìm kiếm như thế nào, vì chất lượng của Chunk ảnh hưởng rất lớn tới việc Rerank.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |
