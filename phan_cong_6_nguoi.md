# Phân công chi tiết cho 6 người - Lab 18 Production RAG

Ngày lập kế hoạch: 2026-05-04
Mục tiêu: hoàn thành pipeline chạy end-to-end, có báo cáo RAGAS, failure analysis, report nhóm và đủ reflection cá nhân.

---

## 1) Chiến lược chia việc

Vì repo có 5 module chính (M1-M5), cách chia hợp lý cho 6 người là:
- 5 người phụ trách 5 module theo chiều dọc (code + test + handoff).
- 1 người làm Integration + QA + Report để ghép toàn hệ thống và chốt deliverable.

Nguyên tắc phối hợp:
- Mỗi người làm trên 1 nhánh riêng: feat/m1, feat/m2, feat/m3, feat/m4, feat/m5, feat/integration.
- Mỗi module phải pass test riêng trước khi merge.
- Người Integration chỉ merge khi module owner xác nhận xong checklist DoD.

---

## 2) Phân công cụ thể từng người

## Người 1 - Module M1 Chunking

Phạm vi chính:
- Implement toàn bộ TODO trong src/m1_chunking.py.

Làm vào đâu:
- src/m1_chunking.py
  - chunk_semantic
  - chunk_hierarchical
  - chunk_structure_aware
  - compare_strategies

Làm như thế nào:
1. Implement chunk_semantic:
- Tách câu bằng regex theo dấu chấm/câu hỏi/câu cảm thán và đoạn xuống dòng.
- Encode câu bằng sentence-transformers.
- Tính cosine similarity giữa các câu liên tiếp.
- Nếu similarity < threshold thì mở chunk mới.
- Trả list Chunk có metadata chunk_index + strategy=semantic.

2. Implement chunk_hierarchical:
- Gom paragraph thành parent chunk theo parent_size.
- Mỗi parent có parent_id trong metadata.
- Tách parent thành child theo child_size, gắn parent_id vào child.parent_id.
- Trả tuple (parents, children).

3. Implement chunk_structure_aware:
- Dùng regex tách section theo markdown headers (#, ##, ###).
- Mỗi chunk giữ header + nội dung section.
- Metadata có section và strategy=structure.

4. Implement compare_strategies:
- Chạy 4 chiến lược trên từng document.
- Tính stats: num_chunks, avg_length, min_length, max_length.
- Trả dict đủ keys: basic, semantic, hierarchical, structure.

Xác nhận hoàn thành (DoD):
- pytest tests/test_m1.py -v pass.
- Không còn TODO trong phần M1.
- Gửi note handoff: input/output của từng hàm + giả định threshold.

---

## Người 2 - Module M2 Hybrid Search

Phạm vi chính:
- Implement BM25 + Dense + RRF trong src/m2_search.py.

Làm vào đâu:
- src/m2_search.py
  - segment_vietnamese
  - BM25Search.index
  - BM25Search.search
  - DenseSearch.index
  - DenseSearch.search
  - reciprocal_rank_fusion

Làm như thế nào:
1. segment_vietnamese:
- Dùng underthesea.word_tokenize(text, format="text").
- Có fallback khi thiếu package (trả text gốc) để pipeline không vỡ.

2. BM25:
- index: segment + tokenize + build BM25Okapi.
- search: tokenize query, lấy score, sort giảm dần, map ra SearchResult method=bm25.

3. Dense (Qdrant):
- recreate collection với cosine distance và size EMBEDDING_DIM.
- Encode text chunks bằng EMBEDDING_MODEL.
- Upsert payload gồm text + metadata.
- search theo query vector, trả SearchResult method=dense.

4. RRF:
- Tính score theo 1/(k+rank) cho từng danh sách kết quả.
- Gộp theo document text, sort giảm dần.
- Trả method=hybrid.

Xác nhận hoàn thành (DoD):
- pytest tests/test_m2.py -v pass.
- Query chứa "nghỉ phép" trả kết quả liên quan ở top.
- Handoff ghi rõ cấu hình Qdrant cần chạy trước (docker compose up -d).

---

## Người 3 - Module M3 Reranking

Phạm vi chính:
- Implement cross-encoder rerank và benchmark trong src/m3_rerank.py.

Làm vào đâu:
- src/m3_rerank.py
  - CrossEncoderReranker._load_model
  - CrossEncoderReranker.rerank
  - benchmark_reranker
  - (tùy chọn) FlashrankReranker.rerank

Làm như thế nào:
1. _load_model:
- Ưu tiên FlagReranker (BAAI/bge-reranker-v2-m3).
- Nếu lỗi môi trường thì fallback CrossEncoder.

2. rerank:
- Tạo pairs (query, doc_text).
- Predict score.
- Zip score với documents, sort giảm dần.
- Trả list RerankResult đúng rank, đủ original_score + rerank_score.

3. benchmark_reranker:
- Chạy n_runs lần bằng time.perf_counter.
- Trả avg_ms, min_ms, max_ms.

Xác nhận hoàn thành (DoD):
- pytest tests/test_m3.py -v pass.
- Tài liệu về "nghỉ phép" được ưu tiên cao hơn tài liệu không liên quan.
- Benchmark trả số dương hợp lệ.

---

## Người 4 - Module M4 Evaluation

Phạm vi chính:
- Implement RAGAS evaluation + failure analysis trong src/m4_eval.py.

Làm vào đâu:
- src/m4_eval.py
  - evaluate_ragas
  - failure_analysis

Làm như thế nào:
1. evaluate_ragas:
- Tạo Dataset.from_dict từ question/answer/contexts/ground_truth.
- Gọi ragas.evaluate với 4 metrics:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall
- Xuất aggregate score + per_question (danh sách EvalResult).

2. failure_analysis:
- Tính avg score theo từng câu hỏi.
- Sort tăng dần, lấy bottom_n.
- Xác định worst_metric và map diagnosis/suggested_fix theo diagnostic tree.

Xác nhận hoàn thành (DoD):
- pytest tests/test_m4.py -v pass.
- evaluate_ragas trả đủ 4 metrics dạng số.
- failure_analysis có diagnosis và suggested_fix.

---

## Người 5 - Module M5 Enrichment

Phạm vi chính:
- Implement enrichment pipeline trong src/m5_enrichment.py.

Làm vào đâu:
- src/m5_enrichment.py
  - summarize_chunk
  - generate_hypothesis_questions
  - contextual_prepend
  - extract_metadata
  - enrich_chunks

Làm như thế nào:
1. Thiết kế dual-mode:
- Nếu có OPENAI_API_KEY: chạy LLM path.
- Nếu không có key: chạy fallback deterministic để test vẫn pass.

2. summarize_chunk:
- Fallback: lấy 1-2 câu đầu làm tóm tắt.

3. generate_hypothesis_questions:
- LLM generate n câu hỏi.
- Fallback: tạo 1-3 câu hỏi theo mẫu từ topic chính.

4. contextual_prepend:
- Luôn giữ nguyên original text trong output.
- Prepend 1 câu context ngắn có document_title.

5. extract_metadata:
- Trả dict với thông tin cơ bản (topic/entities/category/language), có thể rỗng một phần nhưng phải là dict.

6. enrich_chunks:
- Chạy theo methods.
- Trả list EnrichedChunk, giữ nguyên original_text.

Xác nhận hoàn thành (DoD):
- pytest tests/test_m5.py -v pass.
- Enrichment không làm mất văn bản gốc.
- Có hướng dẫn ngắn cách bật API mode.

---

## Người 6 - Integration + QA + Report Owner

Phạm vi chính:
- Ghép M1-M5 vào pipeline, chạy full lab, chốt output nộp bài.

Làm vào đâu:
- src/pipeline.py (ghép logic retrieval-rerank-eval)
- main.py (luồng baseline -> production -> so sánh)
- analysis/failure_analysis.md
- analysis/group_report.md
- analysis/reflections/reflection_<Ten>.md (nhắc từng người tự nộp)
- reports/ragas_report.json, reports/naive_baseline_report.json (auto-generate)

Làm như thế nào:
1. Chạy và kiểm tra tích hợp:
- python src/pipeline.py
- python main.py
- python check_lab.py

2. Đồng bộ hợp nhất:
- Merge lần lượt M1 -> M2 -> M3 -> M4 -> M5.
- Sau mỗi merge chạy smoke test pipeline.

3. Chốt báo cáo:
- Điền bảng so sánh naive vs production trong analysis/group_report.md.
- Làm bottom-5 failure theo Error Tree trong analysis/failure_analysis.md.
- Thu reflection từ mọi thành viên.

4. Chốt chất lượng trước nộp:
- pytest tests/ -v
- ruff check src/
- grep -r "# TODO" src/m*.py

Xác nhận hoàn thành (DoD):
- python check_lab.py không báo lỗi thiếu file quan trọng.
- Có report JSON, group report, failure analysis, reflection cá nhân.
- Chuẩn bị slide nói 4 ý bắt buộc trong ASSIGNMENT_GROUP.md.

---

## 3) Kế hoạch thời gian 2 giờ (đề xuất cho 6 người)

0-10 phút:
- Cả nhóm setup môi trường, chạy baseline, tạo nhánh cá nhân.

10-65 phút:
- Người 1-5 implement module song song.
- Người 6 chuẩn bị khung báo cáo + checklist tích hợp.

65-85 phút:
- Merge theo thứ tự phụ thuộc: M1 -> M2 -> M3 -> M5 -> M4.
- Mỗi lần merge chạy test liên quan.

85-105 phút:
- Chạy main.py tạo report, so sánh scores.
- Làm failure analysis bottom-5.

105-120 phút:
- Hoàn thiện group_report + reflection.
- Chạy check_lab.py lần cuối và chốt nộp.

---

## 4) Checklist bàn giao giữa các vai

Checklist module owner gửi cho người 6:
- File đã sửa và danh sách hàm đã implement.
- Lệnh test đã chạy và kết quả pass.
- Giả định/fallback đang dùng.
- Điểm có thể ảnh hưởng metric RAGAS.

Checklist người 6 xác nhận trước merge:
- Không phá interface cũ của module.
- Không còn TODO trong module đã merge.
- Pipeline vẫn chạy sau merge.
- Kết quả báo cáo được tạo đúng vị trí thư mục reports.

---

## 5) Lệnh chuẩn cả nhóm dùng

Setup:
- pip install -r requirements.txt
- docker compose up -d

Test từng module:
- pytest tests/test_m1.py -v
- pytest tests/test_m2.py -v
- pytest tests/test_m3.py -v
- pytest tests/test_m4.py -v
- pytest tests/test_m5.py -v

Run toàn bộ:
- python main.py
- python check_lab.py

Kiểm tra quality:
- ruff check src/
- grep -r "# TODO" src/m*.py

---

## 6) Tiêu chí chấm điểm cần bám khi làm việc

Cá nhân (A1-A5):
- Đúng logic module.
- Pass test module.
- Có xử lý đặc thù tiếng Việt khi phù hợp.
- Code sạch, dễ đọc.
- Không còn TODO.

Nhóm (B1-B4):
- Pipeline chạy end-to-end.
- Có ít nhất 1 metric RAGAS đạt 0.75 (càng nhiều càng tốt).
- Failure analysis có diagnosis + suggested fix rõ ràng.
- Presentation đủ 4 ý: score, biggest win, case study, next step.

Bonus:
- Faithfulness >= 0.85.
- Có enrichment tích hợp thật.
- Có latency breakdown.
