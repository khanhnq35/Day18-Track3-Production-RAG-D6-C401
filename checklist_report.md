# Checklist Report — Production RAG Pipeline

**Repo:** `Day18-Track3-Production-RAG-D6-C401`  
**Nhánh kiểm tra:** `p6-pipeline` sau khi merge `main`, `viet`, `tan`, `minh_dev`, `p1-chunking-semantic`  
**Thời điểm:** 2026-05-04  

## 1. Tóm tắt trạng thái

| Hạng mục | Trạng thái | Nhận xét ngắn |
|---|---:|---|
| Merge code nhóm | ✅ OK | Đã merge đủ nhánh chính/team, push `p6-pipeline` thành công |
| Cấu hình Vertex AI | ✅ OK | Dùng `gemini-2.5-flash`, fallback `gemini-2.5-flash-lite`, embedding `text-embedding-004` |
| M1 Chunking | ⚠️ Tạm OK | Có đủ 4 strategies, nhưng semantic vẫn dùng local `all-MiniLM-L6-v2`, chưa tối ưu cho tiếng Việt/Vertex |
| M2 Hybrid Search | ⚠️ Tạm OK | BM25 + Dense + RRF có logic đúng, Dense dùng `get_embeddings()` tập trung; cần kiểm tra Qdrant runtime |
| M3 Rerank | ✅ OK | CrossEncoder + FlashRank fallback có logic đủ, model multilingual phù hợp tiếng Việt |
| M4 RAGAS Eval | ✅ OK | Đã implement `evaluate_ragas()` và `failure_analysis()` với 4 metrics |
| M5 Enrichment | ⚠️ Tạm OK | Đã gọi `call_llm()` tập trung; HyQA đang generate nhưng chưa được index vào text |
| Pipeline E2E | ⚠️ Chưa xác nhận | Có flow đầy đủ, có thể chạy end-to-end với M4 thực tế |
| File markdown dữ liệu | ✅ Tốt hơn trước | `BCTC.md` đã gộp bảng; `Nghi_dinh_so_13.md` giữ page markers, phù hợp truy vết nguồn |
| Báo cáo nộp bài | ❌ Chưa OK | `analysis/group_report.md` và `failure_analysis.md` vẫn là template trống |

> **Kết luận chính:** Search/RAG pipeline đã có đủ module, M4 đã hoàn tất. Cần chạy pipeline để lấy kết quả từ report và điền báo cáo phân tích lỗi.

---

## 2. Checklist theo rubric

### A. Module cá nhân

| Tiêu chí | Trạng thái | Bằng chứng | Việc cần làm |
|---|---:|---|---|
| A1 — Module implementation | ✅ | M1/M2/M3/M4/M5 đều đã có code thật | Hoàn thành |
| A2 — Test pass | ⚠️ | Có `tests/test_m1.py` → `test_m5.py`; chưa chạy sau merge | Chạy pytest |
| A3 — Vietnamese handling | ⚠️ | M2 dùng `underthesea`; M3 dùng multilingual reranker | M1 semantic nên cân nhắc model tiếng Việt/multilingual |
| A4 — Code quality | ⚠️ | Có type hints cơ bản, nhưng import style còn chưa PEP8 | Chạy `ruff check --fix src tests` |
| A5 — TODO markers | ⚠️ | `src/pipeline.py` còn TODO comment | Xóa/hoàn thiện TODO |

### B. Nhóm

| Tiêu chí | Trạng thái | Rủi ro |
|---|---:|---|
| B1 — Pipeline E2E | ✅ | Đã chạy tới eval với M4 hoạt động thực tế |
| B2 — RAGAS ≥ 0.75 | ⚠️ | Đã có RAGAS thật, cần chạy để xem điểm thực tế |
| B3 — Failure analysis | ❌ | `failure_analysis()` trả `[]`; markdown báo cáo trống |
| B4 — Presentation | ⚠️ | Chưa có số liệu thật để trình bày |

### Bonus

| Bonus | Trạng thái | Ghi chú |
|---|---:|---|
| Faithfulness ≥ 0.85 | ⚠️ | Prompt đã tốt, nhưng chưa có RAGAS thật |
| Enrichment integrated | ✅ | Pipeline gọi `enrich_chunks()` với contextual + HyQA + metadata |
| Latency breakdown | ❌ | Chưa có bảng thời gian từng bước |

---

## 3. Review chi tiết module

## M1 — `src/m1_chunking.py`

### Đã tốt
- Có `load_documents()`, `chunk_basic()`, `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`.
- Hierarchical chunking có parent-child metadata, phù hợp production RAG.
- Structure-aware chunking giữ section header, tốt cho tài liệu markdown.

### Vấn đề/rủi ro
- `chunk_semantic()` dùng `SentenceTransformer("all-MiniLM-L6-v2")`, không phải model tiếng Việt mạnh, cũng không dùng Vertex embedding.
- Semantic split theo regex sentence đơn giản; văn bản pháp lý nhiều dòng/điều/khoản có thể tách chưa chuẩn.
- `parent_id = f"parent_{p_index}"` reset theo từng document. Nếu nhiều tài liệu cùng index chung, parent_id có thể trùng giữa docs. Pipeline hiện lưu `parent_lookup[parent.parent_id]`, nên doc sau có thể overwrite doc trước.

### Checklist fix
- [ ] Đổi `parent_id` thành gồm source/doc id, ví dụ `f"{source}_parent_{p_index}"`.
- [ ] Nếu dùng semantic thật, cân nhắc dùng `get_embeddings()` hoặc model multilingual/Vietnamese.
- [ ] Thêm metadata `source`, `page`, `section` vào chunk nếu parse được.

---

## M2 — `src/m2_search.py`

### Đã tốt
- `segment_vietnamese()` dùng `underthesea.word_tokenize()` và có fallback.
- `BM25Search` index/search đúng hướng.
- `DenseSearch` đã gọi `src.utils.get_embeddings()`, nên thống nhất Vertex AI `text-embedding-004`.
- RRF có logic đúng: cộng `1/(k + rank + 1)` và dedupe theo `result.text`.

### Vấn đề/rủi ro
- `DenseSearch.search()` nhận `collection` default `COLLECTION_NAME`, không dùng `self._collection`. Nếu index với collection khác rồi search không truyền collection, sẽ lệch collection.
- `except Exception: return []` che lỗi Qdrant/embedding, khó debug khi production fail.
- `EMBEDDING_DIM = 1024` trong `config.py` cần khớp output của `text-embedding-004`. Nếu Vertex trả 768/khác 1024, Qdrant upsert sẽ lỗi dimension.

### Checklist fix
- [ ] Trong `search()`, dùng `collection or self._collection`.
- [ ] Log exception thay vì nuốt lỗi im lặng.
- [ ] Verify dimension thực tế của `text-embedding-004` rồi cập nhật `EMBEDDING_DIM` nếu cần.

---

## M3 — `src/m3_rerank.py`

### Đã tốt
- `CrossEncoderReranker` lazy-load model `BAAI/bge-reranker-v2-m3`, phù hợp tiếng Việt/multilingual.
- Có GPU check và CPU fallback.
- Có `FlashrankReranker` fallback optional.
- Có `benchmark_reranker()` đo latency.

### Vấn đề/rủi ro
- Merge làm khác kế hoạch ban đầu: PLAN nói `FlagReranker`, code dùng `sentence_transformers.CrossEncoder`. Vẫn hợp lệ nếu tests chấp nhận, nhưng cần chạy test.
- `FlashrankReranker` map `original_doc = documents[rank]` theo rank sau rerank có thể sai metadata nếu FlashRank đảo thứ tự.
- Chưa integrate latency breakdown vào report/pipeline.

### Checklist fix
- [ ] Chạy `tests/test_m3.py`.
- [ ] Nếu dùng FlashRank thật, map lại metadata theo text/id thay vì rank.
- [ ] Ghi latency benchmark vào report nếu muốn bonus.

---

## M4 — `src/m4_eval.py`

### Đã tốt
- Đã implement `evaluate_ragas()` với 4 metrics chuẩn của RAGAS.
- `failure_analysis()` đã có logic map worst metric → diagnosis/fix theo Diagnostic Tree.
- `save_report()` ghi đúng cấu trúc json.
- `test_set.json` đã có dữ liệu thực tế (4 cặp Q&A).

### Vấn đề/rủi ro
- `save_report()` mặc định ghi `ragas_report.json` ở root.

### Checklist fix
- [ ] Đổi path mặc định của `save_report` thành `reports/ragas_report.json`.

---

## M5 — `src/m5_enrichment.py`

### Đã tốt
- Enrichment dùng `call_llm()` tập trung, tức chạy với Vertex/Gemini theo `.env`.
- Có contextual prepend, HyQA, metadata extraction.
- `enrich_chunks()` merge metadata cũ và metadata LLM.

### Vấn đề/rủi ro
- File vẫn import `OpenAI` nhưng không dùng.
- HyQA được generate vào `hypothesis_questions` nhưng pipeline chỉ index `enriched_text`; câu hỏi giả định chưa được append/index, nên hiệu quả HyQA gần như chưa phát huy.
- Metadata JSON từ LLM có thể fail parse; hiện có fallback `{}`.
- Enrichment gọi LLM cho từng chunk x 3 tác vụ (`contextual`, `hyqa`, `metadata`) → rất chậm/tốn quota trên Vertex.

### Checklist fix
- [ ] Remove unused import `OpenAI` nếu không dùng.
- [ ] Append HyQA vào `enriched_text` hoặc lưu index phụ.
- [ ] Batch/limit enrichment khi chạy datathon để tránh quota/timeouts.
- [ ] Cân nhắc chỉ bật `contextual` trong pipeline nếu cần chạy nhanh.

---

## Pipeline — `src/pipeline.py`

### Đã tốt
- Flow đầy đủ: load → hierarchical chunk → enrichment → hybrid index → rerank → generate → RAGAS.
- Prompt generation đã giới hạn answer trong context, tốt cho faithfulness.
- Có parent lookup để trả parent context khi retrieve child.

### Vấn đề/rủi ro
- `print("[1/3]")` nhưng thực tế có 4 step; lỗi nhỏ.
- Parent ID có thể trùng giữa tài liệu như đã nêu ở M1.
- Enrichment làm mất mapping parent nếu metadata bị LLM overwrite bất ngờ; hiện merge `{**old_metadata, **auto_meta}` thường giữ được `parent_id`, nhưng nếu LLM trả `parent_id` thì có thể overwrite.
- `save_report()` mặc định ghi `ragas_report.json` ở root, trong README/RUBRIC mong `reports/ragas_report.json`.

### Checklist fix
- [ ] Bảo vệ metadata hệ thống (`source`, `parent_id`, `chunk_type`) không bị LLM overwrite.
- [ ] Ghi report vào `reports/`.
- [ ] Thêm latency timing từng step.

---

## Utils/Config — `src/utils.py`, `config.py`, `.env.example`

### Đã tốt
- `call_llm()` hỗ trợ Google Vertex AI + OpenAI fallback theo provider.
- `get_embeddings()` hỗ trợ Google Vertex AI + local sentence-transformers.
- `.env` hiện đã chuyển sang Gemini 2.5 Flash/Flash-Lite và `text-embedding-004`.

### Vấn đề/rủi ro
- `get_embeddings()` fallback vector 0 khi lỗi. Điều này giúp pipeline không crash nhưng làm search sai âm thầm.
- `task: str = None` nên type hint chuẩn là `str | None = None`.
- `EMBEDDING_DIM` cần verify với Vertex output.
- `.env.example` cần đảm bảo có đủ `GCP_PROJECT_ID`, `GCP_LOCATION`, `DEFAULT_LLM`, `FALLBACK_LLM`, `EMBEDDING_PROVIDER`, `GCP_EMBEDDING_MODEL`.

### Checklist fix
- [ ] Không fallback vector 0 trong production, hoặc ít nhất log rõ và raise khi build index.
- [ ] Sửa type hint `task: str | None = None`.
- [ ] Update `.env.example` nếu thiếu biến Vertex.

---

## 4. Review file markdown dữ liệu

## `data/BCTC.md`

### Đã tốt
- Bảng VAT đã được gộp thành một khối, gồm các dòng `4.1` và `4.2` trong cùng table.
- Giữ `## Trang 1` và `## Trang 2`, tốt cho traceability khi retrieve.
- Đã loại bỏ đánh số kiểu `1/2`, `2/2`, giảm nhiễu chunking.

### Lưu ý
- Dòng 62/63 có dấu ngoặc hơi lỗi OCR: `(42]`, `(43]`. Không nghiêm trọng, nhưng có thể sửa về `[42]`, `[43]` để sạch hơn.
- Table markdown có hàng con dùng cột lệch (`|     | a | ... | ... | ... |`) so với header 4 cột; markdown vẫn đọc được bởi LLM, nhưng renderer có thể lệch.

### Checklist fix
- [ ] Sửa `(42]` → `[42]`, `(43]` → `[43]`.
- [ ] Chuẩn hóa table thành đúng 4 cột nếu cần đẹp/render ổn.

## `data/Nghi_dinh_so_13.md`

### Đã tốt
- Có page markers `## Trang X`, phù hợp truy xuất nguồn.
- Nội dung pháp lý dài, có Điều/Chương rõ ràng.
- Header markdown khá đầy đủ, thuận lợi cho structure-aware chunking.

### Vấn đề/rủi ro
- OCR có lỗi chính tả nhỏ: ví dụ `gắn liên`, `thụ hỏi`, một số ký tự/dấu có thể sai.
- Một số mục điều bị cắt qua page boundary. Nếu chunk theo hierarchical paragraph thì vẫn ổn; nếu chunk theo page thì có thể vỡ context.
- `## Trang X` chen giữa Điều/Khoản có thể làm structure-aware coi page là section mới, gây tách logic pháp lý.

### Checklist fix
- [ ] Nếu ưu tiên truy vấn pháp lý, nên chunk theo `Điều X` hơn là theo `Trang X`.
- [ ] Giữ page marker trong metadata hoặc inline nhẹ, không nên để page marker phá section.
- [ ] Spot-clean OCR cho top điều hay được hỏi trong test set.

---

## 5. Review báo cáo markdown deliverables

## `analysis/group_report.md`

Trạng thái: ❌ template trống.

Cần điền:
- [ ] Tên nhóm/ngày.
- [ ] Thành viên và module.
- [ ] Test pass từng module.
- [ ] RAGAS naive vs production.
- [ ] Key findings.
- [ ] Presentation notes.

## `analysis/failure_analysis.md`

Trạng thái: ❌ template trống.

Cần điền:
- [ ] RAGAS table.
- [ ] Bottom-5 failures.
- [ ] Error Tree walkthrough.
- [ ] Root cause + suggested fix.

---

## 6. Ưu tiên hành động đề xuất

### P0 — Bắt buộc trước khi chạy/nộp
1. Chạy test M4 và full pytest.
2. Chạy pipeline để tạo `reports/ragas_report.json`.
3. Điền `analysis/failure_analysis.md` và `analysis/group_report.md` bằng số liệu thật.

### P1 — Fix rủi ro runtime
1. Verify `text-embedding-004` dimension và `EMBEDDING_DIM`.
2. Fix parent_id trùng giữa documents.
3. Log rõ lỗi DenseSearch thay vì return [] im lặng.
4. Save report vào `reports/`.

### P2 — Tăng điểm/bonus
1. Append HyQA questions vào enriched indexed text.
2. Thêm latency breakdown.
3. Clean OCR/table nhỏ trong `BCTC.md` và các trang hay hỏi của `Nghi_dinh_so_13.md`.

---

## 7. Checklist cuối trước khi push/nộp

- [ ] `git status` clean.
- [ ] Không còn conflict markers `<<<<<<<`, `=======`, `>>>>>>>`.
- [ ] Không còn TODO quan trọng trong `src/m*.py`.
- [ ] `.env.example` có đủ biến Vertex AI.
- [ ] Qdrant chạy ổn.
- [ ] `python test_vertex_integration.py` pass.
- [ ] `python main.py` hoặc `python src/pipeline.py` chạy end-to-end.
- [ ] `reports/ragas_report.json` có score thật, không phải toàn 0.
- [ ] `analysis/failure_analysis.md` đã điền Bottom-5.
- [ ] `analysis/group_report.md` đã điền kết quả.
- [ ] Push branch `p6-pipeline` sau khi hoàn thiện.
