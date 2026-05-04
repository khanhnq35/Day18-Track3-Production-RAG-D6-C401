# 📋 PLAN — Phân Chia Công Việc Lab 18: Production RAG (6 người)

**Lab:** Lab 18 — Production RAG Pipeline  
**Thời gian:** 2 giờ  
**Số thành viên:** 6 người  
**Điểm tối đa:** 100 (60 cá nhân + 40 nhóm + 10 bonus, cap 100)

---

## 1. Tổng Quan Phân Công

```
  P1 ──→ M1a (Semantic + Structure-Aware Chunking)
  P2 ──→ M2  (Hybrid Search: BM25 + Dense + RRF)
  P3 ──→ M3  (Reranking + Latency Benchmark)          ──┐
  P4 ──→ M4  (RAGAS Evaluation + Failure Analysis)      ├──→ pipeline.py → RAGAS → Report
  P5 ──→ M5  (Enrichment Pipeline)                    ──┘
  P6 ──→ M1b (Hierarchical Chunking) + Pipeline + LLM Generation
```

> ⚠️ **Lưu ý Git:** Mỗi người code trên **branch riêng** (vd: `p1-chunking`, `p2-search`, ...). P6 (Integration Lead) merge tất cả vào `main` khi ghép pipeline.

---

## 2. Cấu Trúc Điểm

| Hạng mục | Điểm | Tiêu chí |
|----------|------|----------|
| **A1** Module implementation đúng logic | 15 | Code review: logic đúng, không hardcode |
| **A2** Test pass (`pytest`) | 15 | Auto-grade: số tests pass / tổng |
| **A3** Vietnamese-specific handling | 10 | underthesea / bge-m3 / bge-reranker |
| **A4** Code quality | 10 | Comments, type hints, `ruff check` pass |
| **A5** TODO markers hoàn thành | 10 | `grep "# TODO" src/m*.py` = 0 |
| **B1** Pipeline chạy end-to-end | 10 | `python src/pipeline.py` exit code 0 |
| **B2** RAGAS ≥ 0.75 (any metric) | 10 | ≥ 2 metrics đạt 0.75 = full 10đ |
| **B3** Failure analysis có insight | 10 | Bottom-5 + Error Tree walkthrough |
| **B4** Presentation rõ ràng | 10 | 4 điểm trình bày đầy đủ, có số liệu |
| **Bonus** Faithfulness ≥ 0.85 | +5 | LLM generation + good prompt |
| **Bonus** Enrichment integrated | +3 | Contextual prepend / HyQA |
| **Bonus** Latency breakdown | +2 | Bảng thời gian từng bước |

> ⚠️ **Điểm liệt:** Nếu nhóm KHÔNG có RAGAS evaluation (M4) hoặc KHÔNG có failure analysis → max phần nhóm = **20/40 điểm**.

---

## 3. Chi Tiết Phân Công

### P1 — M1a: Semantic + Structure-Aware Chunking

| Thông tin | Chi tiết |
|-----------|----------|
| **File** | `src/m1_chunking.py` |
| **Branch** | `p1-chunking-semantic` |
| **TODOs** | TODO 1, TODO 3, TODO 4 |
| **Test** | `pytest tests/test_m1.py` |

**Implement:**

| Hàm | Mô tả |
|-----|-------|
| `chunk_semantic()` | Encode sentences bằng `SentenceTransformer("all-MiniLM-L6-v2")`, so sánh cosine similarity liên tiếp, tách chunk mới khi `sim < threshold` |
| `chunk_structure_aware()` | Regex split markdown headers `#{1,3}`, pair header + content, gắn `section` vào metadata |
| `compare_strategies()` | Chạy cả 4 strategies → thu thập stats: `num_chunks`, `avg_length`, `min_length`, `max_length` → in bảng so sánh |

**Test criteria:**
- Semantic: `list[Chunk]` không rỗng, nhóm theo topic
- Structure-Aware: giữ headers, có `section` trong metadata
- `compare_strategies()` trả về stats cho cả 4 strategies

---

### P2 — M2: Hybrid Search (BM25 + Dense + RRF)

| Thông tin | Chi tiết |
|-----------|----------|
| **File** | `src/m2_search.py` |
| **Branch** | `p2-search` |
| **TODOs** | TODO 1–6 (full module) |
| **Test** | `pytest tests/test_m2.py` |

**Implement:**

| Hàm | Mô tả |
|-----|-------|
| `segment_vietnamese()` | `underthesea.word_tokenize(text, format="text")` |
| `BM25Search.index()` | Segment → tokenize → `BM25Okapi` |
| `BM25Search.search()` | Segment query → `get_scores` → top-k |
| `DenseSearch.index()` | Encode bằng `bge-m3` → upload `PointStruct` vào Qdrant |
| `DenseSearch.search()` | Encode query → `client.search()` |
| `reciprocal_rank_fusion()` | Merge rankings: `score = Σ 1/(k + rank)` |

**Vietnamese-specific (A3):** `underthesea` segmentation + `BAAI/bge-m3` embedding

**Dependency:** Qdrant phải chạy (`docker compose up -d`)

---

### P3 — M3: Reranking + Latency Benchmark

| Thông tin | Chi tiết |
|-----------|----------|
| **File** | `src/m3_rerank.py` |
| **Branch** | `p3-rerank` |
| **TODOs** | TODO 1–4 (full module) |
| **Test** | `pytest tests/test_m3.py` |
| **Bonus** | Latency breakdown report (+2đ) |

**Implement:**

| Hàm | Mô tả |
|-----|-------|
| `CrossEncoderReranker._load_model()` | Load `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)` |
| `CrossEncoderReranker.rerank()` | Predict scores → sort descending → top-k `RerankResult` |
| `FlashrankReranker.rerank()` | Lightweight alternative (optional) |
| `benchmark_reranker()` | `time.perf_counter()` × n_runs → `{avg_ms, min_ms, max_ms}` |

**Vietnamese-specific (A3):** `bge-reranker-v2-m3` (multilingual cross-encoder)

**Bonus (+2đ):** Tạo bảng latency breakdown cho từng bước pipeline (chunk → search → rerank → generate)

---

### P4 — M4: RAGAS Evaluation + Failure Analysis

| Thông tin | Chi tiết |
|-----------|----------|
| **File** | `src/m4_eval.py` |
| **Branch** | `p4-eval` |
| **TODOs** | TODO 1–3 (full module) |
| **Test** | `pytest tests/test_m4.py` |
| **Phụ trách thêm** | Điền `analysis/failure_analysis.md` |

> ⚠️ **CRITICAL ROLE** — Nếu M4 không implement → điểm liệt phần nhóm (max 20/40).

**Implement:**

| Hàm | Mô tả |
|-----|-------|
| `evaluate_ragas()` | `Dataset.from_dict` → `ragas.evaluate` với 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) → extract scores |
| `failure_analysis()` | Sort avg score ascending → bottom-N → map Diagnostic Tree |

**Diagnostic mapping:**

| Metric thấp | Diagnosis | Suggested Fix |
|-------------|-----------|---------------|
| `faithfulness < 0.85` | LLM hallucinating | Tighten prompt, lower temperature |
| `context_recall < 0.75` | Missing relevant chunks | Improve chunking or add BM25 |
| `context_precision < 0.75` | Too many irrelevant chunks | Add reranking or metadata filter |
| `answer_relevancy < 0.80` | Answer doesn't match question | Improve prompt template |

**Deliverable:** Điền `analysis/failure_analysis.md` — Bottom-5 failures + Error Tree walkthrough

---

### P5 — M5: Enrichment Pipeline

| Thông tin | Chi tiết |
|-----------|----------|
| **File** | `src/m5_enrichment.py` |
| **Branch** | `p5-enrichment` |
| **TODOs** | TODO 1–5 (full module) |
| **Test** | `pytest tests/test_m5.py` |
| **Bonus** | Enrichment integrated (+3đ) |

**Implement:**

| Hàm | Mô tả |
|-----|-------|
| `summarize_chunk()` | LLM (`gpt-4o-mini`) tóm tắt 2-3 câu. Hoặc extractive fallback: lấy 2 câu đầu |
| `generate_hypothesis_questions()` | LLM generate N câu hỏi chunk có thể trả lời → bridge vocabulary gap |
| `contextual_prepend()` | LLM viết 1 câu context + prepend vào chunk (Anthropic style — giảm 49% retrieval failure) |
| `extract_metadata()` | LLM extract JSON `{topic, entities, category, language}` |
| `enrich_chunks()` | Chạy full pipeline trên list chunks → `list[EnrichedChunk]` |

**API:** Dùng `gpt-4o-mini` (rẻ, nhanh). `OPENAI_API_KEY` đã có sẵn trong `.env`.

**Bonus (+3đ):** Contextual prepend hoặc HyQA integrated vào pipeline

---

### P6 — M1b: Hierarchical Chunking + Pipeline + LLM Generation (Integration Lead)

| Thông tin | Chi tiết |
|-----------|----------|
| **Files** | `src/m1_chunking.py` (TODO 2) + `src/pipeline.py` (TODO LLM) |
| **Branch** | `p6-pipeline` |
| **TODOs** | M1 TODO 2 + Pipeline TODO LLM |
| **Test** | `pytest tests/test_m1.py` (partial) |
| **Bonus** | Faithfulness ≥ 0.85 (+5đ) |
| **Vai trò đặc biệt** | **Integration Lead** — chịu trách nhiệm B1 (pipeline e2e) |

**Implement:**

| Hàm / Task | Mô tả |
|-------------|-------|
| `chunk_hierarchical()` | Parent (2048 chars) → children (256 chars), mỗi child có `parent_id` |
| Pipeline LLM Generation | Replace fallback answer trong `pipeline.py` bằng `gpt-4o-mini` call |
| Pipeline Integration | Merge branches P1–P5 → chạy `python src/pipeline.py` end-to-end |

**LLM Generation prompt (cho Bonus +5đ):**
```python
# System prompt — key cho Faithfulness ≥ 0.85
"Trả lời câu hỏi CHỈ dựa trên context được cung cấp. "
"Nếu context không chứa đủ thông tin → trả lời 'Không tìm thấy thông tin trong tài liệu.'"
"KHÔNG được bịa thêm thông tin ngoài context."
```

**Workflow Integration:**
1. Tạo branch `p6-pipeline` từ `main`
2. Implement `chunk_hierarchical()` trước
3. Khi P1–P5 xong → merge tất cả branches vào `p6-pipeline`
4. Fix integration bugs → test `python src/pipeline.py`
5. Merge `p6-pipeline` vào `main`

---

## 4. Phân Công Phần Nhóm (B1–B4)

| Hạng mục | Người chính | Hỗ trợ |
|----------|-------------|--------|
| **B1** Pipeline end-to-end (10đ) | **P6** | P1–P5 đảm bảo module chạy |
| **B2** RAGAS ≥ 0.75 (10đ) | **P4** | P6 (LLM prompt tuning) |
| **B3** Failure analysis (10đ) | **P4** | P1–P3 (giải thích module mình) |
| **B4** Presentation (10đ) | **Tất cả** | Xem phân chia bên dưới |

### Phân chia Presentation (5 phút ≈ 50 giây/người)

| Thứ tự | Người | Nội dung |
|--------|-------|----------|
| 1 | P6 | Mở đầu: Pipeline architecture overview |
| 2 | P4 | RAGAS scores: naive vs production (bảng so sánh) |
| 3 | P2 hoặc P5 | Biggest win: module nào cải thiện nhiều nhất + tại sao |
| 4 | P4 | Case study: 1 question → Error Tree walkthrough |
| 5 | P3 | Latency breakdown: thời gian từng bước |
| 6 | P1 | Next step: nếu có thêm 1 giờ sẽ optimize gì |

---

## 5. Deliverables

| File | Người chịu trách nhiệm |
|------|------------------------|
| `src/m1_chunking.py` (TODO 1, 3, 4) | P1 |
| `src/m1_chunking.py` (TODO 2) | P6 |
| `src/m2_search.py` | P2 |
| `src/m3_rerank.py` | P3 |
| `src/m4_eval.py` | P4 |
| `src/m5_enrichment.py` | P5 |
| `src/pipeline.py` | P6 |
| `analysis/failure_analysis.md` | P4 (chính) + P6 (hỗ trợ) |
| `analysis/group_report.md` | P6 |
| `analysis/reflections/reflection_[Tên].md` | Mỗi người tự viết |

---

## 6. Timeline (2 giờ)

### Phase 1: Setup (0:00 – 0:15) — TẤT CẢ

```bash
git clone <repo-url> && cd lab18-production-rag
docker compose up -d                    # Qdrant
pip install -r requirements.txt
cp .env.example .env                    # Điền OPENAI_API_KEY
python naive_baseline.py                # Tạo baseline report
```

Mỗi người tạo branch riêng:
```bash
git checkout -b p1-chunking-semantic    # P1
git checkout -b p2-search              # P2
git checkout -b p3-rerank              # P3
git checkout -b p4-eval                # P4
git checkout -b p5-enrichment          # P5
git checkout -b p6-pipeline            # P6
```

### Phase 2: Implement Cá Nhân (0:15 – 1:30)

| Slot | P1 | P2 | P3 | P4 | P5 | P6 |
|------|----|----|----|----|----|----|
| 0:15–0:45 | `chunk_semantic()` | `segment_vietnamese()` + BM25 index/search | `_load_model()` + `rerank()` | `evaluate_ragas()` | `summarize_chunk()` + `generate_hyqa()` | `chunk_hierarchical()` |
| 0:45–1:15 | `chunk_structure_aware()` | Dense index/search + RRF | `FlashrankReranker` + `benchmark` | `failure_analysis()` | `contextual_prepend()` + `extract_metadata()` | Pipeline LLM Generation |
| 1:15–1:30 | `compare_strategies()` → `pytest` | `pytest` | `pytest` | `pytest` | `enrich_chunks()` → `pytest` | Integration test |

**Checkpoint 1:30:** Mỗi người push branch → báo P6.

### Phase 3: Ghép Pipeline (1:30 – 1:45) — P6 Lead

```bash
# P6 thực hiện:
git checkout p6-pipeline
git merge p1-chunking-semantic
git merge p2-search
git merge p3-rerank
git merge p4-eval
git merge p5-enrichment
python src/pipeline.py                  # Test end-to-end
python main.py                          # Tạo full report
```

Fix integration bugs nếu có (cả nhóm hỗ trợ).

### Phase 4: Evaluation & Analysis (1:45 – 2:15)

| Task | Ai làm |
|------|--------|
| Review RAGAS scores | P4 |
| Điền `failure_analysis.md` (bottom-5 + Error Tree) | P4 + tham vấn P1–P3 |
| Điền `group_report.md` | P6 |
| Viết `reflection_[Tên].md` | Mỗi người tự viết |
| `python check_lab.py` | P6 |

### Phase 5: Presentation (2:15 – 2:30)

- 5 phút trình bày theo phân chia ở mục 4
- Push final code lên `main`, nộp link repo

---

## 7. Chiến Lược Tối Đa Điểm

### Tier 1 — Chắc chắn lấy (80đ)

| Hạng mục | Điểm | Cách đạt |
|----------|------|----------|
| A1 Logic đúng | 15 | Implement đúng theo TODO guide có sẵn |
| A2 Test pass | 15 | Chạy `pytest tests/test_m*.py` → fix đến 100% |
| A4 Code quality | 10 | Type hints + Google docstrings + `ruff check` pass |
| A5 TODO done | 10 | `grep -r "# TODO" src/m*.py \| wc -l` = 0 |
| B1 Pipeline e2e | 10 | P6 integration, pipeline có fallback nếu module fail |
| B2 RAGAS ≥ 0.75 | 10 | LLM generation + Enrichment → ≥ 2 metrics đạt 0.75 |
| B3 Failure analysis | 10 | P4 điền đầy đủ bottom-5 + Error Tree |

### Tier 2 — Nên target (90đ)

| Hạng mục | Điểm | Cách đạt |
|----------|------|----------|
| A3 Vietnamese | 10 | `underthesea` (P2) + `bge-m3` (P2) + `bge-reranker` (P3) |
| B4 Presentation | 10 | 6 người × 50 giây, chuẩn bị số liệu cụ thể |

### Tier 3 — Bonus (+10đ → cap 100)

| Bonus | Điểm | Ai làm | Cách đạt |
|-------|------|--------|----------|
| Faithfulness ≥ 0.85 | +5 | P6 | System prompt strict: "CHỈ dựa trên context, KHÔNG bịa" |
| Enrichment integrated | +3 | P5 | Contextual prepend + HyQA trong `enrich_chunks()` |
| Latency breakdown | +2 | P3 | `time.perf_counter()` từng bước: chunk/search/rerank/generate |

---

## 8. Quản Lý Rủi Ro

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|-----------|
| M4 không implement kịp | 🔴 **Điểm liệt** | P4 ưu tiên `evaluate_ragas()` trước, `failure_analysis()` sau. P6 backup |
| Qdrant không chạy | 🟡 | Fallback: chỉ dùng BM25 (pipeline vẫn chạy, Dense trả `[]`) |
| Merge conflict M1 (P1 + P6) | 🟡 | P1 chỉ sửa hàm TODO 1, 3, 4 — P6 chỉ sửa hàm TODO 2. Không overlap code |
| Không đủ thời gian ghép | 🟡 | Pipeline có fallback cho mọi module. Chạy được ngay cả khi 1-2 module dùng basic |
| RAGAS scores thấp | 🟢 | LLM generation (P6) là key. Nếu chưa đạt 0.75 → tune prompt |

---

## 9. Checklist Trước Khi Nộp

- [ ] `python main.py` chạy không lỗi
- [ ] `python check_lab.py` pass
- [ ] `grep -r "# TODO" src/m*.py | wc -l` = 0
- [ ] `ruff check src/` pass
- [ ] `reports/ragas_report.json` tồn tại
- [ ] `reports/naive_baseline_report.json` tồn tại
- [ ] `analysis/failure_analysis.md` đã điền bottom-5
- [ ] `analysis/group_report.md` đã điền đầy đủ
- [ ] 6 file `analysis/reflections/reflection_[Tên].md` đã có
- [ ] Tất cả code đã merge vào `main` và push lên GitHub
