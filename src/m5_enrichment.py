"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
from dataclasses import dataclass, field
from openai import OpenAI


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENROUTER_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


# Cấu hình Client OpenRouter (Dùng chung cho toàn bộ module)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk bằng OpenRouter.
    """
    if not text.strip():
        return ""
        
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini", # OpenRouter model ID
            messages=[
                {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "" # Trả về rỗng nếu lỗi để không làm hỏng flow


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    if not text.strip():
        return []
        
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Dựa trên đoạn văn sau, hãy tạo chính xác {n_questions} câu hỏi ngắn gọn mà đoạn văn này có thể trả lời. Mỗi câu hỏi nằm trên một dòng riêng biệt. Không đánh số thứ tự."},
                {"role": "user", "content": text},
            ],
            max_tokens=300
        )
        # Tách các dòng và làm sạch (loại bỏ dòng trống, số thứ tự nếu AI lỡ tay viết vào)
        raw_output = resp.choices[0].message.content.strip()
        questions = [q.strip("- ").strip("123456789. ") for q in raw_output.split("\n") if q.strip()]
        return questions[:n_questions] # Đảm bảo trả về đúng số lượng yêu cầu
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    """
    if not text.strip():
        return text
        
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Viết 1 câu cực ngắn (dưới 20 từ) mô tả đoạn văn này nói về chủ đề gì trong tài liệu gốc. Chỉ trả về 1 câu duy nhất."},
                {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
            ],
            max_tokens=80,
        )
        context = resp.choices[0].message.content.strip()
        return f"[Ngữ cảnh: {context}]\n\n{text}"
    except Exception as e:
        print(f"Error in contextual_prepend: {e}")
        return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if not text.strip():
        return {}
        
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract metadata từ đoạn văn dưới dạng JSON. Các field: topic (chủ đề), entities (danh sách tên người/tổ chức), category (loại tài liệu). Trả về JSON nguyên bản, không dùng markdown block."},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(chunks: list[dict],
                  methods: list[str] | None = None) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks (dạng dict).
    Methods có thể gồm: "summary", "hyqa", "contextual", "metadata", "full"
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]
    
    if "full" in methods:
        methods = ["summary", "hyqa", "contextual", "metadata"]

    enriched = []
    print(f"Enriching {len(chunks)} chunks using {methods}...")
    
    for i, c in enumerate(chunks):
        raw_text = c.get("text", "")
        old_metadata = c.get("metadata", {})
        
        # Các biến lưu kết quả
        enriched_text = raw_text
        summary = ""
        questions = []
        auto_meta = {}

        # 1. Contextual Prepend (Thay đổi text hiển thị)
        if "contextual" in methods:
            doc_title = old_metadata.get("source", "")
            enriched_text = contextual_prepend(raw_text, document_title=doc_title)
            
        # 2. Summarization
        if "summary" in methods:
            summary = summarize_chunk(raw_text)
            
        # 3. HyQA (Câu hỏi giả định)
        if "hyqa" in methods:
            questions = generate_hypothesis_questions(raw_text)
            
        # 4. Auto Metadata
        if "metadata" in methods:
            auto_meta = extract_metadata(raw_text)
            
        # Tạo object EnrichedChunk theo đúng định nghĩa trong file
        enriched.append(EnrichedChunk(
            original_text=raw_text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**old_metadata, **auto_meta},
            method="+".join(methods)
        ))
        
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
