"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    
    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
        
    # 2. Encode sentences
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast
    embeddings = model.encode(sentences)
    
    # 3. Compare consecutive sentences
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b): 
        n_a, n_b = norm(a), norm(b)
        if n_a == 0 or n_b == 0:
            return 0.0
        return dot(a, b) / (n_a * n_b)
        
    # 4. Group sentences
    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunk_meta = {**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            chunks.append(Chunk(text=" ".join(current_group), metadata=chunk_meta))
            current_group = []
        current_group.append(sentences[i])
        
    # Don't forget last group
    if current_group:
        chunk_meta = {**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        chunks.append(Chunk(text=" ".join(current_group), metadata=chunk_meta))
        
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    # TODO: Implement hierarchical chunking
    # 1. Split text into parents:
    #    paragraphs = text.split("\n\n")
    #    Gom paragraphs cho đến khi đạt parent_size → 1 parent chunk
    #    pid = f"parent_{p_index}"
    #    parent = Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
    #
    # 2. Split each parent into children:
    #    Slide window child_size trên parent text
    #    child = Chunk(text=child_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid)
    #
    # 3. Return (parents_list, children_list)
    #
    # Production pattern:
    #   - Index CHILDREN vào vector DB (nhỏ → embedding chính xác)
    #   - Khi retrieve child → lookup parent_id → trả parent cho LLM (đủ context)
    return [], []


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    
    # 1. Split by markdown headers
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    # 2. Pair headers with their content
    chunks = []
    current_header = ""
    current_content = ""
    
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip()
                chunk_meta = {**metadata, "section": current_header, "strategy": "structure", "chunk_index": len(chunks)}
                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
            
    # Don't forget last section
    if current_content.strip() or current_header:
        chunk_text = f"{current_header}\n{current_content}".strip()
        chunk_meta = {**metadata, "section": current_header, "strategy": "structure", "chunk_index": len(chunks)}
        chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
        
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {
        "basic": {"chunks": [], "num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0},
        "semantic": {"chunks": [], "num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0},
        "hierarchical": {"chunks": [], "parents": 0, "children": 0, "num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0},
        "structure": {"chunks": [], "num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    }
    
    # 1. Run strategies
    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})
        
        # basic
        results["basic"]["chunks"].extend(chunk_basic(text, metadata=meta))
        # semantic
        results["semantic"]["chunks"].extend(chunk_semantic(text, metadata=meta))
        # hierarchical
        p, c = chunk_hierarchical(text, metadata=meta)
        results["hierarchical"]["chunks"].extend(c) # use children for stats
        results["hierarchical"]["parents"] += len(p)
        results["hierarchical"]["children"] += len(c)
        # structure
        results["structure"]["chunks"].extend(chunk_structure_aware(text, metadata=meta))
        
    # 2. Collect stats
    for strategy, data in results.items():
        chunks = data.pop("chunks")
        if not chunks:
            continue
        lengths = [len(c.text) for c in chunks]
        data["num_chunks"] = len(chunks)
        data["avg_length"] = int(sum(lengths) / len(lengths))
        data["min_length"] = min(lengths)
        data["max_length"] = max(lengths)
        
    # 3. Print comparison table
    print(f"{'Strategy':<14} | {'Chunks':<8} | {'Avg Len':<7} | {'Min':<5} | {'Max':<5}")
    print("-" * 47)
    for s in ["basic", "semantic", "hierarchical", "structure"]:
        data = results[s]
        if s == "hierarchical":
            chunks_str = f"{data.get('parents', 0)}p/{data.get('children', 0)}c"
        else:
            chunks_str = str(data.get("num_chunks", 0))
        print(f"{s:<14} | {chunks_str:<8} | {data.get('avg_length', 0):<7} | {data.get('min_length', 0):<5} | {data.get('max_length', 0):<5}")
        
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
