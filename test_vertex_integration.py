import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMBEDDING_PROVIDER, GCP_EMBEDDING_MODEL, GCP_PROJECT_ID
from src.utils import call_llm, get_embeddings


def test_vertex_ai() -> None:
    """Test Vertex AI LLM and text-embedding-004 integration."""
    print("=" * 60)
    print("TESTING GCP VERTEX AI INTEGRATION")
    print("=" * 60)

    print(f"\nProject ID: {GCP_PROJECT_ID}")
    print(f"Embedding Model: {GCP_EMBEDDING_MODEL}")

    # 1. Test LLM
    print("\n--- 1. Testing Gemini LLM ---")
    sys_prompt = "Bạn là trợ lý hữu ích."
    user_prompt = "Chào bạn, hãy giới thiệu ngắn gọn về bản thân."
    answer = call_llm(sys_prompt, user_prompt)
    if answer:
        print(f"Success! Response: {answer[:100]}...")
    else:
        print("Failed to get response from Gemini.")

    # 2. Test Embedding (text-embedding-004 on Vertex)
    print(f"\n--- 2. Testing {GCP_EMBEDDING_MODEL} Embedding on Vertex ---")
    if EMBEDDING_PROVIDER != "google":
        print("Skipping Vertex Embedding test: EMBEDDING_PROVIDER is not 'google'.")
        return

    try:
        texts = ["Học máy là một lĩnh vực của trí tuệ nhân tạo."]
        vectors = get_embeddings(texts)
        if vectors and len(vectors[0]) > 0:
            print(f"Success! Embedding dimension: {len(vectors[0])}")
        else:
            print("Failed to get embeddings.")
    except Exception as exc:
        print(f"Embedding test error: {exc}")


if __name__ == "__main__":
    test_vertex_ai()
