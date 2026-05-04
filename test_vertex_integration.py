import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import call_llm, get_embeddings
from config import GCP_PROJECT_ID, GCP_EMBEDDING_ENDPOINT_ID

def test_vertex_ai():
    print("=" * 60)
    print("TESTING GCP VERTEX AI INTEGRATION")
    print("=" * 60)
    
    print(f"\nProject ID: {GCP_PROJECT_ID}")
    print(f"Endpoint ID: {GCP_EMBEDDING_ENDPOINT_ID}")
    
    # 1. Test LLM
    print("\n--- 1. Testing Gemini LLM ---")
    sys_prompt = "Bạn là trợ lý hữu ích."
    user_prompt = "Chào bạn, hãy giới thiệu ngắn gọn về bản thân."
    answer = call_llm(sys_prompt, user_prompt)
    if answer:
        print(f"Success! Response: {answer[:100]}...")
    else:
        print("Failed to get response from Gemini.")
        
    # 2. Test Embedding (Jina v3 on Vertex)
    print("\n--- 2. Testing Jina v3 Embedding on Vertex ---")
    if not GCP_EMBEDDING_ENDPOINT_ID:
        print("Skipping Embedding test: GCP_EMBEDDING_ENDPOINT_ID is empty.")
    else:
        try:
            texts = ["Học máy là một lĩnh vực của trí tuệ nhân tạo."]
            vectors = get_embeddings(texts)
            if vectors and len(vectors[0]) > 0:
                print(f"Success! Embedding dimension: {len(vectors[0])}")
            else:
                print("Failed to get embeddings.")
        except Exception as e:
            print(f"Embedding test error: {e}")

if __name__ == "__main__":
    test_vertex_ai()
