import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LLM_PROVIDER, DEFAULT_LLM, FALLBACK_LLM, GCP_PROJECT_ID, GCP_LOCATION,
                    EMBEDDING_PROVIDER, GCP_EMBEDDING_ENDPOINT_ID, EMBEDDING_TASK, 
                    EMBEDDING_DIM, EMBEDDING_MODEL)

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """
    Hàm gọi LLM chung hỗ trợ cả GCP Vertex AI và OpenAI.
    
    Args:
        system_prompt: Nội dung system prompt.
        user_prompt: Nội dung user prompt.
        temperature: Độ sáng tạo của model (mặc định 0.0).
        
    Returns:
        Câu trả lời từ LLM hoặc chuỗi rỗng nếu có lỗi.
    """
    try:
        if LLM_PROVIDER == "google":
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            # Khởi tạo Vertex AI nếu project_id được cấu hình
            if GCP_PROJECT_ID:
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            
            def generate_gemini(model_name):
                model = GenerativeModel(model_name)
                # Vertex AI không có role system riêng biệt theo cách OpenAI làm trong SDK đơn giản, 
                # gộp chung vào prompt hoặc dùng system_instruction nếu dùng SDK bản mới.
                # Ở đây gộp chung để đảm bảo tính tương thích cao.
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                response = model.generate_content(full_prompt)
                return response.text

            try:
                return generate_gemini(DEFAULT_LLM)
            except Exception as e:
                print(f"Gemini Default Error: {e}, falling back to {FALLBACK_LLM}")
                return generate_gemini(FALLBACK_LLM)
        else:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return resp.choices[0].message.content
    except Exception as e:
        print(f"LLM Error in utils.call_llm: {e}")
        return ""

def get_embeddings(texts: list[str], task: str = None) -> list[list[float]]:
    """
    Lấy embeddings hỗ trợ cả GCP Vertex AI (Jina v3 Endpoint) và Local.
    
    Args:
        texts: Danh sách chuỗi văn bản cần embed.
        task: Loại task (retrieval.query hoặc retrieval.passage). Mặc định lấy từ config.
        
    Returns:
        Danh sách các vector (list of lists of floats).
    """
    task = task or EMBEDDING_TASK
    try:
        if EMBEDDING_PROVIDER == "google":
            import json
            from google.cloud import aiplatform
            
            if GCP_PROJECT_ID:
                aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            
            endpoint = aiplatform.Endpoint(GCP_EMBEDDING_ENDPOINT_ID)
            
            # Payload theo định dạng Jina v3 trên Vertex AI Model Garden
            payload = {
                "model": "jina-embeddings-v3",
                "task": task,
                "dimensions": EMBEDDING_DIM,
                "input": texts
            }
            
            response = endpoint.raw_predict(
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            
            result = json.loads(response.text)
            # Trích xuất vector từ response (định dạng Jina v3: {"data": [{"embedding": [...]}, ...]})
            if "data" in result:
                return [item["embedding"] for item in result["data"]]
            else:
                raise ValueError(f"Unexpected response format: {result}")
        else:
            # Chạy local dùng sentence-transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
            return model.encode(texts).tolist()
    except Exception as e:
        print(f"Embedding Error in utils.get_embeddings: {e}")
        # Trả về list vector 0 nếu lỗi để tránh crash pipeline (tùy chọn)
        return [[0.0] * EMBEDDING_DIM] * len(texts)
