import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LLM_PROVIDER, DEFAULT_LLM, FALLBACK_LLM, GCP_PROJECT_ID, GCP_LOCATION,
                    EMBEDDING_PROVIDER, GCP_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
                    EMBEDDING_DIM, EMBEDDING_MODEL)

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """
    Hàm gọi LLM dựa trên LLM_PROVIDER (openai hoặc google).

    Args:
        system_prompt: Nội dung system prompt.
        user_prompt: Nội dung user prompt.
        temperature: Độ sáng tạo của model (mặc định 0.0).

    Returns:
        Câu trả lời từ LLM hoặc chuỗi rỗng nếu có lỗi.
    """
    if LLM_PROVIDER == "openai":
        try:
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
            print(f"OpenAI Error: {e}")
            return ""
    elif LLM_PROVIDER == "google":
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            if GCP_PROJECT_ID:
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

            model = GenerativeModel(DEFAULT_LLM)
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Google Vertex AI Error: {e}")
            return ""
    else:
        print(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
        return ""

def get_embeddings(texts: list[str], task: str = None) -> list[list[float]]:
    """
    Lấy embeddings hỗ trợ OpenAI, GCP Vertex AI (Native) và Local.

    Args:
        texts: Danh sách chuỗi văn bản cần embed.
        task: Không dùng cho native (SDK tự xử lý).

    Returns:
        Danh sách các vector (list of lists of floats).
    """
    try:
        if EMBEDDING_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI()

            # OpenAI hỗ trợ batch lên đến 2048 texts
            all_embeddings = []
            for i in range(0, len(texts), 2048):
                batch = texts[i:i+2048]
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                all_embeddings.extend([data.embedding for data in response.data])
            return all_embeddings
        elif EMBEDDING_PROVIDER == "google":
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            if GCP_PROJECT_ID:
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

            model = TextEmbeddingModel.from_pretrained(GCP_EMBEDDING_MODEL)

            # Giới hạn của Vertex AI thường là 250 instances mỗi request
            all_embeddings = []
            for i in range(0, len(texts), 250):
                batch = texts[i:i+250]
                results = model.get_embeddings(batch)
                all_embeddings.extend([r.values for r in results])
            return all_embeddings
        else:
            # Chạy local dùng sentence-transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
            return model.encode(texts).tolist()
    except Exception as e:
        print(f"Embedding Error in utils.get_embeddings: {e}")
        # Trả về list vector 0 nếu lỗi để tránh crash pipeline (tùy chọn)
        return [[0.0] * EMBEDDING_DIM] * len(texts)
