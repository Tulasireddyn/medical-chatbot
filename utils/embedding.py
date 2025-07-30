import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_embeddings():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def embed_fn(texts):
        model = genai.GenerativeModel("embedding-001")  # Gemini embedding model
        return [model.embed_content(content=text, task_type="retrieval_document")["embedding"] for text in texts]

    return embed_fn
