import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings

def init_pinecone(index_name="medical-chatbot-index", dimension=768):
    # Get credentials from environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    region = os.getenv("PINECONE_ENV")

    if not api_key or not region:
        raise ValueError("PINECONE_API_KEY or PINECONE_ENV not set in .env file")

    # ✅ New Pinecone client setup
    pc = Pinecone(api_key=api_key)

    # Check if index exists, else create
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # or "gcp" depending on your Pinecone account
                region=region
            )
        )

    return pc.Index(index_name)  # Return the actual index object


def get_retriever(index, namespace="default"):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Use the actual LangChain Pinecone wrapper
    vectorstore = LangchainPinecone(index, embed, namespace=namespace)
    return vectorstore.as_retriever()
